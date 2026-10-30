import sys
from pathlib import Path
# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import time
from models.segmentation.spikingresformer import SpikingResformerSeg, SpikingResformerSSD, SpikingResformerFPN
from models.segmentation.Unet import *
from dataset.dataset import SARZarrDataset
from dataset.fast_dataset import fastSARZarrDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from spikingjelly.activation_based import encoding, functional
import datetime
from torch.utils.data import random_split
import math
import torch.nn as nn
import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


parser = argparse.ArgumentParser(description='LIF SAR Training')
parser.add_argument('-model', default='s', type=str, help='type of model image or line')
parser.add_argument('-type', default='image', type=str, help='type of model image or line')
parser.add_argument('-path', default='./anomaly', type=str, help='where to save the results')
parser.add_argument('-epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('-batch', default=64, type=int, help='batch size for training')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-data_dir', default="./../../data/v2/RFInject", type=str, help='where SAR data is stored')
parser.add_argument('-loss', default='CrossEntropy', type=str, help='type of loss function MSE or MAE or FDCC')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('--H', type=int, default=224, help='dimension for images based on energy or firing rate')
parser.add_argument('--W', type=int, default=224, help='dimension for images based on energy or firing rate')
parser.add_argument('-j', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
parser.add_argument('-only_mag', default=False, help='use only magnitude channel')
parser.add_argument('-only_ang', default=False, help='use only angle channel')
parser.add_argument('-fast_dataset', default=True, help='use fast dataset protocol')
parser.add_argument('-decoding', default='temporal', type=str, help='decoding method')
parser.add_argument('-scale', default=2000, type=int, help='scaling for the network output')
parser.add_argument('-initialization', default='None', type=str, help='type of weight initialization')
parser.add_argument('-alpha', default=0.5, type=float, help='weighting factor for MSE and IoU in combined loss')
parser.add_argument('-head', default='standard', help='use standard, SSD, or FPN head instead of simple regression head')
parser.add_argument('-step_mode', default='m', type=str, help='mode for stepping the network during training: m for multi stepping, s for single stepping with spikingjelly')
parser.add_argument('-unet_size', default=1, type=float, help='scaling factor for the UNet architecture')
parser.add_argument('-log', default=False, help='use logarithmic scaling for input data')
parser.add_argument('-normalize_bbox', default=False, help='whether to normalize bbox coordinates by the image dimensions')
parser.add_argument('-optimizer', default='Adam', type=str, help='optimizer to use: Adam or adamW')
parser.add_argument('-fused', default=False, help='use fused optimizers if available (only for AdamW)')
parser.add_argument('-learnable_anchors', default=False, help='whether to use learnable anchors in the detection head')

args = parser.parse_args()
last_rfi = None
last_gt_bbox = None
last_pred_bbox = None 
scale_t = args.scale / args.T

def get_device_and_world_size():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        world_size = 1
        device = torch.device("cpu")
    return device, world_size

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) representing boxes (x1, y1, x2, y2).
        boxes2: Tensor of shape (M, 4) representing boxes (x1, y1, x2, y2).

    Returns:
        IoU matrix of shape (N, M).
    """

    # Extract coordinates
    x11, y11, x12, y12 = boxes1.unbind(dim=-1)
    x21, y21, x22, y22 = boxes2.unbind(dim=-1)

    # Calculate intersection area
    x_left = torch.max(x11[:, None], x21)
    y_top = torch.max(y11[:, None], y21)
    x_right = torch.min(x12[:, None], x22)
    y_bottom = torch.min(y12[:, None], y22)

    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

    # Calculate areas of each box
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    # Calculate union area
    union = area1[:, None] + area2 - intersection

    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)  # Avoid division by zero

    return iou

def visualize_bboxes(rfi_image, gt_bbox, pred_bbox, scale_x=224, scale_y=224, sample_idx=0, rfi_idx=0):
    """
    Visualize ground truth and predicted bounding boxes on the RFI image for a specific sample in the batch.
    Args:
        rfi_image: Tensor of shape (B, C, H, W) batch of input images.
        gt_bbox: Tensor of shape (B, 4) [x1, y1, x2, y2] in original scale for each sample.
        pred_bbox: Tensor of shape (B, 4) [x1, y1, x2, y2] in original scale for each sample.
        scale_x, scale_y: Scaling factors for denormalization.
        sample_idx: Index of the sample in the batch to visualize (default: 0).
    """
    # Convert tensors to numpy and select the specified sample
    if isinstance(rfi_image, torch.Tensor):
        rfi_image = rfi_image.detach().cpu().numpy()
    if isinstance(gt_bbox, torch.Tensor):
        gt_bbox = gt_bbox.detach().cpu().numpy()
    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox = pred_bbox.detach().cpu().numpy()

    # Select the first sample in the batch
    rfi_image = rfi_image[sample_idx]  # Shape: (C, H, W) or (H, W)
    gt_bbox = gt_bbox[sample_idx]      # Shape: (4,)
    pred_bbox = pred_bbox[sample_idx]  # Shape: (4,)

    # Squeeze to remove any extra dimensions (e.g., (1, 4) -> (4,))
    gt_bbox = np.squeeze(gt_bbox)
    pred_bbox = np.squeeze(pred_bbox)
    
    gt_bbox = gt_bbox[rfi_idx]
    pred_bbox = pred_bbox[rfi_idx]

    # Ensure bboxes are 1D arrays of shape (4,)
    assert gt_bbox.ndim == 1 and gt_bbox.shape[0] == 4, f"gt_bbox must be (4,), got {gt_bbox.shape}"
    assert pred_bbox.ndim == 1 and pred_bbox.shape[0] == 4, f"pred_bbox must be (4,), got {pred_bbox.shape}"

    # If image is (C, H, W), take the first channel (magnitude)
    if rfi_image.ndim == 3:
        rfi_image = rfi_image[0]  # Shape: (H, W)
    if rfi_image.ndim == 2:
        rfi_image = rfi_image.T  # Transpose to (W, H) for plotting

    # Normalize image for visualization
    rfi_image = (rfi_image - rfi_image.min()) / (rfi_image.max() - rfi_image.min() + 1e-8)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(rfi_image, cmap='viridis', aspect='auto')

    # Extract scalar values for bbox coordinates (x1, y1, x2, y2)
    gt_x1, gt_y1, gt_x2, gt_y2 = map(float, gt_bbox)
    pred_x1, pred_y1, pred_x2, pred_y2 = map(float, pred_bbox)
    
    print("gt:", gt_x1, gt_y1, gt_x2, gt_y2)
    print("pred:", pred_x1, pred_y1, pred_x2, pred_y2)

    gt_x1 *= scale_x
    gt_x2 *= scale_x
    gt_y1 *= scale_y
    gt_y2 *= scale_y
    pred_x1 *= scale_x
    pred_x2 *= scale_x
    pred_y1 *= scale_y
    pred_y2 *= scale_y
    

    # Compute width and height for Rectangle patch
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    pred_width = pred_x2 - pred_x1
    pred_height = pred_y2 - pred_y1

    # Draw ground truth bbox (red)
    gt_rect = patches.Rectangle(
        (gt_x1, gt_y1), gt_width, gt_height,
        linewidth=2, edgecolor='r', facecolor='none', label='Ground Truth'
    )
    ax.add_patch(gt_rect)

    # Draw predicted bbox (blue)
    pred_rect = patches.Rectangle(
        (pred_x1, pred_y1), pred_width, pred_height,
        linewidth=2, edgecolor='b', facecolor='none', label='Predicted'
    )
    ax.add_patch(pred_rect)

    ax.legend()
    ax.set_title(f"Ground Truth (Red) vs Predicted (Blue) Bounding Box (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join('./', 'last_bbox_visualization.png'))
    plt.show(block=True)
    
# --- Global variables to store the last batch's data for visualization ---
last_rfi = None
last_gt_bbox = None
last_pred_bbox = None

def sigint_handler(signum, frame):
    """
    Handler for SIGINT (Ctrl+C). Visualizes the last seen ground truth and predicted bbox.
    """
    global last_rfi, last_gt_bbox, last_pred_bbox
    if last_rfi is not None and last_gt_bbox is not None and last_pred_bbox is not None:
        print("\nCtrl+C detected. Visualizing last ground truth and predicted bbox...")
        visualize_bboxes(last_rfi, last_gt_bbox, last_pred_bbox)
    sys.exit(0)  # Exit after visualization

def train (rank, world_size, num_epochs, training_metrics_dict, validation_metrics_dict, training_loader,
           validation_loader, net, loss_fn, optimizer, plateau_scheduler, training_path, resume, multigpu):
    
    global last_rfi, last_gt_bbox, last_pred_bbox

    # --- Register SIGINT handler ---
    signal.signal(signal.SIGINT, sigint_handler)
    
    if multigpu:
        setup(rank, world_size)
        net = DDP(net, device_ids=[rank])
    net = net.to(rank)
    
    start_epoch = 0
    max_test_loss = -1
    min_test_loss = float('inf')
    encoder = encoding.PoissonEncoder()

    if resume:
        checkpoint = torch.load(args.resume, map_location=get_device_and_world_size()[0])
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_loss = checkpoint['max_test_loss']
        if 'min_test_loss' in checkpoint:
            min_test_loss = checkpoint['min_test_loss']
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch}_lr{args.lr}_type{args.type}_only_mag{args.only_mag}_only_ang{args.only_ang}_daytime{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    os.makedirs(out_dir, exist_ok=True)
    print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    scale_x = 22542.463211026985
    scale_y = 1418.2935352358766
    
    if args.only_mag or args.only_ang:
        in_channels = 1
    else:
        in_channels = 2
    print("start of traning")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_iou = 0
        train_samples = 0
        train_iou = 0
        prova_iou = 0
        train_boxes = 0
        for echo, expanded_RFI, bbox, _ in training_loader:
            #try:
            RFI = expanded_RFI.view(-1, in_channels, args.H, args.W)
            if args.only_mag:
                expanded_RFI = expanded_RFI[:, 0:1, :, :]
            if args.only_ang:
                expanded_RFI = expanded_RFI[:, 1:2, :, :]

            RFI_split = torch.split(RFI, echo.shape[0], dim=0)
            RFI = RFI_split[0].to(rank, non_blocking=True)
            encoded_img = encoder(RFI)
            bbox = bbox.to(rank, non_blocking=True)

            # --- Store the last batch's RFI and ground truth bbox ---
            last_rfi = RFI.cpu()
            last_gt_bbox = bbox.cpu()

            if args.type == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))

                # if args.step_mode == 's':
                #     if args.decoding in ['rate', 'sum']:
                #         output_sum = torch.zeros(encoded_img.shape[0], 10, 4, device=encoded_img.device)
                #     else:
                #         all_outputs = []

                #     for t in range(args.T):
                #         loc_preds_step = net(encoded_img)
                #         if args.decoding in ['rate', 'sum']:
                #             output_sum += loc_preds_step
                #             loc_preds_scaled = loc_preds_step * (scale_t)
                #             loc_preds_flat = loc_preds_scaled.view(-1, 4)
                #             if not args.normalize_bbox:
                #                 x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                #                 x11_normalized = x11 / scale_x
                #                 x12_normalized = x12 / scale_x
                #                 y11_normalized = y11 / scale_y
                #                 y12_normalized = y12 / scale_y
                #                 boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
                #             else:
                #                 boxes_normalized = bbox.view(-1, bbox.shape[-1])
                #             loss = loss_fn(loc_preds_flat, boxes_normalized) / args.T
                #             loss.backward()
                #             optimizer.zero_grad()
                #             functional.reset_net(net)
                #         else:
                #             all_outputs.append(loc_preds_step)

                #     # --- Final decoding ---
                #     if args.decoding == 'rate':
                #         x = output_sum / args.T
                #         x_scaled = x * args.scale
                #     elif args.decoding == 'sum':
                #         x = output_sum
                #         x_scaled = x * scale_t
                #     else:  # temporal
                #         x = torch.stack(all_outputs, dim=0)
                #         probs = x.softmax(dim=0)
                #         x = (probs * torch.arange(x.shape[0], device=x.device, dtype=torch.float32)).sum(dim=0)
                #         x_scaled = x * scale_t

                #     if not args.decoding in ['rate', 'sum']:
                #         x_scaled_ = x_scaled.view(-1, x_scaled.shape[-1])
                #         if not args.normalize_bbox:
                #             x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                #             x11_normalized = x11 / scale_x
                #             x12_normalized = x12 / scale_x
                #             y11_normalized = y11 / scale_y
                #             y12_normalized = y12 / scale_y
                #             boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
                #         else:
                #             boxes_normalized = bbox.view(-1, bbox.shape[-1])
                            
                #         loss = loss_fn(x_scaled_, boxes_normalized)
                #         loss.backward()
                #         optimizer.step()

                #     # --- Store the last predicted bbox (for visualization) ---
                #     x = x.view(-1, bbox.shape[-1])
                #     x1, y1, x2, y2 = x.unbind(dim=-1)
                #     x1_scaled = x1 * scale_x
                #     x2_scaled = x2 * scale_x
                #     y1_scaled = y1 * scale_y
                #     y2_scaled = y2 * scale_y
                #     last_pred_bbox = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dim=-1).cpu()

                #     train_boxes += bbox.shape[0]
                #     train_samples += encoded_img.numel()
                #     train_loss += loss.item() * encoded_img.numel()
                #     iou = calculate_iou(x, boxes_normalized)
                #     numiou = iou.numel()
                #     mean_iou_t = iou.mean().item()
                #     train_iou += mean_iou_t * numiou
                #     train_boxes += numiou

                # else:
                # --- Non-step_mode branch ---
            x = net(encoded_img)


            if args.decoding == 'rate':
                x = torch.mean(x, dim=0)
                x_scaled = x * (scale_t)
                last_pred_bbox = x.cpu()

            elif args.decoding == 'temporal':
                probs = x.softmax(dim=0)
                indices = torch.arange(x.shape[0], device=x.device, dtype=torch.float32)
                indices = indices.view(-1, *([1] * (x.dim() - 1)))
                x = (probs * indices).sum(dim=0)
                last_pred_bbox = x.cpu()
            elif args.decoding == 'sum':
                x = torch.sum(x, dim=0)
                x_scaled = x * (scale_t)
                last_pred_bbox = x.cpu()

            if not args.normalize_bbox:
                x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                x11_normalized = x11 / scale_x
                x12_normalized = x12 / scale_x
                y11_normalized = y11 / scale_y
                y12_normalized = y12 / scale_y
                boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
            else:
                boxes_normalized = bbox.view(-1, bbox.shape[-1])
            x = x.view(-1, x.shape[-1])
            bbox = bbox.view(-1, bbox.shape[-1])
                
            loss = loss_fn(x, boxes_normalized)
            loss.backward()
            optimizer.step()

            train_boxes += bbox.shape[0]
            train_samples += encoded_img.numel()
            train_loss += loss.item() * encoded_img.numel()
            iou = calculate_iou(x, boxes_normalized)
            numiou = iou.numel()
            mean_iou_t = iou.mean().item()
            train_iou += mean_iou_t * numiou
            train_boxes += numiou

            functional.reset_net(net)

            #except KeyboardInterrupt:
                # Let the signal handler take care of visualization
            #    raise
        plateau_scheduler.step(train_loss / train_samples)
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_iou /= train_boxes
        prova_iou /= train_boxes
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_iou', train_iou, epoch)

        net.eval()
        test_loss = 0
        test_iou = 0
        test_samples = 0
        test_iou = 0
        full_test_iou = 0
        test_boxes = 0
        with torch.no_grad():
            for echo, expanded_RFI, bbox, _  in validation_loader:
                RFI = expanded_RFI.view(-1, in_channels, args.H, args.W)
                if args.only_mag:
                    expanded_RFI = expanded_RFI[:,0:1,:,:]         
                    
                if args.only_ang:
                    expanded_RFI = expanded_RFI[:,1:2,:,:]
                
                RFI_split = torch.split(RFI, echo.shape[0], dim=0)
                RFI = RFI_split[0].to(rank, non_blocking=True)
                encoded_img = encoder(RFI)

                bbox = bbox.to(rank, non_blocking=True)

                if args.type == 'line':
                    encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))
                    
                    
                if args.step_mode == 's':
    

                    # Initialize accumulators
                    if args.decoding in ['rate', 'sum']:
                        output_sum = torch.zeros(encoded_img.shape[0], 10, 4, device=encoded_img.device)
                    else:  # temporal
                        all_outputs = []

                    # First pass: Compute outputs (no backward)
                    for t in range(args.T):
                        loc_preds_step = net(encoded_img)  # NO reset_net (Option 1)
                        # functional.reset_net(net)  # REMOVE THIS LINE (or use Option 2 above)

                        if args.decoding in ['rate', 'sum']:
                            # Accumulate outputs for rate/sum
                            output_sum += loc_preds_step

                            # For rate/sum: Compute loss per step and accumulate gradients
                            loc_preds_scaled = loc_preds_step * (scale_t)
                            loc_preds_flat = loc_preds_scaled.view(-1, 4)
                            
                            if not args.normalize_bbox:
                                x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                                x11_normalized = x11 / scale_x
                                x12_normalized = x12 / scale_x
                                y11_normalized = y11 / scale_y
                                y12_normalized = y12 / scale_y
                                boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
                            else:
                                boxes_normalized = bbox.view(-1, bbox.shape[-1])                            
                                    
                            loss = loss_fn(loc_preds_flat, boxes_normalized) / args.T
                        else:  # temporal
                            all_outputs.append(loc_preds_step)  # Store for later decoding
                    # Second pass: Compute final output and loss
                    if args.decoding == 'rate':
                        x = output_sum / args.T
                        x_scaled = x * args.scale
                    elif args.decoding == 'sum':
                        x = output_sum
                        x_scaled = x * scale_t
                    else:  # temporal
                        x = torch.stack(all_outputs, dim=0)
                        probs = x.softmax(dim=0)
                        x = (probs * torch.arange(x.shape[0], device=x.device, dtype=torch.float32)).sum(dim=0)
                        x_scaled = x * scale_t
                    if not args.decoding in ['rate', 'sum']:
                        # Flatten for loss computation
                        x_scaled_ = x_scaled.view(-1, x_scaled.shape[-1])
                        if not args.normalize_bbox:
                            x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                            x11_normalized = x11 / scale_x
                            x12_normalized = x12 / scale_x
                            y11_normalized = y11 / scale_y
                            y12_normalized = y12 / scale_y
                            boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
                        else:
                            boxes_normalized = bbox.view(-1, bbox.shape[-1])
                        
                        loss = loss_fn(x_scaled_, boxes_normalized)

                    # --- Final Decoding ---
                    if args.decoding == 'temporal':
                        # Stack all outputs and apply temporal decoding
                        x = torch.stack(all_outputs, dim=0)  # [T, B, 10, 4]
                        probs = x.softmax(dim=0)
                        x = (probs * torch.arange(x.shape[0], device=x.device, dtype=torch.float32)).sum(dim=0)
                        x_scaled = x * scale_t
                    elif args.decoding == 'rate':
                        x = output_sum / args.T  # Mean over T
                        x_scaled = x * args.scale
                    elif args.decoding == 'sum': 
                        x = output_sum  # Sum over T
                        x_scaled = x * scale_t
                        
                    
                    test_boxes += bbox.shape[0]
                    test_samples += encoded_img.numel()
                    test_loss += loss.item() * encoded_img.numel()
                    #print(f" x_scaled_ max: {x_scaled_.max()}, min: {x_scaled_.min()}, bbox max: {bbox.max()}, min: {bbox.min()}")
                    x = x.view(-1, bbox.shape[-1])
                    
                    x1, y1, x2, y2 = x.unbind(dim=-1)
                    x1_scaled = x1 * scale_x 
                    x2_scaled = x2 * scale_x 
                    y1_scaled = y1 * scale_y
                    y2_scaled = y2 * scale_y
                    x_scaled_ = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dim=-1)
                    
                    bbox = bbox.view(-1, bbox.shape[-1])
                    #print(f"x shape: {x.shape}, boxes_normalized shape: {boxes_normalized.shape}")
                    #print(f"x_scaled_ shape: {x_scaled_.shape}, bbox shape: {bbox.shape}")
                    iou = calculate_iou(x, boxes_normalized)
                    iou_ = 0 #calculate_iou(x_scaled_, bbox)    
                    
                    numiou = iou.numel() 
                    
                    mean_iou_t = iou.mean().item()  # Mean IoU for the batch
                    test_iou += mean_iou_t * numiou  # Sum IoU for all box pairs
                    #mean_iou_tt = iou_.mean().item()  # Mean IoU for the batch
                    #full_test_iou += mean_iou_tt * numiou
                    train_boxes += numiou        # Count box pairs
                    
                else:
                    x = net(encoded_img)
                
                    if args.decoding == 'rate':
                        x = torch.mean(x, dim=0)  
                        x_scaled = x * (scale_t)  # Scale
                        
                        x = x.view(-1, x.shape[-1])
                        x1, y1, x2, y2 = x.unbind(dim=-1)
                        
                        
                        x1_scaled = x1 * scale_x 
                        x2_scaled = x2 * scale_x 
                        y1_scaled = y1 * scale_y
                        y2_scaled = y2 * scale_y
                        
                        #print(f"x1_scaled max: {x1_scaled.max()}, min: {x1_scaled.min()}, x2_scaled max: {x2_scaled.max()}, min: {x2_scaled.min()}, y1_scaled max: {y1_scaled.max()}, min: {y1_scaled.min()}, y2_scaled max: {y2_scaled.max()}, min: {y2_scaled.min()}")
                        
                        x_scaled_ = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dim=-1)
                        
                    elif args.decoding == 'temporal':
                        # Use STE to approximate argmax

                        probs = x.softmax(dim=0)
                        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.float32)
                        # Reshape indices to match the shape of probs for broadcasting
                        indices = indices.view(-1, *([1] * (x.dim() - 1)))
                        x = (probs * indices).sum(dim=0)
                        x_scaled = x * scale_t
                        
                        x = x.view(-1, x.shape[-1]) 
                        x1, y1, x2, y2 = x.unbind(dim=-1)
                        x1_scaled = x1 * scale_x 
                        x2_scaled = x2 * scale_x 
                        y1_scaled = y1 * scale_y
                        y2_scaled = y2 * scale_y
                        x_scaled_ = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dim=-1)
                    elif args.decoding == 'sum':
                        x = torch.sum(x, dim=0)  
                        x_scaled = x * (scale_t)  # Scale
                        
                        x_scaled_ = x.view(-1, x.shape[-1]) 
                        x1, y1, x2, y2 = x.unbind(dim=-1)
                        x1_scaled = x1 * scale_x 
                        x2_scaled = x2 * scale_x 
                        y1_scaled = y1 * scale_y
                        y2_scaled = y2 * scale_y
                        x_scaled_ = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dim=-1)

                    if not args.normalize_bbox:
                        x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                        x11_normalized = x11 / scale_x
                        x12_normalized = x12 / scale_x
                        y11_normalized = y11 / scale_y
                        y12_normalized = y12 / scale_y
                        boxes_normalized = torch.stack([x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1)
                    else:
                        boxes_normalized = bbox.view(-1, bbox.shape[-1])
                    
                    normal_bbox = bbox / torch.max(bbox)
                    
                    #print(f"x: {x.shape} , boxes_normalized: {boxes_normalized.shape}")
                    #print(f"x shape: {x.shape}, min: {x.min()}, max: {x.max()}")
                    #print(f"bbox shape: {bbox.shape}, min: {bbox.min()}, max: {bbox.max()}")
                    
                    x = x.view(-1, x.shape[-1])
                    bbox = bbox.view(-1, bbox.shape[-1])
                    
                    loss = loss_fn(x, boxes_normalized)

                    test_boxes += bbox.shape[0]
                    test_samples += encoded_img.numel()
                    test_loss += loss.item() * encoded_img.numel()
                    #print(f" x_scaled_ max: {x_scaled_.max()}, min: {x_scaled_.min()}, bbox max: {bbox.max()}, min: {bbox.min()}")
                    #print(f"x_scaled shape")
                    iou = calculate_iou(x, boxes_normalized)
                    iou_ = 0 #calculate_iou(x_scaled_, bbox)
                
                    numiou = iou.numel() 
                    
                    mean_iou_t = iou.mean().item()  # Mean IoU for the batch
                    test_iou += mean_iou_t * numiou  # Sum IoU for all box pairs
                    #mean_iou_tt = iou_.mean().item()  # Mean IoU for the batch
                    #full_test_iou += mean_iou_tt * numiou
                    test_boxes += numiou        # Count box pairs

                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        
        full_test_iou /= test_boxes
        test_iou /= test_boxes
        writer.add_scalar('test_loss', test_loss, epoch)
    
        writer.add_scalar('test_iou', test_iou, epoch)


        save_max = False
        if test_loss > max_test_loss:
            max_test_loss = test_loss
            save_max = True
        
        save_min = False
        if test_loss < min_test_loss:
            min_test_loss = test_loss  
            save_min = True 

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_loss': max_test_loss,
            'min_test_loss': min_test_loss
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pt'))
        if save_min:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_min.pt'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pt'))

        print(args)
        print(out_dir)
        print(f"epoch ={epoch+1}: train_loss ={train_loss: .4f}, full_scale_iou ={prova_iou: .4f}, train_iou ={train_iou: .4f}, test_loss ={test_loss: .4f}, full_scale_test_iou ={full_test_iou: .4f} test_iou ={test_iou: .4f}, max_test_loss ={max_test_loss: .4f}, min_test_loss ={min_test_loss: .4f}")
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epoch - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        with open(os.path.join(out_dir, 'training_log.txt'), 'a', encoding='utf-8') as log_file:
            log_file.write(f'epoch ={epoch}, train_loss ={train_loss: .4f}, full_scale_iou ={prova_iou: .4f}, train_iou ={train_iou: .4f}, test_loss ={test_loss: .4f}, full_scale_test_iou ={full_test_iou: .4f}, test_iou ={test_iou: .4f}, max_test_loss ={max_test_loss: .4f}, min_test_loss ={min_test_loss: .4f}\n')
            log_file.write(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s\n')
            log_file.write(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epoch - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n\n')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def init_snn_weights(module, k=0.8, lambda_=0.2, theta=1.0):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        std = math.sqrt(2 * theta**2 / (lambda_**2 * n_in))
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, (theta * (1 - k)) / (2 * lambda_))


def init_weight(module):

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def ciou_loss(pred, target, eps=1e-6):
    # IoU
    iou = calculate_iou(pred.view(-1, pred.shape[-1]), target.view(-1, target.shape[-1]))
    #print(f"pred shape: {pred.shape}, target shape: {target.shape}, iou shape: {iou.shape}")
    # Box centers
    pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
    pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
    target_ctr_x = (target[..., 0] + target[..., 2]) / 2
    target_ctr_y = (target[..., 1] + target[..., 3]) / 2

    # Distance between centers
    ctr_distance = (pred_ctr_x.unsqueeze(1) - target_ctr_x)**2 + (pred_ctr_y.unsqueeze(1) - target_ctr_y)**2

    # Enclosing box diagonal
    xc1 = torch.min(pred[..., 0].unsqueeze(1), target[..., 0])
    yc1 = torch.min(pred[..., 1].unsqueeze(1), target[..., 1])
    xc2 = torch.max(pred[..., 2].unsqueeze(1), target[..., 2])
    yc2 = torch.max(pred[..., 3].unsqueeze(1), target[..., 3])
    enclosing_diag = (xc2 - xc1).clamp(min=1e-6)**2 + (yc2 - yc1).clamp(min=1e-6)**2

    # Width/height
    pred_w = (pred[..., 2] - pred[..., 0]).clamp(min=1e-6)
    pred_h = (pred[..., 3] - pred[..., 1]).clamp(min=1e-6)
    target_w = (target[..., 2] - target[..., 0]).clamp(min=1e-6)
    target_h = (target[..., 3] - target[..., 1]).clamp(min=1e-6)

    # Aspect ratio
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(pred_w.unsqueeze(1) / pred_h.unsqueeze(1)) -
        torch.atan(target_w / target_h), 2
    )
    alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - (ctr_distance / enclosing_diag + alpha * v)
    return 1 - ciou.mean()


def diou_loss(pred, target, eps=1e-6):
    # IoU
    iou = calculate_iou(pred.view(-1, pred.shape[-1]), target.view(-1, target.shape[-1]))

    # Box centers
    pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
    pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
    target_ctr_x = (target[..., 0] + target[..., 2]) / 2
    target_ctr_y = (target[..., 1] + target[..., 3]) / 2

    # Distance between centers
    ctr_distance = (pred_ctr_x.unsqueeze(1) - target_ctr_x)**2 + (pred_ctr_y.unsqueeze(1) - target_ctr_y)**2

    # Enclosing box diagonal
    xc1 = torch.min(pred[..., 0].unsqueeze(1), target[..., 0])
    yc1 = torch.min(pred[..., 1].unsqueeze(1), target[..., 1])
    xc2 = torch.max(pred[..., 2].unsqueeze(1), target[..., 2])
    yc2 = torch.max(pred[..., 3].unsqueeze(1), target[..., 3])
    enclosing_diag = (xc2 - xc1).clamp(min=1e-6)**2 + (yc2 - yc1).clamp(min=1e-6)**2

    # DIoU
    diou = iou - ctr_distance / enclosing_diag
    return 1 - diou.mean()

def MSE_ciou_loss(pred, target, alpha=0.5):
    mse_loss = nn.MSELoss()(pred, target)
    ciou = ciou_loss(pred, target)
    return alpha * mse_loss + (1 - alpha) * ciou

def MSE_diou_loss(pred, target, alpha=0.5):
    mse_loss = nn.MSELoss()(pred, target)
    diou = diou_loss(pred, target)
    return alpha * mse_loss + (1 - alpha) * diou

if __name__ == "__main__":


    print(sys.path)
    
    device, world_size = get_device_and_world_size()
    if args.only_mag or args.only_ang:
        in_channels = 1
    elif args.log:
        in_channels = 3
    else:
        in_channels = 2  
    
    
    if args.head == 'ssd':
        if args.model == 's':
            model = SpikingResformerSSD(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 256, 512],
            [1, 4, 8],
            [2, 1, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'xs':
            model = SpikingResformerSSD(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2, ],
            [64, 256],
            [1, 4],
            [2, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'xxs':
            model = SpikingResformerSSD(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 1, ],
            [32, 64],
            [1, 2],
            [2, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'm':
            model = SpikingResformerSSD(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 384, 768],
            [1, 6, 12],
            [2, 1, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'unet':
            model = Unet(
            in_channels = in_channels,
            step_mode= args.step_mode,
            T= args.T,
            head_type= 'ssd',
            model_size=args.unet_size,
            learnable_anchors=args.learnable_anchors
        )
    elif args.head == 'fpn':
        if args.model == 's':
            model = SpikingResformerFPN(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 256, 512],
            [1, 4, 8],
            [2, 1, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'xs':
            model = SpikingResformerFPN(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2, ],
            [64, 256],
            [1, 4],
            [2, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'm':
            model = SpikingResformerFPN(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 384, 768],
            [1, 6, 12],
            [2, 1, 1],
            in_channels=in_channels,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'unet':
            model = Unet(
            in_channels = in_channels,
            step_mode= args.step_mode,
            T= args.T,
            head_type= 'fpn',
            model_size=args.unet_size,
            learnable_anchors=args.learnable_anchors
        )
    elif args.head == 'standard':
        if args.model == 's':
            model = SpikingResformerSeg(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 256, 512],
            [1, 4, 8],
            [2, 1, 1],
            in_channels=in_channels,
            num_classes=10,
            T=args.T,
            decoding=args.decoding
            )
        elif args.model == 'xs':
            model = SpikingResformerSeg(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2, ],
            [64, 256],
            [1, 4],
            [2, 1],
            in_channels=in_channels,
            num_classes=10,
            T=args.T,
            step_mode=args.step_mode,
            decoding=args.decoding
            )
        elif args.model == 'm':
            model = SpikingResformerSeg(
            [
                ['DSSA', 'GWFFN'] * 1,
                ['DSSA', 'GWFFN'] * 2,
                ['DSSA', 'GWFFN'] * 3, ],
            [64, 384, 768],
            [1, 6, 12],
            [2, 1, 1],
            in_channels=in_channels,
            num_classes=10,
            T=args.T,
            decoding=args.decoding
            )
    
    if args.initialization == 'snn':
        model.apply(init_snn_weights)
    elif args.initialization == 'standard':
        model.apply(init_weight)
    
    if args.fast_dataset:
        print(f"fast dataset with shape {(args.H,args.W)}")
        dataset = fastSARZarrDataset(root_dir=args.data_dir, target_shape=(args.H,args.W), device='cuda', log=args.log, fast=True, normalize_bbox=args.normalize_bbox)
    else:
        dataset = SARZarrDataset(root_dir=args.data_dir, target_shape=(args.H,args.W), log=args.log, device='cuda')



    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.j, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=args.j, pin_memory=True)
    if args.loss == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif args.loss == 'MAE':
        loss_fn = torch.nn.L1Loss()
    elif args.loss == 'CIoU':
        loss_fn = ciou_loss
    elif args.loss == 'DIoU':
        loss_fn = diou_loss
    elif args.loss == 'MSE_CIoU':
        loss_fn = lambda pred, target: MSE_ciou_loss(pred, target, alpha=args.alpha)
    elif args.loss == 'MSE_DIoU':
        loss_fn = lambda pred, target: MSE_diou_loss(pred, target, alpha=args.alpha)
        
    training_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    validation_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=args.fused)

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print(args._get_args())
    
    cfg = {}
    print(args)
#    train(num_epochs=args.epoch, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
#          training_loader=training_loader, validation_loader=validation_loader, net=model, loss_fn=loss_fn,
#          optimizer=optimizer_1d, plateau_scheduler=plateau_scheduler_1d, training_path=args.path, resume=False)
    epoch = args.epoch
    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            args=(args.epoch, model, training_metrics_dict, validation_metrics_dict, training_loader, validation_loader, model, loss_fn, optimizer, plateau_scheduler, args.path, False, True),
            nprocs=world_size,
            join=True
            )
        cleanup()
    else:
        train(rank='cuda:0', world_size=1, num_epochs=args.epoch, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
            training_loader=training_loader, validation_loader=validation_loader, net=model, loss_fn=loss_fn,
            optimizer=optimizer, plateau_scheduler=plateau_scheduler, training_path=args.path, resume=False, multigpu=False)

