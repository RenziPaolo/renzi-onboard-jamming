import sys
from pathlib import Path
# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import os
import torch

from models.segmentation.spikingresformer import SpikingResformerSeg, SpikingResformerSSD, SpikingResformerFPN
from models.segmentation.Unet import *
from dataset.dataset import SARZarrDataset
from dataset.fast_dataset import fastSARZarrDataset
import argparse
from spikingjelly.activation_based import encoding
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ast
import math

def get_device_and_world_size():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        world_size = 1
        device = torch.device("cpu")
    return device, world_size

# --- Global variables for visualization ---
last_rfi = None
last_gt_bbox = None
last_pred_bbox = None

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

def load_args_from_file(args_file_path):
    """
    Load arguments from the args.txt file generated during training.
    The file is expected to contain a string representation of the args Namespace.
    """
    with open(args_file_path, 'r') as f:
        lines = f.readlines()

    # The first line contains the Namespace object
    namespace_str = lines[0].strip()

    # Extract the content inside the Namespace parentheses
    if not namespace_str.startswith("Namespace(") or not namespace_str.endswith(")"):
        raise ValueError("Invalid Namespace format in args.txt")

    # Remove "Namespace(" and the trailing ")"
    content = namespace_str[len("Namespace("):-1]

    # Parse the key-value pairs
    args_dict = {}
    for pair in content.split(', '):
        if '=' not in pair:
            continue  # Skip malformed pairs
        key, value = pair.split('=', 1)
        key = key.strip()

        # Handle string values (remove quotes)
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        # Handle boolean values
        elif value == 'True':
            value = True
        elif value == 'False':
            value = False
        # Handle None
        elif value == 'None':
            value = None
        # Handle numeric values (int or float)
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Leave as string if not numeric

        args_dict[key] = value

    # Convert to Namespace
    args = argparse.Namespace(**args_dict)
    return args

def test(
    rank,
    world_size,
    test_loader,
    net,
    loss_fn,
    args,
    scale_x=22542.463211026985,
    scale_y=1418.2935352958766,
    visualize=False,
    output_dir="test_results"
):
    """
    Test a trained model on a dataset.
    """
    global last_rfi, last_gt_bbox, last_pred_bbox

    # --- Setup device ---
    device = "cpu"
    net = net.to(device)
    net.eval()

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "test_log.txt")

    # --- Initialize metrics ---
    test_loss = 0.0
    test_iou = 0.0
    test_ciou = 0.0
    test_diou = 0.0
    test_samples = 0
    test_boxes = 0

    # --- Test loop ---
    with torch.no_grad():
        for batch_idx, (echo, expanded_RFI, bbox, _) in enumerate(test_loader):
            # --- Prepare input ---
            if args.only_mag or args.only_ang:
                in_channels = 1
            else:
                in_channels = 2

            RFI = expanded_RFI.view(-1, in_channels, args.H, args.W)
            if args.only_mag:
                RFI = RFI[:, 0:1, :, :]
            if args.only_ang:
                RFI = RFI[:, 1:2, :, :]

            RFI_split = torch.split(RFI, echo.shape[0], dim=0)
            RFI = RFI_split[0].to(device, non_blocking=True)
            bbox = bbox.to(device, non_blocking=True)

            # --- Encode input ---
            encoder = encoding.PoissonEncoder()
            encoded_img = encoder(RFI)

            if args.type == "line":
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))

            # --- Forward pass ---
            x = net(encoded_img)

            # --- Decode predictions ---
            if args.decoding == "rate":
                x = torch.mean(x, dim=0)
                x_scaled = x * args.scale
            elif args.decoding == "temporal":
                probs = x.softmax(dim=0)
                indices = torch.arange(x.shape[0], device=x.device, dtype=torch.float32)
                indices = indices.view(-1, *([1] * (x.dim() - 1)))
                x = (probs * indices).sum(dim=0)
                x_scaled = x * scale_x
            elif args.decoding == "sum":
                x = torch.sum(x, dim=0)
                x_scaled = x * args.scale

            # --- Normalize bbox if needed ---
            if not args.normalize_bbox:
                x11, y11, x12, y12 = bbox.view(-1, bbox.shape[-1]).unbind(dim=-1)
                x11_normalized = x11 / scale_x
                x12_normalized = x12 / scale_x
                y11_normalized = y11 / scale_y
                y12_normalized = y12 / scale_y
                boxes_normalized = torch.stack(
                    [x11_normalized, y11_normalized, x12_normalized, y12_normalized], dim=-1
                )
            else:
                boxes_normalized = bbox.view(-1, bbox.shape[-1])

            # --- Reshape predictions ---
            x = x.view(-1, x.shape[-1])
            bbox = bbox.view(-1, bbox.shape[-1])

            # --- Compute loss ---
            loss = loss_fn(x, boxes_normalized)
            test_loss += loss.item() * encoded_img.numel()
            test_samples += encoded_img.numel()

            # --- Compute IoU ---
            iou = calculate_iou(x, boxes_normalized)
            test_iou += iou.mean().item() * iou.numel()
            test_boxes += iou.numel()

            # --- Optional: Compute CIoU/DIoU ---
            if args.loss in ["CIoU", "MSE_CIoU"]:
                test_ciou += (1 - ciou_loss(x, boxes_normalized)).item() * iou.numel()
            if args.loss in ["DIoU", "MSE_DIoU"]:
                test_diou += (1 - diou_loss(x, boxes_normalized)).item() * iou.numel()

            # --- Store last batch for visualization ---
            if visualize:
                last_rfi = RFI.cpu()
                last_gt_bbox = bbox.cpu()
                last_pred_bbox = x.cpu()

            # --- Reset SNN ---
            if hasattr(net, "reset_net"):
                net.reset_net()

    # --- Final metrics ---
    test_loss /= test_samples
    test_iou /= test_boxes
    test_ciou /= test_boxes if test_boxes > 0 else 1
    test_diou /= test_boxes if test_boxes > 0 else 1

    # --- Log results ---
    with open(log_file, "a") as f:
        f.write(
            f"Test Loss: {test_loss:.4f}, "
            f"Test IoU: {test_iou:.4f}, "
            f"Test CIoU: {test_ciou:.4f}, "
            f"Test DIoU: {test_diou:.4f}\n"
        )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test IoU: {test_iou:.4f}, "
        f"Test CIoU: {test_ciou:.4f}, "
        f"Test DIoU: {test_diou:.4f}"
    )

    # --- Visualize last batch ---
    if visualize and last_rfi is not None:
        visualize_bboxes(
            last_rfi,
            last_gt_bbox,
            last_pred_bbox,
            scale_x=scale_x,
            scale_y=scale_y,
            sample_idx=0,
            rfi_idx=0
        )

    return {
        "test_loss": test_loss,
        "test_iou": test_iou,
        "test_ciou": test_ciou,
        "test_diou": test_diou,
    }

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


def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Test a trained model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model and args.txt.")
    parser.add_argument('--data_dir', default="./../../data/v2/RFInject", type=str, help='where SAR data is stored')
    parser.add_argument("--batch", type=int, default=1, help="Batch size (overrides args.txt if provided).")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions.")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory for results.")
    parser.add_argument("--fast_dataset", action="store_true", help="Use fast dataset loader (overrides args.txt if provided).")
    parser.add_argument("--j", type=int, default=None, help="Number of workers for DataLoader (overrides args.txt if provided).")
    args_cli = parser.parse_args()

    # --- Load args from args.txt ---
    args_file_path = os.path.join(args_cli.model_dir, "args.txt")
    args = load_args_from_file(args_file_path)

    # --- Override args with CLI arguments if provided ---
    if args_cli.batch is not None:
        args.batch = args_cli.batch
    if args_cli.fast_dataset:
        args.fast_dataset = args_cli.fast_dataset
    if args_cli.j is not None:
        args.j = args_cli.j

    # --- Set device ---
    device, world_size = get_device_and_world_size()

    device = 'cpu'
    
    # --- Load dataset ---
    if args.fast_dataset:
        dataset = fastSARZarrDataset(
            root_dir=args.data_dir,
            target_shape=(args.H, args.W),
            device=device,
            log=args.log,
            fast=True,
            normalize_bbox=args.normalize_bbox
        )
    else:
        dataset = SARZarrDataset(
            root_dir=args.data_dir,
            target_shape=(args.H, args.W),
            log=args.log,
            device=device
        )

    # Split into train/val/test (use test set if available, else val)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.j,
        pin_memory=True
    )

    # --- Load model ---
    in_channels = 1 if (args.only_mag or args.only_ang) else 2
    if args.log:
        in_channels = 3

    # Recreate model architecture
    if args.head == "ssd":
        if args.model == "s":
            model = SpikingResformerSSD(
                [['DSSA', 'GWFFN'] * 1, ['DSSA', 'GWFFN'] * 2, ['DSSA', 'GWFFN'] * 3],
                [64, 256, 512],
                [1, 4, 8],
                [2, 1, 1],
                in_channels=in_channels,
                T=args.T,
                step_mode=args.step_mode,
                decoding=args.decoding
            )
        elif args.model == "unet":
            model = Unet(
                in_channels=in_channels,
                step_mode=args.step_mode,
                T=args.T,
                head_type="ssd",
                model_size=getattr(args, "unet_size", "s"),
                learnable_anchors=getattr(args, "learnable_anchors", False) 
            )
        # Add other model sizes (xs, xxs, m) as in your training script
    elif args.head == "fpn":
        # Add FPN model definitions
        pass
    elif args.head == "standard":
        # Add standard model definitions
        pass

    # Load checkpoint
    model_path = os.path.join(args_cli.model_dir, "checkpoint_latest.pt")  # or checkpoint_max.pt, checkpoint_min.pt
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)

    # --- Load loss function ---
    if args.loss == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif args.loss == "CIoU":
        loss_fn = ciou_loss
    elif args.loss == "DIoU":
        loss_fn = diou_loss
    elif args.loss == "MSE_CIoU":
        loss_fn = lambda pred, target: MSE_ciou_loss(pred, target, alpha=args.alpha)
    elif args.loss == "MSE_DIoU":
        loss_fn = lambda pred, target: MSE_diou_loss(pred, target, alpha=args.alpha)

    # --- Run test ---
    if world_size > 1:
        # Multi-GPU testing
        torch.multiprocessing.spawn(
            test,
            args=(world_size, test_loader, model, loss_fn, args, args_cli.visualize, args_cli.output_dir),
            nprocs=world_size,
            join=True
        )
    else:
        test(
            rank=device,
            world_size=1,
            test_loader=test_loader,
            net=model,
            loss_fn=loss_fn,
            args=args,
            visualize=args_cli.visualize,
            output_dir=args_cli.output_dir
        )

if __name__ == "__main__":
    main()