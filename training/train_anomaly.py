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
from models.anomalyDetection.CSNNanomaly import CSNNanomaly_image, CSNNanomaly_line, CSNNanomaly_image_VAE, CSNNanomaly_line_VAE, CSNNanomaly_image_LIF, CSNNanomaly_line_LIF, CSNNanomaly_image_VAE_LIF, CSNNanomaly_line_VAE_LIF
from models.anomalyDetection.FDCCpimage import AnomalyImageDetectionModel
from models.anomalyDetection.FDCCp import AnomalyDetectionModel1D
from dataset.dataset import SARZarrDataset
from dataset.fast_dataset import fastSARZarrDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.nn.functional as F
import argparse
from spikingjelly.activation_based import encoding, functional
import datetime
import piq
from torch.utils.data import random_split
import math
import torch.nn as nn

H = 224
W = 224

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='LIF SAR Training')
parser.add_argument('-model', default='image', type=str, help='type of model image or line')
parser.add_argument('-path', default='./anomaly', type=str, help='where to save the results')
parser.add_argument('-epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('-batch', default=64, type=int, help='batch size for training')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-data_dir', default="./../../data/v2/RFInject", type=str, help='where SAR data is stored')
parser.add_argument('-loss', default='MSE', type=str, help='type of loss function MSE or MAE or FDCC')
parser.add_argument('-vae', default=False, help='use VAE architecture')
parser.add_argument('-spike', default=False, help='using spike architecture')

parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('--H', type=float, default=224, help='dimension for images based on energy or firing rate')
parser.add_argument('--W', type=float, default=224, help='dimension for images based on energy or firing rate')
parser.add_argument('-j', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
parser.add_argument('-latent_dim', default=1, type=int, help='latent dimension for VAE')
parser.add_argument('-only_mag', default=False, help='use only magnitude channel')
parser.add_argument('-only_ang', default=False, help='use only angle channel')
parser.add_argument('-fast_dataset', default=False, help='use fast dataset protocol')
parser.add_argument('-lif', default=False, help='use LIF neurons instead of IF')
parser.add_argument('-initialization', default='None', type=str, help='type of weight initialization')

args = parser.parse_args()

if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_device_and_world_size():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        world_size = 1
        device = torch.device("cpu")
    return device, world_size

def calculate_iou(pred, target, threshold=0.5):
    # Threshold predictions to get binary masks
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold

    # Calculate intersection and union
    intersection = (pred & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | target).float().sum((1, 2))         # Will be zero if both are 0

    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()  # Return mean IoU over the batch

def get_device_and_world_size():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        world_size = 1
        device = torch.device("cpu")
    return device, world_size

def train (rank, world_size, num_epochs, training_metrics_dict, validation_metrics_dict, training_loader,
           validation_loader, net, loss_fn, optimizer, plateau_scheduler, training_path, resume, multigpu):
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
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch}_{args.opt}_lr{args.lr}_model{args.model}_vae{args.vae}_only_mag{args.only_mag}_only_ang{args.only_ang}_lif{args.lif}_daytime{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    os.makedirs(out_dir, exist_ok=True)
    print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    print("start of traning")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_iou = 0
        train_samples = 0
        train_ssim = 0
        for echo, _, _, _, _ in training_loader:
            optimizer.zero_grad()
            if args.only_mag:
                echo = echo[:,0:1,:,:]
            
            if args.only_ang:
                echo = echo[:,1:2,:,:]
  
            if args.only_mag or args.only_ang:
                in_channels = 1
            else:
                in_channels = 2
            img = echo.to(rank)

            out_fr = 0.
            encoded_img = encoder(img)
            if args.model == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))
            x_dec_2, x_enc_2, out_fr_ = net(encoded_img)
            out_fr += out_fr_
            out_fr = out_fr / args.T
            loss = loss_fn(out_fr, encoded_img)
            loss.backward()
            optimizer.step()

            train_samples += img.numel()
            train_loss += loss.item() * img.numel()
            if args.model == 'image':
                train_ssim += piq.ssim(out_fr, encoded_img)

            functional.reset_net(net)
        plateau_scheduler.step(train_loss / train_samples)
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        if args.model == 'image':
            train_ssim = train_ssim.mean()

        writer.add_scalar('train_loss', train_loss, epoch)
        if args.model == 'image':
            writer.add_scalar('train_ssim', train_ssim, epoch)

        net.eval()
        test_loss = 0
        test_iou = 0
        test_samples = 0
        test_ssim = 0
        with torch.no_grad():
            for echo, _, _, _, _ in validation_loader:
                if args.only_mag:
                    echo = echo[:,0:1,:,:]         
                    
                if args.only_ang:
                    echo = echo[:,1:2,:,:]


                img = echo.to(rank)
                out_fr = 0.
                encoded_img = encoder(img)

                if args.model == 'line':
                    encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))

                encoded_line = encoder(encoded_img)
                x_dec_2, x_enc_2, out_fr_ = net(encoded_line)
                out_fr += out_fr_
                out_fr = out_fr / args.T
                loss = loss_fn(out_fr, encoded_img)



                test_samples += img.numel()
                test_loss += loss.item() * img.numel()
                if args.model == 'image':
                    test_ssim += piq.ssim(out_fr, encoded_img)
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        if args.model == 'image':
            test_ssim = test_ssim.mean()
        writer.add_scalar('test_loss', test_loss, epoch)
        if args.model == 'image':
            writer.add_scalar('test_ssim', test_ssim, epoch)

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
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_ssim ={train_ssim: .4f}, test_loss ={test_loss: .4f}, test_ssim ={test_ssim: .4f}, max_test_loss ={max_test_loss: .4f}, min_test_loss ={min_test_loss: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epoch - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        with open(os.path.join(out_dir, 'training_log.txt'), 'a', encoding='utf-8') as log_file:
            log_file.write(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_ssim ={train_ssim: .4f}, test_loss ={test_loss: .4f}, test_ssim ={test_ssim: .4f}, max_test_loss ={max_test_loss: .4f}, min_test_loss ={min_test_loss: .4f}\n')
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


if __name__ == "__main__":

    def pseudo_huber_loss(self, input, target, delta=1.0):
        """
        Pseudo-Huber loss for each anomaly channel.
        """
        error = torch.abs(input - target)
        quadratic = torch.clamp(error, max=delta)
        linear = error - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()

    def compute_anomaly_scores(self, x):
        """
        Compute anomaly scores using the pseudo-Huber function.
        """
        anomaly_scores = self.forward(x)
        # Apply pseudo-Huber function
        anomaly_scores = torch.sqrt(anomaly_scores**2 + 1) - 1
        return anomaly_scores
    
    print(sys.path)
    
    device, world_size = get_device_and_world_size()
    if args.only_mag or args.only_ang:
        in_channels = 1
    else:
        in_channels = 2  
    if args.lif:
        if args.model == 'image':
            model = CSNNanomaly_image_LIF(T=args.T, in_channels=in_channels, inner_channels=4)
            if args.vae:
                model = CSNNanomaly_image_VAE_LIF(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=4)
        else:
            model = CSNNanomaly_line_LIF(T=args.T, in_channels=in_channels, inner_channels=4)
            if args.vae:
                model = CSNNanomaly_line_VAE_LIF(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=4)
 
    else:
        if args.model == 'image':
            model = CSNNanomaly_image(T=args.T, in_channels=in_channels, inner_channels=4)
            if args.vae:
                model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=4)
        else:
            model = CSNNanomaly_line(T=args.T, in_channels=in_channels, inner_channels=4)
            if args.vae:
                model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=4)

    if args.initialization == 'snn':
        model.apply(init_snn_weights)
    elif args.initialization == 'default':
        model.apply(init_weight)
    
    if args.fast_dataset:
        print("fast dataset")
        dataset = fastSARZarrDataset(root_dir=args.data_dir, target_shape=(H,W), device='cuda', fast=True)
    else:
        dataset = SARZarrDataset(root_dir=args.data_dir, target_shape=(H,W), device='cuda')



    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.j, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=args.j, pin_memory=True)
    if args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif args.loss == 'MAE':
        loss_fn = torch.nn.L1Loss()
    else:
        loss_fn = pseudo_huber_loss
    training_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    validation_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

