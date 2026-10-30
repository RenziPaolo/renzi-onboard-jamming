import sys
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import time
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from models.anomalyDetection.CSNNanomaly_norse import CSNNanomaly_image_Norse, CSNNanomaly_line_Norse, CSNNanomaly_image_VAE_Norse, CSNNanomaly_line_VAE_Norse
from dataset.dataset import SARZarrDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.nn.functional as F
import argparse
import datetime
import piq
from torch.utils.data import random_split

H = 512
W = 512

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-model', default='image', type=str, help='type of model image or line')
parser.add_argument('-path', default='./anomaly', type=str, help='where to save the results')
parser.add_argument('-epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('-batch', default=64, type=int, help='batch size for training')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-data_dir', default="./../../data/v2/RFInject", type=str, help='where MNIST data is stored')
parser.add_argument('-loss', default='MSE', type=str, help='type of loss function MSE or MAE or FDCC')
parser.add_argument('-vae', action='store_true', help='use VAE architecture')
parser.add_argument('-spike', action='store_true', help='using spike architecture')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('--H', type=float, default=512, help='dimension for images based on energy or firing rate')
parser.add_argument('--W', type=float, default=512, help='dimension for images based on energy or firing rate')
parser.add_argument('-j', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
parser.add_argument('-latent_dim', default=1, type=int, help='latent dimension for VAE')
parser.add_argument('-only_mag', action='store_true', help='use only magnitude channel')
parser.add_argument('-only_ang', action='store_true', help='use only angle channel')

args = parser.parse_args()

if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def get_device_and_world_size():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        world_size = 1
        device = torch.device("cpu")
    return device, world_size

def calculate_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def train(rank, world_size, num_epochs, training_metrics_dict, validation_metrics_dict, training_loader, validation_loader, net, loss_fn, optimizer, plateau_scheduler, training_path, resume, multigpu):
    if multigpu:
        setup(rank, world_size)
        net = DDP(net, device_ids=[rank])
    net = net.to(rank)

    start_epoch = 0
    max_test_loss = -1

    if resume:
        checkpoint = torch.load(args.resume, map_location=get_device_and_world_size()[0])
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_loss = checkpoint['max_test_loss']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch}_{args.opt}_lr{args.lr}_model{args.model}_vae{args.vae}_only_mag{args.only_mag}_only_ang{args.only_ang}_daytime{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_iou = 0
        train_samples = 0
        for echo, _, _, _ in training_loader:
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

            # Norse: No need for PoissonEncoder, input is directly used
            if args.model == 'line':
                img = torch.reshape(img, (-1, in_channels, args.H))

            x_dec_2, x_enc_2, out_fr = net(img)
            loss = loss_fn(out_fr, img)
            loss.backward()
            optimizer.step()

            train_samples += img.numel()
            train_loss += loss.item() * img.numel()
            if args.model == 'image':
                train_ssim = piq.ssim(out_fr, img).item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        if args.model == 'image':
            train_ssim /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        if args.model == 'image':
            writer.add_scalar('train_ssim', train_ssim, epoch)

        net.eval()
        test_loss = 0
        test_iou = 0
        test_samples = 0
        with torch.no_grad():
            for echo, _, _, _ in validation_loader:
                if args.only_mag:
                    echo = echo[:,0:1,:,:]
                if args.only_ang:
                    echo = echo[:,1:2,:,:]

                img = echo.to(rank)
                if args.model == 'line':
                    img = torch.reshape(img, (-1, in_channels, args.H))

                x_dec_2, x_enc_2, out_fr = net(img)
                loss = loss_fn(out_fr, img)

                test_samples += img.numel()
                test_loss += loss.item() * img.numel()
                if args.model == 'image':
                    test_ssim = piq.ssim(out_fr, img).item()

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        if args.model == 'image':
            test_ssim /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        if args.model == 'image':
            writer.add_scalar('test_ssim', test_ssim, epoch)

        save_max = False
        if test_loss > max_test_loss:
            max_test_loss = test_loss
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_loss': max_test_loss
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pt'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pt'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_iou ={train_iou: .4f}, test_loss ={test_loss: .4f}, test_iou ={test_iou: .4f}, max_test_loss ={max_test_loss: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epoch - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    def pseudo_huber_loss(input, target, delta=1.0):
        error = torch.abs(input - target)
        quadratic = torch.clamp(error, max=delta)
        linear = error - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()

    device, world_size = get_device_and_world_size()
    if args.only_mag or args.only_ang:
        in_channels = 1
    else:
        in_channels = 2

    if args.model == 'image':
        model = CSNNanomaly_image_Norse(T=args.T, in_channels=in_channels, inner_channels=4)
        if args.vae:
            model = CSNNanomaly_image_VAE_Norse(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
    else:
        model = CSNNanomaly_line_Norse(T=args.T, in_channels=in_channels, inner_channels=4)
        if args.vae:
            model = CSNNanomaly_line_VAE_Norse(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)

    dataset = SARZarrDataset(root_dir=args.data_dir, target_shape=(H,W), device='cuda')
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

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

    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            args=(args.epoch, training_metrics_dict, validation_metrics_dict, training_loader, validation_loader, model, loss_fn, optimizer, plateau_scheduler, args.path, True),
            nprocs=world_size,
            join=True
        )
        cleanup()
    else:
        train(rank='cuda:0', world_size=1, num_epochs=args.epoch, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
              training_loader=training_loader, validation_loader=validation_loader, net=model, loss_fn=loss_fn,
              optimizer=optimizer, plateau_scheduler=plateau_scheduler, training_path=args.path, resume=False, multigpu=False)
