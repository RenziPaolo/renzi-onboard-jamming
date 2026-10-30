import sys
from pathlib import Path

# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import time
import argparse
import datetime

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision

from spikingjelly.activation_based import encoding, functional
if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-batch', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-model', default='standard', type=str, help='type of model standard or conv')
    parser.add_argument('-dataset', default='mnist', type=str, help='datset to train on')
    parser.add_argument('-case3', default=False, type=bool, help='case3 or not')

    args = parser.parse_args()
    print(args)
    if args.dataset == 'mnist':
        print("initializing MNIST dataset")
        if args.model == 'standard':
            from models.classification.snn import SAR_RF_Interference_Classifier_SNN as SNN
            net = SNN(h=28, w=28, classes=10, channels=1, tau=args.tau)
        elif args.model == 'conv':
            from models.classification.CSNN import CSNN as SNN
            net = SNN(h=28, w=28, classes=10, channels=1, T=args.T, use_cupy=False)
            

        print(net)

        net.to(args.device)


        train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True 
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        
        if args.case3:
            
            from torch.utils.data import Dataset, DataLoader

            class CustomMNIST(Dataset):
                def __init__(self, images, labels):
                    self.images = images.float() / 255.0
                    self.labels = labels

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    return self.images[idx].unsqueeze(0), self.labels[idx]
            
            train_images = train_dataset.data[train_dataset.targets == 0]
            train_labels = train_dataset.targets[train_dataset.targets == 0]
            
            train_dataset = CustomMNIST(train_images, train_labels)
            
            test_images = test_dataset.data[test_dataset.targets == 0]
            test_labels = test_dataset.targets[test_dataset.targets == 0]
            
            test_dataset = CustomMNIST(test_images, test_labels)
        
    if args.dataset == 'CIFAR10':
        print("initializing CIFAR10 dataset")
        if args.model == 'standard':
            from models.classification.snn import SAR_RF_Interference_Classifier_SNN as SNN
            net = SNN(h=32, w=32, classes=10, channels=3, tau=args.tau)
        elif args.model == 'conv':
            from models.classification.CSNN import CSNN as SNN
            net = SNN(h=32, w=32, classes=10, channels=3, T=args.T, use_cupy=False)
            

        print(net)

        net.to(args.device)

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        
    if args.dataset == 'EuroSAT':
        print("initializing EuroSAT dataset")
        if args.model == 'standard':
            from models.classification.snn import SAR_RF_Interference_Classifier_SNN as SNN
            net = SNN(h=64, w=64, classes=10, channels=3, tau=args.tau)
        elif args.model == 'conv':
            from models.classification.CSNN import CSNN as SNN
            net = SNN(h=64, w=64, classes=10, channels=3, T=args.T, use_cupy=True)
            

        print(net)

        net.to(args.device)

        dataset = torchvision.datasets.EuroSAT(
            root=args.data_dir,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        from torch.utils.data import random_split
        transform_eurosat = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.3443, 0.3809, 0.4082), (0.1814, 0.1235, 0.1126))])
        dataset = torchvision.datasets.EuroSAT(root=args.data_dir, download=True, transform=transform_eurosat)
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)  # 80% for training
        val_size = int(0.1 * dataset_size)    # 10% for validation
        test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    net = torch.nn.DataParallel(net)
    
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch}_{args.opt}_lr{args.lr}_model{args.model}_dataset{args.dataset}_case3{args.case3}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pt'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pt'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    net.eval()

    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(args.device)
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

if __name__ == '__main__':
    main()