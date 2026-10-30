import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from tqdm import tqdm
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, EuroSAT
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path
import spikingjelly.activation_based as sj
# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.classification.conv import SAR_RF_Interference_Classifier_simp
from models.classification.linear import SimpleLinearModel
from models.classification.snn import SAR_RF_Interference_Classifier_SNN
from models.classification.CSNN import CSNN
import matplotlib.pyplot as plt

def get_transforms(augmentation, strength=1.0):
    if augmentation == "gaussian_noise":
        class GaussianNoise(object):
            def __init__(self, mean=0., std=0.5):
                self.std = std
                self.mean = mean
            def __call__(self, img):
                img = np.array(img)
                noise = np.random.normal(self.mean, self.std, img.shape)
                noisy_img = img + noise
                return torch.from_numpy(np.clip(noisy_img, 0, 1)).float()
        return transforms.Compose([
            transforms.ToTensor(),
            GaussianNoise(mean=0., std=strength),
        ])
    elif augmentation == "RFI":
        class RFInterference(object):
            def __init__(self, streak_prob=0.25, spike_prob=0.25, grid_prob=0.25, streak_width=20, spike_intensity=1.0, grid_intensity=1.0):
                self.streak_prob = streak_prob
                self.spike_prob = spike_prob
                self.grid_prob = grid_prob
                self.streak_width = streak_width
                self.spike_intensity = spike_intensity
                self.grid_intensity = grid_intensity

            def __call__(self, img):
                _, h, w = img.shape
                noisy_img = img.clone()

                # Add random streaks
                if torch.rand(1) < self.streak_prob:
                    if torch.rand(1) < 0.5:  # Horizontal streak
                        y = torch.randint(0, h - self.streak_width, (1,)).item()
                        noisy_img[:, y:y+self.streak_width, :] = torch.randn_like(noisy_img[:, y:y+self.streak_width, :]) * self.spike_intensity
                    else:  # Vertical streak
                        x = torch.randint(0, w - self.streak_width, (1,)).item()
                        noisy_img[:, :, x:x+self.streak_width] = torch.randn_like(noisy_img[:, :, x:x+self.streak_width]) * self.spike_intensity

                # Add random spikes
                if torch.rand(1) < self.spike_prob:
                    spike_mask = (torch.rand_like(noisy_img) < 0.1).float()
                    noisy_img = noisy_img + spike_mask * self.spike_intensity * torch.randn_like(noisy_img)

                # Add grid-like artifacts
                if torch.rand(1) < self.grid_prob:
                    grid = torch.rand(h, w) < 0.2
                    noisy_img[0, grid] = self.grid_intensity * torch.randn(1).item()

                # Clip to valid range
                noisy_img = torch.clamp(noisy_img, 0, 1)
                return noisy_img

        return transforms.Compose([
            transforms.ToTensor(),
            RFInterference(streak_prob=strength*0.75, spike_prob=strength*0.75, grid_prob=strength*0.5),
        ])
    else:  # no augmentation
        return transforms.Compose([
            transforms.ToTensor(),
        ])

def extract_features(dataloader, device, model, data_name):
    model.eval()
    features = []
    energies = []
    spikes = []

    with torch.no_grad():
        if args.model_type in ['snn', 'csnn']:
            
            spike_count = 0
            for module in model.modules():
                if isinstance(module, sj.neuron.BaseNode):
                    last_spiking_layer = module
            
            def hook_fn(module, input, output):
                nonlocal spike_count
                spike_count += output.sum().item()

            hook = last_spiking_layer.register_forward_hook(hook_fn)
            
            for batch in tqdm(dataloader, desc=f"Extracting energy and spikes from {data_name} samples"):
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device)
                feature = model(imgs)
                firing_rate = (spike_count / model.T) / imgs.size(0)
                energy = -torch.logsumexp(feature, dim=1)
                spikes.append(torch.full((imgs.size(0), 1), firing_rate, dtype=torch.float32).cpu())
                energies.append(energy.cpu())
                features.append(feature.cpu())            
                spike_count = 0
            spikes = torch.cat(spikes, dim=0)
            hook.remove()
        else:
            for batch in tqdm(dataloader, desc=f"Extracting energy from {data_name} samples"):
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device)
                feature = model(imgs)
                energy = -torch.logsumexp(feature, dim=1)
                energies.append(energy.cpu())
                features.append(feature.cpu())

    features = torch.cat(features, dim=0)
    energies = torch.cat(energies, dim=0)

    return energies.numpy(), spikes.numpy() if args.model_type in ['snn', 'csnn'] else None

def plot_jensenshannon(e1, e2, path, id_name, ood_name, type_):
    js_div = jensenshannon(e1, e2)
    print(type(js_div))
    if not isinstance(js_div, np.float64):
        js_div = js_div[0]
    print(f"JS Divergence: {js_div}")
    plt.figure()
    sns.histplot(e1, label=id_name, kde=True, bins=50)
    sns.histplot(e2, label=ood_name, kde=True, bins=50)
    plt.legend()
    plt.title(f"Energy Distribution - JS Div (↓): {js_div:.3f}")
    plt.tight_layout()
    plt.savefig(path + f'{type_}_dist_{id_name}_vs_{ood_name}.png', dpi=200, bbox_inches='tight')
    plt.close()

# Pseudocodice per farti capire il ragionamento
# - L'idea sarebbe vedere come si comparano le distribuizioni originali dei dataset, con quelle rumorose
# - Questo codice vale per l'energia delle reti lineari, poi per le SSN DOVREBBE essere lo stesso, solo che invece di calcolare l'energia, calcoli il firing rate
# - Usa sempre i dati di test
# - Vediamo cosa esce e ci ragioniamo sopra

def main():
            # Load model
    if args.model_type == 'linear':
        model = SimpleLinearModel(input_dim=28*28*1, output_dim=10)
    elif args.model_type == 'conv':
        model = SAR_RF_Interference_Classifier_simp(input_channels=1, num_classes=10)
    elif args.model_type == 'snn':
        model = SAR_RF_Interference_Classifier_SNN(h=28, w=28, channels=1, classes=10)
    elif args.model_type == 'csnn':
        model = CSNN(T=10, h=28, w=28, channels=1, classes=10)  

    if args.dataset == 'CIFAR10':
        # Load CIFAR10 test data
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        datasets = [(test_loader, 'cifar10')]
        # Load model
        if args.model_type == 'linear':
            model = SimpleLinearModel(input_dim=32*32*3, output_dim=10)
        elif args.model_type == 'conv':
            model = SAR_RF_Interference_Classifier_simp(input_channels=3, num_classes=10)
        elif args.model_type == 'snn':
            model = SAR_RF_Interference_Classifier_SNN(h=32, w=32, channels=3, classes=10)
        elif args.model_type == 'csnn':
            model = CSNN(T=10, h=32, w=32, channels=3, classes=10)
        
    if args.dataset == 'EuroSAT':
        # Load model
        if args.model_type == 'linear':
            model = SimpleLinearModel(input_dim=64*64*3, output_dim=10)
        elif args.model_type == 'conv':
            model = SAR_RF_Interference_Classifier_simp(input_channels=3, num_classes=10)
        elif args.model_type == 'snn':
            model = SAR_RF_Interference_Classifier_SNN(h=64, w=64, channels=3, classes=10)
        elif args.model_type == 'csnn':
            model = CSNN(T=50, h=64, w=64, channels=3, classes=10)
        
    if args.model_type in ['snn', 'csnn']:
        state = torch.load(args.model_path, map_location=torch.device('cpu'))
        state = state['net']
        new_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        model.load_state_dict(new_state)
    else:
        state = torch.load(args.model_path, map_location=torch.device('cpu'))
        new_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        model.load_state_dict(new_state)
    model.eval()

    if args.dataset == 'mnist':
        dataset_ = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        if args.case3:
        
            from torch.utils.data import Dataset

            class CustomMNIST(Dataset):
                def __init__(self, images, labels):
                    self.images = images.float() / 255.0
                    self.labels = labels

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    return self.images[idx].unsqueeze(0), self.labels[idx]
            
            train_images = dataset_.data[dataset_.targets != 0]
            train_labels = dataset_.targets[dataset_.targets != 0]
            
            dataset_ = CustomMNIST(train_images, train_labels)
            
        id_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)

        
    elif args.dataset == 'fmnist':
        dataset_ = FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        id_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'CIFAR10':
        transform_noise.transforms.insert(0, transforms.ToTensor())
        dataset_ = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        id_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'CIFAR10r':
        if len(transform_noise.transforms) == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((28, 28)),      # Resize to 28x28
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            ] )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((28, 28)),      # Resize to 28x28
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            ] + transform_noise.transforms[1:])
        dataset_ = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        id_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'EuroSAT':
        from torchvision.datasets import EuroSAT
        from torch.utils.data import random_split
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.3443, 0.3809, 0.4082), (0.1814, 0.1235, 0.1126))])
        dataset_ = EuroSAT(root='./data', download=True, transform=transform)
        dataset_size = len(dataset_)
        train_size = int(0.8 * dataset_size)  # 80% for training
        val_size = int(0.1 * dataset_size)    # 10% for validation
        test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
        train_dataset, val_dataset, test_dataset = random_split(
            dataset_, [train_size, val_size, test_size]
        )
        id_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)



    
    
    for cond in args.conditions:
        transform_noise = get_transforms(cond, strength=1.0)
        

        if args.dataset == 'mnist':
            dataset_ = MNIST(root='./data', train=False, download=True, transform=transform_noise)
            
            if args.case3:
            
                from torch.utils.data import Dataset

                class CustomMNIST(Dataset):
                    def __init__(self, images, labels):
                        self.images = images.float() / 255.0
                        self.labels = labels

                    def __len__(self):
                        return len(self.labels)

                    def __getitem__(self, idx):
                        return self.images[idx].unsqueeze(0), self.labels[idx]
                
                train_images = dataset_.data[dataset_.targets != 0]
                train_labels = dataset_.targets[dataset_.targets != 0]
                
                dataset_ = CustomMNIST(train_images, train_labels)
                
            ood_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)

            
        elif args.dataset == 'fmnist':
            dataset_ = FashionMNIST(root='./data', train=False, download=True, transform=transform_noise)
            ood_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
        elif args.dataset == 'CIFAR10':
            transform_noise.transforms.insert(0, transforms.ToTensor())
            dataset_ = CIFAR10(root='./data', train=False, download=True, transform=transform_noise)
            ood_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
        elif args.dataset == 'CIFAR10r':
            if len(transform_noise.transforms) == 1:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((28, 28)),      # Resize to 28x28
                    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                ] )
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((28, 28)),      # Resize to 28x28
                    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                ] + transform_noise.transforms[1:])
            dataset_ = CIFAR10(root='./data', train=False, download=True, transform=transform)
            ood_dataloader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False)
        elif args.dataset == 'EuroSAT':
            from torchvision.datasets import EuroSAT
            from torch.utils.data import random_split
            transform = transforms.Compose(transform_noise.transforms + [transforms.Normalize((0.3443, 0.3809, 0.4082), (0.1814, 0.1235, 0.1126))])
            dataset_ = EuroSAT(root='./data', download=True, transform=transform)
            dataset_size = len(dataset_)
            train_size = int(0.8 * dataset_size)  # 80% for training
            val_size = int(0.1 * dataset_size)    # 10% for validation
            test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
            train_dataset, val_dataset, test_dataset = random_split(
                dataset_, [train_size, val_size, test_size]
            )
            ood_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


        
        # Calcola energia per il dataset di base
        id_energies, id_spikes = extract_features(id_dataloader, args.device, model, args.dataset) # E poi MNIST, EuroSAT

        # Calcola energia per il dataset con rumore 
        ood_energies, ood_spikes = extract_features(ood_dataloader, args.device, model, args.dataset + ' ' +cond) # E poi gli altri casi di rumore

        # Plotta
        plot_jensenshannon(id_energies, ood_energies, args.output_dir, args.dataset, str(cond), type_='energy')
        if args.model_type in ['snn', 'csnn']:
            plot_jensenshannon(id_spikes, ood_spikes, args.output_dir, args.dataset, str(cond), type_='spike')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model on MNIST with different augmentations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model .pt file")
    parser.add_argument("--output_dir", type=str, default="./", required=True, help="Directory to save the output JSON file")
    parser.add_argument("--conditions", nargs="+", default=["gaussian_noise", "RFI"],
                        help="List of conditions to evaluate (none, blur, salt_pepper, gaussian_noise)")
    parser.add_argument("--model_type", type=str, default='linear', required=True, help="which model to use: linear, conv, snn, csnn")
    parser.add_argument("--type", type=str, default='standard', required=True, help="standard or spike")
    parser.add_argument("--batch_size", type=int, default=1, required=True, help="batch size for evaluation")
    parser.add_argument('--dataset', default='mnist', type=str, help='which dataset to use mnist or CIFAR10')
    parser.add_argument('--case3', default=False, type=bool, help='case3 or not')
    parser.add_argument('--tau', type=float, default=None, help='threshold for classification based on energy or firing rate')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    args = parser.parse_args()
    print("Arguments:", args)
    main()