from sklearn import metrics
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
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

torch.manual_seed(42)

def plot_noise_data_to_file(data, y_key, output_dir):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique noise types and values
    noise_types = set(key.split('.')[0][:-1] for key in data.keys())
    noise_types = [nt for nt in noise_types if nt != 'none']
    strenght_values = [key.split('.')[0][-1]+'.'+key.split('.')[1] for key in data.keys()]
    noise_values = sorted(set(strenght_values))
    print(noise_values)
    print(noise_types)

    # Create a plot for each noise type
    for noise_type in noise_types:
        y_values = [data[noise_type+nv][y_key] for nv in noise_values]

        plt.figure(figsize=(8, 5))
        plt.plot(noise_values, y_values, marker='o', label=noise_type)
        plt.title(f'{y_key} vs Noise Strenght for {noise_type}')
        plt.xlabel('Noise Strenght')
        plt.ylabel(y_key)
        plt.grid(True)
        plt.legend()

        # Save the plot to the specified directory
        output_path = os.path.join(output_dir, f'{noise_type}_{y_key}_{args.dataset}_plot.png')
        plt.savefig(output_path, format='png', dpi=200, bbox_inches='tight')
        plt.close()


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

def evaluate_model_spike(model, dataloader, tau=None):
    # Find the last spiking layer
    last_spiking_layer = None
    for module in model.modules():
        if isinstance(module, sj.neuron.BaseNode):
            last_spiking_layer = module

    if last_spiking_layer is None:
        raise ValueError("No spiking layer found in the model.")

    spike_count = 0
    logits = []
    spikes = []
    y = []
    def hook_fn(module, input, output):
        nonlocal spike_count
        spike_count += output.sum().item()

    hook = last_spiking_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            logits.append(outputs)
            # Calculate firing rate for this batch
            firing_rate = (spike_count / model.T) / images.size(0)
            
            if tau is not None:
                y.append((torch.tensor(firing_rate).unsqueeze(0) <= tau).int())
            # Convert to tensor and reshape to [batch_size, 1]
            spikes.append(torch.full((images.size(0), 1), firing_rate, dtype=torch.float32))
            spike_count = 0

    hook.remove()

    return torch.cat(logits, dim=0), torch.cat(spikes, dim=0), torch.cat(y, dim=0) if tau is not None else None

def evaluate_model_standard(model, dataloader, tau=None):
    logits = []
    y = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            if tau is not None:
                y.append((calculate_energy(outputs) <= tau).int())
                    
            logits.append(outputs)
    return torch.cat(logits, dim=0), torch.cat(y, dim=0) if tau is not None else None

def calculate_energy(logits, T=1.0):
    # logits shape: [batch_size, num_classes]
    exp_logits = torch.exp((logits) / T)
    sum_exp = torch.sum(exp_logits, dim=1)
    log_sum_exp = torch.log(sum_exp)
    energy = -T * log_sum_exp
    return energy

def calculate_metrics(y_true, y_pred):
  metrics_dict = {}
  current_acc = metrics.accuracy_score(y_true, y_pred)
  current_prec = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
  current_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
  current_f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
  current_auroc = metrics.roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
  current_auprc = metrics.auc(metrics.precision_recall_curve(y_true, y_pred))

  metrics_dict["accuracy"] = current_acc
  metrics_dict["precision"] = current_prec
  metrics_dict["recall"] = current_recall
  metrics_dict["f1_score"] = current_f1_score
  metrics_dict["auroc"] = current_auroc
  metrics_dict["auprc"] = current_auprc

  print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}")
  return metrics_dict

# Main function
def main(model_path, output_dir, conditions, model_type, type, batch_size, dataset, tau):

    # Load model
    if model_type == 'linear':
        model = SimpleLinearModel(input_dim=28*28*1, output_dim=10)
    elif model_type == 'conv':
        model = SAR_RF_Interference_Classifier_simp(input_channels=1, num_classes=10)
    elif model_type == 'snn':
        model = SAR_RF_Interference_Classifier_SNN(h=28, w=28, channels=1, classes=10)
    elif model_type == 'csnn':
        model = CSNN(T=10, h=28, w=28, channels=1, classes=10)  

    if dataset == 'CIFAR10':
        # Load CIFAR10 test data
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        datasets = [(test_loader, 'cifar10')]
        # Load model
        if model_type == 'linear':
            model = SimpleLinearModel(input_dim=32*32*3, output_dim=10)
        elif model_type == 'conv':
            model = SAR_RF_Interference_Classifier_simp(input_channels=3, num_classes=10)
        elif model_type == 'snn':
            model = SAR_RF_Interference_Classifier_SNN(h=32, w=32, channels=3, classes=10)
        elif model_type == 'csnn':
            model = CSNN(T=10, h=32, w=32, channels=3, classes=10)
        
    if dataset == 'EuroSAT':
        # Load model
        if model_type == 'linear':
            model = SimpleLinearModel(input_dim=64*64*3, output_dim=10)
        elif model_type == 'conv':
            model = SAR_RF_Interference_Classifier_simp(input_channels=3, num_classes=10)
        elif model_type == 'snn':
            model = SAR_RF_Interference_Classifier_SNN(h=64, w=64, channels=3, classes=10)
        elif model_type == 'csnn':
            model = CSNN(T=50, h=64, w=64, channels=3, classes=10)
        
    if model_type in ['snn', 'csnn']:
        state = torch.load(model_path, map_location=torch.device('cpu'))
        state = state['net']
        new_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        model.load_state_dict(new_state)
    else:
        state = torch.load(model_path, map_location=torch.device('cpu'))
        new_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        model.load_state_dict(new_state)
    model.eval()

    # Evaluate under different conditions
    #for data, name in datasets:
    results = {}
    metrics_dict = {}
    for strenght in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        for cond in conditions:
            transform_noise = get_transforms(cond, strenght)
            if dataset == 'mnist':
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
                    
                data = DataLoader(dataset_, batch_size=batch_size, shuffle=False)

                
            elif dataset == 'fmnist':
                dataset_ = FashionMNIST(root='./data', train=False, download=True, transform=transform_noise)
                data = DataLoader(dataset_, batch_size=batch_size, shuffle=False)
            elif dataset == 'CIFAR10':
                transform_noise.transforms.insert(0, transforms.ToTensor())
                dataset_ = CIFAR10(root='./data', train=False, download=True, transform=transform_noise)
                data = DataLoader(dataset_, batch_size=batch_size, shuffle=False)
            elif dataset == 'CIFAR10r':
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
                data = DataLoader(dataset_, batch_size=batch_size, shuffle=False)
            elif dataset == 'EuroSAT':
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
                data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


            result = {}
            if type == 'spike':
                logits, spikes, y = evaluate_model_spike(model, data, tau=tau)
                result['spikes_avg'] = (torch.mean(spikes).item())
                result['spikes_dev'] = (torch.std(spikes).item())
            elif type == 'standard':
                logits, y = evaluate_model_standard(model, data, tau=tau)
            if tau is not None:
                metric_dict = calculate_metrics(y_true=torch.ones(y.shape) if cond == "none" else torch.zeros(y.shape), y_pred=y)
                
            metric_dict["tau"] = tau
            
            result['logits_avg'] = (torch.mean(logits).item())
            result['logits_dev'] = (torch.std(logits).item())
            
            energy = calculate_energy(logits, T=1.0)
            avg_energy = torch.mean(energy).item()
            std_energy = torch.std(energy).item()
            result['energy_avg'] = avg_energy
            result['energy_dev'] = std_energy
            
            results[cond+str(strenght)] = result
            metrics_dict[cond+str(strenght)] = metric_dict
            print(f"Average logit value for {cond} at {strenght} on dataset {dataset}: {result['logits_avg']:.4f}")
            print(f"standard deviation logit value for {cond} at {strenght} on dataset {dataset}: {result['logits_dev']:.4f}")
            print(f"Average energy value for {cond} at {strenght} on dataset {dataset}: {result['energy_avg']:.4f}")
            print(f"standard deviation energy value for {cond} at {strenght} on dataset {dataset}: {result['energy_dev']:.4f}")
            if type == 'spike':
                print(f"Average firing rate for {cond} at {strenght} on dataset {dataset}: {result['spikes_avg']:.4f}")

    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    plot_noise_data_to_file(results, 'energy_avg', output_dir)
    if type == 'spike':
        plot_noise_data_to_file(results, 'spikes_avg', output_dir)
    output_path = os.path.join(output_dir, "results"+dataset+"_multiple.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    output_path = os.path.join(output_dir, "metrics"+dataset+"_tau"+str(tau)+".json")
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Results saved to {output_path}")

# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model on MNIST with different augmentations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model .pt file")
    parser.add_argument("--output_dir", type=str, default="./", required=True, help="Directory to save the output JSON file")
    parser.add_argument("--conditions", nargs="+", default=["none", "gaussian_noise", "RFI"],
                        help="List of conditions to evaluate (none, blur, salt_pepper, gaussian_noise)")
    parser.add_argument("--model", type=str, default='linear', required=True, help="which model to use: linear, conv, snn, csnn")
    parser.add_argument("--type", type=str, default='standard', required=True, help="standard or spike")
    parser.add_argument("--batch", type=int, default=64, required=True, help="batch size for evaluation")
    parser.add_argument('-dataset', default='mnist', type=str, help='which dataset to use mnist or CIFAR10')
    parser.add_argument('-case3', default=False, type=bool, help='case3 or not')
    parser.add_argument('--tau', type=float, default=None, help='threshold for classification based on energy or firing rate')
    args = parser.parse_args()
    print("Arguments:", args)
    main(args.model_path, args.output_dir, args.conditions, args.model, args.type, args.batch, args.dataset, args.tau)