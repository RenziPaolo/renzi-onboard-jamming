import json

import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import random_split
import os
import torchvision.transforms as transforms
import numpy as np
import argparse
import sys
from pathlib import Path
import spikingjelly.activation_based as sj
# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset.fast_dataset import fastSARZarrDataset
from models.anomalyDetection.CSNNanomaly import CSNNanomaly_image, CSNNanomaly_line, CSNNanomaly_image_VAE, CSNNanomaly_line_VAE, CSNNanomaly_image_LIF, CSNNanomaly_line_LIF, CSNNanomaly_image_VAE_LIF, CSNNanomaly_line_VAE_LIF
import matplotlib.pyplot as plt
from spikingjelly.activation_based import encoding
import piq

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

def extract_features_echo(dataloader, model, only_mag, only_ang, model_type, save=True, path=None):
    model.eval()
    features = []
    energies = []
    spikes = []

    with torch.no_grad():

            
        spike_count = 0
        modules = []
        hooks = []
        ssims = []
        klds = []
        def hook_fn(module, input, output):
            nonlocal spike_count
            spike_count += output.sum().item()
            
        for module in model.modules():
            if isinstance(module, sj.neuron.BaseNode):
                #hook = last_spiking_layer.register_forward_hook(hook_fn)
                modules.append(module)
        
        for i in [0,1]:
            hook = modules[i].register_forward_hook(hook_fn)
            hooks.append(hook)
        
        encoder = encoding.PoissonEncoder()
        for echo, _, _, _ in tqdm(dataloader, desc="testing"):
            
            if only_mag:
                echo = echo[:,0:1,:,:]         
                
            if only_ang:
                echo = echo[:,1:2,:,:]
            
            if only_mag or only_ang:
                in_channels = 1
            else:
                in_channels = 2
            encoded_img = encoder(echo)
            if model_type == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))
            x_dec_2, x_enc_2, feature = model(encoded_img)
            firing_rate = (spike_count / model.T) / encoded_img.size(0)
            energy = -torch.logsumexp(feature, dim=1)
            spikes.append(torch.full((encoded_img.size(0), 1), firing_rate, dtype=torch.float32).cpu())
            energies.append(energy.cpu())
            features.append(feature.cpu())
            if args.model == 'image':
                ssim = piq.ssim(feature, encoded_img).item()    
                ssims.append(ssim)
            klds.append(torch.nn.functional.kl_div(feature, encoded_img, reduction='batchmean').item())          
            spike_count = 0
        spikes = torch.cat(spikes, dim=0)
            
    for hook in hooks:
        hook.remove()
    if save and path is not None:
        if args.model == 'image':
            with open(path + f'/ssim_values_echo.txt', 'w') as f:
                f.write("SSIM values: " + str(ssims) + "\n")
                f.write("mean SSIM: " + str(np.mean(ssims)) + "\n")

        with open(path + f'/kld_values_echo.txt', 'w') as f:
            f.write("KLD values: " + str(klds) + "\n")
            f.write("mean KLD: " + str(np.mean(klds)) + "\n")
    
    features = torch.cat(features, dim=0)
    energies = torch.cat(energies, dim=0)

    return features.numpy(), spikes.numpy() 

def extract_features_RFI(dataloader, model, only_mag, only_ang, model_type, save=False, path=None):
    model.eval()
    features = []
    energies = []
    spikes = []

    with torch.no_grad():

            
        spike_count = 0
        modules = []
        hooks = []
        ssims = []
        klds = []
        def hook_fn(module, input, output):
            nonlocal spike_count
            spike_count += output.sum().item()
            
        for module in model.modules():
            if isinstance(module, sj.neuron.BaseNode):
                #hook = last_spiking_layer.register_forward_hook(hook_fn)
                modules.append(module)
        
        for i in [1]:
            hook = modules[i].register_forward_hook(hook_fn)
            hooks.append(hook)
        
        encoder = encoding.PoissonEncoder()
        for echo, expanded_RFI, _, _ in tqdm(dataloader, desc="testing"):
            if only_mag or only_ang:
                in_channels = 1
            else:
                in_channels = 2
            RFI = expanded_RFI.view(-1, in_channels, args.H, args.W)
            if only_mag:
                expanded_RFI = expanded_RFI[:,0:1,:,:]         
                
            if only_ang:
                expanded_RFI = expanded_RFI[:,1:2,:,:]
            
            RFI_split = torch.split(RFI, echo.shape[0], dim=0)
            RFI = RFI_split[0]
            encoded_img = encoder(RFI)
            if model_type == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))
            x_dec_2, x_enc_2, feature = model(encoded_img)
            firing_rate = (spike_count / model.T) / encoded_img.size(0)
            energy = -torch.logsumexp(feature, dim=1)
            spikes.append(torch.full((encoded_img.size(0), 1), firing_rate, dtype=torch.float32).cpu())
            energies.append(energy.cpu())
            features.append(feature.cpu())     
            if args.model == 'image':
                ssim = piq.ssim(feature, encoded_img).item()    
                ssims.append(ssim)
            klds.append(torch.nn.functional.kl_div(feature, encoded_img, reduction='batchmean').item())
            spike_count = 0
        spikes = torch.cat(spikes, dim=0)
            
    for hook in hooks:
        hook.remove()
    if save and path is not None:
        if args.model == 'image':
            with open(path + f'/ssim_values_RFI.txt', 'w') as f:
                f.write("SSIM values: " + str(ssims) + "\n")
                f.write("mean SSIM: " + str(np.mean(ssims)) + "\n")
            print(path + f'/ssim_values_RFI.txt')
            print("mean SSIM: " + str(np.mean(ssims)) + "\n")

        with open(path + f'/kld_values_RFI.txt', 'w') as f:
            f.write("KLD values: " + str(klds) + "\n")
            f.write("mean KLD: " + str(np.mean(klds)) + "\n")
        print(path + f'/kld_values_RFI.txt')
    print("mean KLD: " + str(np.mean(klds)) + "\n")

    features = torch.cat(features, dim=0)
    energies = torch.cat(energies, dim=0)

    return features.unsqueeze(1).numpy(), spikes.unsqueeze(1).numpy() 

def plot_jensenshannon(e1, e2, path, id_name, ood_name, type_, tau=None):
    print(e1.shape, e2.shape)
    with open(path + f'/{type_}_values_{id_name}_vs_{ood_name}{args.batch}.txt', 'w') as f:
        f.write("ID shape: " + str(e1.shape) + "\n")
        f.write("OOD shape: " + str(e2.shape) + "\n")
        f.write(f"e1_flat shape: {e1.shape}, min: {e1.min()}, max: {e1.max()}")
        f.write(f"e2_flat shape: {e2.shape}, min: {e2.min()}, max: {e2.max()}")
        f.write(f"e1_flat NaN count: {np.isnan(e1).sum()}")
        f.write(f"e2_flat NaN count: {np.isnan(e2).sum()}")

    js_div = jensenshannon(e1, e2)
    if not isinstance(js_div, np.float64):
        js_div = js_div[0]
    print(f"JS Divergence: {js_div}")

    print(f"e1_flat shape: {e1.shape}, min: {e1.min()}, max: {e1.max()}")
    print(f"e2_flat shape: {e2.shape}, min: {e2.min()}, max: {e2.max()}")
    print(f"e1_flat NaN count: {np.isnan(e1).sum()}")
    print(f"e2_flat NaN count: {np.isnan(e2).sum()}")

    min_val = min(e1.min(), e2.min())
    max_val = max(e1.max(), e2.max())

    # Boxplot
    plt.figure()
    ax = plt.gca()
    sns.boxplot(x=e1, color='blue', label=id_name, fill=True)
    sns.boxplot(x=e2, color='red', label=ood_name, fill=True)
    if tau is not None:
        ax.axvline(x=tau, color='black', linestyle='--', label=f'τ: {tau:.2f}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Category')
    plt.xlabel('spikes')
    plt.title(f"Spike Distribution - JS: {js_div:.3f}")
    plt.tight_layout()
    if os.path.exists(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_box.png'):
        os.remove(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_box.png')
    plt.savefig(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_box.png', dpi=200, bbox_inches='tight')
    plt.close()

    # KDE plot
    plt.figure()
    ax = plt.gca()
    sns.kdeplot(e1, color='blue', label=id_name, fill=True)
    sns.kdeplot(e2, color='red', label=ood_name, fill=True)
    if tau is not None:
        ax.axvline(x=tau, color='black', linestyle='--', label=f'τ: {tau:.2f}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Category')
    plt.xlabel('spikes')
    plt.title(f"Spike Distribution - JS: {js_div:.3f}")
    plt.tight_layout()
    if os.path.exists(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_kde.png'):
        os.remove(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_kde.png')
    plt.savefig(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_kde.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Histogram
    plt.figure()
    ax = plt.gca()
    sns.histplot(e1, color='blue', label=id_name+ f" (n: {e1.shape[0]})", binrange=(min_val, max_val), kde=True)
    sns.histplot(e2, color='red', label=ood_name+ f" (n: {e2.shape[0]})", binrange=(min_val, max_val), kde=True)
    if tau is not None:
        ax.axvline(x=tau, color='black', linestyle='--', label=f'τ: {tau:.2f}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Category')
    plt.xlabel('spikes')
    plt.title(f"Spike Distribution - JS: {js_div:.2f}")
    plt.tight_layout()
    if os.path.exists(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_hist.png'):
        os.remove(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_hist.png')
    plt.savefig(path + f'/{type_}_dist_{id_name}_vs_{ood_name}{args.batch}_hist.png', dpi=200, bbox_inches='tight')
    plt.close()

    return e1.min(), e1.max(), e2.min(), e2.max(), js_div
# Pseudocodice per farti capire il ragionamento
# - L'idea sarebbe vedere come si comparano le distribuizioni originali dei dataset, con quelle rumorose
# - Questo codice vale per l'energia delle reti lineari, poi per le SSN DOVREBBE essere lo stesso, solo che invece di calcolare l'energia, calcoli il firing rate
# - Usa sempre i dati di test
# - Vediamo cosa esce e ci ragioniamo sopra

def main(model_path, output_dir, batch_size, tau):
    
    
    if args.only_mag or args.only_ang:
        in_channels = 1
    else:
        in_channels = 2
    # Load model
    if args.model == 'image':
        model = CSNNanomaly_image(T=args.T, in_channels=in_channels, inner_channels=4)
        if args.vae:
            model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
    else:
        model = CSNNanomaly_line(T=args.T, in_channels=in_channels, inner_channels=4)
        if args.vae:
            model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
       
    state = torch.load(model_path, map_location=torch.device('cpu'))
    state = state['net']
    new_state = {}
    for key, value in state.items():
        new_key = key.replace("module.", "")
        new_state[new_key] = value
    try:
        model.load_state_dict(new_state)
    except RuntimeError as e:
        if args.lif:
            if args.model == 'image':
                model = CSNNanomaly_image_LIF(T=args.T, in_channels=in_channels, inner_channels=4)
                if args.vae:
                    model = CSNNanomaly_image_VAE_LIF(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
            else:
                model = CSNNanomaly_line_LIF(T=args.T, in_channels=in_channels, inner_channels=4)
                if args.vae:
                    model = CSNNanomaly_line_VAE_LIF(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
    
        else:
            if args.model == 'image':
                model = CSNNanomaly_image(T=args.T, in_channels=in_channels, inner_channels=4)
                if args.vae:
                    model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
            else:
                model = CSNNanomaly_line(T=args.T, in_channels=in_channels, inner_channels=4)
                if args.vae:
                    model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2)
        state = torch.load(model_path, map_location=torch.device('cpu'))
        state = state['net']
        new_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        try:
            model.load_state_dict(new_state)
        except RuntimeError as e:
            if args.lif:
                if args.model == 'image':
                    model = CSNNanomaly_image_LIF(T=args.T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                    if args.vae:
                        model = CSNNanomaly_image_VAE_LIF(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                else:
                    model = CSNNanomaly_line_LIF(T=args.T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                    if args.vae:
                        model = CSNNanomaly_line_VAE_LIF(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
        
            else:
                if args.model == 'image':
                    model = CSNNanomaly_image(T=args.T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                    if args.vae:
                        model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                else:
                    model = CSNNanomaly_line(T=args.T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                    if args.vae:
                        model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=args.T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
            state = torch.load(model_path, map_location=torch.device('cpu'))
            state = state['net']
            new_state = {}
            for key, value in state.items():
                new_key = key.replace("module.", "")
                new_state[new_key] = value
    model.eval()

    dataset = fastSARZarrDataset(root_dir=args.data_dir, target_shape=(args.H,args.W), fast=True)
    
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
    # Calcola energia per il dataset di base
    _, id_spikes = extract_features_echo(data , model, output_dir) 

    # Calcola energia per il dataset con rumore 
    _, ood_spikes = extract_features_RFI(data, model, output_dir) 
    while len(id_spikes.shape) >1:
        id_spikes = id_spikes.squeeze(-1)
        print("id_spikes shape after squeeze:", id_spikes.shape)
    while len(ood_spikes.shape) >1:
        ood_spikes = ood_spikes.squeeze(-1)
        print("ood_spikes shape after squeeze:", ood_spikes.shape)
    # Plotta
    id_min, id_max, ood_min, ood_max, js_div = plot_jensenshannon(e1=id_spikes, e2=ood_spikes, path=output_dir, id_name="normal SAR", ood_name="RFI injected", type_='spikes')

def main_multi(model, output_dir, only_mag, only_ang, model_type, data, tau):
    metric_dict = {}
    y_echo = []
    y_RFI = []
    



    # Calcola energia per il dataset di base
    _, id_spikes = extract_features_echo(data, model, only_mag, only_ang, model_type, path=output_dir) 

    # Calcola energia per il dataset con rumore 
    _, ood_spikes = extract_features_RFI(data, model, only_mag, only_ang, model_type, path=output_dir) 
    while len(id_spikes.shape) >1:
        id_spikes = id_spikes.squeeze(-1)
    while len(ood_spikes.shape) >1:
        ood_spikes = ood_spikes.squeeze(-1)
    # Plotta
    id_min, id_max, ood_min, ood_max, js_div = plot_jensenshannon(e1=id_spikes, e2=ood_spikes, path=output_dir, id_name="normal SAR", ood_name="RFI injected", type_='spikes', tau=tau)
    
    print(f"Threshold (tau) for classification: {tau}") 
    print(f"ID spikes min: {id_min}, max: {id_max}")
    print(f"OOD spikes min: {ood_min}, max: {ood_max}")
    y_echo.append((torch.tensor(id_spikes) <= tau).int())
    y_RFI.append((torch.tensor(ood_spikes) <= tau).int())
    
    y_echo = torch.cat(y_echo, dim=0)
    y_RFI = torch.cat(y_RFI, dim=0)
    
    print(f"y_echo max: {y_echo.max()} min: {y_echo.min()}, y_RFI max: {y_RFI.max()}, min: {y_RFI.min()}")

    print("echo:")
    metric_dict["echo"] = calculate_metrics(y_true=torch.zeros(y_echo.shape), y_pred=y_echo)
    print("RFI:")
    metric_dict["RFI"] = calculate_metrics(y_true=torch.ones(y_RFI.shape), y_pred=y_RFI)
    print("total:")
    metric_dict["total"] = calculate_metrics(y_true=torch.cat([torch.zeros(y_echo.shape), torch.ones(y_RFI.shape)]), y_pred=torch.cat([y_echo, y_RFI]))

    metric_dict["js_divergence"] = js_div
    
    with open(output_dir + f"/metrics{tau}.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    

def calculate_metrics(y_true, y_pred):
  metrics_dict = {}
  current_acc = metrics.accuracy_score(y_true, y_pred)
  current_prec = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
  current_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
  current_f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
  current_auroc = metrics.roc_auc_score(y_true, y_pred)
  current_auprc = metrics.average_precision_score(y_true, y_pred)

  metrics_dict["accuracy"] = current_acc
  metrics_dict["precision"] = current_prec
  metrics_dict["recall"] = current_recall
  metrics_dict["f1_score"] = current_f1_score
  metrics_dict["auroc"] = current_auroc
  metrics_dict["auprc"] = current_auprc
  
  print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}, auroc: {current_auroc*100:.2f}, auprc: {current_auprc*100:.2f}")
  return metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model on MNIST with different augmentations.")
    parser.add_argument("--model_path", type=str, help="Path to the model .pt file")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the output JSON file")
    parser.add_argument("--model", type=str, default='image', help="which model to use: image or line")
    parser.add_argument('--vae', default=False, help='use VAE architecture')
    parser.add_argument("--batch", type=int, default=1, help="batch size for evaluation")
    parser.add_argument('--ase3', default=False, type=bool, help='case3 or not')
    parser.add_argument('--tau', type=float, default=None, help='threshold for classification based on energy or firing rate')
    parser.add_argument('--H', type=float, default=224, help='dimension for images based on energy or firing rate')
    parser.add_argument('--W', type=float, default=224, help='dimension for images based on energy or firing rate')
    parser.add_argument('--T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-data_dir', default="./../../data/v2/RFInject", type=str, help='where MNIST data is stored')
    parser.add_argument('--latent_dim', default=1, type=int, help='latent dimension for VAE')
    parser.add_argument('--only_mag', default=False, help='use only magnitude channel')
    parser.add_argument('--only_ang', default=False, help='use only angle channel')
    parser.add_argument('--lif', default=False, help='use LIF neurons instead of IF')
    parser.add_argument('--multy', default=False, help='multiple')
    args = parser.parse_args()
    print("Arguments:", args)
    
    if args.multy:
        base_dirs = ["./line", "./image"]  # Current directory
        data_dir = "/home/ricky/uni/tesi/RFInject/data/v2/RFInject/"

        dataset = fastSARZarrDataset(root_dir=data_dir, target_shape=(args.H,args.W), fast=True)
        
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = int(0.1 * len(dataset))    # 10% for validation
        test_size = len(dataset) - train_size - val_size  # Remaining for test

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        print(len(train_dataset), len(val_dataset), len(test_dataset))
        data = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4)
        
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

        for base_dir in base_dirs:
            for dir_name in os.listdir(base_dir):
                dir_path = os.path.join(base_dir, dir_name)
                if os.path.isdir(dir_path):
                    checkpoint_min = os.path.join(dir_path, "checkpoint_min.pt")
                    checkpoint_latest = os.path.join(dir_path, "checkpoint_latest.pt")

                # Determine model type
                model_type = "image" if "image" in dir_name else "line"
                args.model = model_type
                # Determine T
                T = 100 if "T100" in dir_name else 10
                


                # Determine flags
                lif = "lifTrue" in dir_name
                vae = "vaeTrue" in dir_name
                only_mag = "only_magTrue" in dir_name
                only_ang = "only_angTrue" in dir_name

                if os.path.exists(checkpoint_min):
                    model_path = checkpoint_min
                elif os.path.exists(checkpoint_latest):
                    model_path = checkpoint_latest
                else:
                    continue  # Skip if no checkpoint file is found
                
                if only_mag or only_ang:
                    in_channels = 1
                else:
                    in_channels = 2
                # Load model
                if model_type == 'image':
                    model = CSNNanomaly_image(T=T, in_channels=in_channels, inner_channels=4)
                    if vae:
                        model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                else:
                    model = CSNNanomaly_line(T=T, in_channels=in_channels, inner_channels=4)
                    if vae:
                        model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                
                state = torch.load(model_path, map_location=torch.device('cpu'))
                state = state['net']
                new_state = {}
                for key, value in state.items():
                    new_key = key.replace("module.", "")
                    new_state[new_key] = value
                try:
                    model.load_state_dict(new_state)
                except RuntimeError as e:
                    if lif:
                        if model_type == 'image':
                            model = CSNNanomaly_image_LIF(T=T, in_channels=in_channels, inner_channels=4)
                            if vae:
                                model = CSNNanomaly_image_VAE_LIF(w=args.W, h=args.H, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                        else:
                            model = CSNNanomaly_line_LIF(T=T, in_channels=in_channels, inner_channels=4)
                            if vae:
                                model = CSNNanomaly_line_VAE_LIF(w=args.W, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                
                    else:
                        if model_type == 'image':
                            model = CSNNanomaly_image(T=T, in_channels=in_channels, inner_channels=4)
                            if vae:
                                model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                        else:
                            model = CSNNanomaly_line(T=T, in_channels=in_channels, inner_channels=4)
                            if vae:
                                model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2)
                    state = torch.load(model_path, map_location=torch.device('cpu'))
                    state = state['net']
                    new_state = {}
                    for key, value in state.items():
                        new_key = key.replace("module.", "")
                        new_state[new_key] = value
                    try:
                        model.load_state_dict(new_state)
                    except RuntimeError as e:
                        if lif:
                            if model_type == 'image':
                                model = CSNNanomaly_image_LIF(T=T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                                if vae:
                                    model = CSNNanomaly_image_VAE_LIF(w=args.W, h=args.H, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                            else:
                                model = CSNNanomaly_line_LIF(T=T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                                if vae:
                                    model = CSNNanomaly_line_VAE_LIF(w=args.W, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                    
                        else:
                            if model_type == 'image':
                                model = CSNNanomaly_image(T=T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                                if vae:
                                    model = CSNNanomaly_image_VAE(w=args.W, h=args.H, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                            else:
                                model = CSNNanomaly_line(T=T, in_channels=in_channels, inner_channels=4, smaller_kernel=True)
                                if vae:
                                    model = CSNNanomaly_line_VAE(w=args.W, latent_dim=args.latent_dim, T=T, in_channels=in_channels, inner_channels=2, smaller_kernel=True)
                        state = torch.load(model_path, map_location=torch.device('cpu'))
                        state = state['net']
                        new_state = {}
                        for key, value in state.items():
                            new_key = key.replace("module.", "")
                            new_state[new_key] = value
                model.eval()
                            
                # Calcola energia per il dataset di base
                _, id_spikes = extract_features_echo(train_data, model, only_mag, only_ang, model_type, save=False) 

                # Calcola energia per il dataset con rumore 
                _, ood_spikes = extract_features_RFI(train_data, model, only_mag, only_ang, model_type, save=False)
                
                tau = (id_spikes.min() + ood_spikes.min()) / 2
                
                print(dir_path)
                print(f"model_path:{model}, dir_name:{dir_path}, only_mag:{only_mag}, only_ang:{only_ang}, model_type:{model_type}, tau:{tau}")
                main_multi(model, dir_path, only_mag, only_ang, model_type=model_type, data=data, tau=tau)
    else:
        main(args.model_path, args.output_dir, args.batch, args.tau)