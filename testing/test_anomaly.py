import sys
from pathlib import Path 
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from sklearn import metrics
import argparse
from dataset.fast_dataset import fastSARZarrDataset
import torch
import os
import json
import spikingjelly.activation_based as sj
from spikingjelly.activation_based import encoding
from torch.utils.data import random_split
from tqdm import tqdm
# Add the project root to Python's module search path

    
from models.anomalyDetection.CSNNanomaly import CSNNanomaly_image, CSNNanomaly_line, CSNNanomaly_image_VAE, CSNNanomaly_line_VAE, CSNNanomaly_image_LIF, CSNNanomaly_line_LIF, CSNNanomaly_image_VAE_LIF, CSNNanomaly_line_VAE_LIF
torch.manual_seed(42)

def evaluate_model_spike(model, dataloader, tau=None):
    # Find the last spiking layer
    modules = []
    hooks = []
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
    
    spike_count = 0
    logits = []
    spikes_echo = []
    spikes_RFI = []
    y_echo = []
    y_RFI = []
    

    with torch.no_grad():
        encoder = encoding.PoissonEncoder()
        for echo, expanded_RFI, _, _ in tqdm(dataloader, desc="testing"):
            if args.only_mag:
                echo = echo[:,0:1,:,:]
                expanded_RFI = expanded_RFI[:,0:1,:,:]
                in_channels = 1
            elif args.only_ang:
                echo = echo[:,1:2,:,:]
                expanded_RFI = expanded_RFI[:,1:2,:,:]
                in_channels = 1
            else:
                in_channels = 2
                
            encoded_img = encoder(echo)
            if args.model == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))

            x_dec_2, x_enc_2, out_fr_ = model(encoded_img)
            logits.append(out_fr_)
            # Calculate firing rate for this batch
            firing_rate = (spike_count / model.T) / echo.size(0)
            if tau is not None:
                y_echo.append((torch.tensor(firing_rate).unsqueeze(0) >= tau).int())
            # Convert to tensor and reshape to [batch_size, 1]
            spikes_echo.append(torch.full((echo.size(0), 1), firing_rate, dtype=torch.float32))
            spike_count = 0
            RFI = expanded_RFI.view(-1, in_channels, args.H, args.W)
            RFI_split = torch.split(RFI, echo.shape[0], dim=0)
            RFI = RFI_split[0]
            encoded_img = encoder(RFI)
            if args.model == 'line':
                encoded_img = torch.reshape(encoded_img, (-1, in_channels, args.H))
            x_dec_2, x_enc_2, out_fr_ = model(encoded_img)
            logits.append(out_fr_)
            firing_rate = (spike_count / model.T) / RFI.size(0)
            if tau is not None:
                y_RFI.append((torch.tensor(firing_rate).unsqueeze(0) >= tau).int())
            # Convert to tensor and reshape to [batch_size, 1]
            spikes_RFI.append(torch.full((RFI.size(0), 1), firing_rate, dtype=torch.float32))
            spike_count = 0   
                
            
    for hook in hooks:
        hook.remove()

    return torch.cat(logits, dim=0), (torch.cat(spikes_echo, dim=0), torch.cat(spikes_RFI, dim=0)), (torch.cat(y_echo, dim=0), torch.cat(y_RFI, dim=0)) if tau is not None else None

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

  metrics_dict["accuracy"] = current_acc
  metrics_dict["precision"] = current_prec
  metrics_dict["recall"] = current_recall
  metrics_dict["f1_score"] = current_f1_score

  print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}")
  return metrics_dict

# Main function
def main(model_path, output_dir, batch_size, tau):
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
    data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # Evaluate under different conditions
    #for data, name in datasets:

    result = {}
    metric_dict = {}
    logits, spikes, y = evaluate_model_spike(model, data, tau=tau)
    result['spikes_echo_avg'] = (torch.mean(spikes[0]).item())
    result['spikes_echo_dev'] = (torch.std(spikes[0]).item())
    result['spikes_RFI_avg'] = (torch.mean(spikes[1]).item())
    result['spikes_RFI_dev'] = (torch.std(spikes[1]).item())
    if tau is not None:
        print("echo:")
        metric_dict["echo"] = calculate_metrics(y_true=torch.zeros(y[0].shape), y_pred=y[0])
        print("RFI:")
        metric_dict["RFI"] = calculate_metrics(y_true=torch.ones(y[1].shape), y_pred=y[1])

    result['logits_avg'] = (torch.mean(logits).item())
    result['logits_dev'] = (torch.std(logits).item())
    
    energy = calculate_energy(logits, T=1.0)
    avg_energy = torch.mean(energy).item()
    result['energy'] = (avg_energy)
    
    print(f"Average logit value  : {result['logits_avg']:.4f}")
    print(f"standard deviation logit value  : {result['logits_dev']:.4f}")
    print(f"Average energy value  : {result['energy']:.4f}")
    print(f"Average firing rate echo : {result['spikes_echo_avg']:.4f}")
    print(f"Average firing rate RFI : {result['spikes_RFI_avg']:.4f}")

    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
        
    output_path = os.path.join(output_dir, "metrics_tau"+str(tau)+".json")
    with open(output_path, "w") as f:
        json.dump(metric_dict, f, indent=4)

    print(f"Results saved to {output_path}")

# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model on SAR with different augmentations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model .pt file")
    parser.add_argument("--output_dir", type=str, default="./", required=True, help="Directory to save the output JSON file")
    parser.add_argument("--model", type=str, default='image', required=True, help="which model to use: image or line")
    parser.add_argument('--vae', default=False, help='use VAE architecture')
    parser.add_argument("--batch", type=int, default=64, required=True, help="batch size for evaluation")
    parser.add_argument('--ase3', default=False, type=bool, help='case3 or not')
    parser.add_argument('--tau', type=float, default=None, help='threshold for classification based on energy or firing rate')
    parser.add_argument('--H', type=float, default=224, help='dimension for images based on energy or firing rate')
    parser.add_argument('--W', type=float, default=224, help='dimension for images based on energy or firing rate')
    parser.add_argument('--T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-data_dir', default="./../../data/v2/RFInject", type=str, help='where SAR data is stored')
    parser.add_argument('--latent_dim', default=1, type=int, help='latent dimension for VAE')
    parser.add_argument('--only_mag', default=False, help='use only magnitude channel')
    parser.add_argument('--only_ang', default=False, help='use only angle channel')
    parser.add_argument('--lif', default=False, help='use LIF neurons instead of IF')
    args = parser.parse_args()
    print("Arguments:", args)
    main(args.model_path, args.output_dir, args.batch, args.tau)