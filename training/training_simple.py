import sys
from pathlib import Path

# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import torch
import gc
import json
from sklearn import metrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.classification.conv import SAR_RF_Interference_Classifier_simp
from models.classification.linear import SimpleLinearModel
import argparse
if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

torch.manual_seed(42)

def train(num_epochs, training_metrics_dict, validation_metrics_dict, training_loader, validation_loader, model, loss_fn, optimizer, plateau_scheduler, training_path, cfg):
    best_vloss = 10000000000

    results_path = training_path +'/'+ cfg['model'] + str(args.case3)
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Train the model for one epoch
        loss_avg, y_pred_list, y_true_list = train_one_epoch(training_loader, model, loss_fn, optimizer)
        print("TRAIN:")
        calculate_metrics(y_true_list, y_pred_list, training_metrics_dict, loss_avg, epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            vy_pred_list = []
            vy_true_list = []
            vloss_epoch = 0
            for vdata, vtarget in tqdm(validation_loader, desc="Validation", unit="batch"):
                vdata, vtarget = vdata.to('cuda'), vtarget.to('cuda')
                vy_pred = model(vdata)
                vloss = loss_fn(vy_pred, vtarget)
                vloss_epoch += vloss.detach().item()
                vy_true_list += vtarget.detach().cpu().tolist()
                vy_pred_list += torch.argmax(vy_pred.detach(), dim=1).cpu().tolist()

            vloss_avg = vloss_epoch / len(validation_loader)
            plateau_scheduler.step(vloss_avg)
            print("VALIDATION:")
            calculate_metrics(vy_true_list, vy_pred_list, validation_metrics_dict, vloss_avg, epoch)

        print(f'LOSS train {loss_avg} valid {vloss_avg}')
        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            save_model(model, results_path)

    name_training_metrics = '/training_metrics.json'
    name_validation_metrics = '/validation_metrics.json'
    save_metrics(training_metrics_dict, results_path + name_training_metrics)
    save_metrics(validation_metrics_dict, results_path + name_validation_metrics)
    save_hydra_config(cfg, results_path + '/parameters.txt')

def train_one_epoch(training_loader, model, loss_fn, optimizer):
    y_pred_list = []
    y_true_list = []
    loss_epoch = 0
    for data, target in tqdm(training_loader, desc="Training", unit="batch"):
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        y_pred = model(data)
        loss_value = loss_fn(y_pred, target)
        loss_value.backward()
        optimizer.step()
        loss_epoch += loss_value.detach().item()
        y_true_list += target.detach().cpu().tolist()
        y_pred_list += torch.argmax(y_pred.detach(), dim=1).cpu().tolist()
        del loss_value
        gc.collect()
        torch.cuda.empty_cache()

    loss_avg = loss_epoch / len(training_loader)
    return loss_avg, y_pred_list, y_true_list

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, f"model{args.dataset}.pt")
    torch.save(model.state_dict(), model_path)
    print("Model saved:", model_path)

def calculate_metrics(y_true, y_pred, metrics_dict, loss, epoch, inference_time_list=None, test=False):
    current_acc = metrics.accuracy_score(y_true, y_pred)
    current_prec = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    current_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    current_f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    if not inference_time_list is None:
        inference_time_one_sample = round(torch.tensor(inference_time_list).mean().item(), 3)
        inference_time_tot = round(torch.tensor(inference_time_list).sum().item(), 3)
        metrics_dict["inference_time_one_sample"].append(inference_time_one_sample)
        metrics_dict["inference_time_tot"].append(inference_time_tot)
    if not test:
        metrics_dict["epoch"].append(epoch)
    metrics_dict["loss"].append(loss)
    metrics_dict["accuracy"].append(current_acc)
    metrics_dict["precision"].append(current_prec)
    metrics_dict["recall"].append(current_recall)
    metrics_dict["f1_score"].append(current_f1_score)
    print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}")

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(metrics, file, indent=4)
        print("Metrics saved:", path)

def save_hydra_config(cfg, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as f:
        f.write("Hydra Configuration Parameters:\n")
        f.write("-" * 40 + "\n")
        def recursive_print(cfg, f, indent=0):
            for key, value in cfg.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    recursive_print(value, f, indent + 4)
                else:
                    f.write(f"{' ' * indent}{key}: {value}\n")
        recursive_print(cfg, f)
    print(f"Hydra configuration saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-model', default='standard', type=str, help='type of model standard or conv')
    parser.add_argument('-path', default='./results', type=str, help='where to save the results')
    parser.add_argument('-epoch', default=100, type=int, help='number of epochs to train')
    parser.add_argument('-batch', default=64, type=int, help='batch size for training')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-data_dir', default="./../../data", type=str, help='where MNIST data is stored')
    parser.add_argument('-dataset', default='mnist', type=str, help='which dataset to use mnist or CIFAR10')
    parser.add_argument('-case3', default=False, type=bool, help='case3 or not')
    args = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

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
            
            print(train_dataset[0][0].shape)
            
    elif args.dataset == 'CIFAR10':
        transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_cifar)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_cifar)
    elif args.dataset == 'EuroSAT':
        from torch.utils.data import random_split
        transform_eurosat = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.3443, 0.3809, 0.4082), (0.1814, 0.1235, 0.1126))])
        dataset = datasets.EuroSAT(root=args.data_dir, download=True, transform=transform_eurosat)
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)  # 80% for training
        val_size = int(0.1 * dataset_size)    # 10% for validation
        test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # Initialize model, loss, optimizer, and scheduler
    
    if args.dataset == 'CIFAR10':
        model_conv = SAR_RF_Interference_Classifier_simp(input_channels = 3, num_classes=10).to('cuda')
        model_linear = SimpleLinearModel(input_dim=32*32*3, output_dim=10).to('cuda')
    if args.dataset == 'mnist':
        model_conv = SAR_RF_Interference_Classifier_simp(input_channels = 1, num_classes=10).to('cuda')
        model_linear = SimpleLinearModel(input_dim=28*28, output_dim=10).to('cuda')
    if args.dataset == 'EuroSAT':
        model_conv = SAR_RF_Interference_Classifier_simp(input_channels = 3, num_classes=10).to('cuda')
        model_linear = SimpleLinearModel(input_dim=64*64*3, output_dim=10).to('cuda')
    if args.model == 'standard':
        model = model_linear
    elif args.model == 'conv':
        model = model_conv
    model.cuda()
    model = torch.nn.DataParallel(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Initialize metrics dictionaries
    training_metrics_dict = {"epoch": [], "loss": [], "accuracy": [], "precision": [], "recall": [], "f1_score": []}
    validation_metrics_dict = {"epoch": [], "loss": [], "accuracy": [], "precision": [], "recall": [], "f1_score": []}

    # Train the model
    train(
        num_epochs=args.epoch,
        training_metrics_dict=training_metrics_dict,
        validation_metrics_dict=validation_metrics_dict,
        training_loader=train_loader,
        validation_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        plateau_scheduler=plateau_scheduler,
        training_path=args.path,
        cfg={"model": args.model, "epochs": args.epoch, "batch_size": args.batch}
    )