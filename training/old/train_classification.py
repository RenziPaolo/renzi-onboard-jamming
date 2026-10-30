import sys
from pathlib import Path

# Add the project root to Python's module search path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import os

os.environ['MPLCONFIGDIR'] = "/work/project"
import torch
import gc
import json
from sklearn import metrics
from models.classification.conv import SAR_RF_Interference_Classifier, SAR_RF_Interference_Classifier_simp
from models.classification.LCNNGWHA import LCNN_GWHA
from models.classification.snn import SAR_RF_Interference_Classifier_SNN
from dataset.dataset import SARZarrDataset
from tqdm import tqdm
H = 512
W = 512

def train (num_epochs, training_metrics_dict, validation_metrics_dict, training_loader,
           validation_loader, model, loss_fn, optimizer, plateau_scheduler, training_path, cfg):
    
    best_vloss = 10000000000

    for epoch in range(num_epochs):
        
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Compute the average loss and the predictions vs true values
        # Train the model for the single epoch
        #model.train()
        loss_avg, y_pred_list, y_true_list = train_one_epoch(training_loader, model, loss_fn, optimizer)

        # Calculate the metrics
        print("TRAIN:")
        calculate_metrics(y_true_list, y_pred_list, training_metrics_dict, loss_avg, epoch) 
        
        # Validation part (disable the gradient computation)
        # Set the model in validation mode
        model.eval() 
        with torch.no_grad():
            vy_pred_list = []
            vy_true_list = []
            vloss_epoch = 0
            for vdata in tqdm(training_loader , desc="Validation", unit="batch"):
                
                # Divide the tuple data in inputs tensor and labels tensor (batch_size dimension)
                #print(data)
                vecho, vexpanded_RFI, _, _ = vdata
                
                vRFI = vexpanded_RFI.view(-1, 2, H, W)
                
                vy_true = torch.Tensor([1.]*vecho.shape[0]+[0.]*vecho.shape[0]).unsqueeze(dim=1).to('cuda')

                #print("vecho.shape", vecho.shape)
                #print("vRFI.shape", vRFI.shape)
                
                vRFI_split = torch.split(vRFI, vecho.shape[0], dim=0)
                vRFI = vRFI_split[0]
                
                v_data = ((vecho, torch.Tensor([[1.]*vecho.shape[0]]).mT), (vRFI, torch.Tensor([[0.]*vRFI.shape[0]]).mT))  

                #print("v_data",v_data[0][0].shape, v_data[0][1].shape, v_data[1][0].shape, v_data[1][1].shape)
        

                #print("vx",vx.shape)
                #print("vy_pred",vy_pred.shape)
                #print("vy_true",vy_true.shape)                
                vy_pred = model(torch.cat((vecho, vRFI), dim=0).to('cuda'))
                vloss = loss_fn(vy_pred, vy_true)
                #print("vloss",vloss)
                vloss_epoch += vloss.detach().item()
                vy_true_list += vy_true.squeeze(dim=1).detach().cpu().tolist()
                vy_pred_list += (torch.sigmoid(vy_pred.detach()) >=0.5 ).int().cpu().tolist()
            #print("vloss_epoch", vloss_epoch)  
            vloss_avg = vloss_epoch / len(validation_loader)
            
            plateau_scheduler.step(vloss_avg)

            print("VALIDATION:")
            calculate_metrics(vy_true_list, vy_pred_list, validation_metrics_dict, vloss_avg, epoch)

        print(f'LOSS train {loss_avg} valid {vloss_avg}')

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            save_path = os.path.join(training_path, "results")
            parent_dir = save_model(model, save_path) # the name_path is: "your_path/2024-06-25_14:06:10/model.pt"
    
    results_path = parent_dir

    name_training_metrics = '/training_metrics.json'
    name_validation_metrics = '/validation_metrics.json'

    save_metrics(training_metrics_dict, results_path + name_training_metrics)
    save_metrics(validation_metrics_dict, results_path + name_validation_metrics)
            
    save_hydra_config(cfg, results_path+'/parameters.txt')

def train_one_epoch(training_loader, model, loss_fn, optimizer):

    # Initialization of lists (predictions and true values) and the value of loss_epoch
    y_pred_list = []
    y_true_list = []
    loss_epoch = 0

    for data in tqdm(training_loader , desc="Training", unit="batch"):
        # Divide the tuple data in inputs tensor and labels tensor (batch_size dimension)
        #print(data)
        echo, expanded_RFI, _, _ = data
        
        RFI = expanded_RFI.view(-1, 2, H, W)
        RFI_split = torch.split(RFI, echo.shape[0], dim=0)
        RFI = RFI_split[0]        
        #print("x_data",x)
        
        # Put the gradient to zero for every batch
        optimizer.zero_grad()
        
        echo = echo.to('cuda')
        RFI = RFI.to('cuda')
        
        y_true = torch.Tensor([1.]*echo.shape[0]+[0.]*echo.shape[0]).unsqueeze(dim=1).to('cuda')
        # Forward pass for echo
        y_pred = model(torch.cat((echo, RFI), dim=0).to('cuda'))
        loss_value = loss_fn(y_pred, y_true)



        # Compute the loss and the gradient
        #y_pred = torch.argmax(y_pred, dim)
        #print("y_pred", y_pred)
        #print("y_true", y_true)
        #print("y_pred", y_pred.shape)
        #print("y_true", y_true.shape)
        #print('echo.shape[0]', echo.shape[0])
        #print("RFI.shape[0]", RFI.shape[0])
        #loss_value = loss_fn(y_pred, y_true)

        loss_value.backward()

        # Update the weights
        optimizer.step()

        #print("loss_report", loss_report)
        # Summing the values of the loss in each batch of the epoch
        loss_epoch += loss_value.detach().item()

        # Add the "batch" predictions and true values to the corrispettive lists
        #print("y_true", y_true.tolist())
        #y_true_tmp = torch.argmax(y_true).cpu().item()
        #print("y_true_tmp", y_true_tmp)
        
        y_true_list += y_true.squeeze(dim=1).detach().cpu().tolist()
        #print("y_pred", y_pred.cpu().tolist())
        y_pred_list += (torch.sigmoid(y_pred.detach()) >=0.5 ).int().cpu().tolist()

        #print("y_pred_list",y_pred_list)
        #print("y_true_list",y_true_list)

        del loss_value
        gc.collect()

        '''
        print("y_true_list", y_true_list)
        print("y_pred_list", y_pred_list)
        print("y_pred", y_pred)
        print("torch.argmax(y_pred)", torch.argmax(y_pred))
        '''
        

        torch.cuda.empty_cache()

    loss_avg = loss_epoch / len(training_loader)   
    
    return loss_avg, y_pred_list, y_true_list
        
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved:", path)
    return path

def calculate_metrics(y_true, y_pred, metrics_dict, loss, epoch, inference_time_list = None, test=False):

  current_acc = metrics.accuracy_score(y_true, y_pred)
  current_prec = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
  current_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
  current_f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)

  if not inference_time_list == None:
    inference_time_one_sample = round(torch.tensor(inference_time_list).mean().item(),3)
    inference_time_tot = round(torch.tensor(inference_time_list).sum().item(),3)
    metrics_dict["inference_time_one_sample"].append(inference_time_one_sample)
    metrics_dict["inference_time_tot"].append(inference_time_tot)

  if test == False:
    metrics_dict["epoch"].append(epoch)

  metrics_dict["loss"].append(loss)
  metrics_dict["accuracy"].append(current_acc)
  metrics_dict["precision"].append(current_prec)
  metrics_dict["recall"].append(current_recall)
  metrics_dict["f1_score"].append(current_f1_score)

  print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}")

def save_metrics(metrics, path):
  with open(path, "w") as file:
    json.dump(metrics, file, indent=4)
    print("Metrics saved:", path)

def save_hydra_config(cfg, save_path):
    """
    Save Hydra configuration parameters to a specified text file.

    :param cfg: Hydra config object
    :param save_path: Path to save the text file
    """
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Write the configuration to a text file
    with open(save_path, 'w') as f:
        f.write("Hydra Configuration Parameters:\n")
        f.write("-" * 40 + "\n")
        
        # Recursively write the parameters in cfg
        def recursive_print(cfg, f, indent=0):
            for key, value in cfg.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    recursive_print(value, f, indent + 4)
                else:
                    f.write(f"{' ' * indent}{key}: {value}\n")
        
        recursive_print(cfg, f)

    print(f"Hydra configuration saved to {save_path}")
    
if __name__ == "__main__":
    print(sys.path)
    model_cnn = SAR_RF_Interference_Classifier().to('cuda')
    model_cnn_simp = SAR_RF_Interference_Classifier_simp().to('cuda')
    model_snn = SAR_RF_Interference_Classifier_SNN(H, W).to('cuda')
    model_lcnn = LCNN_GWHA(num_classes=1).to('cuda')
    
    if torch.cuda.device_count() > 1:
        dataset = SARZarrDataset(root_dir='/mnt/ssd2/renzi/RFInject', target_shape=(H,W), device='cuda')
        print("Using", torch.cuda.device_count(), "GPUs!")
        model_cnn = torch.nn.DataParallel(model_cnn)
        model_cnn_simp = torch.nn.DataParallel(model_cnn_simp)
        model_snn = torch.nn.DataParallel(model_snn)
        model_lcnn = torch.nn.DataParallel(model_lcnn)
        model_cnn.cuda()
        model_cnn_simp.cuda()
        model_snn.cuda()
        model_lcnn.cuda()
    
    else:
        dataset = SARZarrDataset(root_dir='../../data/v1/2_processed/', target_shape=(H,W), device='cuda')
    from torch.utils.data import random_split


    print(model_lcnn.device_ids)

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    training_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    validation_metrics_dict = {"epoch":[], "loss":[], "accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-4)
    optimizer_snn = torch.optim.Adam(model_snn.parameters(), lr=1e-3)
    optimizer_lcnn = torch.optim.Adam(model_lcnn.parameters(), lr=1e-4)
    optimizer_cnn_simp = torch.optim.Adam(model_cnn_simp.parameters(), lr=1e-4)
    plateau_scheduler_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.1, patience=5)
    plateau_scheduler_snn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_snn, mode='min', factor=0.1, patience=5)
    plateau_scheduler_lcnn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lcnn, mode='min', factor=0.1, patience=5)
    plateau_scheduler_cnn_simp = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn_simp, mode='min', factor=0.1, patience=5)
    training_path_cnn = './training_runs_cnn'
    training_path_snn = './training_runs_snn'
    training_path_lcnn = './training_runs_lcnn'
    training_path_cnn_simp = './training_runs_cnn_simp'
    
    cfg = {}
    '''
    train(num_epochs=10, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
          training_loader=training_loader, validation_loader=validation_loader, model=model_cnn, loss_fn=loss_fn,
          optimizer=optimizer_cnn, plateau_scheduler=plateau_scheduler_cnn, training_path=training_path_cnn, cfg=cfg)
    train(num_epochs=10, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
          training_loader=training_loader, validation_loader=validation_loader, model=model_cnn_simp, loss_fn=loss_fn,
          optimizer=optimizer_cnn_simp, plateau_scheduler=plateau_scheduler_cnn_simp, training_path=training_path_cnn_simp, cfg=cfg)
    '''
    train(num_epochs=10, training_metrics_dict=training_metrics_dict, validation_metrics_dict=validation_metrics_dict,
          training_loader=training_loader, validation_loader=validation_loader, model=model_lcnn, loss_fn=loss_fn,
          optimizer=optimizer_lcnn, plateau_scheduler=plateau_scheduler_lcnn, training_path=training_path_lcnn, cfg=cfg)