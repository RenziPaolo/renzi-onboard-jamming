import torch
import zarr
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
import ujson as json

class fastSARZarrDataset(Dataset):
    def __init__(self, root_dir, fast=True, rfi=range(0,10), target_shape=(224, 224), direct=False, paths=False, device='cpu', verbose=False, log=False, small=False, normalize_bbox=False):
        """
        Args:
            root_dir (str): Path to the directory containing all .zarr files.
            array_type (str): Type of array to load ('echo' or 'RFI').
        """
        self.root_dir = root_dir
        self.rfi = rfi
        self.target_shape = target_shape
        self.zarr_files = []
        self.bursts_per_file = []
        self.burst_to_bbox_tensor = {}
        self.burst_to_category_tensor = {}
        self.idx_to_bbox_tensor = []
        self.idx_to_category_tensor = []
        self.cumulative_bursts = [0]
        self.direct = direct
        self.device = device
        if paths:
            self.paths = []
        else:
            self.paths =None
        self.verbose = verbose
        self.log = log
        i = [0, 0]
        j = 0

        # Scan all .zarr files and count bursts
        if target_shape == (224, 224) and fast:
            saved = torch.load(f"{root_dir}/echo_RFI.pt")
            self.tensor_echo_fast = saved["echo"]
            self.split_RFI_fast = saved["RFI"]
        elif target_shape == (512, 512) and fast:
            saved = torch.load(f"{root_dir}/echo_RFI_512.pt")
            self.tensor_echo_fast = saved["echo"]
            self.split_RFI_fast = saved["RFI"]
        elif target_shape == (256, 256) and fast:
            saved = torch.load(f"{root_dir}/echo_RFI_256.pt")
            self.tensor_echo_fast = saved["echo"]
            self.split_RFI_fast = saved["RFI"]
        elif target_shape == (256, 256) and fast and log:
            saved = torch.load(f"{root_dir}/echo_RFI_256_log.pt")
            self.tensor_echo_fast = saved["echo"]
            self.split_RFI_fast = saved["RFI"]
        for file_name in tqdm(sorted(os.listdir(root_dir)), desc="Creating datasets") :
            file_path = os.path.join(root_dir, file_name)
            if file_name.endswith('.zarr'):
                z = zarr.open(file_path, mode='r')
                bursts = [key for key in z.keys() if key.startswith('burst_')]
                if not fast:
                    self.zarr_files.append(z)
                    if paths:
                        self.paths.append(file_path)
                    self.bursts_per_file.append(len(bursts))
                    self.cumulative_bursts.append(self.cumulative_bursts[-1] + len(bursts))
                
                num_bursts = len(bursts)
                bboxes_tensor = torch.zeros((num_bursts, 10, 4), dtype=torch.float32)
                categories_tensor = torch.ones((num_bursts, 10, 1), dtype=torch.int64)
                                    
                for burst in bursts:
                    
                    echo = z[burst]['echo']
                    h, w = echo.shape
                    
                    #all_bboxes = []
                    #all_categories = []
                    rfi_params = z[burst].attrs.asdict()['rfi_params']
                    bboxes_tensor = torch.tensor([
                        [
                            rfi['AffectedRg'][0] / w if normalize_bbox else rfi['AffectedRg'][0],
                            rfi['AffectedAz'][0] / h if normalize_bbox else rfi['AffectedAz'][0],
                            rfi['AffectedRg'][1] / w if normalize_bbox else rfi['AffectedRg'][1],
                            rfi['AffectedAz'][1] / h if normalize_bbox else rfi['AffectedAz'][1],
                        ]
                        for rfi in rfi_params[:10]
                    ], dtype=torch.float32)
                    
                    categories_tensor = torch.tensor([[1] for _ in range(10)], dtype=torch.int64)
                    if self.verbose:
                        print("bbox shape", bboxes_tensor.shape,"categories shape", categories_tensor.shape)
                    if not fast:
                        self.burst_to_bbox_tensor[burst] = bboxes_tensor
                        self.burst_to_category_tensor[burst] = categories_tensor
                    self.idx_to_bbox_tensor.append((bboxes_tensor, j, burst))
                    self.idx_to_category_tensor.append((categories_tensor, j, burst))
                    
                z.store.close()  # Close the file after counting bursts
                j += 1
        self.idx_to_bbox_tensor = [a for a, _, _ in sorted(self.idx_to_bbox_tensor, key=lambda x: (int(x[1]), int(x[2].replace("burst_",""))))]
        self.idx_to_category_tensor = [a for a, _, _ in sorted(self.idx_to_category_tensor, key=lambda x: (x[1], x[2]))]
        self.total_bursts = self.cumulative_bursts[-1]

        if self.verbose:
            print(self.total_bursts)
            print(len(self.zarr_files))
            print(len(self.bursts_per_file))
            if paths:
                print(len(self.paths))

        self.fast = fast
        self.small = small
        

    def __len__(self):
        if self.small:
            return 100
        if self.fast:
            return self.tensor_echo_fast.shape[0]
        return self.total_bursts

    def __getitem__(self, idx):
        if self.fast:
            return self.tensor_echo_fast[idx], self.split_RFI_fast[idx], self.idx_to_bbox_tensor[idx], self.idx_to_category_tensor[idx]
        
        if self.verbose:
            print(f"getting index {idx}")
        # Find which file and burst the index corresponds to
        file_idx = 0
        while idx >= self.cumulative_bursts[file_idx + 1]:
            file_idx += 1
            if self.verbose:
                print(f"file index {file_idx}")
        burst_idx = idx - self.cumulative_bursts[file_idx]
        if self.verbose:
            print(f"burst index {burst_idx}")
        burst_key = f'burst_{burst_idx}'
        bbox = self.burst_to_bbox_tensor[burst_key]
        category = self. burst_to_category_tensor[burst_key]
       
        label = (bbox, category)
        if self.paths:
            path = self.paths[file_idx]
            if self.verbose:
                print(path)
            
        # Load the burst
        z = self.zarr_files[file_idx]
        if self.verbose:
            print(z)
            for key in z.keys():
                print(key)
        echo = z[burst_key]['echo']
        rfi = z[burst_key]['rfi']
            
        if self.direct:
            # Directly return tensors without resizing
            echo = echo[:]
            rfi = rfi[:][self.rfi]
            z.store.close()  # Close the file immediately
            return echo, echo + rfi, label, path if self.paths else torch.tensor(idx)
        
        # Convert to tensor
        tensor_echo = torch.from_numpy(echo[:])
        tensor_rfi = torch.from_numpy(rfi[:])[self.rfi]
        del echo, rfi
        z.store.close()  # Close the file immediately
        
        if tensor_echo.is_complex():
            real_part_echo = tensor_echo.real.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            imag_part_echo = tensor_echo.imag.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            del tensor_echo
            # Resize real and imaginary parts
            real_part_echo = F.interpolate(
                real_part_echo,
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Shape: (1, 224, 224)

            imag_part_echo = F.interpolate(
                imag_part_echo,
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Shape: (1, 224, 224)


        else:
            # If not complex, just resize as before
            tensor_echo = tensor_echo.unsqueeze(0).float()
            tensor_echo = F.interpolate(
                tensor_echo.unsqueeze(0),
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)


        if tensor_rfi.is_complex():
            real_part_rfi = tensor_rfi.real.unsqueeze(0) # Shape: (1, 1, H, W)
            imag_part_rfi = tensor_rfi.imag.unsqueeze(0)  # Shape: (1, 1, H, W)

            del tensor_rfi

            # Resize real and imaginary parts
            real_part_rfi = F.interpolate(
                real_part_rfi,
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Shape: (1, 224, 224)

            imag_part_rfi = F.interpolate(
                imag_part_rfi,
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Shape: (1, 224, 224)


        else:
            # If not complex, just resize as before
            tensor_rfi = tensor_rfi.unsqueeze(0).float()
            tensor_rfi = F.interpolate(
                tensor_rfi,
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        real_part_label = real_part_echo + real_part_rfi
        imag_part_label = imag_part_echo + imag_part_rfi
        del real_part_rfi, imag_part_rfi
        # Recombine into a complex tensor
        tensor_echo = torch.stack([real_part_echo, imag_part_echo], dim=1)
        del real_part_echo, imag_part_echo
        # Recombine into a complex tensor
        tensor_RFI = torch.stack([real_part_label, imag_part_label], dim=1)
        del real_part_label, imag_part_label    
        tensor_echo = tensor_echo.squeeze(0)
        tensor_RFI = tensor_RFI.reshape(tensor_RFI.shape[0]*tensor_RFI.shape[1], self.target_shape[0], self.target_shape[1])
        #print(tensor_echo.shape, tensor_label.shape)

        split_RFI = tensor_RFI.split(2, dim=0)  # List of 10 tensors, each [2, 224, 224]
        # Stack all 10 expanded RFI tensors into a single tensor of shape [10, 3, 224, 224]
        split_RFI = torch.stack(split_RFI, dim=0)  # Shape: [10, 3, 224, 224]
        
        tensor_echo = channelwise_min_max_normalize(tensor=tensor_echo)
        split_RFI = channelwise_min_max_normalize(tensor=split_RFI)
        
        return tensor_echo, split_RFI, label, path if self.paths else torch.tensor(idx)
        
    def close(self):
        """Close all open Zarr files."""
        for z in self.zarr_files:
            z.store.close()

    def set_device(self, device):
        """Set the device for the tensors."""
        self.device = device

    def get_average_original_shape(self):
        total_h, total_w, count = 0, 0, 0
        for file_name in sorted(os.listdir(self.root_dir)):
            file_path = os.path.join(self.root_dir, file_name)
            if file_name.endswith('.zarr'):
                z = zarr.open(file_path, mode='r')
                bursts = [key for key in z.keys() if key.startswith('burst_')]
                for burst in bursts:
                    echo = z[burst]['echo']
                    h, w = echo.shape
                    total_h += h
                    total_w += w
                    count += 1
                z.store.close()
        if count == 0:
            raise RuntimeError("No burst data found in the provided directory.")
        return total_h / count, total_w / count

def channelwise_min_max_normalize(tensor):
    # tensor shape: (batch, channels, height, width)
    min_vals = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero
    return normalized_tensor

    
if __name__ == "__main__":
    dataset = fastSARZarrDataset(root_dir='../../data/v1/2_processed/', paths=True, device='cuda')
    avg_h, avg_w = dataset.get_average_original_shape()
    print(f"Average original height: {avg_h}, Average original width: {avg_w}")