import torch
import zarr
import os
import bisect
from tqdm import tqdm
import torch.nn.functional as F
import ujson as json
import gc

class SARZarrDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, rfi=range(0, 10), target_shape=(512, 512), direct=False, paths=False, device='cpu', log=False, verbose=False):
        self.root_dir = root_dir
        self.rfi = rfi
        self.target_shape = target_shape
        self.direct = direct
        self.device = device
        self.verbose = verbose
        self.log = log
        self.zarr_paths = []
        self.bursts_per_file = []
        self.cumulative_bursts = [0]
        self.paths = [] if paths else None

        # Index files and bursts (lazy load)
        for file_name in tqdm(os.listdir(root_dir), desc="Indexing datasets"):
            file_path = os.path.join(root_dir, file_name)
            if file_name.endswith('.zarr'):
                z = zarr.open(file_path, mode='r')
                bursts = [key for key in z.keys() if key.startswith('burst_')]
                self.zarr_paths.append(file_path)
                if paths:
                    self.paths.append(file_path)
                self.bursts_per_file.append(len(bursts))
                self.cumulative_bursts.append(self.cumulative_bursts[-1] + len(bursts))
                z.store.close()

    def __len__(self):
        return self.cumulative_bursts[-1]

    def _resize_and_normalize(self, tensor):
        # Ensure tensor is a torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor)

        # Handle complex tensors
        if tensor.is_complex():
            real = tensor.real
            imag = tensor.imag
            if self.log:
                log = torch.log(real.abs() + 1e-8)  # Logarithmic scaling for real part
        else:
            real = tensor
            imag = None  # Not used for non-complex tensors
            log = None

        # Reshape real and imag to (C, H, W) or (1, H, W)
        if real.dim() == 2:  # (H, W)
            real = real.unsqueeze(0)  # (1, H, W)
        elif real.dim() == 3:  # (C, H, W)
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unexpected tensor shape: {real.shape}")

        # Resize real part
        real = F.interpolate(
            real.unsqueeze(0),  # Add batch dim: (1, C, H, W)
            size=self.target_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dim: (C, H, W)

        # Resize imag part if complex
        if imag is not None:
            if imag.dim() == 2:  # (H, W)
                imag = imag.unsqueeze(0)  # (1, H, W)
            imag = F.interpolate(
                imag.unsqueeze(0),  # Add batch dim: (1, C, H, W)
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim: (C, H, W)

        if log is not None:
            if imag.dim() == 2:  # (H, W)
                imag = imag.unsqueeze(0)  # (1, H, W)
            imag = F.interpolate(
                imag.unsqueeze(0),  # Add batch dim: (1, C, H, W)
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim: (C, H, W)
        
        
        # Recombine real and imag if complex
        if tensor.is_complex() and not self.log:
            tensor = torch.stack([real, imag], dim=1)  # (C, 2, H, W)
        elif tensor.is_complex() and self.log:
            tensor = torch.stack([real, log, imag], dim=1)  # (C, 3, H, W)
        else:
            tensor = real  # (C, H, W)

        # Normalize
        return channelwise_min_max_normalize(tensor)

    def __getitem__(self, idx):
        if self.verbose:
            print(f"Getting index {idx}")

        # Find file and burst index
        file_idx = bisect.bisect_right(self.cumulative_bursts, idx) - 1
        burst_idx = idx - self.cumulative_bursts[file_idx]
        burst_key = f'burst_{burst_idx}'

        # Load Zarr data
        z = zarr.open(self.zarr_paths[file_idx], mode='r')
        echo = z[burst_key]['echo'][:]
        rfi = z[burst_key]['rfi'][:][self.rfi]
        z.store.close()

        # Load bbox and category
        json_file = os.path.join(
            self.root_dir,
            os.path.basename(self.zarr_paths[file_idx])[1:-5] + f"RFI_{self.rfi[0]}{burst_key}.json"
        )
        if not os.path.exists(json_file):
            bbox = torch.zeros(4, dtype=torch.float32)
            category = torch.zeros(1, dtype=torch.int64)
        else:
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            bbox = torch.tensor(annotation['bbox'], dtype=torch.float32)
            category = torch.tensor(annotation['category_id'], dtype=torch.int64)

        # Process tensors
        tensor_echo = torch.from_numpy(echo)
        tensor_rfi = torch.from_numpy(rfi)
        del echo, rfi

        if self.direct:
            return tensor_echo, tensor_echo + tensor_rfi, (bbox, category), self.paths[file_idx] if self.paths else torch.tensor(idx)

        tensor_echo = self._resize_and_normalize(tensor_echo)
        tensor_rfi = self._resize_and_normalize(tensor_rfi)

        # Stack RFI channels
        split_RFI = tensor_rfi.split(2, dim=0)
        split_RFI = torch.stack(split_RFI, dim=0)

        # Cleanup
        del tensor_rfi
        gc.collect()

        return tensor_echo, split_RFI, (bbox, category), self.paths[file_idx] if self.paths else torch.tensor(idx)

def channelwise_min_max_normalize(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)

def save_in_batches(dataset, batch_size=100, output_path="/mnt/ssd2/renzi/RFInject/echo_RFI512.pt"):
    echo_batch, rfi_batch = [], []
    for i, (echo, rfi, _, _) in enumerate(tqdm(dataset, desc="Saving in batches")):
        echo_batch.append(echo.unsqueeze(0))
        rfi_batch.append(rfi.unsqueeze(0))
        if (i + 1) % batch_size == 0:
            torch.save({
                "echo": torch.cat(echo_batch, dim=0),
                "RFI": torch.cat(rfi_batch, dim=0)
            }, f"{output_path}_{i//batch_size}.pt")
            echo_batch, rfi_batch = [], []
    if echo_batch:
        torch.save({
            "echo": torch.cat(echo_batch, dim=0),
            "RFI": torch.cat(rfi_batch, dim=0)
        }, f"{output_path}_final.pt")

if __name__ == "__main__":
    dataset = SARZarrDataset(
        root_dir='/mnt/ssd2/renzi/RFInject',
        paths=True,
        device='cuda',
        target_shape=(512, 512)
    )
    save_in_batches(dataset)