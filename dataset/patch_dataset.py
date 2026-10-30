import torch
import zarr
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
import gc

class SARZarrPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        rfi=range(0, 10),
        target_shape=(512, 512),
        patch_size=(256, 256),
        direct=False,
        paths=False,
        device='cpu',
        log=False,
        verbose=False,
    ):
        self.root_dir = root_dir
        self.rfi = rfi
        self.target_shape = target_shape
        self.patch_size = patch_size  # (height, width)
        self.direct = direct
        self.device = device
        self.verbose = verbose
        self.log = log
        self.zarr_paths = []
        self.bursts_per_file = []
        self.cumulative_bursts = [0]
        self.paths = [] if paths else None
        self.patch_indices = []  # List of (file_idx, burst_idx, patch_row, patch_col)
        self.cumulative_patches = [0]

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

        # Precompute patch indices for all bursts
        self._precompute_patch_indices()

    def _precompute_patch_indices(self):
        """Precompute all patch indices for all bursts."""
        total_patches = 0
        for file_idx, num_bursts in enumerate(self.bursts_per_file):
            for burst_idx in range(num_bursts):
                # Calculate number of patches per burst
                h_patches = self.target_shape[0] // self.patch_size[0]
                w_patches = self.target_shape[1] // self.patch_size[1]
                for h in range(h_patches):
                    for w in range(w_patches):
                        self.patch_indices.append((file_idx, burst_idx, h, w))
                total_patches += h_patches * w_patches
            self.cumulative_patches.append(total_patches)

    def __len__(self):
        return len(self.patch_indices)

    def _resize_and_normalize(self, tensor):
        # (Same as your original implementation)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor)

        if tensor.is_complex():
            real = tensor.real
            imag = tensor.imag
            if self.log:
                log = torch.log(real.abs() + 1e-8)
            else:
                log = None
        else:
            real = tensor
            imag = None
            log = None

        if real.dim() == 2:
            real = real.unsqueeze(0)
        elif real.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {real.shape}")


        if imag is not None:
            if imag.dim() == 2:
                imag = imag.unsqueeze(0)

        if log is not None:
            if log.dim() == 2:
                log = log.unsqueeze(0)


        if tensor.is_complex() and not self.log:
            tensor = torch.stack([real, imag], dim=1)
        elif tensor.is_complex() and self.log:
            tensor = torch.stack([real, log, imag], dim=1)
        else:
            tensor = real

        return channelwise_min_max_normalize(tensor)

    def _get_patch(self, tensor, h_idx, w_idx):
        """Extract a patch from the tensor at (h_idx, w_idx)."""
        h_start = h_idx * self.patch_size[0]
        h_end = h_start + self.patch_size[0]
        w_start = w_idx * self.patch_size[1]
        w_end = w_start + self.patch_size[1]
        return tensor[..., h_start:h_end, w_start:w_end]

    def _compute_overlap_fraction(self, bbox, h_idx, w_idx):
        """Compute the fraction of the patch overlapping with the RFI bbox.
        Args:
            bbox: [x_min, y_min, x_max, y_max] in normalized coordinates (0-1).
            h_idx, w_idx: Patch indices.
        Returns:
            overlap_fraction: Fraction of the patch area overlapping with the bbox.
        """
        # Convert bbox to pixel coordinates (assuming bbox is in [0,1] range)
        H, W = self.target_shape
        x_min, y_min, x_max, y_max = bbox * torch.tensor([W, H, W, H], dtype=torch.float32)

        # Patch boundaries in pixel coordinates
        h_start = h_idx * self.patch_size[0]
        h_end = h_start + self.patch_size[0]
        w_start = w_idx * self.patch_size[1]
        w_end = w_start + self.patch_size[1]

        # Compute intersection
        inter_x_min = max(x_min, w_start)
        inter_y_min = max(y_min, h_start)
        inter_x_max = min(x_max, w_end)
        inter_y_max = min(y_max, h_end)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Patch area
        patch_area = self.patch_size[0] * self.patch_size[1]

        # Overlap fraction
        overlap_fraction = inter_area / patch_area if patch_area > 0 else 0.0
        return overlap_fraction # Return as Python float

    def __getitem__(self, idx):
        if self.verbose:
            print(f"Getting patch index {idx}")

        # Get patch metadata
        file_idx, burst_idx, h_idx, w_idx = self.patch_indices[idx]
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
            # No resizing/normalization
            patch_echo = self._get_patch(tensor_echo, h_idx, w_idx)
            patch_rfi = self._get_patch(tensor_rfi, h_idx, w_idx)
            overlap_fraction = self._compute_overlap_fraction(bbox, h_idx, w_idx)
            return (
                patch_echo,
                patch_echo + patch_rfi,
                overlap_fraction,
                (bbox, category),
                self.paths[file_idx] if self.paths else torch.tensor(idx),
            )

        # Resize and normalize
        tensor_echo = self._resize_and_normalize(tensor_echo)
        tensor_rfi = self._resize_and_normalize(tensor_rfi)

        # Extract patches
        patch_echo = self._get_patch(tensor_echo, h_idx, w_idx)
        patch_rfi = self._get_patch(tensor_rfi, h_idx, w_idx)

        # Stack RFI channels (if needed)
        split_RFI = patch_rfi.split(2, dim=0)
        split_RFI = torch.stack(split_RFI, dim=0)

        # Compute overlap fraction
        overlap_fraction = self._compute_overlap_fraction(bbox, h_idx, w_idx)

        # Cleanup
        del tensor_echo, tensor_rfi, patch_rfi
        gc.collect()

        return (
            patch_echo,
            split_RFI,
            overlap_fraction,
            (bbox, category),
            self.paths[file_idx] if self.paths else torch.tensor(idx),
        )

def channelwise_min_max_normalize(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)

def save_in_batches(dataset, batch_size=100, output_path="/mnt/ssd2/renzi/RFInject/echo_RFI_patches.pt"):
    echo_batch, rfi_batch, overlap_batch = [], [], []
    for i, (echo, rfi, overlap, _, _) in enumerate(tqdm(dataset, desc="Saving in batches")):
        echo_batch.append(echo.unsqueeze(0))
        rfi_batch.append(rfi.unsqueeze(0))
        overlap_batch.append(torch.tensor([overlap]))
        if (i + 1) % batch_size == 0:
            torch.save({
                "echo": torch.cat(echo_batch, dim=0),
                "RFI": torch.cat(rfi_batch, dim=0),
                "overlap_fraction": torch.cat(overlap_batch, dim=0),
            }, f"{output_path}_{i//batch_size}.pt")
            echo_batch, rfi_batch, overlap_batch = [], [], []
    if echo_batch:
        torch.save({
            "echo": torch.cat(echo_batch, dim=0),
            "RFI": torch.cat(rfi_batch, dim=0),
            "overlap_fraction": torch.cat(overlap_batch, dim=0),
        }, f"{output_path}_final.pt")
        
import argparse
import os
import torch
from pathlib import Path

def cache_patches(dataset, cache_dir, force_recompute=False):
    """Cache all patches to disk for faster reuse."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Caching patches"):
        cache_path = cache_dir / f"patch_{idx}.pt"
        if not force_recompute and cache_path.exists():
            continue  # Skip if already cached

        # Get patch data
        patch_echo, patch_rfi, overlap_fraction, (bbox, category), metadata = dataset[idx]

        # Save to disk
        torch.save({
            "patch_echo": patch_echo,
            "patch_rfi": patch_rfi,
            "overlap_fraction": overlap_fraction,
            "bbox": bbox,
            "category": category,
            "metadata": metadata,
        }, cache_path)

    print(f"Cached {len(dataset)} patches to {cache_dir}")

def load_cached_patches(cache_dir, dataset):
    """Load cached patches instead of recomputing."""
    cache_dir = Path(cache_dir)
    cached_patches = []

    for idx in tqdm(range(len(dataset)), desc="Loading cached patches"):
        cache_path = cache_dir / f"patch_{idx}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached patch {idx} not found at {cache_path}")

        cached_data = torch.load(cache_path)
        cached_patches.append((
            cached_data["patch_echo"],
            cached_data["patch_rfi"],
            cached_data["overlap_fraction"],
            (cached_data["bbox"], cached_data["category"]),
            cached_data["metadata"],
        ))

    return cached_patches

def validate_patches(dataset, num_samples=5):
    """Validate a few patches to ensure correctness."""
    print("\nValidating patches...")
    for idx in range(min(num_samples, len(dataset))):
        patch_echo, patch_rfi, overlap_fraction, (bbox, category), metadata = dataset[idx]

        # Check shapes
        assert patch_echo.shape[1:] == dataset.patch_size, f"Patch shape mismatch: {patch_echo.shape}"
        assert 0 <= overlap_fraction <= 1, f"Invalid overlap fraction: {overlap_fraction}"

        print(f"Patch {idx}:")
        print(f"  - Echo shape: {patch_echo.shape}")
        print(f"  - RFI shape: {patch_rfi.shape}")
        print(f"  - Overlap fraction: {overlap_fraction:.4f}")
        print(f"  - Bbox: {bbox}")
        print(f"  - Category: {category}")

def main():
    parser = argparse.ArgumentParser(description="Test and cache SAR patches.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to Zarr files.")
    parser.add_argument("--cache_dir", type=str, default="/mnt/ssd2/renzi/RFInject", help="Directory to cache patches.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="Patch size (height, width).")
    parser.add_argument("--target_shape", type=int, nargs=2, default=[512, 512], help="Target shape for resizing.")
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute and overwrite cache.")
    parser.add_argument("--validate", action="store_true", help="Run validation on patches.")
    parser.add_argument("--save_batches", action="store_true", help="Save patches in batches.")
    args = parser.parse_args()

    # Initialize dataset
    dataset = SARZarrPatchDataset(
        root_dir=args.root_dir,
        rfi=range(0, 10),
        target_shape=tuple(args.patch_size),
        patch_size=tuple(args.patch_size),
        direct=False,
        verbose=False,
    )

    # Cache patches
    cache_patches(dataset, args.cache_dir, force_recompute=args.force_recompute)

    # Load cached patches (optional)
    if args.validate or args.save_batches:
        cached_patches = load_cached_patches(args.cache_dir, dataset)

    # Validate patches
    if args.validate:
        validate_patches(dataset)

    # Save patches in batches (optional)
    if args.save_batches:
        save_in_batches(
            dataset,
            batch_size=args.batch_size,
            output_path=os.path.join(args.cache_dir, "batches", "patches"),
        )

if __name__ == "__main__":
    main()