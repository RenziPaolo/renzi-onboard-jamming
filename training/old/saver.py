import os
import torch
import logging
from pathlib import Path

class SimpleCheckpointSaver:
    def __init__(
        self,
        model,
        optimizer,
        args,
        model_ema=None,
        amp_scaler=None,
        checkpoint_dir='./checkpoints',
        recovery_dir=None,
        decreasing=True,
        max_history=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir or checkpoint_dir
        self.decreasing = decreasing  # If True, lower metric is better
        self.max_history = max_history
        self.best_metric = None
        self.best_epoch = None
        self.history = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.recovery_dir, exist_ok=True)
        self.logger = logging.getLogger('train')

    def save_checkpoint(self, epoch, metric):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args,
        }
        if self.model_ema is not None:
            state['model_ema'] = self.model_ema.state_dict()
        if self.amp_scaler is not None:
            state['amp_scaler'] = self.amp_scaler.state_dict()

        # Save as last.pth.tar
        last_path = os.path.join(self.checkpoint_dir, 'last.pth.tar')
        torch.save(state, last_path)
        self.logger.info(f'Saved checkpoint to {last_path}')

        # Update best metric and save best checkpoint
        is_best = self._is_best(metric)
        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
            best_path = os.path.join(self.checkpoint_dir, 'best.pth.tar')
            torch.save(state, best_path)
            self.logger.info(f'New best metric: {metric:.4f} (epoch {epoch})')
            self.logger.info(f'Saved best checkpoint to {best_path}')

        # Manage history
        self._update_history(epoch, metric, is_best)
        self._cleanup_old_checkpoints()

        return self.best_metric, self.best_epoch

    def _is_best(self, metric):
        if self.best_metric is None:
            return True
        return (metric < self.best_metric) if self.decreasing else (metric > self.best_metric)

    def _update_history(self, epoch, metric, is_best):
        self.history.append((epoch, metric, is_best))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _cleanup_old_checkpoints(self):
        # Keep only the last `max_history` checkpoints
        files = os.listdir(self.checkpoint_dir)
        checkpoints = [f for f in files if f.startswith('checkpoint_') and f.endswith('.pth.tar')]
        checkpoints.sort()
        if len(checkpoints) > self.max_history:
            for f in checkpoints[:-self.max_history]:
                os.remove(os.path.join(self.checkpoint_dir, f))
                self.logger.info(f'Removed old checkpoint: {f}')

    def save_recovery(self, epoch, batch_idx=0):
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args,
        }
        if self.model_ema is not None:
            state['model_ema'] = self.model_ema.state_dict()
        if self.amp_scaler is not None:
            state['amp_scaler'] = self.amp_scaler.state_dict()
        recovery_path = os.path.join(self.recovery_dir, f'recovery_{epoch}_{batch_idx}.pth.tar')
        torch.save(state, recovery_path)
        self.logger.info(f'Saved recovery checkpoint to {recovery_path}')
        
def simple_get_outdir(path: str = None, *path_components: str) -> Path:
    """
    Return the absolute path to an output directory.

    Args:
        path: Base output directory path (default: './output').
        *path_components: Additional subdirectory components to append.

    Returns:
        Path: Absolute path to the output directory.
    """
    if path is None:
        path = os.getenv('OUT_DIR', './output')

    path = Path(path).expanduser().absolute()
    for component in path_components:
        path = path / component

    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    return path