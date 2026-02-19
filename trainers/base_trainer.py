"""Abstract base trainer for face detection and rPPG."""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict

import torch
from utils.gpu_utils import setup_gpu


def make_face_ckpt_name(cfg, tag: str = "best") -> str:
    """Generate descriptive face model checkpoint filename.

    Format: face_{backbone}_a{alpha}_{size}px_{tag}.pth
    Example: face_mobilenetv2_a050_256px_best.pth
    """
    backbone = cfg.FACE_MODEL.BACKBONE.replace("-", "").replace("_", "")
    alpha = f"{cfg.FACE_MODEL.BACKBONE_ALPHA:.2f}".replace(".", "")
    size = cfg.FACE_MODEL.INPUT_SIZE
    return f"face_{backbone}_a{alpha}_{size}px_{tag}.pth"


def make_rppg_ckpt_name(cfg, tag: str = "best") -> str:
    """Generate descriptive rPPG model checkpoint filename.

    Format: rppg_{name}_{frames}f_{size}px_{tag}.pth
    Example: rppg_fcatt_128f_72px_best.pth
    """
    name = cfg.RPPG_MODEL.NAME.lower()
    frames = cfg.RPPG_MODEL.FRAMES
    size = cfg.RPPG_MODEL.INPUT_SIZE
    return f"rppg_{name}_{frames}f_{size}px_{tag}.pth"


class BaseTrainer(ABC):
    """Abstract base trainer with shared logic."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = setup_gpu(cfg.DEVICE, cfg.GPU_MEMORY_LIMIT_MB)

        # Ensure output directories exist
        os.makedirs(cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT.LOG_DIR, exist_ok=True)

        # Reproducibility
        torch.manual_seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.SEED)

    @abstractmethod
    def train(self, data_loaders: dict) -> dict:
        """Run full training pipeline. Returns training history."""
        ...

    @abstractmethod
    def validate(self, data_loader) -> float:
        """Run validation. Returns validation loss."""
        ...

    @abstractmethod
    def test(self, data_loader) -> dict:
        """Run testing. Returns metrics dict."""
        ...

    def save_checkpoint(self, model, optimizer, epoch: int, path: str, **extra):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, model, optimizer, path: str) -> int:
        """Load model checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)

    def save_history(self, history: dict, path: str):
        """Save training history as JSON."""
        # Convert numpy/tensor values to Python types
        serializable = {}
        for key, values in history.items():
            serializable[key] = [float(v) if hasattr(v, 'item') else v for v in values]
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
