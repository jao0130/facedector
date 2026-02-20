"""
PyTorch dataset for rPPG training.
Loads preprocessed NPY cached data (face-cropped, resized, chunked).
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class rPPGVideoDataset(Dataset):
    """
    Dataset for rPPG training from NPY cache.

    Each sample is a pre-processed video chunk:
        - video: [T, H, W, C] stored as NPY
        - bvp: [T] stored as NPY
        - spo2: [T] or scalar stored as NPY

    __getitem__ returns:
        video_chunk: Tensor [3, T, H, W] float32 (NCTHW format for 3D CNN)
        bvp_label: Tensor [T] float32
        spo2_label: Tensor [1] float32
    """

    def __init__(self, file_list: List[str], cfg=None):
        """
        Args:
            file_list: list of base paths (without suffix). Each base has:
                {base}_input.npy, {base}_bvp.npy, {base}_spo2.npy
            cfg: configuration node
        """
        self.file_list = file_list
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_path = self.file_list[idx]

        # Load video chunk: [T, H, W, C] or [C, T, H, W]
        video = np.load(base_path + "_input.npy").astype(np.float32)
        if video.ndim == 4:
            if video.shape[-1] in (1, 3):
                # [T, H, W, C] -> [C, T, H, W]
                video = np.transpose(video, (3, 0, 1, 2))
            # else assume already [C, T, H, W]

        # Load BVP label: [T]
        bvp = np.load(base_path + "_bvp.npy").astype(np.float32)

        # Load SpO2 label: scalar or [T]
        spo2_path = base_path + "_spo2.npy"
        if os.path.exists(spo2_path):
            spo2 = np.load(spo2_path).astype(np.float32)
            if spo2.ndim > 0:
                spo2 = np.mean(spo2)  # Average to scalar
        else:
            spo2 = np.float32(0.0)

        video_tensor = torch.from_numpy(video)
        bvp_tensor = torch.from_numpy(bvp)
        spo2_tensor = torch.tensor([float(spo2)], dtype=torch.float32)

        return video_tensor, bvp_tensor, spo2_tensor


class rPPGUnlabeledDataset(Dataset):
    """
    Dataset for unlabeled face video chunks (e.g. VoxCeleb2).
    Only loads video data — no BVP/SpO2 labels.

    __getitem__ returns:
        video_chunk: Tensor [3, T, H, W] float32
    """

    def __init__(self, file_list: List[str], cfg=None):
        self.file_list = file_list
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        base_path = self.file_list[idx]
        video = np.load(base_path + "_input.npy").astype(np.float32)
        if video.ndim == 4 and video.shape[-1] in (1, 3):
            video = np.transpose(video, (3, 0, 1, 2))
        return torch.from_numpy(video)


def _discover_npy_files(cached_path) -> List[str]:
    """
    Discover preprocessed NPY file base paths from cached directory(ies).
    Looks for *_input.npy files and returns base paths (without _input.npy suffix).

    Args:
        cached_path: str or list of str — one or more directories to scan.
    """
    if isinstance(cached_path, str):
        paths = [cached_path] if cached_path else []
    else:
        paths = [p for p in cached_path if p]

    all_bases = set()
    for p in paths:
        if not os.path.isdir(p):
            print(f"[Dataset] WARNING: Directory not found, skipping: {p}")
            continue
        found = glob.glob(os.path.join(p, "**", "*_input.npy"), recursive=True)
        for f in found:
            all_bases.add(f.replace("_input.npy", ""))
    return sorted(all_bases)


def _split_by_ratio(file_list: List[str], begin: float, end: float) -> List[str]:
    """Split file list by begin/end ratio."""
    n = len(file_list)
    start_idx = int(n * begin)
    end_idx = int(n * end)
    return file_list[start_idx:end_idx]


def create_rppg_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Factory: creates train/valid/test DataLoaders from config.

    Training data: PURE (split by TRAIN/VALID ratios)
    Test data:     UBFC (independent, from TEST_CACHED_PATH)

    Returns:
        dict with 'train', 'valid', and optionally 'test' DataLoaders.
    """
    # Merge single path + multi-path list
    cached_paths = []
    if cfg.RPPG_DATA.CACHED_PATH:
        cached_paths.append(cfg.RPPG_DATA.CACHED_PATH)
    cached_paths.extend(cfg.RPPG_DATA.CACHED_PATHS)

    all_files = _discover_npy_files(cached_paths)
    print(f"[rPPG Data] {cfg.RPPG_DATA.DATASET}: {len(all_files)} chunks from {len(cached_paths)} source(s)")

    # Split training dataset by ratios
    train_files = _split_by_ratio(all_files, cfg.RPPG_DATA.TRAIN_BEGIN, cfg.RPPG_DATA.TRAIN_END)
    valid_files = _split_by_ratio(all_files, cfg.RPPG_DATA.VALID_BEGIN, cfg.RPPG_DATA.VALID_END)

    print(f"[rPPG Data] Train: {len(train_files)}, Valid: {len(valid_files)}")

    loaders = {}

    if train_files:
        train_ds = rPPGVideoDataset(train_files, cfg)
        loaders['train'] = DataLoader(
            train_ds, batch_size=cfg.RPPG_TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True,
        )

    if valid_files:
        valid_ds = rPPGVideoDataset(valid_files, cfg)
        loaders['valid'] = DataLoader(
            valid_ds, batch_size=cfg.RPPG_TRAIN.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

    # Independent test set (e.g. UBFC)
    test_cached = cfg.RPPG_DATA.TEST_CACHED_PATH
    if test_cached and os.path.isdir(test_cached):
        test_files = _discover_npy_files(test_cached)
        if test_files:
            print(f"[rPPG Data] Test ({cfg.RPPG_DATA.TEST_DATASET}): {len(test_files)} chunks in {test_cached}")
            test_ds = rPPGVideoDataset(test_files, cfg)
            loaders['test'] = DataLoader(
                test_ds, batch_size=cfg.RPPG_TRAIN.BATCH_SIZE,
                shuffle=False, num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
            )

    return loaders


def create_rppg_semi_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Factory: creates labeled train/valid + unlabeled DataLoaders for semi-supervised training.

    Returns:
        dict with 'train', 'valid', 'unlabeled', and optionally 'test' DataLoaders.
    """
    # Labeled data (same as supervised)
    loaders = create_rppg_dataloaders(cfg)

    # Unlabeled data — merge single path + multi-path list
    unlabeled_paths = []
    if cfg.RPPG_SEMI.UNLABELED_PATH:
        unlabeled_paths.append(cfg.RPPG_SEMI.UNLABELED_PATH)
    unlabeled_paths.extend(cfg.RPPG_SEMI.UNLABELED_PATHS)

    if unlabeled_paths:
        unlabeled_files = _discover_npy_files(unlabeled_paths)
        print(f"[rPPG Semi] Unlabeled: {len(unlabeled_files)} chunks from {len(unlabeled_paths)} source(s)")

        if unlabeled_files:
            unlabeled_ds = rPPGUnlabeledDataset(unlabeled_files, cfg)
            loaders['unlabeled'] = DataLoader(
                unlabeled_ds,
                batch_size=cfg.RPPG_SEMI.UNLABELED_BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
    else:
        print("[rPPG Semi] WARNING: No unlabeled data paths configured")

    return loaders
