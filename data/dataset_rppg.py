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

        # Load video chunk: [T, H, W, C] uint8 -> [C, T, H, W] float32
        video = np.load(base_path + "_input.npy")
        if video.ndim == 4:
            if video.shape[-1] in (1, 3):
                video = np.transpose(video, (3, 0, 1, 2))
        video_tensor = torch.from_numpy(video.copy()).float()

        # Load BVP label: [T]
        bvp = np.load(base_path + "_bvp.npy")

        # Apply DiffNorm to BVP label when LABEL_TYPE == "DiffNormalized".
        # The model's first layer (DiffNormalizeLayer) produces a difference-style
        # signal from the input frames. Applying the same transform to the label
        # aligns the training target with the model's output representation and
        # improves Pearson correlation during training.
        label_type = "DiffNormalized"
        if self.cfg is not None:
            label_type = getattr(self.cfg.RPPG_DATA, 'LABEL_TYPE', 'DiffNormalized')
        if label_type == "DiffNormalized":
            bvp = _diff_normalize_bvp(bvp)

        bvp_tensor = torch.from_numpy(bvp.copy()).float()

        # Load SpO2 label: scalar or [T]
        spo2_path = base_path + "_spo2.npy"
        if os.path.exists(spo2_path):
            spo2 = np.load(spo2_path)
            spo2 = float(np.mean(spo2))
        else:
            spo2 = 0.0
        spo2_tensor = torch.tensor([spo2], dtype=torch.float32)

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
        video = np.load(base_path + "_input.npy")
        if video.ndim == 4 and video.shape[-1] in (1, 3):
            video = np.transpose(video, (3, 0, 1, 2))
        return torch.from_numpy(video.copy()).float()


def _diff_normalize_bvp(bvp: np.ndarray) -> np.ndarray:
    """
    Apply DiffNormalize to a 1D BVP signal (mirrors DiffNormalizeLayer for video).

    Formula:
        diff[t] = bvp[t] - bvp[t-1]
        norm[t] = diff[t] / (|bvp[t]| + |bvp[t-1]| + ε)
        norm    = norm / (std(norm) + ε)
        norm    = clamp(norm, -5, 5)
        result  = [0, norm[0], norm[1], ..., norm[T-2]]  (pad first sample with 0)
    """
    bvp = bvp.astype(np.float32)
    diff = bvp[1:] - bvp[:-1]
    denom = np.abs(bvp[1:]) + np.abs(bvp[:-1]) + 1e-7
    norm = diff / denom
    norm = norm / (norm.std() + 1e-7)
    norm = np.clip(norm, -5.0, 5.0)
    return np.concatenate([[0.0], norm]).astype(np.float32)


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


def _split_by_session(file_list: List[str], begin: float, end: float) -> List[str]:
    """
    Session-level split: groups chunks by their session prefix (everything before _chunkNNN),
    then splits sessions by ratio.  Prevents the same subject/session from appearing in
    both train and validation sets.

    File naming convention expected:  <session>_chunk<NNN>
    e.g.  01-01_chunk000  (PURE),  subject01_chunk000  (UBFC),
          P001_F01_REST_chunk000  (MCD-rPPG)
    """
    import re
    from collections import defaultdict

    session_groups: dict = defaultdict(list)
    for f in file_list:
        basename = os.path.basename(f)
        m = re.match(r'^(.+)_chunk\d+$', basename)
        session_key = m.group(1) if m else basename
        # Include parent directory to keep datasets separated when sorting
        group_key = os.path.dirname(f) + '/' + session_key
        session_groups[group_key].append(f)

    sessions = sorted(session_groups.keys())
    n = len(sessions)
    start_idx = int(n * begin)
    end_idx   = int(n * end)

    result = []
    for s in sessions[start_idx:end_idx]:
        result.extend(sorted(session_groups[s]))
    return result


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

    # Split by session (not individual chunks) to prevent subject leakage
    train_files = _split_by_session(all_files, cfg.RPPG_DATA.TRAIN_BEGIN, cfg.RPPG_DATA.TRAIN_END)
    valid_files = _split_by_session(all_files, cfg.RPPG_DATA.VALID_BEGIN, cfg.RPPG_DATA.VALID_END)

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

    # Independent test set(s): single TEST_CACHED_PATH + multi TEST_CACHED_PATHS
    # Collect (name, path) pairs from both config keys.
    test_entries: List[Tuple[str, str]] = []

    single_path = cfg.RPPG_DATA.TEST_CACHED_PATH
    if single_path and os.path.isdir(single_path):
        name = cfg.RPPG_DATA.TEST_DATASET or os.path.basename(single_path.rstrip('/\\'))
        test_entries.append((name, single_path))

    for p in getattr(cfg.RPPG_DATA, 'TEST_CACHED_PATHS', []):
        if p and os.path.isdir(p):
            # Derive a short name from the directory, e.g.
            # "D:/PreprocessedData_UBFC" → "UBFC"
            basename = os.path.basename(p.rstrip('/\\'))
            name = basename.replace('PreprocessedData_', '').replace('PreprocessedData', 'test')
            test_entries.append((name, p))

    for name, path in test_entries:
        files = _discover_npy_files(path)
        if files:
            print(f"[rPPG Data] Test ({name}): {len(files)} chunks")
            ds = rPPGVideoDataset(files, cfg)
            loader = DataLoader(
                ds, batch_size=cfg.RPPG_TRAIN.BATCH_SIZE,
                shuffle=False, num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
            )
            # Use "test_{name}" key; first entry also occupies "test" for
            # backward compatibility with code that checks `if 'test' in loaders`.
            key = f'test_{name}'
            loaders[key] = loader
            if 'test' not in loaders:
                loaders['test'] = loader

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
                pin_memory=False,
                drop_last=True,
            )
    else:
        print("[rPPG Semi] WARNING: No unlabeled data paths configured")

    return loaders
