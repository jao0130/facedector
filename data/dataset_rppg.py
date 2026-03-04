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
        spo2_tensor = torch.tensor([spo2 - 80.0], dtype=torch.float32)

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


def _filter_by_subject(file_list: List[str], subject_ids: List[str]) -> List[str]:
    """
    Keep only files whose session name starts with one of the given subject IDs.

    PURE naming: '01-01_chunk000' → subject '01' (digits before first '-')
    Files not matching the expected pattern are silently skipped.
    """
    import re
    sid_set = set(subject_ids)
    result  = []
    for f in file_list:
        m = re.match(r'^(\d+)-', os.path.basename(f))
        if not m:
            continue   # 非 PURE 格式（如 MCD-rPPG），跳過
        if m.group(1) in sid_set:
            result.append(f)
    return result


def create_rppg_spo2_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Factory for SpO2 fine-tune (Stage-3).

    Train : MCD-rPPG sessions 0.0–0.8  +  PURE subjects in PURE_TRAIN_SUBJECTS
    Val   : MCD-rPPG sessions 0.8–1.0  (SpO2 MAE 監控)
    Test  : PURE subjects in PURE_TEST_SUBJECTS
    """
    ft = cfg.RPPG_SPO2_FINETUNE
    bs = cfg.RPPG_TRAIN.BATCH_SIZE

    # ── MCD-rPPG: session-based train/val ─────────────────────────────────────
    cached_paths = []
    if cfg.RPPG_DATA.CACHED_PATH:
        cached_paths.append(cfg.RPPG_DATA.CACHED_PATH)
    cached_paths.extend(cfg.RPPG_DATA.CACHED_PATHS)

    mcd_files   = _discover_npy_files(cached_paths)
    train_files = _split_by_session(mcd_files, cfg.RPPG_DATA.TRAIN_BEGIN, cfg.RPPG_DATA.TRAIN_END)
    valid_files = _split_by_session(mcd_files, cfg.RPPG_DATA.VALID_BEGIN, cfg.RPPG_DATA.VALID_END)
    print(f"[SpO2 Data] MCD-rPPG — train: {len(train_files)}, val: {len(valid_files)} chunks")

    # ── PURE: subject-based split ─────────────────────────────────────────────
    pure_path          = getattr(ft, 'PURE_PATH', '')
    pure_train_subjects = list(getattr(ft, 'PURE_TRAIN_SUBJECTS', []))
    pure_test_subjects  = list(getattr(ft, 'PURE_TEST_SUBJECTS',  []))
    pure_test_files = []

    if pure_path and os.path.isdir(pure_path):
        pure_all = _discover_npy_files(pure_path)

        if pure_train_subjects:
            pure_train = _filter_by_subject(pure_all, pure_train_subjects)
            train_files = train_files + pure_train
            print(f"[SpO2 Data] PURE train (subj {pure_train_subjects}): {len(pure_train)} chunks")

        if pure_test_subjects:
            pure_test_files = _filter_by_subject(pure_all, pure_test_subjects)
            print(f"[SpO2 Data] PURE test  (subj {pure_test_subjects}): {len(pure_test_files)} chunks")
    else:
        if pure_path:
            print(f"[SpO2 Data] WARNING: PURE_PATH not found: {pure_path}")

    print(f"[SpO2 Data] Total — train: {len(train_files)}, val: {len(valid_files)}")

    loaders: Dict[str, DataLoader] = {}

    if train_files:
        loaders['train'] = DataLoader(
            rPPGVideoDataset(train_files, cfg),
            batch_size=bs, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        )
    if valid_files:
        loaders['valid'] = DataLoader(
            rPPGVideoDataset(valid_files, cfg),
            batch_size=bs, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
        )
    if pure_test_files:
        test_ds = rPPGVideoDataset(pure_test_files, cfg)
        test_loader = DataLoader(
            test_ds, batch_size=bs, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
        )
        loaders['test']      = test_loader
        loaders['test_PURE'] = test_loader

    return loaders


def create_rppg_joint_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Factory for joint multi-task semi-supervised training.

    Labeled  : MCD-rPPG + UBFC  (CACHED_PATHS)  — K-fold train/val split
    Test     : PURE              (TEST_CACHED_PATHS) — all chunks, not in training
    Unlabeled: CelebV-HQ        (RPPG_SEMI.UNLABELED_PATHS)

    K-fold strategy:
        All labeled sessions are sorted, divided into N_FOLDS equal parts.
        Fold FOLD_IDX is held out as validation; the rest form the training set.
        This ensures no session leakage while enabling full K-fold CV.
    """
    jt       = cfg.RPPG_JOINT
    n_folds  = getattr(jt, 'N_FOLDS',   5)
    fold_idx = getattr(jt, 'FOLD_IDX',  0)
    bs       = cfg.RPPG_TRAIN.BATCH_SIZE

    # ── Labeled data ──────────────────────────────────────────────────────────
    labeled_paths = []
    if cfg.RPPG_DATA.CACHED_PATH:
        labeled_paths.append(cfg.RPPG_DATA.CACHED_PATH)
    labeled_paths.extend(cfg.RPPG_DATA.CACHED_PATHS)

    all_labeled = _discover_npy_files(labeled_paths)
    print(f"[Joint Data] Labeled: {len(all_labeled)} chunks "
          f"from {len(labeled_paths)} source(s)")

    # K-fold split — _split_by_session handles [begin, end) by ratio
    fold_w   = 1.0 / n_folds
    v_begin  = fold_idx * fold_w
    v_end    = (fold_idx + 1) * fold_w

    train_files = (
        _split_by_session(all_labeled, 0.0, v_begin) +
        _split_by_session(all_labeled, v_end,  1.0)
    )
    valid_files = _split_by_session(all_labeled, v_begin, v_end)
    print(f"[Joint Data] Fold {fold_idx}/{n_folds} — "
          f"train: {len(train_files)}, val: {len(valid_files)}")

    # ── Test: PURE (independent, never in train/val) ──────────────────────────
    test_paths: List[str] = []
    if cfg.RPPG_DATA.TEST_CACHED_PATH:
        test_paths.append(cfg.RPPG_DATA.TEST_CACHED_PATH)
    test_paths.extend(getattr(cfg.RPPG_DATA, 'TEST_CACHED_PATHS', []))

    # ── Unlabeled: CelebV-HQ ─────────────────────────────────────────────────
    unlabeled_paths: List[str] = []
    if cfg.RPPG_SEMI.UNLABELED_PATH:
        unlabeled_paths.append(cfg.RPPG_SEMI.UNLABELED_PATH)
    unlabeled_paths.extend(cfg.RPPG_SEMI.UNLABELED_PATHS)

    loaders: Dict[str, DataLoader] = {}

    if train_files:
        loaders['train'] = DataLoader(
            rPPGVideoDataset(train_files, cfg),
            batch_size=bs, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        )
    if valid_files:
        loaders['valid'] = DataLoader(
            rPPGVideoDataset(valid_files, cfg),
            batch_size=bs, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
        )

    for path in test_paths:
        if not path or not os.path.isdir(path):
            continue
        files = _discover_npy_files(path)
        if not files:
            continue
        name = (os.path.basename(path.rstrip('/\\'))
                .replace('PreprocessedData_', '')
                .replace('PreprocessedData', 'test'))
        print(f"[Joint Data] Test ({name}): {len(files)} chunks")
        loader = DataLoader(
            rPPGVideoDataset(files, cfg),
            batch_size=bs, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
        )
        loaders[f'test_{name}'] = loader
        if 'test' not in loaders:
            loaders['test'] = loader

    if unlabeled_paths:
        unlab_files = _discover_npy_files(unlabeled_paths)
        print(f"[Joint Data] Unlabeled: {len(unlab_files)} chunks "
              f"from {len(unlabeled_paths)} source(s)")
        if unlab_files:
            loaders['unlabeled'] = DataLoader(
                rPPGUnlabeledDataset(unlab_files, cfg),
                batch_size=cfg.RPPG_SEMI.UNLABELED_BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.NUM_WORKERS, pin_memory=False, drop_last=True,
            )
    else:
        print("[Joint Data] WARNING: No unlabeled data paths configured")

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
