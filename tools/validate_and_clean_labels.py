"""
Label validation and cleanup tool.
Validates all datasets, fixes CelebA bbox from landmarks, and outputs cleaned unified labels.

Usage:
    python tools/validate_and_clean_labels.py

Output:
    D:/cleaned_labels/
        pure.json       - validated PURE samples
        wider.json      - validated WIDER samples
        celeba.json     - CelebA with bbox re-derived from landmarks
        300w_train.json - 70% of 300W for training
        300w_test.json  - 30% of 300W for testing
        stats.json      - validation statistics
"""

import json
import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

LANDMARK_ORDER = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
OUTPUT_DIR = "D:/cleaned_labels"
SEED = 42


def bbox_from_landmarks(landmarks: dict, padding: float = 0.3) -> dict:
    """Derive bounding box from landmarks with padding."""
    xs = [landmarks[n]['x'] for n in LANDMARK_ORDER]
    ys = [landmarks[n]['y'] for n in LANDMARK_ORDER]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min
    size = max(w, h)

    # Add padding proportional to face size
    pad = size * padding
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # Make roughly square bbox
    half = (size / 2) + pad
    x_min = max(0.0, cx - half)
    y_min = max(0.0, cy - half * 1.3)  # extend more upward for forehead
    x_max = min(1.0, cx + half)
    y_max = min(1.0, cy + half * 0.8)  # less extension below mouth

    return {'x_min': float(x_min), 'y_min': float(y_min),
            'x_max': float(x_max), 'y_max': float(y_max)}


def validate_sample(bbox: dict, landmarks: dict, strict: bool = True) -> str:
    """Validate a single sample. Returns '' if valid, else reason string."""
    b = bbox
    x_min, y_min, x_max, y_max = b['x_min'], b['y_min'], b['x_max'], b['y_max']

    # NaN check
    vals = [x_min, y_min, x_max, y_max]
    lm_pts = []
    for n in LANDMARK_ORDER:
        lx, ly = landmarks[n]['x'], landmarks[n]['y']
        vals.extend([lx, ly])
        lm_pts.append([lx, ly])
    lm_pts = np.array(lm_pts)

    if any(np.isnan(v) for v in vals):
        return 'nan'

    # Range check
    if x_min < -0.01 or y_min < -0.01 or x_max > 1.01 or y_max > 1.01:
        return 'bbox_range'
    if lm_pts.min() < -0.01 or lm_pts.max() > 1.01:
        return 'lm_range'

    # Inverted bbox
    if x_min >= x_max or y_min >= y_max:
        return 'bbox_inverted'

    # Bbox too small
    area = (x_max - x_min) * (y_max - y_min)
    if area < 0.001:
        return 'bbox_tiny'

    # Inter-ocular distance
    iod = np.sqrt((lm_pts[0, 0] - lm_pts[1, 0]) ** 2 +
                  (lm_pts[0, 1] - lm_pts[1, 1]) ** 2)
    if iod < 0.008:
        return 'iod_tiny'

    # Landmarks should be inside bbox (with margin)
    if strict:
        margin = 0.15 * max(x_max - x_min, y_max - y_min)
        if (lm_pts[:, 0].min() < x_min - margin or
                lm_pts[:, 0].max() > x_max + margin or
                lm_pts[:, 1].min() < y_min - margin or
                lm_pts[:, 1].max() > y_max + margin):
            return 'lm_outside_bbox'

    return ''


def verify_image_readable(path: str) -> bool:
    """Quick check that image can be opened."""
    if not os.path.exists(path):
        return False
    try:
        img = cv2.imread(path)
        return img is not None and img.shape[0] > 0 and img.shape[1] > 0
    except Exception:
        return False


def process_pure(sample_limit: int = 0) -> list:
    """Process PURE dataset. All samples valid, optionally subsample."""
    print("\n[1/4] Processing PURE...")
    labels_dir = Path("D:/PURE_labels")
    pure_dir = Path("D:/PURE")
    samples = []

    for label_file in sorted(labels_dir.glob("*.json")):
        if label_file.name == "stats.json":
            continue
        session_id = label_file.stem

        with open(label_file, 'r', encoding='utf-8') as f:
            session_labels = json.load(f)

        for image_name, d in session_labels.items():
            if not isinstance(d, dict) or 'bbox' not in d:
                continue

            # Resolve image path
            img_path = pure_dir / session_id / session_id / image_name
            if not img_path.exists():
                img_path = pure_dir / session_id / image_name
            if not img_path.exists():
                continue

            reason = validate_sample(d['bbox'], d['landmarks'])
            if reason:
                continue

            samples.append({
                'image_path': str(img_path),
                'bbox': d['bbox'],
                'landmarks': d['landmarks'],
                'source': 'pure',
            })

    total = len(samples)

    # Subsample if requested (PURE is very homogeneous)
    if sample_limit > 0 and len(samples) > sample_limit:
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(samples), sample_limit, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    print(f"  PURE: {total} valid -> {len(samples)} kept")
    return samples


def process_wider() -> list:
    """Process WIDER FACE dataset with strict validation."""
    print("\n[2/4] Processing WIDER FACE...")
    labels_dir = Path("D:/WIDER_FACE_labels")
    img_dir = Path("D:/WIDER_FACE/WIDER_train/images")
    samples = []
    skipped = defaultdict(int)
    total = 0

    for label_file in sorted(labels_dir.glob("*.json")):
        if label_file.name == "stats.json":
            continue
        category = label_file.stem

        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for image_name, d in data.items():
            if not isinstance(d, dict) or 'bbox' not in d:
                continue
            total += 1

            img_path = img_dir / category / image_name
            if not img_path.exists():
                skipped['missing_image'] += 1
                continue

            reason = validate_sample(d['bbox'], d['landmarks'], strict=True)
            if reason:
                skipped[reason] += 1
                continue

            samples.append({
                'image_path': str(img_path),
                'bbox': d['bbox'],
                'landmarks': d['landmarks'],
                'source': 'wider',
            })

    print(f"  WIDER: {total} total -> {len(samples)} valid")
    if skipped:
        for k, v in skipped.items():
            print(f"    skipped {k}: {v}")
    return samples


def process_celeba() -> list:
    """Process CelebA: re-derive bbox from landmarks, strict validation."""
    print("\n[3/4] Processing CelebA...")
    labels_dir = Path("D:/CelebA_labels")
    img_dir = Path("D:/CelebA/Img/img_align_celeba/img_align_celeba")

    if not img_dir.exists():
        # Try alternate path
        img_dir = Path("D:/CelebA")
        nested = img_dir / "Img" / "img_align_celeba" / "img_align_celeba"
        if nested.exists():
            img_dir = nested

    samples = []
    skipped = defaultdict(int)
    total = 0
    fixed_bbox = 0

    for label_file in sorted(labels_dir.glob("celeba_*.json")):
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for image_name, d in data.items():
            if not isinstance(d, dict) or 'landmarks' not in d:
                continue
            total += 1

            img_path = img_dir / image_name
            if not img_path.exists():
                skipped['missing_image'] += 1
                continue

            lm = d['landmarks']
            # Check landmarks are valid first
            lm_pts = []
            valid_lm = True
            for n in LANDMARK_ORDER:
                if n not in lm:
                    valid_lm = False
                    break
                x, y = lm[n]['x'], lm[n]['y']
                if np.isnan(x) or np.isnan(y) or x < 0 or x > 1 or y < 0 or y > 1:
                    valid_lm = False
                    break
                lm_pts.append([x, y])

            if not valid_lm:
                skipped['bad_landmarks'] += 1
                continue

            lm_pts = np.array(lm_pts)

            # Check inter-ocular distance
            iod = np.sqrt((lm_pts[0, 0] - lm_pts[1, 0]) ** 2 +
                          (lm_pts[0, 1] - lm_pts[1, 1]) ** 2)
            if iod < 0.008:
                skipped['iod_tiny'] += 1
                continue

            # Always re-derive bbox from landmarks for CelebA
            new_bbox = bbox_from_landmarks(lm)
            fixed_bbox += 1

            # Final validation with new bbox
            reason = validate_sample(new_bbox, lm, strict=True)
            if reason:
                skipped[f'post_fix_{reason}'] += 1
                continue

            samples.append({
                'image_path': str(img_path),
                'bbox': new_bbox,
                'landmarks': lm,
                'source': 'celeba',
            })

    print(f"  CelebA: {total} total -> {len(samples)} valid (re-derived {fixed_bbox} bboxes)")
    if skipped:
        for k, v in skipped.items():
            print(f"    skipped {k}: {v}")
    return samples


def process_300w() -> tuple:
    """Process 300W: split into train (70%) and test (30%)."""
    print("\n[4/4] Processing 300W...")
    labels_dir = Path("D:/300W_labels")
    img_dir = Path("D:/300W")
    samples = []
    skipped = defaultdict(int)
    total = 0

    for label_file in sorted(labels_dir.glob("300w_*.json")):
        subset = label_file.stem  # e.g. "300w_afw"

        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for image_key, d in data.items():
            if not isinstance(d, dict) or 'bbox' not in d:
                continue
            total += 1

            img_path = img_dir / image_key
            if not img_path.exists():
                skipped['missing_image'] += 1
                continue

            reason = validate_sample(d['bbox'], d['landmarks'], strict=True)
            if reason:
                skipped[reason] += 1
                continue

            samples.append({
                'image_path': str(img_path),
                'bbox': d['bbox'],
                'landmarks': d['landmarks'],
                'source': f'300w_{subset.split("_")[-1]}',
            })

    print(f"  300W: {total} total -> {len(samples)} valid")
    if skipped:
        for k, v in skipped.items():
            print(f"    skipped {k}: {v}")

    # Split 70% train / 30% test
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(samples))
    split_idx = int(len(samples) * 0.7)
    train_samples = [samples[i] for i in indices[:split_idx]]
    test_samples = [samples[i] for i in indices[split_idx:]]

    print(f"  Split: {len(train_samples)} train, {len(test_samples)} test")
    return train_samples, test_samples


def spot_check_samples(samples: list, name: str, n: int = 5):
    """Randomly verify a few samples by loading images."""
    rng = np.random.RandomState(SEED + 1)
    check_indices = rng.choice(len(samples), min(n, len(samples)), replace=False)
    ok = 0
    for i in check_indices:
        s = samples[i]
        if verify_image_readable(s['image_path']):
            ok += 1
        else:
            print(f"  [WARN] {name}: cannot read {s['image_path']}")
    print(f"  Spot check {name}: {ok}/{len(check_indices)} images readable")


def compute_stats(samples: list) -> dict:
    """Compute distribution statistics for a sample list."""
    if not samples:
        return {}

    bbox_sizes = []
    iods = []
    for s in samples:
        b = s['bbox']
        area = (b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min'])
        bbox_sizes.append(area)
        lm = s['landmarks']
        le, re = lm['left_eye'], lm['right_eye']
        iod = np.sqrt((le['x'] - re['x']) ** 2 + (le['y'] - re['y']) ** 2)
        iods.append(iod)

    bbox_sizes = np.array(bbox_sizes)
    iods = np.array(iods)
    return {
        'count': len(samples),
        'bbox_size_mean': float(bbox_sizes.mean()),
        'bbox_size_std': float(bbox_sizes.std()),
        'iod_mean': float(iods.mean()),
        'iod_std': float(iods.std()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all datasets
    # PURE: subsample to 25K (from 123K) to reduce homogeneity dominance
    pure_samples = process_pure(sample_limit=25000)
    wider_samples = process_wider()
    celeba_samples = process_celeba()
    w300_train, w300_test = process_300w()

    # Spot check
    print("\n" + "=" * 60)
    print("Spot checking image accessibility...")
    for name, samples in [('PURE', pure_samples), ('WIDER', wider_samples),
                           ('CelebA', celeba_samples), ('300W_train', w300_train),
                           ('300W_test', w300_test)]:
        if samples:
            spot_check_samples(samples, name)

    # Save cleaned labels
    print("\n" + "=" * 60)
    print("Saving cleaned labels...")

    def save_json(data, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  Saved {path} ({len(data)} samples)")

    save_json(pure_samples, "pure.json")
    save_json(wider_samples, "wider.json")
    save_json(celeba_samples, "celeba.json")
    save_json(w300_train, "300w_train.json")
    save_json(w300_test, "300w_test.json")

    # Statistics
    all_train = pure_samples + wider_samples + celeba_samples + w300_train
    stats = {
        'pure': compute_stats(pure_samples),
        'wider': compute_stats(wider_samples),
        'celeba': compute_stats(celeba_samples),
        '300w_train': compute_stats(w300_train),
        '300w_test': compute_stats(w300_test),
        'total_train_pool': compute_stats(all_train),
    }
    save_json(stats, "stats.json")

    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print(f"  PURE:       {stats['pure']['count']:>6} samples  (bbox_size={stats['pure']['bbox_size_mean']:.3f}±{stats['pure']['bbox_size_std']:.3f}, IOD={stats['pure']['iod_mean']:.3f}±{stats['pure']['iod_std']:.3f})")
    print(f"  WIDER:      {stats['wider']['count']:>6} samples  (bbox_size={stats['wider']['bbox_size_mean']:.3f}±{stats['wider']['bbox_size_std']:.3f}, IOD={stats['wider']['iod_mean']:.3f}±{stats['wider']['iod_std']:.3f})")
    print(f"  CelebA:     {stats['celeba']['count']:>6} samples  (bbox_size={stats['celeba']['bbox_size_mean']:.3f}±{stats['celeba']['bbox_size_std']:.3f}, IOD={stats['celeba']['iod_mean']:.3f}±{stats['celeba']['iod_std']:.3f})")
    print(f"  300W train: {stats['300w_train']['count']:>6} samples  (bbox_size={stats['300w_train']['bbox_size_mean']:.3f}±{stats['300w_train']['bbox_size_std']:.3f}, IOD={stats['300w_train']['iod_mean']:.3f}±{stats['300w_train']['iod_std']:.3f})")
    print(f"  300W test:  {stats['300w_test']['count']:>6} samples  (bbox_size={stats['300w_test']['bbox_size_mean']:.3f}±{stats['300w_test']['bbox_size_std']:.3f}, IOD={stats['300w_test']['iod_mean']:.3f}±{stats['300w_test']['iod_std']:.3f})")
    total = sum(stats[k]['count'] for k in ['pure', 'wider', 'celeba', '300w_train'])
    print(f"  {'─' * 50}")
    print(f"  Train pool: {total:>6} samples")
    print(f"  Test:       {stats['300w_test']['count']:>6} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
