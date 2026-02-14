"""
Download and convert WIDER FACE dataset to the same label format as PURE.

WIDER FACE provides bbox-only annotations. This script:
1. Downloads WIDER FACE training images + annotations
2. Parses the official annotation format
3. Selects the largest face per image (for single-face detection model)
4. Generates 5-point landmarks using MediaPipe
5. Outputs JSON labels identical to PURE format

Usage:
    python data/convert_wider_face.py
    python data/convert_wider_face.py --wider_dir D:/WIDER_FACE --output_dir D:/WIDER_FACE_labels
    python data/convert_wider_face.py --skip-download  # if already downloaded
"""

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from generate_labels import FaceLabelGenerator


# WIDER FACE download URLs
WIDER_FACE_URLS = {
    'train_images': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip',
    'annotations': 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip',
}

# Backup URLs (Google Drive via gdown)
WIDER_FACE_URLS_BACKUP = {
    'train_images': 'https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M',
    'annotations': 'https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
}


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """Download a file with progress bar."""
    output = Path(output_path)
    if output.exists():
        print(f"  Already exists: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {desc}: {url}")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get('Content-Length', 0))
            with open(output_path, 'wb') as f:
                downloaded = 0
                block_size = 8192 * 16
                with tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        if output.exists():
            output.unlink()
        raise RuntimeError(f"Download failed: {e}")


def extract_zip(zip_path: str, extract_to: str, desc: str = "Extracting"):
    """Extract a zip file."""
    print(f"  {desc}: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def download_wider_face(wider_dir: str):
    """Download and extract WIDER FACE dataset."""
    wider_path = Path(wider_dir)
    wider_path.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    images_dir = wider_path / "WIDER_train" / "images"
    anno_file = wider_path / "wider_face_split" / "wider_face_train_bbx_gt.txt"

    if images_dir.exists() and anno_file.exists():
        print("WIDER FACE already downloaded and extracted.")
        return

    print("=== Downloading WIDER FACE Dataset ===\n")

    # Download training images
    train_zip = wider_path / "WIDER_train.zip"
    if not images_dir.exists():
        download_file(WIDER_FACE_URLS['train_images'], str(train_zip), "Training images (~1.6GB)")
        extract_zip(str(train_zip), str(wider_path))

    # Download annotations
    anno_zip = wider_path / "wider_face_split.zip"
    if not anno_file.exists():
        download_file(WIDER_FACE_URLS['annotations'], str(anno_zip), "Annotations")
        extract_zip(str(anno_zip), str(wider_path))

    print("\nDownload complete!")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {anno_file}")


def parse_wider_annotations(anno_path: str) -> Dict[str, List[Dict]]:
    """
    Parse WIDER FACE annotation file.

    Format:
        image_path
        number_of_faces
        x1 y1 w h blur expression illumination invalid occlusion pose
        ...

    Returns:
        Dict mapping image_path -> list of face dicts with bbox info
    """
    annotations = {}

    with open(anno_path, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break

            image_path = line
            num_faces = int(f.readline().strip())

            faces = []
            for _ in range(max(num_faces, 1)):
                parts = f.readline().strip().split()
                if len(parts) >= 4:
                    x1, y1, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    invalid = int(parts[7]) if len(parts) > 7 else 0
                    blur = int(parts[4]) if len(parts) > 4 else 0

                    # Skip invalid or zero-area faces
                    if invalid == 1 or w <= 0 or h <= 0:
                        continue

                    faces.append({
                        'x1': x1, 'y1': y1, 'w': w, 'h': h,
                        'blur': blur,
                    })

            if faces:
                annotations[image_path] = faces

    return annotations


def select_largest_face(faces: List[Dict]) -> Dict:
    """Select the largest face (by area) from a list of faces."""
    return max(faces, key=lambda f: f['w'] * f['h'])


def convert_wider_face(
    wider_dir: str,
    output_dir: str,
    model_path: str = "models/face_landmarker.task",
    min_face_size: int = 20,
    max_images: int = 0,
):
    """
    Convert WIDER FACE annotations to PURE-compatible JSON format.

    Args:
        wider_dir: Path to WIDER FACE root directory
        output_dir: Path to save converted labels
        model_path: Path to MediaPipe model
        min_face_size: Minimum face size in pixels (skip tiny faces)
        max_images: Maximum images to process (0 = all)
    """
    wider_path = Path(wider_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Paths
    images_dir = wider_path / "WIDER_train" / "images"
    anno_file = wider_path / "wider_face_split" / "wider_face_train_bbx_gt.txt"

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Parse annotations
    print("Parsing WIDER FACE annotations...")
    annotations = parse_wider_annotations(str(anno_file))
    print(f"  Found {len(annotations)} images with valid faces")

    # Initialize MediaPipe landmark generator
    print("Initializing MediaPipe Face Landmarker...")
    generator = FaceLabelGenerator(model_path)

    # Process images grouped by event category
    stats = {'total': 0, 'processed': 0, 'landmark_ok': 0, 'skipped_small': 0, 'landmark_failed': 0}
    category_labels = {}
    processed_count = 0

    items = list(annotations.items())
    if max_images > 0:
        items = items[:max_images]

    try:
        for image_rel_path, faces in tqdm(items, desc="Converting"):
            stats['total'] += 1

            # Select largest face
            face = select_largest_face(faces)

            # Skip tiny faces
            if face['w'] < min_face_size or face['h'] < min_face_size:
                stats['skipped_small'] += 1
                continue

            # Full image path
            image_path = images_dir / image_rel_path
            if not image_path.exists():
                continue

            # Read image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            h, w = image.shape[:2]

            # Convert bbox to normalized coordinates with 10% padding
            x_min = face['x1'] / w
            y_min = face['y1'] / h
            x_max = (face['x1'] + face['w']) / w
            y_max = (face['y1'] + face['h']) / h

            # Add 10% padding (same as PURE)
            bw = x_max - x_min
            bh = y_max - y_min
            x_min = max(0.0, x_min - bw * 0.1)
            y_min = max(0.0, y_min - bh * 0.1)
            x_max = min(1.0, x_max + bw * 0.1)
            y_max = min(1.0, y_max + bh * 0.1)

            # Try to get landmarks from MediaPipe
            result = generator.process_image(str(image_path))

            if result and 'landmarks' in result:
                landmarks = result['landmarks']
                stats['landmark_ok'] += 1
            else:
                # Estimate landmarks from bbox (fallback)
                landmarks = _estimate_landmarks_from_bbox(x_min, y_min, x_max, y_max)
                stats['landmark_failed'] += 1

            # Group by event category
            category = image_rel_path.split('/')[0]  # e.g., "0--Parade"
            image_name = Path(image_rel_path).name

            if category not in category_labels:
                category_labels[category] = {}

            category_labels[category][image_name] = {
                'bbox': {
                    'x_min': float(x_min),
                    'y_min': float(y_min),
                    'x_max': float(x_max),
                    'y_max': float(y_max),
                },
                'landmarks': landmarks,
                'image_width': w,
                'image_height': h,
            }

            stats['processed'] += 1

    finally:
        generator.close()

    # Save per-category JSON files
    print("\nSaving labels...")
    for category, labels in category_labels.items():
        cat_path = output_path / f"{category}.json"
        with open(cat_path, 'w') as f:
            json.dump(labels, f, indent=2)

    # Save statistics
    stats_path = output_path / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Conversion Complete ===")
    print(f"  Total images: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Landmarks OK: {stats['landmark_ok']}")
    print(f"  Landmarks estimated: {stats['landmark_failed']}")
    print(f"  Skipped (tiny): {stats['skipped_small']}")
    print(f"  Categories: {len(category_labels)}")
    print(f"  Output: {output_path}")


def _estimate_landmarks_from_bbox(
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> Dict:
    """
    Estimate 5-point landmarks from bbox when MediaPipe fails.
    Uses typical face proportions relative to the bounding box.
    """
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = x_max - x_min
    bh = y_max - y_min

    return {
        'left_eye': {'x': cx - bw * 0.18, 'y': cy - bh * 0.12},
        'right_eye': {'x': cx + bw * 0.18, 'y': cy - bh * 0.12},
        'nose': {'x': cx, 'y': cy + bh * 0.02},
        'left_mouth': {'x': cx - bw * 0.14, 'y': cy + bh * 0.18},
        'right_mouth': {'x': cx + bw * 0.14, 'y': cy + bh * 0.18},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert WIDER FACE dataset"
    )
    parser.add_argument(
        "--wider_dir", type=str, default="D:/WIDER_FACE",
        help="Path to download/store WIDER FACE dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="D:/WIDER_FACE_labels",
        help="Path to save converted labels",
    )
    parser.add_argument(
        "--model_path", type=str, default="models/face_landmarker.task",
        help="Path to MediaPipe model file",
    )
    parser.add_argument(
        "--min_face_size", type=int, default=20,
        help="Minimum face size in pixels (default: 20)",
    )
    parser.add_argument(
        "--max_images", type=int, default=0,
        help="Max images to process (0 = all)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (use existing files)",
    )

    args = parser.parse_args()

    if not args.skip_download:
        download_wider_face(args.wider_dir)

    convert_wider_face(
        wider_dir=args.wider_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        min_face_size=args.min_face_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
