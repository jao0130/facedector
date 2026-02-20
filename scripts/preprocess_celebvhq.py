"""
CelebV-HQ preprocessing for semi-supervised rPPG training (unlabeled data).

CelebV-HQ videos are already face-cropped, so no face detection is needed.
Just resize to 72x72, subsample to ~30fps, and chunk into 128-frame segments.

Supports two sources:
  --source hf    : Download from HuggingFace (SwayStar123/CelebV-HQ)
  --source local : Read from a local directory of extracted MP4 files

Usage:
    # From HuggingFace (downloads ~42GB)
    python scripts/preprocess_celebvhq.py \
        --source hf \
        --hf_cache D:/CelebVHQ_HF \
        --output D:/PreprocessedData_CelebVHQ \
        --max_chunks 20000

    # From local extracted MP4s
    python scripts/preprocess_celebvhq.py \
        --source local \
        --input D:/CelebVHQ/videos \
        --output D:/PreprocessedData_CelebVHQ \
        --max_chunks 20000

Output per chunk:
    {video_id}_chunk{N}_input.npy  [128, 72, 72, 3] uint8

No BVP/SpO2 labels (unlabeled data for semi-supervised training).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ── Constants ──
TARGET_FPS = 30
CHUNK_LENGTH = 128
INPUT_SIZE = 72


def download_from_hf(hf_cache: str) -> str:
    """Download CelebV-HQ videos from HuggingFace."""
    from huggingface_hub import snapshot_download

    print("[CelebV-HQ] Downloading from SwayStar123/CelebV-HQ...")
    print("[CelebV-HQ] This may take a while (~42GB)...")

    path = snapshot_download(
        repo_id="SwayStar123/CelebV-HQ",
        repo_type="dataset",
        local_dir=hf_cache,
        allow_patterns=["**/*.mp4"],
    )
    print(f"[CelebV-HQ] Downloaded to: {path}")
    return path


def collect_videos_local(input_dir: str, max_videos: int = 0) -> List[str]:
    """Recursively collect MP4 files from local directory."""
    videos = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith('.mp4'):
                videos.append(os.path.join(root, f))
                if max_videos > 0 and len(videos) >= max_videos:
                    return videos
    return videos


def video_to_id(video_path: str, base_dir: str) -> str:
    """Generate a unique ID from the video path."""
    rel = os.path.relpath(video_path, base_dir)
    video_id = rel.replace(os.sep, '_').replace('/', '_')
    video_id = video_id.replace('.mp4', '').replace('.MP4', '')
    return video_id


def process_video(video_path: str) -> Optional[np.ndarray]:
    """
    Read a face-cropped video, resize frames to INPUT_SIZE, subsample to TARGET_FPS.

    No face detection needed — CelebV-HQ videos are already face-cropped.

    Returns:
        [N, 72, 72, 3] uint8 array, or None if too short.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        return None

    # Subsample to ~TARGET_FPS
    frame_interval = max(1, round(src_fps / TARGET_FPS))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(resized)

    cap.release()

    if len(frames) < CHUNK_LENGTH:
        return None

    return np.array(frames, dtype=np.uint8)


def save_chunks(frames: np.ndarray, output_dir: str, video_id: str) -> int:
    """Split frames into CHUNK_LENGTH chunks and save as NPY (input only)."""
    n_chunks = frames.shape[0] // CHUNK_LENGTH
    saved = 0

    for i in range(n_chunks):
        start = i * CHUNK_LENGTH
        end = start + CHUNK_LENGTH
        chunk = frames[start:end]  # [128, 72, 72, 3]

        base_name = f"{video_id}_chunk{i:03d}"
        np.save(os.path.join(output_dir, f"{base_name}_input.npy"), chunk)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="CelebV-HQ preprocessing for semi-supervised rPPG")
    parser.add_argument("--source", type=str, required=True, choices=["hf", "local"],
                        help="Data source: 'hf' (HuggingFace download) or 'local' (extracted MP4s)")
    parser.add_argument("--hf_cache", type=str, default="",
                        help="HuggingFace download directory (required for --source hf)")
    parser.add_argument("--input", type=str, default="",
                        help="Local MP4 directory (required for --source local)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for NPY chunks")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to process (0=all)")
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="Stop after saving this many chunks (0=unlimited)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have chunks")
    args = parser.parse_args()

    # Determine input directory
    if args.source == "hf":
        if not args.hf_cache:
            print("Error: --hf_cache is required when --source hf")
            sys.exit(1)
        input_dir = download_from_hf(args.hf_cache)
    else:
        input_dir = args.input
        if not input_dir or not os.path.isdir(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Collect videos
    print(f"[CelebV-HQ] Scanning {input_dir} for MP4 files...")
    videos = collect_videos_local(input_dir, args.max_videos)
    print(f"[CelebV-HQ] Found {len(videos)} videos")

    if not videos:
        print("No videos found. Check the input directory.")
        sys.exit(1)

    total_chunks = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, video_path in enumerate(videos):
        video_id = video_to_id(video_path, input_dir)
        prefix = f"[{i+1}/{len(videos)}]"

        # Skip existing
        if args.skip_existing:
            existing = [f for f in os.listdir(args.output) if f.startswith(video_id + "_chunk")]
            if existing:
                skip_count += 1
                total_chunks += len(existing)
                continue

        # Check chunk limit
        if args.max_chunks > 0 and total_chunks >= args.max_chunks:
            print(f"{prefix} Reached chunk limit ({args.max_chunks}). Stopping.")
            break

        # Process video (resize only, no face detection)
        frames = process_video(video_path)
        if frames is None:
            fail_count += 1
            if (i + 1) % 500 == 0:
                print(f"{prefix} {video_id}: SKIP (too short) | Total chunks: {total_chunks}")
            continue

        # Save chunks
        remaining = args.max_chunks - total_chunks if args.max_chunks > 0 else 999999
        max_save = min(frames.shape[0] // CHUNK_LENGTH, remaining)

        if max_save <= 0:
            break

        saved = save_chunks(frames[:max_save * CHUNK_LENGTH], args.output, video_id)
        total_chunks += saved
        success_count += 1

        if (i + 1) % 200 == 0 or saved > 2:
            print(f"{prefix} {video_id}: {saved} chunks ({frames.shape[0]} frames) | Total: {total_chunks}")

    # Summary
    print()
    print("=" * 60)
    print("CelebV-HQ Preprocessing Complete")
    print(f"  Videos processed: {success_count}")
    print(f"  Videos skipped:   {skip_count}")
    print(f"  Videos failed:    {fail_count}")
    print(f"  Total chunks:     {total_chunks}")
    print(f"  Output:           {args.output}")
    print(f"  Chunk format:     [{CHUNK_LENGTH}, {INPUT_SIZE}, {INPUT_SIZE}, 3] uint8")
    print(f"  Note: Unlabeled data (no BVP/SpO2 labels)")


if __name__ == "__main__":
    main()
