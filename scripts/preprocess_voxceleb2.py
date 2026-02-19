"""
VoxCeleb2 preprocessing for semi-supervised rPPG training.

Extracts face-cropped video chunks from VoxCeleb2 MP4 files:
  1. Read MP4 video frames
  2. Detect face using MediaPipe FaceLandmarker
  3. Crop face ROI with padding (LARGE_BOX_COEF=1.5)
  4. Resize to 72x72
  5. Save 128-frame chunks as {id}_chunk{N}_input.npy

No BVP/SpO2 labels are saved (unlabeled data).

Usage:
    python scripts/preprocess_voxceleb2.py \
        --input D:/VoxCeleb2/dev/mp4 \
        --output D:/PreprocessedData_VoxCeleb2 \
        --max_videos 2000 \
        --max_chunks 5000

VoxCeleb2 structure:
    D:/VoxCeleb2/dev/mp4/id00001/video_id/00001.mp4
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ── Constants ──
TARGET_FPS = 30
CHUNK_LENGTH = 128
INPUT_SIZE = 72
LARGE_BOX_COEF = 1.5
FACE_THRESHOLD = 0.4


def create_face_detector(model_path: Optional[str] = None) -> vision.FaceLandmarker:
    """Create MediaPipe FaceLandmarker for face detection."""
    if model_path is None:
        # Download default model
        model_path = _ensure_mediapipe_model()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=FACE_THRESHOLD,
        min_face_presence_confidence=FACE_THRESHOLD,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _ensure_mediapipe_model() -> str:
    """Download MediaPipe face landmarker model if not cached."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"

    if model_path.exists():
        return str(model_path)

    print("[Preprocess] Downloading MediaPipe face_landmarker model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    urllib.request.urlretrieve(url, str(model_path))
    print(f"[Preprocess] Model saved to {model_path}")
    return str(model_path)


def detect_face_bbox(
    detector: vision.FaceLandmarker,
    frame_rgb: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face and return padded bounding box in pixel coordinates.

    Returns:
        (x1, y1, x2, y2) or None if no face detected.
    """
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    lms = result.face_landmarks[0]

    # Bounding box from all landmarks (normalized coords)
    xs = [l.x for l in lms]
    ys = [l.y for l in lms]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Expand with LARGE_BOX_COEF
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = (x_max - x_min) * LARGE_BOX_COEF
    bh = (y_max - y_min) * LARGE_BOX_COEF

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return (x1, y1, x2, y2)


def process_video(
    video_path: str,
    detector: vision.FaceLandmarker,
    target_fps: int = TARGET_FPS,
) -> Optional[np.ndarray]:
    """
    Extract face-cropped frames from a video.

    Returns:
        [N, H, W, C] uint8 array of face crops, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        return None

    # Frame sampling to match target FPS
    frame_interval = max(1, round(src_fps / target_fps))

    frames = []
    frame_idx = 0
    consecutive_misses = 0
    last_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bbox = detect_face_bbox(detector, frame_rgb)
        if bbox is None:
            consecutive_misses += 1
            if consecutive_misses > 15:
                # Too many missed frames, stop
                break
            if last_bbox is not None:
                # Use last known bbox
                bbox = last_bbox
            else:
                continue
        else:
            consecutive_misses = 0
            last_bbox = bbox

        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(crop_resized)

    cap.release()

    if len(frames) < CHUNK_LENGTH:
        return None

    return np.array(frames, dtype=np.uint8)


def save_chunks(
    frames: np.ndarray,
    output_dir: str,
    video_id: str,
) -> int:
    """
    Split frames into CHUNK_LENGTH chunks and save as NPY.

    Returns:
        Number of chunks saved.
    """
    total_frames = frames.shape[0]
    n_chunks = total_frames // CHUNK_LENGTH
    saved = 0

    for i in range(n_chunks):
        start = i * CHUNK_LENGTH
        end = start + CHUNK_LENGTH
        chunk = frames[start:end]  # [128, 72, 72, 3]

        base_name = f"{video_id}_chunk{i:03d}"
        out_path = os.path.join(output_dir, f"{base_name}_input.npy")
        np.save(out_path, chunk)
        saved += 1

    return saved


def collect_videos(input_dir: str, max_videos: int = 0) -> list:
    """Recursively collect all MP4 files from VoxCeleb2 directory."""
    videos = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                videos.append(os.path.join(root, f))
                if max_videos > 0 and len(videos) >= max_videos:
                    return videos
    return videos


def video_to_id(video_path: str, input_dir: str) -> str:
    """Generate a unique ID from the video path relative to input dir."""
    rel = os.path.relpath(video_path, input_dir)
    # Replace path separators and extension
    video_id = rel.replace(os.sep, '_').replace('/', '_').replace('.mp4', '').replace('.MP4', '')
    return video_id


def main():
    parser = argparse.ArgumentParser(description="VoxCeleb2 preprocessing for semi-supervised rPPG")
    parser.add_argument("--input", type=str, required=True, help="VoxCeleb2 MP4 directory (e.g. D:/VoxCeleb2/dev/mp4)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for NPY chunks")
    parser.add_argument("--max_videos", type=int, default=0, help="Max videos to process (0=all)")
    parser.add_argument("--max_chunks", type=int, default=0, help="Stop after saving this many chunks (0=unlimited)")
    parser.add_argument("--model", type=str, default=None, help="Path to MediaPipe face_landmarker.task model")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos that already have chunks")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect videos
    print(f"[Preprocess] Scanning {input_dir} for MP4 files...")
    videos = collect_videos(input_dir, args.max_videos)
    print(f"[Preprocess] Found {len(videos)} videos")

    if not videos:
        print("No videos found. Check the input directory.")
        sys.exit(1)

    # Create face detector
    print("[Preprocess] Loading MediaPipe FaceLandmarker...")
    detector = create_face_detector(args.model)

    total_chunks = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, video_path in enumerate(videos):
        video_id = video_to_id(video_path, input_dir)
        prefix = f"[{i+1}/{len(videos)}]"

        # Check if already processed
        if args.skip_existing:
            existing = [f for f in os.listdir(output_dir) if f.startswith(video_id + "_chunk")]
            if existing:
                skip_count += 1
                total_chunks += len(existing)
                continue

        # Check chunk limit
        if args.max_chunks > 0 and total_chunks >= args.max_chunks:
            print(f"{prefix} Reached chunk limit ({args.max_chunks}). Stopping.")
            break

        # Process video
        frames = process_video(video_path, detector)
        if frames is None:
            fail_count += 1
            if (i + 1) % 100 == 0:
                print(f"{prefix} {video_id}: SKIP (no face / too short) | Total chunks: {total_chunks}")
            continue

        # Save chunks
        remaining = args.max_chunks - total_chunks if args.max_chunks > 0 else frames.shape[0]
        max_possible = min(frames.shape[0] // CHUNK_LENGTH, remaining // CHUNK_LENGTH if args.max_chunks > 0 else 999999)

        if max_possible <= 0:
            break

        saved = save_chunks(frames[:max_possible * CHUNK_LENGTH], output_dir, video_id)
        total_chunks += saved
        success_count += 1

        if (i + 1) % 50 == 0 or saved > 0:
            print(f"{prefix} {video_id}: {saved} chunks ({frames.shape[0]} frames) | Total: {total_chunks}")

    # Summary
    print()
    print("=" * 60)
    print(f"VoxCeleb2 Preprocessing Complete")
    print(f"  Videos processed: {success_count}")
    print(f"  Videos skipped:   {skip_count}")
    print(f"  Videos failed:    {fail_count}")
    print(f"  Total chunks:     {total_chunks}")
    print(f"  Output:           {output_dir}")
    print(f"  Chunk format:     [{CHUNK_LENGTH}, {INPUT_SIZE}, {INPUT_SIZE}, 3] uint8")


if __name__ == "__main__":
    main()
