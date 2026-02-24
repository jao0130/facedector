"""
PURE dataset preprocessing for rPPG training (labeled data).

PURE structure:
    RawData/
    |-- 01-01/
    |   |-- 01-01/          # PNG frames (Image{timestamp}.png)
    |   |-- 01-01.json      # PPG waveform + SpO2 at ~31Hz
    |-- 01-02/
    ...

JSON format:
    /FullPackage: list of {Timestamp, Value: {waveform, o2saturation, ...}}
    /Image: list of {Timestamp, ...} (one per video frame)

Output:
    {session}_chunk{N}_input.npy  [128,72,72,3] uint8
    {session}_chunk{N}_bvp.npy    [128] float32
    {session}_chunk{N}_spo2.npy   scalar float32

Usage:
    python scripts/preprocess_pure.py \
        --input D:/PURE \
        --output D:/PreprocessedData
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Constants (match training preprocessing) ──
CHUNK_LENGTH = 128
INPUT_SIZE = 72
LARGE_BOX_COEF = 1.5
FACE_THRESHOLD = 0.4
DETECT_EVERY_N = 15


def _ensure_mediapipe_model() -> str:
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"
    if model_path.exists():
        return str(model_path)
    print("[PURE] Downloading MediaPipe face_landmarker model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def create_face_detector(model_path: Optional[str] = None):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if model_path is None:
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


def detect_face_bbox(detector, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    lms = result.face_landmarks[0]
    xs = [l.x for l in lms]
    ys = [l.y for l in lms]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

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


def read_pure_frames(frame_dir: str, detector) -> Optional[np.ndarray]:
    """
    Read PNG frames from PURE session, detect/crop face, resize to INPUT_SIZE.

    Returns:
        [N, 72, 72, 3] uint8 array or None.
    """
    all_png = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if len(all_png) < CHUNK_LENGTH:
        return None

    frames = []
    last_bbox = None
    consecutive_misses = 0

    for idx, png_path in enumerate(all_png):
        img = cv2.imread(png_path)
        if img is None:
            continue
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection every N frames
        if idx % DETECT_EVERY_N == 0 or last_bbox is None:
            bbox = detect_face_bbox(detector, frame_rgb)
        else:
            bbox = last_bbox

        if bbox is None:
            consecutive_misses += 1
            if consecutive_misses > 15:
                break
            if last_bbox is not None:
                bbox = last_bbox
            else:
                continue
        else:
            consecutive_misses = 0  # reset on any valid detection, not only when bbox changes
            last_bbox = bbox

        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(crop_resized)

    if len(frames) < CHUNK_LENGTH:
        return None

    return np.array(frames, dtype=np.uint8)


def read_pure_labels(json_path: str, n_frames: int) -> Optional[Tuple[np.ndarray, float]]:
    """
    Read BVP waveform and SpO2 from PURE JSON, resample to n_frames.

    Returns:
        (bvp [n_frames] float32, spo2_mean float) or None.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    packages = data.get("/FullPackage", [])
    if len(packages) < CHUNK_LENGTH:
        return None

    bvp_raw = np.array([p["Value"]["waveform"] for p in packages], dtype=np.float32)
    spo2_raw = np.array([p["Value"]["o2saturation"] for p in packages], dtype=np.float32)

    # Resample BVP from ~31Hz to match video frame count
    bvp_resampled = np.interp(
        np.linspace(0, len(bvp_raw) - 1, n_frames),
        np.arange(len(bvp_raw)),
        bvp_raw,
    ).astype(np.float32)

    spo2_mean = float(np.mean(spo2_raw))

    return bvp_resampled, spo2_mean


def main():
    parser = argparse.ArgumentParser(description="PURE dataset preprocessing to NPY chunks")
    parser.add_argument("--input", type=str, required=True,
                        help="PURE dataset root directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for NPY chunks")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sessions that already have chunks")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Collect session directories (format: XX-YY)
    sessions = sorted([
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d)) and '-' in d
    ])
    print(f"[PURE] Found {len(sessions)} sessions")

    # Create face detector
    print("[PURE] Loading MediaPipe FaceLandmarker...")
    detector = create_face_detector()

    total_chunks = 0
    success_count = 0
    fail_count = 0

    for i, session in enumerate(sessions):
        session_dir = os.path.join(args.input, session)
        frame_dir = os.path.join(session_dir, session)  # PNG frames in subfolder
        json_path = os.path.join(session_dir, f"{session}.json")

        if not os.path.isdir(frame_dir) or not os.path.isfile(json_path):
            fail_count += 1
            print(f"[{i+1}/{len(sessions)}] {session}: SKIP (missing files)")
            continue

        # Skip existing
        if args.skip_existing:
            existing = [f for f in os.listdir(args.output) if f.startswith(session + "_chunk")]
            if existing:
                n_existing = len(existing) // 3
                total_chunks += n_existing
                print(f"[{i+1}/{len(sessions)}] {session}: SKIP (existing, {n_existing} chunks)")
                continue

        # Read and crop face frames
        frames = read_pure_frames(frame_dir, detector)
        if frames is None:
            fail_count += 1
            print(f"[{i+1}/{len(sessions)}] {session}: SKIP (no face / too short)")
            continue

        n_frames = frames.shape[0]

        # Read and resample labels
        labels = read_pure_labels(json_path, n_frames)
        if labels is None:
            fail_count += 1
            print(f"[{i+1}/{len(sessions)}] {session}: SKIP (bad labels)")
            continue

        bvp, spo2_mean = labels

        # Save chunks
        n_chunks = n_frames // CHUNK_LENGTH
        saved = 0

        for c in range(n_chunks):
            start = c * CHUNK_LENGTH
            end = start + CHUNK_LENGTH
            base_name = f"{session}_chunk{c:03d}"
            np.save(os.path.join(args.output, f"{base_name}_input.npy"), frames[start:end])
            np.save(os.path.join(args.output, f"{base_name}_bvp.npy"), bvp[start:end])
            np.save(os.path.join(args.output, f"{base_name}_spo2.npy"), np.float32(spo2_mean))
            saved += 1

        total_chunks += saved
        success_count += 1
        print(f"[{i+1}/{len(sessions)}] {session}: {saved} chunks "
              f"({n_frames} frames, SpO2={spo2_mean:.1f}) | Total: {total_chunks}")

    print()
    print("=" * 60)
    print(f"PURE Preprocessing Complete")
    print(f"  Sessions processed: {success_count}")
    print(f"  Sessions failed:    {fail_count}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Output:             {args.output}")
    print(f"  Chunk format:       [{CHUNK_LENGTH}, {INPUT_SIZE}, {INPUT_SIZE}, 3] uint8")


if __name__ == "__main__":
    main()
