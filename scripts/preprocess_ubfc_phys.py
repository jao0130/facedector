"""
UBFC-Phys dataset preprocessing for rPPG training.

UBFC-Phys structure (56 subjects × 3 tasks):
    s1/
    |-- vid_s1_T1.avi   # ~35fps webcam video (task 1: social stress)
    |-- vid_s1_T2.avi   # task 2: arithmetic
    |-- vid_s1_T3.avi   # task 3: relaxation
    |-- bvp_s1_T1.csv   # BVP from Empatica E4 @ 64Hz
    |-- bvp_s1_T2.csv
    |-- bvp_s1_T3.csv
    s2/ ...

BVP CSV format (Empatica E4):
    First line: sampling rate (64.0) OR header string → skip if non-numeric
    Remaining lines: one BVP value per line

SpO2: NOT available in UBFC-Phys (E4 wristband has no SpO2 sensor).
      All spo2.npy files are saved as 0.0 → dataset loader treats 0 as missing.

Output:
    s{X}_T{Y}_chunk{N}_input.npy  [128,72,72,3] uint8
    s{X}_T{Y}_chunk{N}_bvp.npy   [128] float32
    s{X}_T{Y}_chunk{N}_spo2.npy  0.0  (missing — masked in SpO2 loss)

Usage:
    python scripts/preprocess_ubfc_phys.py \\
        --input D:/UBFC-Phys \\
        --output D:/PreprocessedData_UBFCPhys
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Constants (match all other preprocessing scripts) ──
TARGET_FPS     = 30
CHUNK_LENGTH   = 128
INPUT_SIZE     = 72
LARGE_BOX_COEF = 1.5
FACE_THRESHOLD = 0.4
DETECT_EVERY_N = 15
BVP_FS         = 64          # Empatica E4 sampling rate
TASKS          = ['T1', 'T2', 'T3']


def _ensure_mediapipe_model() -> str:
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"
    if model_path.exists():
        return str(model_path)
    print("[UBFCPhys] Downloading MediaPipe face_landmarker model...")
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
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


def process_video(video_path: str, detector) -> Optional[Tuple[np.ndarray, int]]:
    """
    Extract face-cropped frames subsampled to ~TARGET_FPS.

    Returns:
        (frames [N,72,72,3] uint8, frame_interval) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        return None

    frame_interval = max(1, round(src_fps / TARGET_FPS))

    frames = []
    frame_idx   = 0
    sampled_idx = 0
    last_bbox   = None
    consecutive_misses = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if sampled_idx % DETECT_EVERY_N == 0 or last_bbox is None:
            bbox = detect_face_bbox(detector, frame_rgb)
        else:
            bbox = last_bbox
        sampled_idx += 1

        if bbox is None:
            consecutive_misses += 1
            if consecutive_misses > 15:
                break
            if last_bbox is not None:
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
        crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
        frames.append(crop_resized)

    cap.release()

    if len(frames) < CHUNK_LENGTH:
        return None

    return np.array(frames, dtype=np.uint8), frame_interval


def load_bvp_csv(csv_path: str) -> Optional[np.ndarray]:
    """
    Load BVP CSV from Empatica E4.

    Handles two common formats:
      A) First line = sampling rate (e.g. "64.00"), rest are values
      B) No header — all lines are BVP values

    Returns [N] float32 array at BVP_FS Hz, or None on error.
    """
    try:
        with open(csv_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError:
        return None

    if len(lines) < CHUNK_LENGTH:
        return None

    # Detect header: first line is a single float (sample rate) close to 64
    try:
        first_val = float(lines[0])
        # If it looks like a sample rate (30–256 Hz) rather than a BVP amplitude
        if 30.0 <= first_val <= 256.0 and ',' not in lines[0]:
            lines = lines[1:]
    except ValueError:
        # Non-numeric header → skip
        lines = lines[1:]

    try:
        values = [float(l.split(',')[0]) for l in lines]
    except ValueError:
        return None

    if len(values) < CHUNK_LENGTH:
        return None

    return np.array(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="UBFC-Phys preprocessing to NPY chunks")
    parser.add_argument("--input", type=str, required=True,
                        help="UBFC-Phys root directory containing s1..s56 folders")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for NPY chunks")
    parser.add_argument("--tasks", type=str, default="T1,T2,T3",
                        help="Comma-separated task list (default: T1,T2,T3)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sessions that already have output chunks")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(',')]

    # Discover subject folders (s1, s2, ..., s56)
    subject_dirs = sorted([
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
        and d.startswith('s') and d[1:].isdigit()
    ], key=lambda d: int(d[1:]))

    print(f"[UBFCPhys] Found {len(subject_dirs)} subjects, tasks: {tasks}")
    print("[UBFCPhys] Loading MediaPipe FaceLandmarker...")
    detector = create_face_detector()

    total_chunks  = 0
    success_count = 0
    fail_count    = 0
    session_total = len(subject_dirs) * len(tasks)
    session_idx   = 0

    for subj in subject_dirs:
        subj_dir = os.path.join(args.input, subj)

        for task in tasks:
            session_idx += 1
            session_name = f"{subj}_{task}"

            video_path = os.path.join(subj_dir, f"vid_{subj}_{task}.avi")
            bvp_path   = os.path.join(subj_dir, f"bvp_{subj}_{task}.csv")

            if not os.path.isfile(video_path) or not os.path.isfile(bvp_path):
                fail_count += 1
                print(f"[{session_idx}/{session_total}] {session_name}: "
                      f"SKIP (missing files)")
                continue

            if args.skip_existing:
                existing = [f for f in os.listdir(args.output)
                            if f.startswith(session_name + "_chunk")]
                if existing:
                    n_ex = len(existing) // 3
                    total_chunks += n_ex
                    print(f"[{session_idx}/{session_total}] {session_name}: "
                          f"SKIP (existing, {n_ex} chunks)")
                    continue

            # Load BVP (64 Hz)
            bvp_raw = load_bvp_csv(bvp_path)
            if bvp_raw is None:
                fail_count += 1
                print(f"[{session_idx}/{session_total}] {session_name}: "
                      f"SKIP (bad BVP CSV)")
                continue

            # Process video → face-cropped frames at ~30fps
            result = process_video(video_path, detector)
            if result is None:
                fail_count += 1
                print(f"[{session_idx}/{session_total}] {session_name}: "
                      f"SKIP (no face / too short)")
                continue

            frames, frame_interval = result
            n_frames = frames.shape[0]

            # Resample BVP from 64Hz to match sampled video frames (30fps)
            # Duration of video in seconds ≈ n_frames / TARGET_FPS
            duration_frames = n_frames / TARGET_FPS
            n_bvp_target = round(duration_frames * BVP_FS)
            n_bvp_target = min(n_bvp_target, len(bvp_raw))

            bvp_clipped = bvp_raw[:n_bvp_target]
            bvp_resampled = np.interp(
                np.linspace(0, len(bvp_clipped) - 1, n_frames),
                np.arange(len(bvp_clipped)),
                bvp_clipped,
            ).astype(np.float32)

            # Align
            min_len = min(n_frames, len(bvp_resampled))
            if min_len < CHUNK_LENGTH:
                fail_count += 1
                print(f"[{session_idx}/{session_total}] {session_name}: "
                      f"SKIP (too short: {min_len})")
                continue

            frames       = frames[:min_len]
            bvp_resampled = bvp_resampled[:min_len]

            # Save 128-frame chunks (no SpO2 → 0.0)
            n_chunks = min_len // CHUNK_LENGTH
            saved = 0
            for c in range(n_chunks):
                start = c * CHUNK_LENGTH
                end   = start + CHUNK_LENGTH
                base  = os.path.join(args.output,
                                     f"{session_name}_chunk{c:03d}")
                np.save(base + "_input.npy", frames[start:end])
                np.save(base + "_bvp.npy",   bvp_resampled[start:end])
                np.save(base + "_spo2.npy",  np.float32(0.0))  # no SpO2 label
                saved += 1

            total_chunks  += saved
            success_count += 1
            print(f"[{session_idx}/{session_total}] {session_name}: "
                  f"{saved} chunks ({n_frames} frames, "
                  f"bvp_raw={len(bvp_raw)}) | Total: {total_chunks}")

    print()
    print("=" * 60)
    print("UBFC-Phys Preprocessing Complete")
    print(f"  Sessions processed : {success_count}")
    print(f"  Sessions failed    : {fail_count}")
    print(f"  Total chunks       : {total_chunks}")
    print(f"  Output             : {args.output}")
    print(f"  SpO2 label         : 0.0 (missing — masked in SpO2 loss)")
    print(f"  Chunk format       : [{CHUNK_LENGTH}, {INPUT_SIZE}, {INPUT_SIZE}, 3] uint8")


if __name__ == "__main__":
    main()
