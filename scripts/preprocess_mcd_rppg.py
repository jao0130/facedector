"""
MCD-rPPG preprocessing for semi-supervised rPPG training (labeled data).

Downloads from HuggingFace (kyegorov/mcd_rppg) and extracts:
  1. Read AVI video frames
  2. Detect face using MediaPipe FaceLandmarker
  3. Crop face ROI with padding (LARGE_BOX_COEF=1.5)
  4. Resize to 72x72
  5. Read ppg_sync .txt (100Hz PPG) → resample to video FPS
  6. Read SpO2 from db.csv
  7. Save 128-frame chunks as:
       {subject}_{camera}_{state}_chunk{N}_input.npy  [128,72,72,3]
       {subject}_{camera}_{state}_chunk{N}_bvp.npy    [128]
       {subject}_{camera}_{state}_chunk{N}_spo2.npy   scalar

Usage:
    python scripts/preprocess_mcd_rppg.py \
        --hf_cache D:/MCD_rPPG_HF \
        --output D:/PreprocessedData_MCD \
        --camera FullHDwebcam \
        --download \
        --max_chunks 8000

MCD-rPPG structure (HuggingFace kyegorov/mcd_rppg):
    video/{subject}_{camera}_{state}.avi
    ppg_sync/{subject}_{camera}_{state}.txt  (100Hz PPG, "{index} {value}" per line)
    db.csv                                   (subject metadata + SpO2)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Constants ──
TARGET_FPS = 30
CHUNK_LENGTH = 128
INPUT_SIZE = 72
LARGE_BOX_COEF = 1.5
FACE_THRESHOLD = 0.4
PPG_SYNC_SAMPLE_RATE = 100  # Hz


def download_dataset(hf_cache: str, camera: str = "FullHDwebcam"):
    """Download MCD-rPPG dataset from HuggingFace (selective)."""
    from huggingface_hub import snapshot_download

    # Download video files for the selected camera, ppg_sync, and db.csv
    allow_patterns = [
        f"video/*_{camera}_*.avi",
        "ppg_sync/*.txt",
        "db.csv",
    ]

    print(f"[MCD-rPPG] Downloading from kyegorov/mcd_rppg (camera={camera})...")
    print(f"[MCD-rPPG] Patterns: {allow_patterns}")

    path = snapshot_download(
        repo_id="kyegorov/mcd_rppg",
        repo_type="dataset",
        local_dir=hf_cache,
        allow_patterns=allow_patterns,
    )
    print(f"[MCD-rPPG] Downloaded to: {path}")
    return path


def load_spo2_map(db_csv_path: str) -> Dict[str, float]:
    """
    Parse db.csv and return mapping: subject_id -> SpO2 value.

    Tries common column names: 'saturation', 'SpO2', 'spo2', 'oxygen_saturation'.
    """
    spo2_map = {}
    if not os.path.isfile(db_csv_path):
        print(f"[MCD-rPPG] WARNING: db.csv not found at {db_csv_path}")
        return spo2_map

    with open(db_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        # Find SpO2 column
        spo2_col = None
        for candidate in ['saturation', 'SpO2', 'spo2', 'oxygen_saturation', 'Saturation']:
            if candidate in columns:
                spo2_col = candidate
                break

        # Find subject ID column
        id_col = None
        for candidate in ['patient_id', 'id', 'subject', 'subject_id', 'ID', 'Subject']:
            if candidate in columns:
                id_col = candidate
                break

        if spo2_col is None or id_col is None:
            print(f"[MCD-rPPG] WARNING: Could not find SpO2/ID columns in db.csv. Columns: {columns}")
            return spo2_map

        print(f"[MCD-rPPG] db.csv: using '{id_col}' for subject ID, '{spo2_col}' for SpO2")

        for row in reader:
            try:
                subject_id = str(row[id_col]).strip()
                spo2_val = float(row[spo2_col])
                spo2_map[subject_id] = spo2_val
            except (ValueError, KeyError):
                continue

    print(f"[MCD-rPPG] Loaded SpO2 for {len(spo2_map)} subjects")
    return spo2_map


def load_ppg_sync(ppg_path: str, n_video_frames: int, video_fps: float) -> Optional[np.ndarray]:
    """
    Load ppg_sync .txt file (100Hz) and resample to match video frames.

    Args:
        ppg_path: path to ppg_sync .txt file
        n_video_frames: number of video frames to align to
        video_fps: video frame rate

    Returns:
        [n_video_frames] float32 array, or None if failed.
    """
    if not os.path.isfile(ppg_path):
        return None

    try:
        ppg_values = []
        with open(ppg_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # Format: "{sample_index} {ppg_value}"
                    ppg_values.append(float(parts[1]))
                else:
                    ppg_values.append(float(parts[0]))

        if len(ppg_values) < 10:
            return None

        ppg_raw = np.array(ppg_values, dtype=np.float64)

        # Validate PPG covers enough of the video duration
        video_duration = n_video_frames / video_fps
        ppg_duration = len(ppg_raw) / PPG_SYNC_SAMPLE_RATE
        if ppg_duration < video_duration * 0.8:
            # PPG signal too short — would cause flat-line artifacts at the end
            return None

        # Resample from PPG_SYNC_SAMPLE_RATE (100Hz) to video_fps
        video_times = np.arange(n_video_frames) / video_fps
        ppg_indices = video_times * PPG_SYNC_SAMPLE_RATE

        # Clip to valid range
        ppg_indices = np.clip(ppg_indices, 0, len(ppg_raw) - 1)

        # Linear interpolation
        ppg_resampled = np.interp(ppg_indices, np.arange(len(ppg_raw)), ppg_raw)

        return ppg_resampled.astype(np.float32)

    except Exception as e:
        print(f"  WARNING: Failed to load PPG sync: {ppg_path}: {e}")
        return None


def create_face_detector(model_path: Optional[str] = None):
    """Create MediaPipe FaceLandmarker."""
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


def _ensure_mediapipe_model() -> str:
    """Download MediaPipe face landmarker model if not cached."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"

    if model_path.exists():
        return str(model_path)

    print("[MCD-rPPG] Downloading MediaPipe face_landmarker model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def detect_face_bbox(detector, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect face and return padded bounding box (x1, y1, x2, y2)."""
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


DETECT_EVERY_N = 15  # Run face detection every Nth sampled frame (~0.5s at 30fps)


def process_video(video_path: str, detector) -> Optional[Tuple[np.ndarray, float]]:
    """
    Extract face-cropped frames from a video.
    Detects face every DETECT_EVERY_N frames and reuses bbox for intermediate frames.

    Returns:
        (frames [N, 72, 72, 3] uint8, actual_fps) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        return None

    # Subsample to TARGET_FPS
    frame_interval = max(1, round(src_fps / TARGET_FPS))
    actual_fps = src_fps / frame_interval

    frames = []
    frame_idx = 0
    sampled_idx = 0
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

        # Only run face detection every N sampled frames
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
            if bbox != last_bbox:
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

    return np.array(frames, dtype=np.uint8), actual_fps


def save_chunks(
    frames: np.ndarray,
    ppg: np.ndarray,
    spo2_val: float,
    output_dir: str,
    video_id: str,
    max_chunks: int = 0,
) -> int:
    """
    Split frames + PPG into CHUNK_LENGTH chunks and save as NPY.

    Returns number of chunks saved.
    """
    total_frames = min(frames.shape[0], ppg.shape[0])
    n_chunks = total_frames // CHUNK_LENGTH
    if max_chunks > 0:
        n_chunks = min(n_chunks, max_chunks)

    saved = 0
    for i in range(n_chunks):
        start = i * CHUNK_LENGTH
        end = start + CHUNK_LENGTH

        chunk_video = frames[start:end]   # [128, 72, 72, 3]
        chunk_ppg = ppg[start:end]         # [128]

        base_name = f"{video_id}_chunk{i:03d}"
        np.save(os.path.join(output_dir, f"{base_name}_input.npy"), chunk_video)
        np.save(os.path.join(output_dir, f"{base_name}_bvp.npy"), chunk_ppg)
        np.save(os.path.join(output_dir, f"{base_name}_spo2.npy"), np.float32(spo2_val))
        saved += 1

    return saved


def collect_videos(data_dir: str, camera: str = "FullHDwebcam") -> List[str]:
    """Collect AVI files for the specified camera from the video directory."""
    video_dir = os.path.join(data_dir, "video")
    if not os.path.isdir(video_dir):
        # Try flat directory
        video_dir = data_dir

    videos = []
    for f in sorted(os.listdir(video_dir)):
        if f.lower().endswith('.avi') and camera in f:
            videos.append(os.path.join(video_dir, f))
    return videos


def parse_video_id(video_path: str) -> Tuple[str, str, str]:
    """
    Parse video filename into (subject_id, camera, state).
    E.g. '1020_FullHDwebcam_before.avi' -> ('1020', 'FullHDwebcam', 'before')
    """
    name = Path(video_path).stem  # '1020_FullHDwebcam_before'
    parts = name.split('_')
    if len(parts) >= 3:
        subject = parts[0]
        state = parts[-1]
        camera = '_'.join(parts[1:-1])
        return subject, camera, state
    return name, "", ""


def find_ppg_sync(data_dir: str, subject: str, camera: str, state: str) -> Optional[str]:
    """Find the ppg_sync file for a given subject, camera, and state."""
    ppg_dir = os.path.join(data_dir, "ppg_sync")
    if not os.path.isdir(ppg_dir):
        return None

    # MCD-rPPG naming: {subject}_{camera}_{state}.txt
    candidates = [
        f"{subject}_{camera}_{state}.txt",
        f"{subject}_{state}.txt",  # fallback without camera
    ]
    for cand in candidates:
        path = os.path.join(ppg_dir, cand)
        if os.path.isfile(path):
            return path

    # Fallback: search for any file containing subject, camera and state
    for f in os.listdir(ppg_dir):
        if subject in f and state in f and f.endswith('.txt'):
            if camera in f or '_' not in f.replace(subject, '').replace(state, ''):
                return os.path.join(ppg_dir, f)

    return None


def main():
    parser = argparse.ArgumentParser(description="MCD-rPPG preprocessing for semi-supervised rPPG")
    parser.add_argument("--hf_cache", type=str, required=True,
                        help="HuggingFace download / local dataset directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for NPY chunks")
    parser.add_argument("--camera", type=str, default="FullHDwebcam",
                        choices=["FullHDwebcam", "USBVideo", "IriunWebcam"],
                        help="Camera type to process (default: FullHDwebcam)")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from HuggingFace first")
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="Stop after saving this many total chunks (0=unlimited)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have chunks")
    parser.add_argument("--mediapipe_model", type=str, default=None,
                        help="Path to MediaPipe face_landmarker.task model")
    args = parser.parse_args()

    data_dir = args.hf_cache

    # Step 1: Download if requested
    if args.download:
        data_dir = download_dataset(args.hf_cache, args.camera)

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Step 2: Load SpO2 map from db.csv
    db_csv = os.path.join(data_dir, "db.csv")
    spo2_map = load_spo2_map(db_csv)

    # Step 3: Collect videos
    videos = collect_videos(data_dir, args.camera)
    print(f"[MCD-rPPG] Found {len(videos)} {args.camera} videos")

    if not videos:
        print("No videos found. Check --hf_cache directory and --camera option.")
        sys.exit(1)

    # Step 4: Create face detector
    print("[MCD-rPPG] Loading MediaPipe FaceLandmarker...")
    detector = create_face_detector(args.mediapipe_model)

    total_chunks = 0
    success_count = 0
    skip_count = 0
    fail_count = 0
    no_ppg_count = 0

    for i, video_path in enumerate(videos):
        subject, camera, state = parse_video_id(video_path)
        video_id = f"{subject}_{camera}_{state}"
        prefix = f"[{i+1}/{len(videos)}]"

        # Skip existing
        if args.skip_existing:
            existing = [f for f in os.listdir(args.output) if f.startswith(video_id + "_chunk")]
            if existing:
                skip_count += 1
                total_chunks += len(existing) // 3  # 3 files per chunk
                continue

        # Check chunk limit
        if args.max_chunks > 0 and total_chunks >= args.max_chunks:
            print(f"{prefix} Reached chunk limit ({args.max_chunks}). Stopping.")
            break

        # Find PPG sync file
        ppg_path = find_ppg_sync(data_dir, subject, camera, state)
        if ppg_path is None:
            no_ppg_count += 1
            if (i + 1) % 100 == 0:
                print(f"{prefix} {video_id}: SKIP (no ppg_sync)")
            continue

        # Process video (face crop)
        result = process_video(video_path, detector)
        if result is None:
            fail_count += 1
            if (i + 1) % 100 == 0:
                print(f"{prefix} {video_id}: SKIP (no face / too short)")
            continue

        frames, actual_fps = result

        # Load and resample PPG
        ppg = load_ppg_sync(ppg_path, frames.shape[0], actual_fps)
        if ppg is None:
            no_ppg_count += 1
            continue

        # Get SpO2 — skip if missing or out of range
        spo2_val = spo2_map.get(subject, None)
        if spo2_val is None or spo2_val < 70.0 or spo2_val > 100.0:
            spo2_val = 0.0  # Mark as unavailable (semi-supervised trainer ignores SpO2)

        # Save chunks
        remaining = args.max_chunks - total_chunks if args.max_chunks > 0 else 999999
        saved = save_chunks(frames, ppg, spo2_val, args.output, video_id, max_chunks=remaining)
        total_chunks += saved
        success_count += 1

        print(f"{prefix} {video_id}: {saved} chunks ({frames.shape[0]} frames, SpO2={spo2_val:.1f}) | Total: {total_chunks}")

    # Summary
    print()
    print("=" * 60)
    print(f"MCD-rPPG Preprocessing Complete ({args.camera})")
    print(f"  Videos processed: {success_count}")
    print(f"  Videos skipped:   {skip_count}")
    print(f"  Videos failed:    {fail_count}")
    print(f"  No PPG sync:      {no_ppg_count}")
    print(f"  Total chunks:     {total_chunks}")
    print(f"  Output:           {args.output}")
    print(f"  Chunk format:     [{CHUNK_LENGTH}, {INPUT_SIZE}, {INPUT_SIZE}, 3] uint8")


if __name__ == "__main__":
    main()
