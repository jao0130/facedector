"""
Generate face detection labels for PURE dataset using MediaPipe.
Outputs bounding boxes and 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth).
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


# Model download URL
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model(model_path: str) -> str:
    """Download MediaPipe face landmarker model if not exists."""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, str(model_path))
        print("Download complete.")
    return str(model_path)


class FaceLabelGenerator:
    """Generate face detection labels using MediaPipe Face Landmarker."""

    # MediaPipe Face Landmarker indices for 5 key points
    # Total 478 landmarks (468 face + 10 iris)
    LANDMARK_INDICES = {
        'left_eye': 468,      # Left iris center
        'right_eye': 473,     # Right iris center
        'nose': 4,            # Nose tip
        'left_mouth': 61,     # Left mouth corner
        'right_mouth': 291,   # Right mouth corner
    }

    # Fallback indices if iris landmarks not available
    LANDMARK_INDICES_FALLBACK = {
        'left_eye': 159,      # Left eye
        'right_eye': 386,     # Right eye
        'nose': 4,
        'left_mouth': 61,
        'right_mouth': 291,
    }

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe Face Landmarker."""
        # Download model if needed
        model_path = download_model(model_path)

        # Create face landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image and extract face labels.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with bbox and landmarks, or None if no face detected
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        height, width = image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect face landmarks
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]

        # Extract all landmark coordinates (normalized [0, 1])
        landmarks_norm = np.array([(lm.x, lm.y) for lm in face_landmarks])

        # Calculate bounding box from all landmarks
        x_coords = landmarks_norm[:, 0]
        y_coords = landmarks_norm[:, 1]

        x_min = max(0, np.min(x_coords))
        y_min = max(0, np.min(y_coords))
        x_max = min(1, np.max(x_coords))
        y_max = min(1, np.max(y_coords))

        # Add padding to bbox (10%)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        padding_x = bbox_width * 0.1
        padding_y = bbox_height * 0.1

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(1, x_max + padding_x)
        y_max = min(1, y_max + padding_y)

        # Extract 5 key landmarks
        num_landmarks = len(face_landmarks)
        indices = self.LANDMARK_INDICES if num_landmarks > 473 else self.LANDMARK_INDICES_FALLBACK

        key_landmarks = {}
        for name, idx in indices.items():
            if idx < num_landmarks:
                lm = face_landmarks[idx]
                key_landmarks[name] = {'x': lm.x, 'y': lm.y}
            else:
                # Fallback
                fallback_idx = self.LANDMARK_INDICES_FALLBACK[name]
                lm = face_landmarks[fallback_idx]
                key_landmarks[name] = {'x': lm.x, 'y': lm.y}

        return {
            'bbox': {
                'x_min': float(x_min),
                'y_min': float(y_min),
                'x_max': float(x_max),
                'y_max': float(y_max),
            },
            'landmarks': key_landmarks,
            'image_width': width,
            'image_height': height,
        }

    def close(self):
        """Release resources."""
        self.face_landmarker.close()


def get_pure_image_paths(pure_dir: str) -> List[Tuple[str, str]]:
    """
    Get all image paths from PURE dataset.

    Args:
        pure_dir: Path to PURE dataset root directory

    Returns:
        List of (session_id, image_path) tuples
    """
    image_paths = []
    pure_path = Path(pure_dir)

    # PURE structure: XX-YY/XX-YY/*.png (e.g., 01-01/01-01/Image*.png)
    for session_dir in sorted(pure_path.iterdir()):
        if not session_dir.is_dir():
            continue
        # Match pattern like "01-01", "02-03", etc.
        if not session_dir.name[0].isdigit():
            continue

        session_id = session_dir.name

        # Images are in a subdirectory with the same name
        image_dir = session_dir / session_id
        if not image_dir.exists():
            # Try direct directory if subdirectory doesn't exist
            image_dir = session_dir

        for img_file in sorted(image_dir.glob("*.png")):
            image_paths.append((session_id, str(img_file)))

    return image_paths


def process_pure_dataset(
    pure_dir: str,
    output_dir: str,
    model_path: str,
    min_confidence: float = 0.5,
    save_interval: int = 100,
) -> Dict[str, int]:
    """
    Process entire PURE dataset and generate labels.

    Args:
        pure_dir: Path to PURE dataset
        output_dir: Path to save labels
        model_path: Path to MediaPipe model
        min_confidence: Minimum detection confidence
        save_interval: Save progress every N images

    Returns:
        Statistics dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image paths
    print("Scanning PURE dataset...")
    image_paths = get_pure_image_paths(pure_dir)
    print(f"Found {len(image_paths)} images")

    # Initialize generator
    generator = FaceLabelGenerator(model_path, min_detection_confidence=min_confidence)

    # Process images
    stats = {'total': len(image_paths), 'detected': 0, 'failed': 0}

    current_session = None
    session_labels = {}

    try:
        for session_id, image_path in tqdm(image_paths, desc="Processing images"):
            # Save previous session when session changes
            if current_session is not None and current_session != session_id:
                _save_session_labels(output_path, current_session, session_labels)
                session_labels = {}

            current_session = session_id

            # Process image
            result = generator.process_image(image_path)

            image_name = Path(image_path).name
            if result:
                session_labels[image_name] = result
                stats['detected'] += 1
            else:
                stats['failed'] += 1

        # Save last session
        if current_session is not None and session_labels:
            _save_session_labels(output_path, current_session, session_labels)

    finally:
        generator.close()

    # Save statistics
    stats_path = output_path / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nProcessing complete:")
    print(f"  Total images: {stats['total']}")
    print(f"  Faces detected: {stats['detected']}")
    print(f"  Detection failed: {stats['failed']}")
    print(f"  Detection rate: {stats['detected']/stats['total']*100:.2f}%")

    return stats


def _save_session_labels(output_path: Path, session_id: str, labels: Dict):
    """Save labels for a session to JSON file."""
    session_path = output_path / f"{session_id}.json"
    with open(session_path, 'w') as f:
        json.dump(labels, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate face detection labels for PURE dataset"
    )
    parser.add_argument(
        "--pure_dir",
        type=str,
        default="D:/PURE",
        help="Path to PURE dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/PURE_labels",
        help="Path to save generated labels",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/face_landmarker.task",
        help="Path to MediaPipe model file",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence (default: 0.5)",
    )

    args = parser.parse_args()

    process_pure_dataset(
        pure_dir=args.pure_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
