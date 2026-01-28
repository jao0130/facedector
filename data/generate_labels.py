"""
Generate face detection labels for PURE dataset using MediaPipe.
Outputs bounding boxes and 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


class FaceLabelGenerator:
    """Generate face detection labels using MediaPipe Face Mesh."""

    # MediaPipe Face Mesh landmark indices for 5 key points
    LANDMARK_INDICES = {
        'left_eye': 468,      # Left eye center (iris)
        'right_eye': 473,     # Right eye center (iris)
        'nose': 4,            # Nose tip
        'left_mouth': 61,     # Left mouth corner
        'right_mouth': 291,   # Right mouth corner
    }

    # Fallback indices if iris landmarks not available
    LANDMARK_INDICES_FALLBACK = {
        'left_eye': 159,      # Left eye outer corner
        'right_eye': 386,     # Right eye outer corner
        'nose': 4,
        'left_mouth': 61,
        'right_mouth': 291,
    }

    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
        )

    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image and extract face labels.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with bbox and landmarks, or None if no face detected
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        height, width = image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # Extract all landmark coordinates
        landmarks_px = []
        for lm in face_landmarks.landmark:
            landmarks_px.append((lm.x * width, lm.y * height))
        landmarks_px = np.array(landmarks_px)

        # Calculate bounding box from all landmarks
        x_coords = landmarks_px[:, 0]
        y_coords = landmarks_px[:, 1]

        x_min = max(0, np.min(x_coords))
        y_min = max(0, np.min(y_coords))
        x_max = min(width, np.max(x_coords))
        y_max = min(height, np.max(y_coords))

        # Add padding to bbox (10%)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        padding_x = bbox_width * 0.1
        padding_y = bbox_height * 0.1

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)

        # Extract 5 key landmarks
        try:
            # Try with iris landmarks first
            key_landmarks = self._extract_key_landmarks(
                face_landmarks, width, height, self.LANDMARK_INDICES
            )
        except (IndexError, KeyError):
            # Fallback to basic landmarks
            key_landmarks = self._extract_key_landmarks(
                face_landmarks, width, height, self.LANDMARK_INDICES_FALLBACK
            )

        # Normalize coordinates to [0, 1]
        bbox_normalized = {
            'x_min': x_min / width,
            'y_min': y_min / height,
            'x_max': x_max / width,
            'y_max': y_max / height,
        }

        landmarks_normalized = {
            name: {'x': x / width, 'y': y / height}
            for name, (x, y) in key_landmarks.items()
        }

        return {
            'bbox': bbox_normalized,
            'landmarks': landmarks_normalized,
            'image_width': width,
            'image_height': height,
        }

    def _extract_key_landmarks(
        self,
        face_landmarks,
        width: int,
        height: int,
        indices: Dict[str, int]
    ) -> Dict[str, Tuple[float, float]]:
        """Extract 5 key landmarks from face mesh."""
        key_landmarks = {}
        for name, idx in indices.items():
            lm = face_landmarks.landmark[idx]
            key_landmarks[name] = (lm.x * width, lm.y * height)
        return key_landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


def get_pure_image_paths(pure_dir: str) -> List[Tuple[str, str]]:
    """
    Get all image paths from PURE dataset.

    Args:
        pure_dir: Path to PURE dataset root directory

    Returns:
        List of (subject_session, image_path) tuples
    """
    image_paths = []
    pure_path = Path(pure_dir)

    # PURE structure: subject/session/*.png
    for subject_dir in sorted(pure_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        if not subject_dir.name.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            continue

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            session_id = f"{subject_dir.name}_{session_dir.name}"

            for img_file in sorted(session_dir.glob("*.png")):
                image_paths.append((session_id, str(img_file)))

    return image_paths


def process_pure_dataset(
    pure_dir: str,
    output_dir: str,
    min_confidence: float = 0.5,
    save_interval: int = 100,
) -> Dict[str, int]:
    """
    Process entire PURE dataset and generate labels.

    Args:
        pure_dir: Path to PURE dataset
        output_dir: Path to save labels
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
    generator = FaceLabelGenerator(min_detection_confidence=min_confidence)

    # Process images
    labels = {}
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
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence (default: 0.5)",
    )

    args = parser.parse_args()

    process_pure_dataset(
        pure_dir=args.pure_dir,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
