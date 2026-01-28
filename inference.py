"""
Inference script for face detection model.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import yaml

from models.face_detector import create_face_detector
from utils.visualization import visualize_detection, visualize_batch


class FaceDetectorInference:
    """Inference wrapper for face detection model."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
    ):
        """
        Initialize inference.

        Args:
            model_path: Path to model weights (.weights.h5)
            config_path: Path to config file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_size = self.config.get('model', {}).get('input_size', 256)

        # Create and load model
        self.model = create_face_detector(self.config)
        self.model.build((None, self.input_size, self.input_size, 3))
        self.model.load_weights(model_path)

        print(f"Model loaded from {model_path}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            preprocessed: Preprocessed image batch [1, H, W, 3]
            original_size: Original image size (H, W)
        """
        original_size = image.shape[:2]

        # Resize
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batch = np.expand_dims(normalized, 0)

        return batch, original_size

    def predict(
        self,
        image: np.ndarray,
        return_original_coords: bool = True,
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image (BGR or RGB)
            return_original_coords: Whether to return coordinates in original image space

        Returns:
            Dictionary with bbox, landmarks, confidence
        """
        # Preprocess
        batch, original_size = self.preprocess(image)

        # Run inference
        predictions = self.model(batch, training=False)

        # Extract results
        bbox = predictions['bbox'][0].numpy()
        landmarks = predictions['landmarks'][0].numpy()
        confidence = float(predictions['confidence'][0].numpy())

        if return_original_coords:
            h, w = original_size
            # Scale bbox to original size
            bbox = np.array([
                bbox[0] * w,
                bbox[1] * h,
                bbox[2] * w,
                bbox[3] * h,
            ])
            # Scale landmarks to original size
            landmarks = landmarks * np.array([[w, h]])

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }

    def predict_batch(
        self,
        images: List[np.ndarray],
        return_original_coords: bool = True,
    ) -> List[Dict]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images
            return_original_coords: Whether to return coordinates in original image space

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_original_coords)
            results.append(result)
        return results

    def draw_detection(
        self,
        image: np.ndarray,
        prediction: Dict,
        draw_landmarks: bool = True,
        draw_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            image: Input image (will be modified in place)
            prediction: Prediction dictionary
            draw_landmarks: Whether to draw landmarks
            draw_confidence: Whether to draw confidence score

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        bbox = prediction['bbox'].astype(int)
        landmarks = prediction['landmarks'].astype(int)
        confidence = prediction['confidence']

        # Draw bounding box
        cv2.rectangle(
            output,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )

        # Draw landmarks
        if draw_landmarks:
            colors = [
                (255, 0, 0),    # left eye - blue
                (255, 0, 0),    # right eye - blue
                (0, 255, 0),    # nose - green
                (0, 0, 255),    # left mouth - red
                (0, 0, 255),    # right mouth - red
            ]
            for i, (lx, ly) in enumerate(landmarks):
                cv2.circle(output, (lx, ly), 3, colors[i], -1)

        # Draw confidence
        if draw_confidence:
            text = f'{confidence:.2f}'
            cv2.putText(
                output,
                text,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return output


def run_inference_on_image(
    model_path: str,
    config_path: str,
    image_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Run inference on a single image.

    Args:
        model_path: Path to model weights
        config_path: Path to config file
        image_path: Path to input image
        output_path: Path to save output (optional)
        show: Whether to display result
    """
    # Initialize detector
    detector = FaceDetectorInference(model_path, config_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    prediction = detector.predict(rgb_image)

    print(f"Detection results for {image_path}:")
    print(f"  BBox: {prediction['bbox']}")
    print(f"  Landmarks: {prediction['landmarks']}")
    print(f"  Confidence: {prediction['confidence']:.4f}")

    # Draw results
    output = detector.draw_detection(image, prediction)

    if output_path:
        cv2.imwrite(output_path, output)
        print(f"Saved to {output_path}")

    if show:
        cv2.imshow('Detection', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_inference_on_video(
    model_path: str,
    config_path: str,
    video_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Run inference on a video.

    Args:
        model_path: Path to model weights
        config_path: Path to config file
        video_path: Path to input video (or 0 for webcam)
        output_path: Path to save output video (optional)
        show: Whether to display result
    """
    # Initialize detector
    detector = FaceDetectorInference(model_path, config_path)

    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"  FPS: {fps}, Size: {width}x{height}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        prediction = detector.predict(rgb_frame)

        # Draw results
        output = detector.draw_detection(frame, prediction)

        if writer:
            writer.write(output)

        if show:
            cv2.imshow('Detection', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")

    if show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames total")


def main():
    parser = argparse.ArgumentParser(description='Face detection inference')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or video (use "0" for webcam)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display results',
    )

    args = parser.parse_args()

    input_path = args.input

    # Check if input is image or video
    if input_path.isdigit() or input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_inference_on_video(
            args.model,
            args.config,
            input_path,
            args.output,
            show=not args.no_show,
        )
    else:
        run_inference_on_image(
            args.model,
            args.config,
            input_path,
            args.output,
            show=not args.no_show,
        )


if __name__ == '__main__':
    main()
