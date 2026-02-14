"""
Inference script for face detection model.
Optimized for real-time webcam/video inference with GPU acceleration.
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import yaml

from models.face_detector import create_face_detector


# ============================================
#  GPU Configuration (must be before model creation)
# ============================================
def setup_gpu(memory_limit_mb: Optional[int] = None):
    """Configure GPU memory growth to avoid OOM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_limit_mb:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=memory_limit_mb,
                        )],
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print("WARNING: No GPU detected. Running on CPU.")


# ============================================
#  Async Video Capture (threaded)
# ============================================
class AsyncCapture:
    """Threaded video capture for non-blocking frame reading."""

    def __init__(self, source, resolution: Optional[Tuple[int, int]] = None):
        if isinstance(source, int) or source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {source}")

        # Set camera resolution
        if resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Minimize capture buffer to get latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False

        # Read first frame synchronously
        self._ret, self._frame = self.cap.read()

        # Start background thread
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        """Continuously grab frames in background thread."""
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame (non-blocking)."""
        with self._lock:
            return self._ret, self._frame

    @property
    def fps(self) -> int:
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
        self.cap.release()


# ============================================
#  Face Detector Inference (Optimized)
# ============================================
class FaceDetectorInference:
    """Inference wrapper for face detection model with GPU optimization."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        confidence_threshold: float = 0.5,
        use_fp16: bool = False,
    ):
        """
        Initialize inference.

        Args:
            model_path: Path to model weights (.weights.h5)
            config_path: Path to config file
            confidence_threshold: Minimum confidence to draw detection
            use_fp16: Enable mixed precision FP16 for faster GPU inference
        """
        # Mixed precision (before model creation)
        if use_fp16 and tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision FP16 enabled")

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.input_size = self.config.get('model', {}).get('input_size', 224)
        self.confidence_threshold = confidence_threshold

        # Create model
        self.model = create_face_detector(self.config, pretrained=False)

        # Build model with dummy forward pass
        dummy_input = tf.zeros((1, self.input_size, self.input_size, 3))
        self.model(dummy_input, training=False)
        self.model.load_weights(model_path)
        print(f"Model loaded from {model_path}")

        # Create compiled predict function with fixed input signature
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1, self.input_size, self.input_size, 3), dtype=tf.float32),
        ])
        def predict_fn(inputs):
            return self.model(inputs, training=False)

        self._predict_fn = predict_fn

        # Warmup: run compiled inference to trigger tracing
        print("Warming up model...")
        for _ in range(3):
            self._predict_fn(dummy_input)
        print("Model ready.")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image in RGB format.

        Returns:
            preprocessed: Preprocessed image batch [1, H, W, 3]
            original_size: Original image size (H, W)
        """
        original_size = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size))
        normalized = resized.astype(np.float32) * (1.0 / 255.0)
        batch = normalized[np.newaxis]
        return batch, original_size

    def predict(
        self,
        image: np.ndarray,
        return_original_coords: bool = True,
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image (RGB)
            return_original_coords: Scale coordinates to original image size

        Returns:
            Dictionary with bbox, landmarks, confidence
        """
        batch, original_size = self.preprocess(image)
        batch_tensor = tf.constant(batch)

        # Use compiled @tf.function for inference
        predictions = self._predict_fn(batch_tensor)

        # Extract results (single .numpy() call each)
        bbox = predictions['bbox'][0].numpy()
        landmarks = predictions['landmarks'][0].numpy()
        confidence = float(predictions['confidence'][0, 0].numpy())

        if return_original_coords:
            h, w = original_size
            scale = np.array([w, h, w, h], dtype=np.float32)
            bbox = bbox * scale
            landmarks = landmarks * np.array([[w, h]], dtype=np.float32)

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
        """Run inference on a batch of images."""
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
        inference_ms: Optional[float] = None,
        fps_display: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw detection results on image (modifies in place for speed).

        Args:
            image: Input image (BGR, modified in place)
            prediction: Prediction dictionary
            draw_landmarks: Whether to draw landmarks
            draw_confidence: Whether to draw confidence score
            inference_ms: Inference time in milliseconds
            fps_display: Smoothed FPS value to display

        Returns:
            Same image with drawn detections
        """
        confidence = prediction['confidence']

        if confidence >= self.confidence_threshold:
            bbox = prediction['bbox'].astype(np.int32)
            landmarks = prediction['landmarks'].astype(np.int32)

            # Bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 2)

            # Landmarks
            if draw_landmarks:
                colors = [
                    (255, 0, 0),    # left eye
                    (255, 0, 0),    # right eye
                    (0, 255, 0),    # nose
                    (0, 0, 255),    # left mouth
                    (0, 0, 255),    # right mouth
                ]
                for i, (lx, ly) in enumerate(landmarks):
                    cv2.circle(image, (lx, ly), 3, colors[i], -1)

            # Confidence text
            if draw_confidence:
                cv2.putText(image, f'{confidence:.2f}',
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Performance overlay
        if inference_ms is not None:
            fps = fps_display or (1000.0 / inference_ms if inference_ms > 0 else 0)
            cv2.putText(image, f'{inference_ms:.1f}ms | {fps:.0f} FPS',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        # GPU indicator
        if tf.config.list_physical_devices('GPU'):
            cv2.putText(image, 'GPU', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        return image


# ============================================
#  Image Inference
# ============================================
def run_inference_on_image(
    model_path: str,
    config_path: str,
    image_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
    confidence_threshold: float = 0.5,
    use_fp16: bool = False,
):
    """Run inference on a single image."""
    detector = FaceDetectorInference(
        model_path, config_path,
        confidence_threshold=confidence_threshold,
        use_fp16=use_fp16,
    )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t_start = time.perf_counter()
    prediction = detector.predict(rgb_image)
    inference_ms = (time.perf_counter() - t_start) * 1000

    print(f"Detection results for {image_path}:")
    print(f"  BBox: {prediction['bbox']}")
    print(f"  Landmarks: {prediction['landmarks']}")
    print(f"  Confidence: {prediction['confidence']:.4f}")
    print(f"  Inference time: {inference_ms:.1f}ms")

    output = detector.draw_detection(
        image.copy(), prediction, inference_ms=inference_ms,
    )

    if output_path:
        cv2.imwrite(output_path, output)
        print(f"Saved to {output_path}")

    if show:
        try:
            cv2.imshow('Detection', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("GUI not available, skipping display")


# ============================================
#  Video / Webcam Inference (Optimized)
# ============================================
def run_inference_on_video(
    model_path: str,
    config_path: str,
    video_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
    confidence_threshold: float = 0.5,
    use_fp16: bool = False,
    resolution: Optional[Tuple[int, int]] = None,
    benchmark: bool = False,
):
    """
    Run optimized inference on video or webcam.

    Optimizations:
    - Async frame capture (separate thread)
    - @tf.function compiled model
    - In-place drawing (no image.copy())
    - EMA-smoothed FPS display
    - Frame skip when behind
    """
    is_webcam = video_path.isdigit()

    detector = FaceDetectorInference(
        model_path, config_path,
        confidence_threshold=confidence_threshold,
        use_fp16=use_fp16,
    )

    # Open video with async capture
    if is_webcam:
        cap = AsyncCapture(video_path, resolution=resolution)
    else:
        cap = AsyncCapture(video_path)

    fps_source = cap.fps
    width, height = cap.width, cap.height

    # Video writer
    writer = None
    if output_path:
        ext = Path(output_path).suffix.lower()
        codec_map = {'.avi': 'XVID', '.mp4': 'mp4v', '.mov': 'mp4v', '.mkv': 'XVID'}
        codec = codec_map.get(ext, 'XVID')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps_source, (width, height))

    total_frames = cap.frame_count if not is_webcam else 0
    mode_str = "webcam" if is_webcam else video_path
    print(f"Processing: {mode_str}")
    print(f"  Source FPS: {fps_source}, Size: {width}x{height}")
    print(f"  Confidence threshold: {confidence_threshold}")
    if total_frames > 0:
        print(f"  Total frames: {total_frames}")
    print("  Press 'q' to quit")

    # GUI check
    gui_available = show
    if show:
        try:
            cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
        except cv2.error:
            gui_available = False
            print("  GUI not available, running headless")

    # FPS tracking with EMA (Exponential Moving Average)
    ema_fps = 0.0
    ema_alpha = 0.1  # Smoothing factor
    frame_count = 0

    # Benchmark mode
    bench_times = [] if benchmark else None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if is_webcam:
                continue  # Retry on webcam frame drop
            break

        # Convert BGR → RGB for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference with timing
        t_start = time.perf_counter()
        prediction = detector.predict(rgb_frame)
        inference_ms = (time.perf_counter() - t_start) * 1000

        # EMA FPS
        current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0
        ema_fps = ema_alpha * current_fps + (1 - ema_alpha) * ema_fps if ema_fps > 0 else current_fps

        # Benchmark tracking
        if bench_times is not None:
            bench_times.append(inference_ms)

        # Draw results (in-place on frame)
        detector.draw_detection(
            frame, prediction,
            inference_ms=inference_ms,
            fps_display=ema_fps,
        )

        if writer:
            writer.write(frame)

        if gui_available:
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if not is_webcam and total_frames > 0 and frame_count % 100 == 0:
            pct = frame_count / total_frames * 100
            print(f"  {frame_count}/{total_frames} ({pct:.1f}%)")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    if gui_available:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")

    # Benchmark report
    if bench_times and len(bench_times) > 10:
        times = bench_times[5:]  # Skip first 5 warmup frames
        avg_ms = np.mean(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        print(f"\n=== Benchmark Results ===")
        print(f"  Frames:    {len(times)}")
        print(f"  Avg:       {avg_ms:.2f} ms ({1000/avg_ms:.1f} FPS)")
        print(f"  P50:       {p50:.2f} ms")
        print(f"  P95:       {p95:.2f} ms")
        print(f"  P99:       {p99:.2f} ms")
        gpu_name = "CPU"
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_name = gpus[0].name
        print(f"  Device:    {gpu_name}")


# ============================================
#  CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Face detection inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image/video (use "0" for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable mixed precision FP16 for faster GPU inference')
    parser.add_argument('--resolution', type=str, default=None,
                        help='Camera resolution WxH (e.g., 640x480, 1280x720)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark and print inference statistics')

    args = parser.parse_args()

    # Setup GPU
    setup_gpu()

    # Parse resolution
    resolution = None
    if args.resolution:
        parts = args.resolution.lower().split('x')
        if len(parts) == 2:
            resolution = (int(parts[0]), int(parts[1]))

    input_path = args.input
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

    if input_path.isdigit() or input_path.lower().endswith(video_exts):
        run_inference_on_video(
            args.model, args.config, input_path, args.output,
            show=not args.no_show,
            confidence_threshold=args.threshold,
            use_fp16=args.fp16,
            resolution=resolution,
            benchmark=args.benchmark,
        )
    else:
        run_inference_on_image(
            args.model, args.config, input_path, args.output,
            show=not args.no_show,
            confidence_threshold=args.threshold,
            use_fp16=args.fp16,
        )


if __name__ == '__main__':
    main()
