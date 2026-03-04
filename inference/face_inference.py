"""Face detection inference with async capture and real-time visualization."""

import time
import threading
from collections import deque

import cv2
import numpy as np
import torch


class AsyncCapture:
    """Threaded video capture for non-blocking frame reading."""

    def __init__(self, source, resolution=None):
        self.cap = cv2.VideoCapture(source)
        if resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)


class FaceDetectorInference:
    """GPU-optimized face detection inference."""

    def __init__(self, cfg):
        from models.face_detector import FaceDetector

        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        self.input_size = cfg.FACE_MODEL.INPUT_SIZE
        self.threshold = cfg.INFERENCE.FACE_CONFIDENCE_THRESHOLD

        # Load model
        self.model = FaceDetector(cfg=cfg).to(self.device)
        if cfg.FACE_MODEL.WEIGHTS:
            checkpoint = torch.load(cfg.FACE_MODEL.WEIGHTS, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Face] Loaded weights: {cfg.FACE_MODEL.WEIGHTS}")
        self.model.eval()

        # Warmup
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        for _ in range(3):
            with torch.no_grad():
                self.model(dummy)
        print(f"[Face] Warmup complete on {self.device}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image to model input tensor with ImageNet normalization."""
        resized = cv2.resize(image, (self.input_size, self.input_size))
        tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> dict:
        """
        Run face detection on RGB image.

        Returns:
            dict with 'bbox' [4], 'landmarks' [5,2], 'confidence' float
        """
        tensor = self.preprocess(image_rgb)
        output = self.model(tensor)

        bbox = output['bbox'][0].cpu().numpy()
        landmarks = output['landmarks'][0].cpu().numpy()
        confidence = output['confidence'][0, 0].cpu().item()

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }

    @staticmethod
    def draw_detection(image: np.ndarray, prediction: dict,
                       original_size: tuple = None) -> np.ndarray:
        """Draw bbox, landmarks, confidence on BGR image."""
        h, w = image.shape[:2]
        bbox = prediction['bbox']
        landmarks = prediction['landmarks']
        confidence = prediction['confidence']

        # Denormalize bbox
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks
        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
        for i, (lx, ly) in enumerate(landmarks):
            px, py = int(lx * w), int(ly * h)
            cv2.circle(image, (px, py), 3, colors[i], -1)

        # Draw confidence
        cv2.putText(image, f"{confidence:.3f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image
