"""
MediaPipe-based face detection for real-time rPPG pipeline.
Drop-in replacement for FaceDetectorInference — same predict() interface.

Uses MediaPipe FaceLandmarker in VIDEO mode with temporal smoothing.
When detection fails, holds the last valid result for up to GRACE_FRAMES frames.

Performance: ~2-8ms per frame on CPU.
"""

import os
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


# MediaPipe 478-point landmark indices
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_NOSE_TIP = 1
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

# Iris indices (available in 478-point model)
_LEFT_IRIS = 468
_RIGHT_IRIS = 473

# Grace period: hold last detection for N frames when detection fails
_GRACE_FRAMES = 10


class MediaPipeFaceDetector:
    """
    MediaPipe FaceLandmarker wrapper with temporal smoothing.

    Returns same format as FaceDetectorInference:
        predict() -> dict with 'bbox' [4], 'landmarks' [5,2], 'confidence' float
    """

    def __init__(self, cfg=None, model_path: str = None,
                 min_detection_confidence: float = 0.4,
                 video_mode: bool = True):
        """
        Args:
            cfg: YACS config (overrides min_detection_confidence)
            model_path: path to face_landmarker.task
            min_detection_confidence: detection threshold
            video_mode: True=VIDEO mode (webcam, temporal smoothing),
                        False=IMAGE mode (single images)
        """
        if cfg is not None:
            min_detection_confidence = cfg.INFERENCE.FACE_CONFIDENCE_THRESHOLD

        # Resolve model path
        if model_path is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task'),
                'models/face_landmarker.task',
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found. Download face_landmarker.task to models/\n"
                f"URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                f"face_landmarker/float16/latest/face_landmarker.task"
            )

        self._video_mode = video_mode
        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

        # Temporal smoothing state (only used in VIDEO mode)
        self._last_result = self._empty_result()
        self._miss_count = 0
        self._smooth_alpha = 0.6  # EMA blending: 60% new, 40% old

        mode_name = f"VIDEO + grace={_GRACE_FRAMES}" if video_mode else "IMAGE"
        print(f"[MediaPipe] FaceLandmarker loaded ({mode_name})")

    def predict(self, image_rgb: np.ndarray) -> dict:
        """
        Run face detection on RGB image.
        VIDEO mode: temporal smoothing + grace period.
        IMAGE mode: independent per-frame detection.

        Args:
            image_rgb: [H, W, 3] uint8 RGB

        Returns:
            dict with:
                'bbox': np.array [4] (x_min, y_min, x_max, y_max) normalized [0,1]
                'landmarks': np.array [5, 2] (x, y) normalized [0,1]
                'confidence': float
        """
        if not image_rgb.flags['C_CONTIGUOUS']:
            image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        try:
            if self._video_mode:
                self._timestamp_ms += 33
                result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            else:
                result = self.landmarker.detect(mp_image)
        except Exception:
            return self._handle_miss() if self._video_mode else self._empty_result()

        if not result.face_landmarks:
            return self._handle_miss() if self._video_mode else self._empty_result()

        face_lm = result.face_landmarks[0]
        num_lm = len(face_lm)

        landmarks = self._extract_5_landmarks(face_lm, num_lm)
        bbox = self._landmarks_to_bbox(face_lm)

        new_result = {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': 0.95,
        }

        # EMA smoothing only in VIDEO mode
        if self._video_mode and self._last_result['confidence'] > 0:
            a = self._smooth_alpha
            new_result['bbox'] = a * bbox + (1 - a) * self._last_result['bbox']
            new_result['landmarks'] = a * landmarks + (1 - a) * self._last_result['landmarks']

        self._last_result = new_result
        self._miss_count = 0
        return new_result

    def _handle_miss(self) -> dict:
        """Handle detection failure with grace period."""
        self._miss_count += 1
        if self._miss_count <= _GRACE_FRAMES and self._last_result['confidence'] > 0:
            # Gradually decay confidence during grace period
            decay = 1.0 - (self._miss_count / (_GRACE_FRAMES + 1))
            return {
                'bbox': self._last_result['bbox'].copy(),
                'landmarks': self._last_result['landmarks'].copy(),
                'confidence': self._last_result['confidence'] * decay,
            }
        # Grace period expired
        self._last_result = self._empty_result()
        return self._empty_result()

    def _extract_5_landmarks(self, face_lm, num_lm: int) -> np.ndarray:
        """
        Extract 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth.
        Uses iris center if available, otherwise averages eye corners.
        """
        # Left eye
        if num_lm > _LEFT_IRIS:
            le = face_lm[_LEFT_IRIS]
            le_x, le_y = le.x, le.y
        else:
            li = face_lm[_LEFT_EYE_INNER]
            lo = face_lm[_LEFT_EYE_OUTER]
            le_x = (li.x + lo.x) / 2
            le_y = (li.y + lo.y) / 2

        # Right eye
        if num_lm > _RIGHT_IRIS:
            re = face_lm[_RIGHT_IRIS]
            re_x, re_y = re.x, re.y
        else:
            ri = face_lm[_RIGHT_EYE_INNER]
            ro = face_lm[_RIGHT_EYE_OUTER]
            re_x = (ri.x + ro.x) / 2
            re_y = (ri.y + ro.y) / 2

        nose = face_lm[_NOSE_TIP]
        lm = face_lm[_LEFT_MOUTH]
        rm = face_lm[_RIGHT_MOUTH]

        return np.array([
            [le_x, le_y],
            [re_x, re_y],
            [nose.x, nose.y],
            [lm.x, lm.y],
            [rm.x, rm.y],
        ], dtype=np.float32)

    @staticmethod
    def _landmarks_to_bbox(face_lm, padding: float = 0.2) -> np.ndarray:
        """Derive bounding box from all face mesh landmarks with padding."""
        xs = [lm.x for lm in face_lm]
        ys = [lm.y for lm in face_lm]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min

        x_min = max(0.0, x_min - w * padding)
        y_min = max(0.0, y_min - h * padding)
        x_max = min(1.0, x_max + w * padding)
        y_max = min(1.0, y_max + h * padding)

        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    @staticmethod
    def _empty_result() -> dict:
        return {
            'bbox': np.zeros(4, dtype=np.float32),
            'landmarks': np.zeros((5, 2), dtype=np.float32),
            'confidence': 0.0,
        }

    @staticmethod
    def draw_detection(image: np.ndarray, prediction: dict) -> np.ndarray:
        """Draw bbox + landmarks on BGR image."""
        h, w = image.shape[:2]
        bbox = prediction['bbox']
        landmarks = prediction['landmarks']
        confidence = prediction['confidence']

        if confidence < 0.3:
            return image

        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
        names = ['LE', 'RE', 'N', 'LM', 'RM']
        for i, (lx, ly) in enumerate(landmarks):
            px, py = int(lx * w), int(ly * h)
            cv2.circle(image, (px, py), 4, colors[i], -1)
            cv2.putText(image, names[i], (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1)

        cv2.putText(image, f"conf: {confidence:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
