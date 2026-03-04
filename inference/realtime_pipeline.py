"""
Real-time pipeline: Webcam -> MediaPipe Face Detection -> ROI Crop -> rPPG -> Display HR/SpO2.

Uses MediaPipe FaceLandmarker (~2ms) instead of custom FaceDetector (~25ms)
for production-grade real-time performance.
"""

import time

import cv2
import numpy as np

from .face_inference import AsyncCapture
from .mediapipe_face import MediaPipeFaceDetector
from .rppg_inference import rPPGInference


class RealtimePipeline:
    """
    End-to-end real-time pipeline combining MediaPipe face detection and rPPG.

    Threading model:
        - Thread 1 (AsyncCapture): Webcam frame grabbing
        - Main thread: Face detection + ROI crop + rPPG + display

    Performance budget per frame (30 FPS = 33ms):
        - MediaPipe face detection: ~2ms
        - ROI crop + buffer:        ~0.1ms
        - rPPG (every 30 frames):   ~7ms on GPU / ~200ms on CPU
        - Draw overlay:             ~0.5ms
        - Total:                    ~3ms (face-only frames)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.face_detector = MediaPipeFaceDetector(cfg=cfg)
        self.rppg = rPPGInference(cfg)
        self.large_box_coef = cfg.RPPG_DATA.LARGE_BOX_COEF

    def run(self):
        """Main loop. Press 'q' to quit."""
        resolution = (self.cfg.INFERENCE.WEBCAM_W, self.cfg.INFERENCE.WEBCAM_H)
        cap = AsyncCapture(self.cfg.INFERENCE.WEBCAM_ID, resolution=resolution)

        # FPS tracking
        fps_ema = 0.0
        alpha = 0.1

        # Video writer
        writer = None
        if self.cfg.INFERENCE.SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.cfg.INFERENCE.SAVE_VIDEO, fourcc, 30.0, resolution,
            )

        vitals = {'hr_bpm': 0.0, 'spo2': 0.0}
        face_detected = False

        # Target FPS limiter
        target_fps = self.cfg.INFERENCE.TARGET_FPS
        min_frame_time = 1.0 / target_fps if target_fps > 0 else 0

        print("[Pipeline] Starting real-time inference. Press 'q' to quit.")

        try:
            while True:
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. Face detection (MediaPipe ~2ms)
                face_pred = self.face_detector.predict(frame_rgb)

                if face_pred['confidence'] > self.cfg.INFERENCE.FACE_CONFIDENCE_THRESHOLD:
                    face_detected = True

                    # 2. Crop face ROI with padding
                    face_roi = self._crop_face(frame_rgb, face_pred['bbox'])

                    # 3. Feed to rPPG buffer
                    predicted = self.rppg.add_frame(face_roi)
                    if predicted:
                        vitals = self.rppg.get_vitals()
                else:
                    face_detected = False

                # 4. Draw overlay
                self._draw_overlay(frame, face_pred, vitals, face_detected)

                # FPS
                elapsed = time.perf_counter() - t_start
                current_fps = 1.0 / max(elapsed, 1e-6)
                fps_ema = alpha * current_fps + (1 - alpha) * fps_ema

                cv2.putText(frame, f"FPS: {fps_ema:.0f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Buffer fill bar
                fill = self.rppg.buffer_fill_ratio
                bar_w = 100
                cv2.rectangle(frame, (10, 35), (10 + bar_w, 50), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 35), (10 + int(bar_w * fill), 50), (0, 200, 0), -1)
                cv2.putText(frame, f"Buffer: {fill:.0%}", (120, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                if writer:
                    writer.write(frame)

                if self.cfg.INFERENCE.DISPLAY_WINDOW:
                    cv2.imshow('VitalSense', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # FPS limiter
                if min_frame_time > 0:
                    remaining = min_frame_time - (time.perf_counter() - t_start)
                    if remaining > 0:
                        time.sleep(remaining)

        finally:
            cap.release()
            self.face_detector.close()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[Pipeline] Stopped.")

    def _crop_face(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop face ROI with padding (LARGE_BOX_COEF)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Denormalize
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

        # Expand box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = (x2 - x1) * self.large_box_coef
        bh = (y2 - y1) * self.large_box_coef

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w, int(cx + bw / 2))
        y2 = min(h, int(cy + bh / 2))

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image
        return roi

    def _draw_overlay(self, frame: np.ndarray, face_pred: dict,
                      vitals: dict, face_detected: bool):
        """Draw face detection + vitals overlay on BGR frame."""
        h, w = frame.shape[:2]

        if face_detected:
            bbox = face_pred['bbox']
            landmarks = face_pred['landmarks']

            # Draw bbox
            x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
            x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmarks
            colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
            names = ['LE', 'RE', 'N', 'LM', 'RM']
            for i, (lx, ly) in enumerate(landmarks):
                px, py = int(lx * w), int(ly * h)
                cv2.circle(frame, (px, py), 3, colors[i], -1)
        else:
            cv2.putText(frame, "No face detected", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Vitals panel (top-right)
        panel_x = w - 200
        panel_y = 10
        cv2.rectangle(frame, (panel_x - 5, panel_y - 5),
                       (w - 5, panel_y + 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x - 5, panel_y - 5),
                       (w - 5, panel_y + 60), (0, 255, 0), 1)

        hr = vitals.get('hr_bpm', 0.0)
        spo2 = vitals.get('spo2', 0.0)

        cv2.putText(frame, f"HR: {hr:.0f} BPM", (panel_x, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"SpO2: {spo2:.1f}%", (panel_x, panel_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
