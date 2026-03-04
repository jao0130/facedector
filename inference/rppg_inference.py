"""rPPG inference with frame buffer for real-time heart rate estimation."""

import numpy as np
import cv2
import torch
from collections import deque

from utils.signal_processing import estimate_hr_fft


class rPPGInference:
    """
    rPPG inference with circular frame buffer.
    Accumulates face-cropped frames and runs FCAtt when buffer is full.
    """

    def __init__(self, cfg):
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        self.buffer_size = cfg.INFERENCE.RPPG_BUFFER_SIZE
        self.predict_interval = cfg.INFERENCE.RPPG_PREDICT_INTERVAL
        self.input_size = cfg.RPPG_MODEL.INPUT_SIZE
        self.fps = cfg.RPPG_MODEL.FPS

        # Frame buffer
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.frame_count = 0

        # Load model (select by config NAME)
        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.model = FCAtt_v3(cfg=cfg).to(self.device)
        elif model_name == 'FCAtt_v2':
            from models.rppg_model_v2 import FCAtt_v2
            self.model = FCAtt_v2(cfg=cfg).to(self.device)
        else:
            from models.rppg_model import FCAtt
            self.model = FCAtt(cfg=cfg).to(self.device)
        if cfg.RPPG_MODEL.WEIGHTS:
            checkpoint = torch.load(cfg.RPPG_MODEL.WEIGHTS, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[rPPG] Loaded weights: {cfg.RPPG_MODEL.WEIGHTS}")
        self.model.eval()

        # Warmup
        dummy = torch.randn(1, 3, self.buffer_size, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        print(f"[rPPG] Warmup complete on {self.device}")

        # Latest results
        self.latest_hr = 0.0
        self.latest_spo2 = 0.0
        self.latest_wave = None

    def add_frame(self, face_roi: np.ndarray) -> bool:
        """
        Add face ROI frame to buffer.

        Args:
            face_roi: [H, W, 3] RGB face crop

        Returns:
            True if a prediction was made this frame.
        """
        # Resize to model input size
        resized = cv2.resize(face_roi, (self.input_size, self.input_size))
        self.frame_buffer.append(resized)
        self.frame_count += 1

        # Predict every N frames when buffer is full
        if (self.frame_count % self.predict_interval == 0 and
                len(self.frame_buffer) >= self.buffer_size):
            self._predict()
            return True
        return False

    @torch.no_grad()
    def _predict(self):
        """Run FCAtt on buffered frames."""
        frames = np.stack(list(self.frame_buffer))  # [T, H, W, C]

        # To tensor: [1, C, T, H, W] — raw pixels, DiffNormalize 內建於模型
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
        tensor = tensor.to(self.device)

        rppg_wave, spo2 = self.model(tensor)

        wave = rppg_wave[0].cpu().numpy()
        self.latest_hr = estimate_hr_fft(wave, self.fps)
        self.latest_spo2 = spo2[0].cpu().item()
        self.latest_wave = wave

    def get_vitals(self) -> dict:
        """Get latest vital signs."""
        return {
            'hr_bpm': self.latest_hr,
            'spo2': self.latest_spo2,
            'rppg_wave': self.latest_wave,
        }

    @property
    def buffer_fill_ratio(self) -> float:
        """How full is the buffer (0.0 to 1.0)."""
        return len(self.frame_buffer) / self.buffer_size
