"""
Export FCAtt rPPG model to ONNX for browser inference.

Replaces ONNX-unsupported ops:
  - torch.fft.rfft/irfft → precomputed DFT matrix multiplication
  - torch.diff → explicit subtraction
  - AdaptiveAvgPool3d/AdaptiveMaxPool3d → fixed-size equivalents

Usage:
    cd D:\\Projects\\facedector
    python tools/export_rppg_onnx.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.defaults import get_config
from models.rppg_model import FCAtt
from models.rppg_modules import FrequencyAttention, ChannelAttention3D


# ── Helpers: replace adaptive pooling ────────────────────────────────────────

class GlobalAvgPool3d(nn.Module):
    """Replace AdaptiveAvgPool3d(1) with global mean."""
    def forward(self, x):
        return x.mean(dim=[2, 3, 4], keepdim=True)


class GlobalMaxPool3d(nn.Module):
    """Replace AdaptiveMaxPool3d(1) with global max."""
    def forward(self, x):
        # torch.amax is well-supported in ONNX
        return torch.amax(x, dim=[2, 3, 4], keepdim=True)


class FixedAvgPool3d_256_18_to_128_1(nn.Module):
    """Replace AdaptiveAvgPool3d((128, 1, 1)) for input [B, C, 256, 18, 18]."""
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=(2, 18, 18), stride=(2, 18, 18))


class SpatialPoolThenRepeat(nn.Module):
    """Replace AdaptiveAvgPool3d((128, 1, 1)) for input [B, C, 64, 18, 18].

    Temporal 64→128 is nearest-neighbor upsampling (each element repeated).
    """
    def forward(self, x):
        # Spatial pool: [B, C, 64, 18, 18] → [B, C, 64, 1, 1]
        pooled = F.avg_pool3d(x, kernel_size=(1, 18, 18), stride=(1, 18, 18))
        # Temporal repeat: 64 → 128 via repeat_interleave
        return pooled.repeat_interleave(2, dim=2)


class FixedSpatialAvgPool3d(nn.Module):
    """Replace AdaptiveAvgPool3d((None, 18, 18)) for input [B, C, T, 36, 36]."""
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))


# ── ONNX-safe ChannelAttention3D ────────────────────────────────────────────

class ChannelAttention3DONNX(nn.Module):
    """ChannelAttention3D with global mean/max instead of adaptive pooling."""

    def __init__(self, orig: ChannelAttention3D):
        super().__init__()
        self.fc = orig.fc
        self.sigmoid = orig.sigmoid

    def forward(self, x):
        avg_out = self.fc(x.mean(dim=[2, 3, 4], keepdim=True))
        max_out = self.fc(torch.amax(x, dim=[2, 3, 4], keepdim=True))
        return x * self.sigmoid(avg_out + max_out)


# ── ONNX-safe FrequencyAttention (DFT matrix) ──────────────────────────────

class FrequencyAttentionONNX(nn.Module):
    """Drop-in replacement using precomputed DFT matrices."""

    def __init__(self, orig: FrequencyAttention):
        super().__init__()
        self.score_refiner = orig.score_refiner
        self.spatial_score_refiner = orig.spatial_score_refiner
        self.pool_size_h = orig.pool_size_h
        self.pool_size_w = orig.pool_size_w

        # Replace adaptive pool with fixed avg pool (36→18 spatial)
        self.pool = FixedSpatialAvgPool3d()

        T = orig.frames
        fps = orig.fps
        K = T // 2 + 1

        # DFT basis (rfft, ortho normalization)
        n = torch.arange(T, dtype=torch.float32)
        k = torch.arange(K, dtype=torch.float32)
        angles = -2.0 * math.pi * n.unsqueeze(1) * k.unsqueeze(0) / T
        scale = 1.0 / math.sqrt(T)
        self.register_buffer("dft_cos", torch.cos(angles) * scale)
        self.register_buffer("dft_sin", torch.sin(angles) * scale)

        # iDFT basis (irfft, ortho normalization with half-spectrum scaling)
        inv_angles = 2.0 * math.pi * n.unsqueeze(0) * k.unsqueeze(1) / T
        bin_scale = torch.ones(K) * 2.0
        bin_scale[0] = 1.0
        if T % 2 == 0:
            bin_scale[-1] = 1.0
        self.register_buffer("idft_cos", torch.cos(inv_angles) * bin_scale.unsqueeze(1) / math.sqrt(T))
        self.register_buffer("idft_sin", torch.sin(inv_angles) * bin_scale.unsqueeze(1) / math.sqrt(T))

        # Frequency mask
        freqs = torch.arange(K, dtype=torch.float32) * fps / T
        self.register_buffer("freq_mask", ((freqs >= orig.freq_low) & (freqs <= orig.freq_high)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        x_pooled = self.pool(x)
        x_flat = x_pooled.view(B, C, T, -1)

        # DFT
        x_t = x_flat.permute(0, 1, 3, 2)
        fft_real = torch.matmul(x_t, self.dft_cos)
        fft_imag = torch.matmul(x_t, self.dft_sin)

        # Frequency mask
        mask = self.freq_mask.view(1, 1, 1, -1)
        fft_real_f = fft_real * mask
        fft_imag_f = fft_imag * mask

        # iDFT
        filtered = (torch.matmul(fft_real_f, self.idft_cos)
                    - torch.matmul(fft_imag_f, self.idft_sin))
        filtered = filtered.permute(0, 1, 3, 2)
        filtered_5d = filtered.view(B, C, T, self.pool_size_h, self.pool_size_w)

        # Temporal refiner
        temporal_scores = self.score_refiner(filtered_5d)
        pattern_map_scores = torch.mean(temporal_scores, dim=2)
        pattern_map_probs = torch.sigmoid(pattern_map_scores)

        # Power map → spatial refiner
        power = fft_real_f ** 2 + fft_imag_f ** 2
        power_map = torch.mean(power, dim=-1)
        spatial_att = power_map.view(B, C, self.pool_size_h, self.pool_size_w)
        spatial_score = self.spatial_score_refiner(spatial_att)
        energy_map_probs = torch.sigmoid(spatial_score)

        # Fused attention
        att_fused = torch.sqrt(pattern_map_probs * energy_map_probs + 1e-20)
        att_up = F.interpolate(att_fused, size=(H, W), mode="bilinear", align_corners=False)
        return x * att_up.unsqueeze(2)


# ── ONNX-safe FCAtt ─────────────────────────────────────────────────────────

class FCAttONNX(FCAtt):
    """Replaces torch.diff and all adaptive pooling with ONNX-compatible ops."""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        batch, channel, length, width, height = x1.shape

        if x2 is not None:
            rppg_wave1, spo2_1 = self.encode_video(x1)
            rppg_wave2, _ = self.encode_video(x2)
            rppg_wave = self.fusion_net(rppg_wave1, rppg_wave2)
        else:
            rppg_wave, spo2_features = self.encode_video(x1)

        rppg_wave = rppg_wave.view(batch, length)

        # Derivatives (replace torch.diff)
        rppg_d1 = rppg_wave.unsqueeze(1)
        vpg_temp = rppg_d1[:, :, 1:] - rppg_d1[:, :, :-1]
        vpg_signal = F.pad(vpg_temp, (0, 1), mode="replicate")
        apg_temp = vpg_signal[:, :, 1:] - vpg_signal[:, :, :-1]
        apg_signal = F.pad(apg_temp, (0, 1), mode="replicate")

        # SpO2
        spo2_branch_for_fusion = spo2_features.view(
            batch, self.num_spo2_branch_output_channels, length,
        )
        fused_features_1d = torch.cat(
            (rppg_d1, vpg_signal, apg_signal, spo2_branch_for_fusion), dim=1,
        )
        spo2_pred = self.spo2_fusion_head(fused_features_1d)
        spo2_pred = spo2_pred * self.spo2_range + self.spo2_min

        return rppg_wave, spo2_pred


# ── Build exportable model ──────────────────────────────────────────────────

def replace_adaptive_pools(model: FCAttONNX):
    """Replace all AdaptiveAvg/MaxPool3d in the model with fixed-size ops."""
    # 1. FrequencyAttention → already replaced during construction
    # (att_mask1 will be swapped after this function)

    # 2. poolspa: input [B, 64, 256, 18, 18] → [B, 64, 128, 1, 1]
    model.poolspa = FixedAvgPool3d_256_18_to_128_1()

    # 3. spo2_branch_pool: input [B, 16, 64, 18, 18] → [B, 16, 128, 1, 1]
    model.spo2_branch_pool = SpatialPoolThenRepeat()

    # 4. ChannelAttention3D instances → replace adaptive pools
    model.channel_attention = ChannelAttention3DONNX(model.channel_attention)
    model.channel_attention1 = ChannelAttention3DONNX(model.channel_attention1)
    model.channel_attention2 = ChannelAttention3DONNX(model.channel_attention2)
    model.channel_attention3 = ChannelAttention3DONNX(model.channel_attention3)


def build_exportable_model(cfg, weights_path: str) -> FCAttONNX:
    """Load FCAtt, replace all non-exportable ops, return ONNX-ready model."""
    orig = FCAtt(cfg=cfg)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    orig.load_state_dict(state)
    orig.eval()

    model = FCAttONNX(cfg=cfg)
    model.load_state_dict(state)
    model.eval()

    # Replace adaptive pools
    replace_adaptive_pools(model)

    # Replace FrequencyAttention
    model.att_mask1 = FrequencyAttentionONNX(orig.att_mask1)

    return model


def validate(model: nn.Module, onnx_path: str, dummy: torch.Tensor):
    """Compare PyTorch vs ONNX Runtime outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed, skipping validation.")
        return True

    with torch.no_grad():
        pt_wave, pt_spo2 = model(dummy)
    pt_wave = pt_wave.numpy()
    pt_spo2 = pt_spo2.numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"video": dummy.numpy()})
    ort_wave, ort_spo2 = ort_out

    wave_err = np.max(np.abs(pt_wave - ort_wave))
    spo2_err = np.max(np.abs(pt_spo2 - ort_spo2))
    print(f"[Validate] rppg_wave max error: {wave_err:.6f}")
    print(f"[Validate] spo2 max error:      {spo2_err:.6f}")

    ok = wave_err < 0.05 and spo2_err < 1.0
    print(f"[Validate] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "realtime.yaml")
    cfg = get_config(cfg_path)

    weights_path = cfg.RPPG_MODEL.WEIGHTS
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), "..", weights_path)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "web", "static", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rppg_fcatt.onnx")

    print(f"[Export] Weights: {weights_path}")
    print(f"[Export] Output:  {output_path}")

    model = build_exportable_model(cfg, weights_path)

    frames = cfg.RPPG_MODEL.FRAMES
    size = cfg.RPPG_MODEL.INPUT_SIZE
    dummy = torch.randn(1, 3, frames, size, size)

    with torch.no_grad():
        wave, spo2 = model(dummy)
    print(f"[Export] Forward OK: rppg_wave {list(wave.shape)}, spo2 {list(spo2.shape)}")

    # Export with UTF-8 encoding for Windows compatibility
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["video"],
        output_names=["rppg_wave", "spo2"],
        opset_version=18,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Done: {output_path} ({file_size:.1f} MB)")

    validate(model, output_path, dummy)


if __name__ == "__main__":
    main()
