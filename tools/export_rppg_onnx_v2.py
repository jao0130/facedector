"""
Export FCAtt_v2 rPPG model to ONNX for browser inference.

V2 vs V1 主要差異：
  - DiffNormalizeLayer 內建於模型（輸入為原始像素 0-255，不需 JS 前處理）
  - FrequencyAttention v2（可學習 soft mask、逐幀時序注意力、功率比）
  - 無 FusionNet / 雙路輸入

ONNX 不支援的 ops 替換：
  - torch.fft.rfft/irfft  → 預計算 DFT 矩陣的 matmul
  - DiffNormalizeLayer.std → 手動計算（避免 multi-dim std 相容性問題）
  - AdaptiveAvg/MaxPool3d  → 固定尺寸 pool（已知輸入形狀）
  - AdaptiveAvgPool1d(1)   → global mean
  - torch.diff             → 顯式相減

Usage:
    cd D:\\Projects\\facedector
    python tools/export_rppg_onnx_v2.py
    python tools/export_rppg_onnx_v2.py --weights PATH --output PATH
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rppg_model_v2 import FCAtt_v2
from models.rppg_modules_v2 import FrequencyAttention, ChannelAttention3D


# ── ONNX-safe DiffNormalizeLayer ─────────────────────────────────────────────

class DiffNormalizeLayerONNX(nn.Module):
    """
    DiffNormalizeLayer with multi-dim std replaced by manual computation.
    torch.std(dim=(2,3,4)) 在部分 ONNX runtime 有相容問題，改用顯式 sqrt(var)。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, :, 1:] - x[:, :, :-1]
        norm = diff / (x[:, :, 1:].abs() + x[:, :, :-1].abs() + 1e-7)
        # Manual global std (biased); difference from unbiased is negligible (N~660K)
        mean = norm.mean(dim=[2, 3, 4], keepdim=True)
        var  = ((norm - mean) ** 2).mean(dim=[2, 3, 4], keepdim=True)
        std  = (var + 1e-7).sqrt()
        norm = norm / (std + 1e-7)
        norm = torch.clamp(norm, -5.0, 5.0)
        pad  = torch.zeros_like(x[:, :, :1])
        return torch.cat([pad, norm], dim=2)


# ── ONNX-safe ChannelAttention3D ─────────────────────────────────────────────

class ChannelAttention3DONNX(nn.Module):
    """ChannelAttention3D：AdaptiveAvg/MaxPool3d(1) → global mean/amax。"""

    def __init__(self, orig: ChannelAttention3D):
        super().__init__()
        self.fc      = orig.fc
        self.sigmoid = orig.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(x.mean(dim=[2, 3, 4], keepdim=True))
        max_out = self.fc(torch.amax(x, dim=[2, 3, 4], keepdim=True))
        return x * self.sigmoid(avg_out + max_out)


# ── ONNX-safe FrequencyAttention v2 ──────────────────────────────────────────

class FrequencyAttentionV2ONNX(nn.Module):
    """
    FrequencyAttention v2 ONNX 版本。

    替換：
      - torch.fft.rfft/irfft    → 預計算 DFT/iDFT 矩陣 matmul
      - torch.fft.rfrtfreq      → 預計算 freqs buffer
      - AdaptiveAvgPool3d((None,ps,ps)) → F.avg_pool3d（固定 kernel/stride）
      - view(B*T, ...) 假設 B=1（瀏覽器推論固定 batch=1）

    保留 v2 特性：
      - 可學習 soft mask（freq_center / freq_width，ONNX 常數折疊後固化）
      - 逐幀時序注意力（保留 T 維度）
      - 功率比（cardiac / total power）
      - 殘差注意力 fusion：1 + tanh(pattern + energy - 1)
    """

    def __init__(self, orig: FrequencyAttention, input_h: int = 36, input_w: int = 36):
        super().__init__()
        self.score_refiner         = orig.score_refiner
        self.spatial_score_refiner = orig.spatial_score_refiner
        self.freq_center           = orig.freq_center   # nn.Parameter
        self.freq_width            = orig.freq_width    # nn.Parameter
        self.sharpness             = orig.sharpness
        self.pool_size_h           = orig.pool_size_h
        self.pool_size_w           = orig.pool_size_w
        self.input_h               = input_h
        self.input_w               = input_w

        T   = orig.frames
        fps = orig.fps
        K   = T // 2 + 1

        # 固定空間 pool：AdaptiveAvgPool3d((None, ps, ps))
        # PyTorch adaptive 公式：stride = floor(H/out), kernel = H - (out-1)*stride
        ph = orig.pool_size_h
        pw = orig.pool_size_w
        self.pool_stride_h = input_h // ph
        self.pool_stride_w = input_w // pw
        self.pool_kernel_h = input_h - (ph - 1) * self.pool_stride_h
        self.pool_kernel_w = input_w - (pw - 1) * self.pool_stride_w

        # DFT 矩陣（rfft，ortho 正規化）
        n = torch.arange(T, dtype=torch.float32)
        k = torch.arange(K, dtype=torch.float32)
        angles = -2.0 * math.pi * n.unsqueeze(1) * k.unsqueeze(0) / T
        scale  = 1.0 / math.sqrt(T)
        self.register_buffer("dft_cos", torch.cos(angles) * scale)   # [T, K]
        self.register_buffer("dft_sin", torch.sin(angles) * scale)   # [T, K]

        # iDFT 矩陣
        inv_angles = 2.0 * math.pi * n.unsqueeze(0) * k.unsqueeze(1) / T
        bin_scale  = torch.ones(K, dtype=torch.float32) * 2.0
        bin_scale[0] = 1.0
        if T % 2 == 0:
            bin_scale[-1] = 1.0
        scale_i = bin_scale.unsqueeze(1) / math.sqrt(T)
        self.register_buffer("idft_cos", torch.cos(inv_angles) * scale_i)  # [K, T]
        self.register_buffer("idft_sin", torch.sin(inv_angles) * scale_i)  # [K, T]

        # 頻率 bin（用於 soft mask 計算）
        freqs = torch.arange(K, dtype=torch.float32) * fps / T
        self.register_buffer("freqs", freqs)  # [K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        # 1. 固定空間 pool：H×W → pool_size_h×pool_size_w
        x_pooled = F.avg_pool3d(
            x,
            kernel_size=(1, self.pool_kernel_h, self.pool_kernel_w),
            stride     =(1, self.pool_stride_h, self.pool_stride_w),
        )                                                          # [B, C, T, ph, pw]
        x_flat = x_pooled.view(B, C, T, -1)                       # [B, C, T, ph*pw]

        # 2. DFT（在 T 維度，等同 rfft dim=2）
        x_t      = x_flat.permute(0, 1, 3, 2)                     # [B, C, ph*pw, T]
        fft_real = torch.matmul(x_t, self.dft_cos)                # [B, C, ph*pw, K]
        fft_imag = torch.matmul(x_t, self.dft_sin)

        # 3. Learnable soft mask（ONNX export 後由 constant folding 固化）
        low_gate  = torch.sigmoid(
            self.sharpness * (self.freqs - (self.freq_center - self.freq_width)))
        high_gate = torch.sigmoid(
            self.sharpness * (-(self.freqs - (self.freq_center + self.freq_width))))
        soft_mask = (low_gate * high_gate).view(1, 1, 1, -1)      # [1, 1, 1, K]

        fft_real_f = fft_real * soft_mask
        fft_imag_f = fft_imag * soft_mask

        # 4. iDFT → filtered wave
        filtered = (torch.matmul(fft_real_f, self.idft_cos)
                    - torch.matmul(fft_imag_f, self.idft_sin))    # [B, C, ph*pw, T]
        filtered = filtered.permute(0, 1, 3, 2)                   # [B, C, T, ph*pw]
        filtered_5d = filtered.view(
            B, C, T, self.pool_size_h, self.pool_size_w)          # [B, C, T, ph, pw]

        # 5. 逐幀時序注意力分數
        temporal_scores    = self.score_refiner(filtered_5d)       # [B, 1, T, ph, pw]
        pattern_map_probs  = torch.sigmoid(temporal_scores)        # [B, 1, T, ph, pw]

        # 6. 功率比 → 空間注意力
        cardiac_pwr = (fft_real_f ** 2 + fft_imag_f ** 2).sum(dim=-1)   # [B, C, ph*pw]
        total_pwr   = (fft_real   ** 2 + fft_imag   ** 2).sum(dim=-1) + 1e-8
        power_ratio = cardiac_pwr / total_pwr                            # [B, C, ph*pw]
        spatial_att = power_ratio.view(B, C, self.pool_size_h, self.pool_size_w)
        spatial_score      = self.spatial_score_refiner(spatial_att)     # [B, 1, ph, pw]
        energy_map_probs   = torch.sigmoid(spatial_score).unsqueeze(2)  # [B, 1, 1, ph, pw]

        # 7. 殘差 fusion：1 + tanh(pattern + energy - 1)
        att_fused = 1.0 + torch.tanh(
            pattern_map_probs + energy_map_probs - 1.0)            # [B, 1, T, ph, pw]

        # 8. 逐幀 upsample（假設 B=1，B*T = T 為常數）
        att_flat = att_fused.view(B * T, 1, self.pool_size_h, self.pool_size_w)
        att_up   = F.interpolate(
            att_flat, size=(H, W), mode='bilinear', align_corners=False)
        att_up   = att_up.view(B, 1, T, H, W)

        return x * att_up


# ── 固定尺寸 pool 模組 ────────────────────────────────────────────────────────

class FixedAvgPool3d_256x18_to_128x1(nn.Module):
    """poolspa: [B, 64, 256, 18, 18] → [B, 64, 128, 1, 1]"""
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=(2, 18, 18), stride=(2, 18, 18))


class SpatialPoolThenRepeat(nn.Module):
    """spo2_branch_pool: [B, 16, 64, 18, 18] → [B, 16, 128, 1, 1]"""
    def forward(self, x):
        pooled = F.avg_pool3d(x, kernel_size=(1, 18, 18), stride=(1, 18, 18))
        return pooled.repeat_interleave(2, dim=2)


# ── ONNX-safe FCAtt_v2 ───────────────────────────────────────────────────────

class FCAtt_v2_ONNX(FCAtt_v2):
    """
    FCAtt_v2 ONNX 版本。

    替換：
      - DiffNormalizeLayer      → DiffNormalizeLayerONNX
      - ChannelAttention3D × 4  → ChannelAttention3DONNX
      - att_mask1               → FrequencyAttentionV2ONNX
      - poolspa                 → FixedAvgPool3d_256x18_to_128x1
      - spo2_branch_pool        → SpatialPoolThenRepeat
      - AdaptiveAvgPool1d(1)    → mean（spo2_fusion_head 第 8 層）
      - torch.diff              → 顯式相減
    """

    def forward(self, x: torch.Tensor,
                x2: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x.shape

        rppg_wave, spo2_features = self.encode_video(x)
        rppg_wave = rppg_wave.view(B, T)

        # torch.diff → 顯式相減（ONNX 相容）
        rppg_1d  = rppg_wave.unsqueeze(1)                          # [B, 1, T]
        vpg_diff = rppg_1d[:, :, 1:] - rppg_1d[:, :, :-1]
        vpg      = F.pad(vpg_diff, (0, 1), mode='replicate')
        apg_diff = vpg[:, :, 1:] - vpg[:, :, :-1]
        apg      = F.pad(apg_diff, (0, 1), mode='replicate')

        spo2_1d = spo2_features.view(B, 16, T)
        fused   = torch.cat([rppg_1d, vpg, apg, spo2_1d], dim=1)  # [B, 19, T]
        spo2_pred = self.spo2_fusion_head(fused)                   # [B, 1]
        spo2_pred = torch.clamp(
            spo2_pred * self.spo2_range + self.spo2_min,
            min=self.spo2_min, max=self.spo2_min + self.spo2_range,
        )
        return rppg_wave, spo2_pred


# ── 建立可匯出模型 ─────────────────────────────────────────────────────────────

def build_exportable_model(weights_path: str) -> FCAtt_v2_ONNX:
    """讀入 FCAtt_v2 checkpoint，替換所有非 ONNX-compatible ops。"""
    ckpt  = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    # 從 checkpoint 反推架構超參（temporal_kernel_size）
    key = 'att_mask1.score_refiner.0.weight'
    temporal_kernel = state[key].shape[2] if key in state else 3
    print(f"[Export] Detected freq_att_temporal_kernel={temporal_kernel}")

    # 建立 ONNX 子類別，載入權重
    model = FCAtt_v2_ONNX(
        frames=128, fps=30,
        freq_att_pool_size=8,
        freq_att_temporal_kernel=temporal_kernel,
        freq_low=0.7, freq_high=2.5,
        freq_att_sharpness=10.0,
        spo2_min=85.0, spo2_range=15.0,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    orig = FCAtt_v2(
        frames=128, fps=30,
        freq_att_pool_size=8,
        freq_att_temporal_kernel=temporal_kernel,
        freq_low=0.7, freq_high=2.5,
        freq_att_sharpness=10.0,
    )
    orig.load_state_dict(state, strict=False)
    orig.eval()

    # 替換 DiffNormalizeLayer
    model.diff_norm = DiffNormalizeLayerONNX()

    # 替換 ChannelAttention3D
    model.channel_attention  = ChannelAttention3DONNX(orig.channel_attention)
    model.channel_attention1 = ChannelAttention3DONNX(orig.channel_attention1)
    model.channel_attention2 = ChannelAttention3DONNX(orig.channel_attention2)
    model.channel_attention3 = ChannelAttention3DONNX(orig.channel_attention3)

    # 替換 FrequencyAttention v2（空間輸入為 36×36）
    model.att_mask1 = FrequencyAttentionV2ONNX(orig.att_mask1, input_h=36, input_w=36)

    # 替換 adaptive pools（已知輸入形狀）
    model.poolspa          = FixedAvgPool3d_256x18_to_128x1()
    model.spo2_branch_pool = SpatialPoolThenRepeat()

    # 替換 spo2_fusion_head 中的 AdaptiveAvgPool1d(1) → global mean
    # 第 7 層（index 7）是 AdaptiveAvgPool1d(1)
    head_layers = list(model.spo2_fusion_head.children())
    new_layers  = []
    for layer in head_layers:
        if isinstance(layer, nn.AdaptiveAvgPool1d):
            new_layers.append(nn.AdaptiveAvgPool1d(1))  # stays — ONNX supports this
        else:
            new_layers.append(layer)
    # AdaptiveAvgPool1d(1) is well-supported in ONNX opset 11+; keep as-is

    return model


# ── 驗證：比對 PyTorch vs ONNX Runtime 輸出 ──────────────────────────────────

def validate(model: nn.Module, onnx_path: str, dummy: torch.Tensor):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Validate] onnxruntime not installed, skipping.")
        return True

    with torch.no_grad():
        pt_wave, pt_spo2 = model(dummy)
    pt_wave = pt_wave.numpy()
    pt_spo2 = pt_spo2.numpy()

    sess    = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {'video': dummy.numpy()})
    ort_wave, ort_spo2 = ort_out

    wave_err = float(np.max(np.abs(pt_wave - ort_wave)))
    spo2_err = float(np.max(np.abs(pt_spo2 - ort_spo2)))
    print(f"[Validate] rppg_wave max error : {wave_err:.6f}")
    print(f"[Validate] spo2     max error  : {spo2_err:.6f}")

    ok = wave_err < 0.05 and spo2_err < 1.0
    print(f"[Validate] {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='checkpoints/rppg_semi/rppg_fcatt_v2_128f_72px_best.pth')
    parser.add_argument('--output',  default='web/static/models/rppg_fcatt_v2.onnx')
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), '..')
    weights_path = args.weights if os.path.isabs(args.weights) \
                   else os.path.join(root, args.weights)
    output_path  = args.output  if os.path.isabs(args.output)  \
                   else os.path.join(root, args.output)

    if not os.path.isfile(weights_path):
        print(f"[Error] Weights not found: {weights_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Export] Weights : {weights_path}")
    print(f"[Export] Output  : {output_path}")

    model = build_exportable_model(weights_path)

    dummy = torch.randn(1, 3, 128, 72, 72)
    with torch.no_grad():
        wave, spo2 = model(dummy)
    print(f"[Export] Forward OK: rppg_wave {list(wave.shape)}, spo2 {list(spo2.shape)}")

    torch.onnx.export(
        model, (dummy,), output_path,
        input_names=['video'],
        output_names=['rppg_wave', 'spo2'],
        opset_version=18,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Done: {output_path} ({size_mb:.1f} MB)")

    validate(model, output_path, dummy)


if __name__ == '__main__':
    main()
