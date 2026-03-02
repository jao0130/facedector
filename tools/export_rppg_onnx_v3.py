"""
Export FCAtt_v3 rPPG model to ONNX for browser inference.

V3 與 V2 的主要架構差異：
  - SpatialStemEncoder (+ TSM)         : 3→64，空間 /4（72→18）
  - FrequencyAttentionBlock × 3        : 每塊含 FrequencyAttention + ChannelAttention3D(V3版)
  - HRBranch (temporal_refiner + Mamba): AdaptiveAvgPool3d 需替換
  - SpO2Branch                         : AdaptiveAvgPool3d 需替換

ONNX 不支援的 ops 替換：
  - DiffNormalizeLayer.std(dim=…)   → 手動 sqrt(var)
  - FrequencyAttention rfft/irfft   → 預計算 DFT 矩陣 matmul
  - ChannelAttention3D adaptive pool → global mean / amax
  - HRBranch.spatial_pool           → F.avg_pool3d(kernel 1×18×18)
  - SpO2Branch.pool                 → F.avg_pool3d(kernel 1×18×18)
  - torch.diff (forward)            → 顯式相減

MambaBlock sequential scan：
  ONNX tracer 會將 for-loop 展開為 128 個靜態節點。
  模型略大但推論正確，onnxruntime-web 可執行。

Usage:
    cd D:\\Projects\\facedector
    python tools/export_rppg_onnx_v3.py
    python tools/export_rppg_onnx_v3.py --weights PATH --output PATH
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

from models.rppg_model_v3 import (
    FCAtt_v3, DiffNormalizeLayer, TemporalShift,
    SpatialStemEncoder, FrequencyAttentionBlock,
    HRBranch, SpO2Branch, ChannelAttention3D, MambaBlock,
)
from models.rppg_modules_v2 import FrequencyAttention, ChannelAttention1D


# ── 共用：ONNX-safe DiffNormalizeLayer ───────────────────────────────────────

class DiffNormalizeLayerONNX(nn.Module):
    """std(dim=…) → 手動 sqrt(var)，避免多維 std 的 ONNX 相容問題。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, :, 1:] - x[:, :, :-1]
        norm = diff / (x[:, :, 1:].abs() + x[:, :, :-1].abs() + 1e-7)
        mean = norm.mean(dim=[2, 3, 4], keepdim=True)
        var  = ((norm - mean) ** 2).mean(dim=[2, 3, 4], keepdim=True)
        std  = (var + 1e-7).sqrt()
        norm = norm / (std + 1e-7)
        norm = torch.clamp(norm, -5.0, 5.0)
        pad  = torch.zeros_like(x[:, :, :1])
        return torch.cat([pad, norm], dim=2)


# ── V3 ChannelAttention3D ONNX（avg/max adaptive → global mean/amax）────────

class ChannelAttention3DONNX(nn.Module):
    """V3 的 ChannelAttention3D：AdaptiveAvgPool3d(1)/AdaptiveMaxPool3d(1) → global 運算。"""

    def __init__(self, orig: ChannelAttention3D):
        super().__init__()
        self.fc      = orig.fc
        self.sigmoid = orig.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(x.mean(dim=[2, 3, 4], keepdim=True))
        mx  = self.fc(torch.amax(x, dim=[2, 3, 4], keepdim=True))
        return x * self.sigmoid(avg + mx)


# ── ONNX-safe FrequencyAttention（rfft → DFT matmul）────────────────────────

class FrequencyAttentionONNX(nn.Module):
    """
    FrequencyAttention ONNX 版本（V3 用，input 18×18，pool → 8×8）。

    替換：
      - torch.fft.rfft/irfft → 預計算 DFT/iDFT 矩陣 matmul
      - AdaptiveAvgPool3d    → 固定 kernel/stride avg_pool3d
      - B=1 假設（瀏覽器推論）
    """

    def __init__(self, orig: FrequencyAttention,
                 input_h: int = 18, input_w: int = 18):
        super().__init__()
        self.score_refiner         = orig.score_refiner
        self.spatial_score_refiner = orig.spatial_score_refiner
        self.freq_center           = orig.freq_center
        self.freq_width            = orig.freq_width
        self.sharpness             = orig.sharpness
        self.pool_size_h           = orig.pool_size_h
        self.pool_size_w           = orig.pool_size_w
        self.input_h               = input_h
        self.input_w               = input_w

        T   = orig.frames
        fps = orig.fps
        K   = T // 2 + 1

        # 固定空間 pool：PyTorch adaptive 公式
        ph = orig.pool_size_h  # 8
        pw = orig.pool_size_w  # 8
        self.pool_stride_h = input_h // ph          # 18//8 = 2
        self.pool_stride_w = input_w // pw
        self.pool_kernel_h = input_h - (ph - 1) * self.pool_stride_h  # 18 - 7*2 = 4
        self.pool_kernel_w = input_w - (pw - 1) * self.pool_stride_w

        # DFT 矩陣（rfft，ortho 正規化）
        n = torch.arange(T, dtype=torch.float32)
        k = torch.arange(K, dtype=torch.float32)
        angles = -2.0 * math.pi * n.unsqueeze(1) * k.unsqueeze(0) / T
        scale  = 1.0 / math.sqrt(T)
        self.register_buffer("dft_cos", torch.cos(angles) * scale)
        self.register_buffer("dft_sin", torch.sin(angles) * scale)

        # iDFT 矩陣
        inv_angles = 2.0 * math.pi * n.unsqueeze(0) * k.unsqueeze(1) / T
        bin_scale  = torch.ones(K, dtype=torch.float32) * 2.0
        bin_scale[0] = 1.0
        if T % 2 == 0:
            bin_scale[-1] = 1.0
        scale_i = bin_scale.unsqueeze(1) / math.sqrt(T)
        self.register_buffer("idft_cos", torch.cos(inv_angles) * scale_i)
        self.register_buffer("idft_sin", torch.sin(inv_angles) * scale_i)

        # 頻率 bin
        freqs = torch.arange(K, dtype=torch.float32) * fps / T
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        # 1. 固定空間 pool → pool_size_h × pool_size_w
        x_pooled = F.avg_pool3d(
            x,
            kernel_size=(1, self.pool_kernel_h, self.pool_kernel_w),
            stride     =(1, self.pool_stride_h, self.pool_stride_w),
        )
        x_flat = x_pooled.view(B, C, T, -1)                       # [B, C, T, ph*pw]

        # 2. DFT
        x_t      = x_flat.permute(0, 1, 3, 2)                     # [B, C, ph*pw, T]
        fft_real = torch.matmul(x_t, self.dft_cos)
        fft_imag = torch.matmul(x_t, self.dft_sin)

        # 3. Learnable soft mask
        low_gate  = torch.sigmoid(
            self.sharpness * (self.freqs - (self.freq_center - self.freq_width)))
        high_gate = torch.sigmoid(
            self.sharpness * (-(self.freqs - (self.freq_center + self.freq_width))))
        soft_mask = (low_gate * high_gate).view(1, 1, 1, -1)

        fft_real_f = fft_real * soft_mask
        fft_imag_f = fft_imag * soft_mask

        # 4. iDFT
        filtered = (torch.matmul(fft_real_f, self.idft_cos)
                    - torch.matmul(fft_imag_f, self.idft_sin))
        filtered = filtered.permute(0, 1, 3, 2)
        filtered_5d = filtered.view(B, C, T, self.pool_size_h, self.pool_size_w)

        # 5. 逐幀時序注意力
        temporal_scores   = self.score_refiner(filtered_5d)
        pattern_map_probs = torch.sigmoid(temporal_scores)

        # 6. 功率比 → 空間注意力
        cardiac_pwr = (fft_real_f ** 2 + fft_imag_f ** 2).sum(dim=-1)
        total_pwr   = (fft_real   ** 2 + fft_imag   ** 2).sum(dim=-1) + 1e-8
        power_ratio = cardiac_pwr / total_pwr
        spatial_att = power_ratio.view(B, C, self.pool_size_h, self.pool_size_w)
        spatial_score    = self.spatial_score_refiner(spatial_att)
        energy_map_probs = torch.sigmoid(spatial_score).unsqueeze(2)

        # 7. 殘差 fusion
        att_fused = 1.0 + torch.tanh(pattern_map_probs + energy_map_probs - 1.0)

        # 8. upsample（B=1 假設）
        att_flat = att_fused.view(B * T, 1, self.pool_size_h, self.pool_size_w)
        att_up   = F.interpolate(att_flat, size=(H, W),
                                 mode='bilinear', align_corners=False)
        att_up   = att_up.view(B, 1, T, H, W)

        return x * att_up


# ── ONNX-safe FrequencyAttentionBlock ────────────────────────────────────────

class FrequencyAttentionBlockONNX(nn.Module):
    """FrequencyAttentionBlock：替換內部 FrequencyAttention 和 ChannelAttention3D。"""

    def __init__(self, orig: FrequencyAttentionBlock,
                 input_h: int = 18, input_w: int = 18):
        super().__init__()
        self.freq_att    = FrequencyAttentionONNX(orig.freq_att, input_h, input_w)
        self.pointwise   = orig.pointwise
        self.channel_attn = ChannelAttention3DONNX(orig.channel_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.freq_att(x)
        out = self.pointwise(out)
        out = self.channel_attn(out)
        return out + x


# ── ONNX-safe HRBranch（fixed spatial pool）──────────────────────────────────

class HRBranchONNX(nn.Module):
    """HRBranch：AdaptiveAvgPool3d((T,1,1)) → F.avg_pool3d(1×18×18)。"""

    def __init__(self, orig: HRBranch):
        super().__init__()
        self.temporal_refiner = orig.temporal_refiner
        self.mamba            = orig.mamba
        self.dropout          = orig.dropout
        self.output_conv      = orig.output_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_refiner(x)                       # [B, 64, T, 18, 18]
        x = F.avg_pool3d(x, kernel_size=(1, 18, 18),
                         stride=(1, 18, 18))               # [B, 64, T, 1, 1]
        x_1d = x.squeeze(-1).squeeze(-1)                   # [B, 64, T]
        x_1d = self.mamba(x_1d)
        x_1d = self.dropout(x_1d)
        x = x_1d.unsqueeze(-1).unsqueeze(-1)               # [B, 64, T, 1, 1]
        return self.output_conv(x)                         # [B, 1, T, 1, 1]


# ── ONNX-safe SpO2Branch（fixed spatial pool）────────────────────────────────

class SpO2BranchONNX(nn.Module):
    """SpO2Branch：AdaptiveAvgPool3d((T,1,1)) → F.avg_pool3d(1×18×18)。"""

    def __init__(self, orig: SpO2Branch):
        super().__init__()
        self.conv1 = orig.conv1
        self.conv2 = orig.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                                  # [B, 32, T, 18, 18]
        x = self.conv2(x)                                  # [B, 16, T, 18, 18]
        return F.avg_pool3d(x, kernel_size=(1, 18, 18),
                            stride=(1, 18, 18))            # [B, 16, T, 1, 1]


# ── ONNX-safe FCAtt_v3 ────────────────────────────────────────────────────────

class FCAtt_v3_ONNX(nn.Module):
    """
    FCAtt_v3 ONNX 版本。

    forward() 中的 torch.diff 替換為顯式相減，
    其餘全部委派給已替換的子模組。
    """

    def __init__(self, orig: FCAtt_v3,
                 input_h: int = 18, input_w: int = 18):
        super().__init__()
        self.frames     = orig.frames
        self.spo2_min   = orig.spo2_min
        self.spo2_range = orig.spo2_range

        self.diff_norm      = DiffNormalizeLayerONNX()
        self.stem           = orig.stem                    # SpatialStemEncoder: ONNX-safe
        self.freq_blocks    = nn.ModuleList([
            FrequencyAttentionBlockONNX(blk, input_h, input_w)
            for blk in orig.freq_blocks
        ])
        self.hr_branch      = HRBranchONNX(orig.hr_branch)
        self.spo2_branch    = SpO2BranchONNX(orig.spo2_branch)
        self.spo2_fusion_head = orig.spo2_fusion_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x.shape

        x = self.diff_norm(x)
        x = self.stem(x)
        for block in self.freq_blocks:
            x = block(x)

        rppg_feat = self.hr_branch(x)
        rppg_wave = rppg_feat.view(B, T)

        spo2_feat = self.spo2_branch(x)

        # torch.diff → 顯式相減（ONNX 相容）
        rppg_1d  = rppg_wave.unsqueeze(1)
        vpg_diff = rppg_1d[:, :, 1:] - rppg_1d[:, :, :-1]
        vpg      = F.pad(vpg_diff, (0, 1), mode='replicate')
        apg_diff = vpg[:, :, 1:] - vpg[:, :, :-1]
        apg      = F.pad(apg_diff, (0, 1), mode='replicate')

        spo2_1d   = spo2_feat.view(B, 16, T)
        fused     = torch.cat([rppg_1d, vpg, apg, spo2_1d], dim=1)
        spo2_pred = self.spo2_fusion_head(fused)
        spo2_pred = spo2_pred * self.spo2_range + self.spo2_min

        return rppg_wave, spo2_pred


# ── 建立可匯出模型 ─────────────────────────────────────────────────────────────

def build_exportable_model(weights_path: str) -> FCAtt_v3_ONNX:
    ckpt  = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    # 從 checkpoint 反推 temporal_kernel_size
    key = 'freq_blocks.0.freq_att.score_refiner.0.weight'
    temporal_kernel = state[key].shape[2] if key in state else 5
    print(f"[Export] Detected freq_att_temporal_kernel={temporal_kernel}")

    # 建立原始 V3 並載入權重
    orig = FCAtt_v3(
        frames=128, fps=30,
        num_freq_blocks=3, pool_size=8,
        temporal_kernel_size=temporal_kernel,
        freq_low=0.7, freq_high=2.5, sharpness=10.0,
        spo2_min=85.0, spo2_range=15.0,
    )
    missing, unexpected = orig.load_state_dict(state, strict=False)
    if missing:
        print(f"[Export] Missing keys ({len(missing)}): {missing[:3]}")
    orig.eval()

    # 建立 ONNX 版本（stem 輸出為 18×18 = 72/4）
    model = FCAtt_v3_ONNX(orig, input_h=18, input_w=18)
    model.eval()
    return model


# ── 驗證 ──────────────────────────────────────────────────────────────────────

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
    parser.add_argument('--weights',
        default='checkpoints/rppg_finetune_v3/rppg_fcatt_v3_128f_72px_finetune_best.pth')
    parser.add_argument('--output',
        default='web/static/models/rppg_fcatt_v3.onnx')
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

    dummy = torch.zeros(1, 3, 128, 72, 72)  # 固定輸入，確保 constant folding
    with torch.no_grad():
        wave, spo2 = model(dummy)
    print(f"[Export] Forward OK: wave {list(wave.shape)}, spo2 {list(spo2.shape)}")

    torch.onnx.export(
        model, (dummy,), output_path,
        input_names=['video'],
        output_names=['rppg_wave', 'spo2'],
        opset_version=18,
        do_constant_folding=True,
    )

    # 如果 PyTorch exporter 建立了外部 .data 檔，合併回單一 ONNX
    data_path = output_path + '.data'
    if os.path.isfile(data_path):
        import onnx as _onnx
        m = _onnx.load(output_path)
        _onnx.save_model(m, output_path, save_as_external_data=False)
        os.remove(data_path)
        print(f"[Export] External data merged into single file.")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Done: {output_path} ({size_mb:.1f} MB)")

    validate(model, output_path, dummy)


if __name__ == '__main__':
    main()
