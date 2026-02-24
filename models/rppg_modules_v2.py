"""
rPPG building block modules v2: DiffNormalize, improved FrequencyAttention, ChannelAttention.

FrequencyAttention improvements over v1:
  1. Learnable soft frequency mask (nn.Parameter center/width, differentiable)
  2. Residual attention: 1 + tanh(pattern + energy - 1) (replaces sqrt(a*b))
  3. Power ratio: cardiac/total power (scale-invariant)
  4. Time-varying attention: per-frame maps (replaces time-averaged)
  5. Reduced pool size: 8 (from 18) for efficiency

Removed from v2.0:
  - FusionNet (dual-input path never used in semi-supervised training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 差分正規化層 ─────────────────────────────────────────────────────────────

class DiffNormalizeLayer(nn.Module):
    """
    內建差分正規化（rPPG-Toolbox 標準公式）：
      (frame[t] - frame[t-1]) / (|frame[t]| + |frame[t-1]| + ε)
    再除以整段 clip 的 std（全域標準化），最後 clamp 到 [-5, 5]。

    輸入：原始像素值 0-255 (float)
    輸出：差分正規化後的序列，第一幀補零
    不含可學習參數，train/eval 行為一致。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        diff = x[:, :, 1:] - x[:, :, :-1]                                       # [B, 3, T-1, H, W]
        norm = diff / (x[:, :, 1:].abs() + x[:, :, :-1].abs() + 1e-7)           # sum-denominator
        norm = norm / (norm.std(dim=2, keepdim=True) + 1e-7)                     # global std per clip
        norm = torch.clamp(norm, -5.0, 5.0)
        pad  = torch.zeros_like(x[:, :, :1])                                     # zeros for t=0
        return torch.cat([pad, norm], dim=2)                                     # [B, 3, T, H, W]


class FrequencyAttention(nn.Module):
    """
    FFT-based frequency attention module v2 with improvements:

    1. Learnable soft frequency mask (replaces hard binary mask)
       - nn.Parameter for center/width, differentiable sigmoid gates
    2. Residual attention: 1 + tanh(pattern + energy - 1)
       - Replaces sqrt(a*b), guarantees output in [0, 2], centered at 1
    3. Power ratio: cardiac_power / total_power
       - Replaces absolute power, scale-invariant across channels
    4. Time-varying attention: per-frame attention maps
       - Replaces time-averaged maps, captures transient events
    5. Reduced pool size: 8 (from 18) for efficiency
    """

    def __init__(self, num_input_channels: int, frames: int = 128, fps: int = 30,
                 pool_size: int = 8, temporal_kernel_size: int = 3,
                 M_intermediate_channels: int = 16,
                 freq_low: float = 0.7, freq_high: float = 2.5,
                 sharpness: float = 10.0):
        super().__init__()
        self.frames = frames
        self.fps = fps
        self.pool_size_h = pool_size
        self.pool_size_w = pool_size
        self.sharpness = sharpness
        self.pool = nn.AdaptiveAvgPool3d((None, pool_size, pool_size))

        # Improvement 1: Learnable soft frequency mask
        self.freq_center = nn.Parameter(
            torch.tensor((freq_low + freq_high) / 2.0))
        self.freq_width = nn.Parameter(
            torch.tensor((freq_high - freq_low) / 2.0))

        # Temporal score refiner
        self.score_refiner = nn.Sequential(
            nn.Conv3d(
                num_input_channels, M_intermediate_channels,
                kernel_size=(temporal_kernel_size, 1, 1),
                padding=((temporal_kernel_size - 1) // 2, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(M_intermediate_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(M_intermediate_channels, 1, kernel_size=(1, 1, 1), bias=True),
        )

        # Spatial score refiner — NO Sigmoid (applied in forward)
        self.spatial_score_refiner = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        # 1. Spatial pooling and reshape
        x_pooled = self.pool(x)  # [B, C, T, ph, pw]
        x_reshaped = x_pooled.view(B, C, T, -1)  # [B, C, T, ph*pw]

        # 2. FFT
        fft_out = torch.fft.rfft(x_reshaped, dim=2, norm='ortho')

        # 3. Learnable soft frequency mask (improvement 1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(x.device)
        low_gate = torch.sigmoid(
            self.sharpness * (freqs - (self.freq_center - self.freq_width)))
        high_gate = torch.sigmoid(
            self.sharpness * (-(freqs - (self.freq_center + self.freq_width))))
        soft_mask = low_gate * high_gate  # [num_freq_bins]

        fft_filtered = fft_out * soft_mask.view(1, 1, -1, 1)

        # 4. Inverse FFT → temporal refiner
        filtered_wave = torch.fft.irfft(fft_filtered, n=T, dim=2, norm='ortho')
        filtered_wave_spatial = filtered_wave.view(
            B, C, T, self.pool_size_h, self.pool_size_w)

        # Improvement 4: Time-varying attention (keep T dimension)
        temporal_scores = self.score_refiner(
            filtered_wave_spatial)              # [B, 1, T, ph, pw]
        pattern_map_probs = torch.sigmoid(temporal_scores)

        # 5. Power ratio → spatial refiner (improvement 3)
        cardiac_power = torch.sum(
            torch.abs(fft_filtered) ** 2, dim=2)        # [B, C, ph*pw]
        total_power = torch.sum(
            torch.abs(fft_out) ** 2, dim=2) + 1e-8      # [B, C, ph*pw]
        power_ratio = cardiac_power / total_power        # [B, C, ph*pw]

        spatial_att = power_ratio.view(B, C, self.pool_size_h, self.pool_size_w)
        spatial_score = self.spatial_score_refiner(spatial_att)  # [B, 1, ph, pw]
        energy_map_probs = torch.sigmoid(spatial_score)
        energy_map_probs = energy_map_probs.unsqueeze(2)  # [B, 1, 1, ph, pw]

        # 6. Residual attention fusion (improvement 2)
        att_map_fused = 1.0 + torch.tanh(
            pattern_map_probs + energy_map_probs - 1.0)  # [B, 1, T, ph, pw]

        # 7. Upsample to original spatial size (per-frame)
        att_flat = att_map_fused.view(
            B * T, 1, self.pool_size_h, self.pool_size_w)
        att_upsampled = F.interpolate(
            att_flat, size=(H, W), mode='bilinear', align_corners=False)
        att_upsampled = att_upsampled.view(B, 1, T, H, W)

        return x * att_upsampled


class ChannelAttention1D(nn.Module):
    """1D channel attention (SE-block style)."""

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        if in_channels < reduction:
            reduction = max(in_channels // 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ChannelAttention3D(nn.Module):
    """3D channel attention with avg + max pooling."""

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention
