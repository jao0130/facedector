"""
rPPG building block modules: FrequencyAttention, TemporalShift, ChannelAttention, FusionNet.
Ported from rPPG-Toolbox FCAtt.py (already PyTorch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """Channel-wise temporal shifting for temporal feature extraction."""

    def __init__(self, shift_ratio: float = 0.125):
        super().__init__()
        self.shift_ratio = shift_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.size()
        shift = int(C * self.shift_ratio)
        if shift == 0:
            return x
        x_shift = x.clone()
        x_shift[:, :shift, :-1] = x[:, :shift, 1:]        # Forward shift
        x_shift[:, shift:2*shift, 1:] = x[:, shift:2*shift, :-1]  # Backward shift
        return x_shift


class FrequencyAttention(nn.Module):
    """
    FFT-based frequency attention module.
    Filters features to cardiac frequency band and learns attention maps.
    """

    def __init__(self, num_input_channels: int, frames: int = 128, fps: int = 30,
                 pool_size: int = 18, temporal_kernel_size: int = 3,
                 M_intermediate_channels: int = 16,
                 freq_low: float = 0.7, freq_high: float = 2.5):
        super().__init__()
        self.frames = frames
        self.fps = fps
        self.pool_size_h = pool_size
        self.pool_size_w = pool_size
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.pool = nn.AdaptiveAvgPool3d((None, pool_size, pool_size))

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

        # Spatial score refiner
        self.spatial_score_refiner = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        # 1. Spatial pooling and reshape
        x_pooled = self.pool(x)
        x_reshaped = x_pooled.view(B, C, T, -1)

        # 2. FFT
        fft_out = torch.fft.rfft(x_reshaped, dim=2, norm='ortho')

        # 3. Frequency mask
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(x.device)
        mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        fft_filtered_complex = fft_out * mask.view(1, 1, -1, 1)

        # 4. Inverse FFT -> temporal refiner
        filtered_wave = torch.fft.irfft(fft_filtered_complex, n=T, dim=2, norm='ortho')
        filtered_wave_spatial = filtered_wave.view(B, C, T, self.pool_size_h, self.pool_size_w)
        temporal_scores = self.score_refiner(filtered_wave_spatial)
        pattern_map_scores = torch.mean(temporal_scores, dim=2)
        pattern_map_probs = torch.sigmoid(pattern_map_scores)

        # 5. Power map -> spatial refiner
        power_map = torch.mean(torch.abs(fft_filtered_complex) ** 2, dim=2)
        spatial_att = power_map.view(B, C, self.pool_size_h, self.pool_size_w)
        spatial_score = self.spatial_score_refiner(spatial_att)
        energy_map_probs = torch.sigmoid(spatial_score)

        # 6. Fused attention
        att_map_fused = torch.sqrt(pattern_map_probs * energy_map_probs + 1e-20)

        # 7. Upsample to original spatial size
        att_map_upsampled = F.interpolate(
            att_map_fused, size=(H, W), mode='bilinear', align_corners=False,
        )
        att_map_upsampled = att_map_upsampled.unsqueeze(2)  # [B, 1, 1, H, W]

        return x * att_map_upsampled


class FusionNet(nn.Module):
    """Fuses two rPPG streams (face + finger) via 3D convolution."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 32, kernel_size=[3, 1, 1], stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=[3, 1, 1], stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


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
