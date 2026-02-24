"""
FCAtt rPPG model: PhysNet encoder-decoder with FrequencyAttention + SpO2 branch.
Ported from rPPG-Toolbox FCAtt.py.

Input: [B, 3, T, H, W] where T=128, H=W=72
Output: (rppg_wave: [B, T], spo2_pred: [B, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rppg_modules import (
    FrequencyAttention, FusionNet, ChannelAttention1D, ChannelAttention3D,
)


class FCAtt(nn.Module):
    """
    PhysNet encoder-decoder with FrequencyAttention and dual output.
    rPPG wave (BVP signal) + SpO2 prediction.
    """

    def __init__(self, cfg=None, frames=128, fps=30, drop_rate1=0.1, drop_rate2=0.2,
                 freq_att_pool_size=18, freq_att_temporal_kernel=7,
                 freq_low=0.7, freq_high=2.5, spo2_min=85.0, spo2_range=15.0):
        super().__init__()

        if cfg is not None:
            frames = cfg.RPPG_MODEL.FRAMES
            fps = cfg.RPPG_MODEL.FPS
            drop_rate1 = cfg.RPPG_MODEL.DROP_RATE1
            drop_rate2 = cfg.RPPG_MODEL.DROP_RATE2
            freq_att_pool_size = cfg.RPPG_MODEL.FREQ_ATT_POOL_SIZE
            freq_att_temporal_kernel = cfg.RPPG_MODEL.FREQ_ATT_TEMPORAL_KERNEL
            freq_low = cfg.RPPG_MODEL.FREQ_LOW
            freq_high = cfg.RPPG_MODEL.FREQ_HIGH
            spo2_min = cfg.RPPG_MODEL.SPO2_MIN
            spo2_range = cfg.RPPG_MODEL.SPO2_RANGE

        self.frames = frames
        self.spo2_min = spo2_min
        self.spo2_range = spo2_range

        # Encoder blocks
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16), nn.ELU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(32), nn.ELU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [5, 1, 1], stride=1, padding=[2, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )

        # Residual connections
        self.residual_conv2 = nn.Conv3d(16, 32, 1, stride=1, padding=0)
        self.residual_conv4 = nn.Conv3d(64, 64, 1, stride=1, padding=0)
        self.residual_conv6 = nn.Conv3d(64, 64, 1, stride=1, padding=0)
        self.residual_conv8 = nn.Conv3d(64, 64, 1, stride=1, padding=0)

        # Channel attention (SE-block) after residual blocks
        self.channel_attention = ChannelAttention3D(64, reduction=4)
        self.channel_attention1 = ChannelAttention3D(64, reduction=4)
        self.channel_attention2 = ChannelAttention3D(64, reduction=4)
        self.channel_attention3 = ChannelAttention3D(64, reduction=4)

        # Frequency attention
        self.att_mask1 = FrequencyAttention(
            num_input_channels=64, frames=frames, fps=fps,
            pool_size=freq_att_pool_size, temporal_kernel_size=freq_att_temporal_kernel,
            freq_low=freq_low, freq_high=freq_high,
        )

        # HR branch: upsample + final conv
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(),
        )
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        # Pooling
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # SpO2 branch
        self.spo2_branch_conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.BatchNorm3d(32), nn.ELU(inplace=True),
        )
        self.spo2_branch_conv2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=[3, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(16), nn.ELU(inplace=True),
        )
        self.spo2_branch_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # SpO2 fusion head: 1 (rPPG) + 1 (VPG) + 1 (APG) + 16 (SpO2 branch) = 19 channels
        self.num_spo2_branch_output_channels = 16
        fusion_head_input_channels = 1 + 1 + 1 + self.num_spo2_branch_output_channels
        self.spo2_fusion_head = nn.Sequential(
            nn.Conv1d(fusion_head_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.ELU(),
            ChannelAttention1D(in_channels=32, reduction=8),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1), nn.ELU(),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Multi-input fusion
        self.fusion_net = FusionNet()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        batch, channel, length, width, height = x1.shape

        if x2 is not None:
            rppg_wave1, spo2_1 = self.encode_video(x1)
            rppg_wave2, _ = self.encode_video(x2)
            rppg_wave = self.fusion_net(rppg_wave1, rppg_wave2)
        else:
            rppg_wave, spo2_features = self.encode_video(x1)

        rppg_wave = rppg_wave.view(batch, length)  # [B, T]

        # Compute derivatives for SpO2 fusion
        rppg_d1 = rppg_wave.unsqueeze(1)  # [B, 1, T]
        vpg_temp = torch.diff(rppg_d1, n=1, dim=-1)
        vpg_signal = F.pad(vpg_temp, (0, 1), mode='replicate')
        apg_temp = torch.diff(vpg_signal, n=1, dim=-1)
        apg_signal = F.pad(apg_temp, (0, 1), mode='replicate')

        # SpO2 prediction
        spo2_branch_for_fusion = spo2_features.view(
            batch, self.num_spo2_branch_output_channels, length,
        )
        fused_features_1d = torch.cat(
            (rppg_d1, vpg_signal, apg_signal, spo2_branch_for_fusion), dim=1,
        )
        spo2_pred = self.spo2_fusion_head(fused_features_1d)  # [B, 1]
        spo2_pred = spo2_pred * self.spo2_range + self.spo2_min

        return rppg_wave, spo2_pred

    def encode_video(self, x: torch.Tensor):
        x = self.ConvBlock1(x)        # [B, 16, T, H, W]
        x = self.MaxpoolSpa(x)        # [B, 16, T, H/2, W/2]

        residual2 = self.residual_conv2(x)  # [B, 32, T, H/2, W/2]
        x = self.ConvBlock2(x)        # [B, 32, T, H/2, W/2]
        x = x + residual2

        x = self.ConvBlock3(x)        # [B, 64, T, H/2, W/2]
        x_att = self.att_mask1(x)
        x = self.MaxpoolSpaTem(x_att) # [B, 64, T/2, H/4, W/4]

        residual4 = self.residual_conv4(x)
        x = self.ConvBlock4(x)
        x = x + residual4
        x = self.channel_attention(x)
        x = self.ConvBlock5(x)
        x = self.channel_attention1(x)

        residual6 = self.residual_conv6(x)
        x = self.ConvBlock6(x)
        x = x + residual6
        x = self.channel_attention2(x)
        xb = self.ConvBlock7(x)

        # HR branch
        residual8 = self.residual_conv8(xb)
        x = self.ConvBlock8(xb)
        x = x + residual8
        x = self.channel_attention3(x)
        x = self.ConvBlock9(x)

        hr_f = self.upsample(x)
        hr_f = self.upsample2(hr_f)
        hr_f_pooled = self.poolspa(hr_f)
        rppg_wave_features = self.ConvBlock10(hr_f_pooled)

        # SpO2 branch
        spo2_b_f = self.spo2_branch_conv1(xb)
        spo2_b_f = self.spo2_branch_conv2(spo2_b_f)
        spo2_branch_features = self.spo2_branch_pool(spo2_b_f)

        return rppg_wave_features, spo2_branch_features


def create_rppg_model(cfg) -> FCAtt:
    """Create rPPG model from config."""
    return FCAtt(cfg=cfg)
