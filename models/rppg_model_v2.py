"""
FCAtt_v2 rPPG model: PhysNet encoder-decoder with DiffNormalize built-in.

架構沿用 v2（PhysNet + FrequencyAttention + 時序 encoder-decoder），
針對訓練/推理一致性進行清理：
  - DiffNormalize 內建為第一層（接受原始 RGB 0-255）
  - FusionNet 及雙路 x2 輸入路徑移除（semi-trainer 從不使用）
  - Dropout 移除（預設從未啟用）

Input:  [B, 3, T, H, W]  raw uint8.float() 或 float 0-255
Output: (rppg_wave: [B, T], spo2_pred: [B, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rppg_modules_v2 import (
    DiffNormalizeLayer, FrequencyAttention,
    ChannelAttention1D, ChannelAttention3D,
)


class FCAtt_v2(nn.Module):
    """
    PhysNet encoder-decoder with built-in DiffNormalize + FrequencyAttention.

    時序流程：
      T  →  MaxpoolSpa    →  FreqAtt  →  MaxpoolSpaTem(T/2)
        →  ConvBlock4~9   →  upsample×2(T)  →  poolspa  →  output
    """

    def __init__(self, cfg=None, frames: int = 128, fps: int = 30,
                 freq_att_pool_size: int = 8, freq_att_temporal_kernel: int = 7,
                 freq_low: float = 0.7, freq_high: float = 2.5,
                 freq_att_sharpness: float = 10.0,
                 spo2_min: float = 85.0, spo2_range: float = 15.0,
                 use_channel_attn: bool = True):
        super().__init__()

        if cfg is not None:
            frames                = cfg.RPPG_MODEL.FRAMES
            fps                   = cfg.RPPG_MODEL.FPS
            freq_att_pool_size    = getattr(cfg.RPPG_MODEL, 'FREQ_ATT_POOL_SIZE', 8)
            freq_att_temporal_kernel = cfg.RPPG_MODEL.FREQ_ATT_TEMPORAL_KERNEL
            freq_low              = cfg.RPPG_MODEL.FREQ_LOW
            freq_high             = cfg.RPPG_MODEL.FREQ_HIGH
            freq_att_sharpness    = getattr(cfg.RPPG_MODEL, 'FREQ_ATT_SHARPNESS', 10.0)
            spo2_min              = cfg.RPPG_MODEL.SPO2_MIN
            spo2_range            = cfg.RPPG_MODEL.SPO2_RANGE

        self.frames         = frames
        self.spo2_min       = spo2_min
        self.spo2_range     = spo2_range
        self.use_channel_attn = use_channel_attn

        # ── 0. 內建差分正規化（不含可學習參數） ──
        self.diff_norm = DiffNormalizeLayer()

        # ── Encoder blocks ──
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
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [5, 1, 1], stride=1, padding=[2, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(inplace=True),
        )

        # ── Residual connections ──
        self.residual_conv2 = nn.Conv3d(16, 32, 1)
        self.residual_conv4 = nn.Conv3d(64, 64, 1)
        self.residual_conv6 = nn.Conv3d(64, 64, 1)
        self.residual_conv8 = nn.Conv3d(64, 64, 1)

        # ── Channel attention (SE-block) ──
        if self.use_channel_attn:
            self.channel_attention  = ChannelAttention3D(64, reduction=4)
            self.channel_attention1 = ChannelAttention3D(64, reduction=4)
            self.channel_attention2 = ChannelAttention3D(64, reduction=4)
            self.channel_attention3 = ChannelAttention3D(64, reduction=4)

        # ── Frequency attention ──
        self.att_mask1 = FrequencyAttention(
            num_input_channels=64, frames=frames, fps=fps,
            pool_size=freq_att_pool_size,
            temporal_kernel_size=freq_att_temporal_kernel,
            freq_low=freq_low, freq_high=freq_high,
            sharpness=freq_att_sharpness,
        )

        # ── HR branch: 時序 upsample + final conv ──
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1],
                               stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1],
                               stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU(),
        )
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1])

        # ── Pooling ──
        self.poolspa      = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.MaxpoolSpa   = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # ── SpO2 branch ──
        self.spo2_branch_conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.BatchNorm3d(32), nn.ELU(inplace=True),
        )
        self.spo2_branch_conv2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=[3, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(16), nn.ELU(inplace=True),
        )
        self.spo2_branch_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # ── SpO2 fusion head: rPPG(1) + VPG(1) + APG(1) + branch(16) = 19 ──
        self.spo2_fusion_head = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            ChannelAttention1D(in_channels=32, reduction=8),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor,
                x2: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x2 保留參數名稱以便與舊 checkpoint 相容，但不使用
        B, C, T, H, W = x.shape

        rppg_wave, spo2_features = self.encode_video(x)
        rppg_wave = rppg_wave.view(B, T)                        # [B, T]

        # VPG / APG（從 rPPG 差分計算）
        rppg_1d = rppg_wave.unsqueeze(1)                        # [B, 1, T]
        vpg = F.pad(torch.diff(rppg_1d, n=1, dim=-1), (0, 1), mode='replicate')
        apg = F.pad(torch.diff(vpg,     n=1, dim=-1), (0, 1), mode='replicate')

        # SpO2 融合
        spo2_1d = spo2_features.view(B, 16, T)                 # [B, 16, T]
        fused   = torch.cat([rppg_1d, vpg, apg, spo2_1d], dim=1)  # [B, 19, T]
        spo2_pred = self.spo2_fusion_head(fused)                # [B, 1]
        spo2_pred = torch.clamp(
            spo2_pred * self.spo2_range + self.spo2_min,
            min=self.spo2_min,
            max=self.spo2_min + self.spo2_range,
        )

        return rppg_wave, spo2_pred

    def encode_video(self, x: torch.Tensor):
        # ── 0. 差分正規化 ──
        x = self.diff_norm(x)                   # [B, 3, T, H, W]  raw → diff-norm

        # ── Spatial encoder ──
        x = self.ConvBlock1(x)                  # [B, 16, T, H,   W  ]
        x = self.MaxpoolSpa(x)                  # [B, 16, T, H/2, W/2]

        residual2 = self.residual_conv2(x)
        x = self.ConvBlock2(x) + residual2      # [B, 32, T, H/2, W/2]

        x = self.ConvBlock3(x)                  # [B, 64, T, H/2, W/2]
        x = self.att_mask1(x)                   # FrequencyAttention
        x = self.MaxpoolSpaTem(x)               # [B, 64, T/2, H/4, W/4]  ← 時序壓縮

        # ── 時空混合 blocks ──
        residual4 = self.residual_conv4(x)
        x = self.ConvBlock4(x) + residual4
        if self.use_channel_attn:
            x = self.channel_attention(x)
        x = self.ConvBlock5(x)
        if self.use_channel_attn:
            x = self.channel_attention1(x)

        residual6 = self.residual_conv6(x)
        x = self.ConvBlock6(x) + residual6
        if self.use_channel_attn:
            x = self.channel_attention2(x)
        xb = self.ConvBlock7(x)                 # SpO2 branch 分叉點

        # ── HR branch: 時序精煉 + upsample 還原 ──
        residual8 = self.residual_conv8(xb)
        x = self.ConvBlock8(xb) + residual8
        if self.use_channel_attn:
            x = self.channel_attention3(x)
        x = self.ConvBlock9(x)

        x = self.upsample(x)                    # [B, 64, T, H/4, W/4]
        x = self.upsample2(x)                   # [B, 64, T, H/4, W/4]  ← 時序還原
        x = self.poolspa(x)                     # [B, 64, T, 1,   1  ]
        rppg_wave = self.ConvBlock10(x)         # [B, 1,  T, 1,   1  ]

        # ── SpO2 branch ──
        spo2 = self.spo2_branch_conv1(xb)
        spo2 = self.spo2_branch_conv2(spo2)
        spo2 = self.spo2_branch_pool(spo2)      # [B, 16, T, 1, 1]

        return rppg_wave, spo2


def create_rppg_model(cfg) -> FCAtt_v2:
    return FCAtt_v2(cfg=cfg)
