"""
FCAtt_v3  —  全新架構，依照 FCATT_v2_SPEC.md 實作。

v3.1 新增模組（全部自訂，不依賴外部套件）：
  - TemporalShift (TSM)       : 零參數，SpatialStemEncoder 中賦予空間卷積時序感知
  - ChannelAttention3D        : SE-block，FrequencyAttentionBlock 中通道選擇
  - MambaBlock                : 純 PyTorch selective SSM，HRBranch 中全域時序建模

設計哲學：
  - 輸入原始 RGB，DiffNormalize 內建為第一層（Option B）
  - 空間 kernel 必須 1×1（SpatialStemEncoder 以外）
  - FrequencyAttention 是主幹，重複堆疊
  - 空間聚合放在最後，Mamba 在聚合後做全域時序

架構：
  Raw RGB [B,3,T,H,W]
      → DiffNormalizeLayer
      → SpatialStemEncoder (+ TSM)         [B, 64, T, H/4, W/4]
      → FrequencyAttentionBlock×3 (+ CAttn) [B, 64, T, H/4, W/4]
      ├─→ HRBranch (+ MambaBlock)  → rppg_wave [B, T]
      └─→ SpO2Branch + FusionHead  → spo2_pred [B, 1]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rppg_modules_v2 import (
    FrequencyAttention,
    ChannelAttention1D,
)


# ── 差分正規化層 ─────────────────────────────────────────────────────────────

class DiffNormalizeLayer(nn.Module):
    """
    內建差分正規化（rPPG-Toolbox 標準公式）：
      (frame[t] - frame[t-1]) / (|frame[t]| + |frame[t-1]| + ε)
    再除以整段 clip 的時序 std（全域標準化）。

    - 接受原始像素值（0-255 float 或 uint8.float()）
    - 第一幀 (t=0) 補零
    - 不含可學習參數，train/eval 行為一致
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        diff = x[:, :, 1:] - x[:, :, :-1]
        norm = diff / (x[:, :, 1:].abs() + x[:, :, :-1].abs() + 1e-7)
        norm = norm / (norm.std(dim=(2, 3, 4), keepdim=True) + 1e-7)
        norm = torch.clamp(norm, -5.0, 5.0)
        pad  = torch.zeros_like(x[:, :, :1])
        return torch.cat([pad, norm], dim=2)                    # [B, 3, T, H, W]


# ── Temporal Shift Module (TSM) ──────────────────────────────────────────────

class TemporalShift(nn.Module):
    """
    零參數時序移位（Lin et al., TSM 2019）。

    將 1/fold_div 通道往前移一幀（paste → present），
    將 1/fold_div 通道往後移一幀（future → present），
    讓純空間卷積獲得短程時序感知，不增加任何可學習參數。
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        fold = max(C // self.fold_div, 1)

        # 前移（past → present）: channels [0 : fold]
        x_fwd = torch.cat([
            torch.zeros_like(x[:, :fold,       :1,  :, :]),
            x[:, :fold, :-1, :, :],
        ], dim=2)                                               # [B, fold, T, H, W]

        # 後移（future → present）: channels [fold : 2*fold]
        x_bwd = torch.cat([
            x[:, fold:2*fold, 1:, :, :],
            torch.zeros_like(x[:, fold:2*fold, :1,  :, :]),
        ], dim=2)                                               # [B, fold, T, H, W]

        # 不移動的通道
        x_mid = x[:, 2*fold:, :, :, :]                         # [B, C-2*fold, T, H, W]

        return torch.cat([x_fwd, x_bwd, x_mid], dim=1)         # [B, C, T, H, W]


# ── Channel Attention 3D ──────────────────────────────────────────────────────

class ChannelAttention3D(nn.Module):
    """
    3D SE-block 通道注意力（avg + max pooling 融合）。
    讓 FrequencyAttentionBlock 能選擇與心跳節律相關的通道。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )                                                       # [B, C, 1, 1, 1]
        return x * attn


# ── Mamba Block（純 PyTorch 自訂 Selective SSM）──────────────────────────────

class MambaBlock(nn.Module):
    """
    Mamba Selective SSM Block — 忠實復現原論文邏輯（Gu & Dao, 2023）。
    不依賴 mamba-ssm CUDA extension，可在任何 PyTorch 環境執行。

    與原版完全一致的設計：
      1. dt_rank（低秩 Δ 投影）：dt_rank = ceil(channels/16) ≪ d_inner
         - x_proj: d_inner → (dt_rank + 2·d_state)，一次得到 Δ_raw、B、C
         - dt_proj: dt_rank → d_inner，展開至完整維度
      2. dt_proj.bias 初始化至 [dt_min, dt_max] log-uniform 範圍
         保證訓練初期 Δ 落在合理區間，避免梯度消失或爆炸
      3. D skip connection：y += D · u（學習殘差旁路）
      4. 離散化：Ā = exp(Δ·A)，B̄ = Δ·B（與 mamba-ssm 實作一致）
      5. Sequential scan（數學等價於 parallel scan，T=128 速度可接受）
      6. 閘控：y = SSM(u) · silu(z)

    Input / Output: [B, channels, T]
    """

    def __init__(self, channels: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        d_inner  = channels * expand
        dt_rank  = math.ceil(channels / 16)          # 低秩：64/16 = 4
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.dt_rank  = dt_rank

        # ── Pre-norm ──
        self.norm = nn.LayerNorm(channels)

        # ── 輸入投影（gated split）──
        self.in_proj  = nn.Linear(channels, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner,  channels,    bias=False)

        # ── Short causal depthwise conv ──
        self.dw_conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # ── 單一投影：d_inner → (dt_rank, B, C)（原版結構）──
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # ── Δ 展開：dt_rank → d_inner（低秩展開）──
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # dt_proj 初始化：weight ~ Uniform，bias = softplus⁻¹(dt)
        # 確保初始 Δ 落在 [dt_min, dt_max] log-uniform 範圍
        nn.init.uniform_(self.dt_proj.weight, -dt_rank ** -0.5, dt_rank ** -0.5)
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # softplus⁻¹(y) = log(exp(y) − 1) = y + log(1 − exp(−y))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ── A：對角衰減矩陣，HiPPO-inspired init（A_n = n）──
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone()
        )                                                        # [d_inner, d_state]

        # ── D：skip connection（可學習）──
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B_batch, C, T = x.shape
        residual = x

        # Pre-norm
        x_n = self.norm(x.permute(0, 2, 1))                     # [B, T, C]

        # Gated split
        xz   = self.in_proj(x_n)                                 # [B, T, 2·d_inner]
        x_in, z = xz.chunk(2, dim=-1)                            # [B, T, d_inner] each

        # Short causal depthwise conv + SiLU
        x_c = self.dw_conv(
            x_in.permute(0, 2, 1)                               # [B, d_inner, T]
        )[:, :, :T].permute(0, 2, 1)                            # [B, T, d_inner]
        x_c = F.silu(x_c)

        # SSM 參數（單一投影，原版結構）
        x_dbl = self.x_proj(x_c)                                 # [B, T, dt_rank+2·d_state]
        dt_raw, B_seq, C_seq = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))                    # [B, T, d_inner] > 0

        # A（永遠為負，保證衰減穩定）
        A = -torch.exp(self.A_log.float())                       # [d_inner, d_state]

        # 離散化：Ā = exp(Δ·A)，B̄ = Δ·B（與 mamba-ssm 實作一致）
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )                                                        # [B, T, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B_seq.unsqueeze(-2)             # [B, T, d_inner, d_state]

        # Sequential scan：h[t] = dA[t]·h[t-1] + dB[t]·u[t]
        u = x_c                                                  # [B, T, d_inner]
        h = torch.zeros(B_batch, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)  # [B, d_inner, d_state]
            y_t = (h * C_seq[:, t].unsqueeze(1)).sum(-1)           # [B, d_inner]
            ys.append(y_t)
        y_ssm = torch.stack(ys, dim=1)                           # [B, T, d_inner]

        # D skip connection（y += D · u）
        y_ssm = y_ssm + self.D * u                               # [B, T, d_inner]

        # 輸出閘控 + 投影
        y = y_ssm * F.silu(z)                                    # [B, T, d_inner]
        y = self.out_proj(y).permute(0, 2, 1)                   # [B, C, T]

        return y + residual


# ── 空間升維編碼器（含 TSM）────────────────────────────────────────────────

class SpatialStemEncoder(nn.Module):
    """
    3 → 64 通道，空間降採樣 /4，時序完全不動。
    只有此模組允許 spatial kernel > 1×1。
    每個空間卷積前插入 TSM，讓空間 conv 獲得短程時序感知。
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()

        self.tsm0 = TemporalShift(fold_div=fold_div)
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.tsm1 = TemporalShift(fold_div=fold_div)
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )

        self.tsm2 = TemporalShift(fold_div=fold_div)
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(self.tsm0(x)))   # [B, 16, T, H/2, W/2]
        x = self.conv2(self.tsm1(x))               # [B, 32, T, H/2, W/2]
        x = self.pool2(self.conv3(self.tsm2(x)))   # [B, 64, T, H/4, W/4]
        return x


# ── 頻域注意力 Block（含 ChannelAttention3D）────────────────────────────────

class FrequencyAttentionBlock(nn.Module):
    """
    FrequencyAttention
      → 1×1×1 pointwise Conv
      → ChannelAttention3D（通道選擇）
      → Residual
    """

    def __init__(self, channels: int = 64, frames: int = 128, fps: int = 30,
                 pool_size: int = 8, temporal_kernel_size: int = 3,
                 M: int = 16, freq_low: float = 0.7, freq_high: float = 2.5,
                 sharpness: float = 10.0, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate
        self.freq_att = FrequencyAttention(
            num_input_channels=channels,
            frames=frames, fps=fps,
            pool_size=pool_size,
            temporal_kernel_size=temporal_kernel_size,
            M_intermediate_channels=M,
            freq_low=freq_low, freq_high=freq_high,
            sharpness=sharpness,
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )
        self.channel_attn = ChannelAttention3D(channels, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stochastic Depth：訓練時按 drop_rate 機率直接回傳 identity
        if self.training and self.drop_rate > 0.0:
            if torch.rand(1, device=x.device).item() < self.drop_rate:
                return x
        out = self.freq_att(x)          # 頻域加權
        out = self.pointwise(out)       # 通道混合
        out = self.channel_attn(out)    # 通道選擇
        return out + x                  # Residual


# ── HR 分支（含 MambaBlock）──────────────────────────────────────────────────

class HRBranch(nn.Module):
    """
    局部時序精煉（3D conv）→ 空間聚合 → Mamba 全域時序建模 → 輸出 rPPG 波形。

    設計意圖：
      temporal_refiner : 局部時序（±1~2 幀），在全空間上執行
      spatial_pool     : 空間壓縮，之後純 1D 時序
      mamba            : 全域時序（整個 128 幀週期），捕捉心跳週期性
    """

    def __init__(self, frames: int = 128, channels: int = 64,
                 mamba_d_state: int = 16, mamba_expand: int = 2):
        super().__init__()
        self.frames = frames

        # 局部時序精煉（spatial kernel=1）
        # ELU：保留負值（對應 BVP 波形下降相），比 SiLU 在 x=-1 多傳遞 2.3× 負值資訊
        self.temporal_refiner = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, channels),
            nn.ELU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.GroupNorm(8, channels),
            nn.ELU(inplace=True),
        )

        # 空間聚合
        self.spatial_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # Mamba：全域時序建模（在 1D 時序上）
        self.mamba = MambaBlock(
            channels=channels,
            d_state=mamba_d_state,
            d_conv=4,
            expand=mamba_expand,
        )

        # Mamba 輸出後的 dropout（Mamba 看到完整時序，僅對輸出特徵正則化）
        self.dropout = nn.Dropout(p=0.1)

        # 輸出投影
        self.output_conv = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_refiner(x)               # [B, 64, T, H/4, W/4]
        x = self.spatial_pool(x)                   # [B, 64, T, 1,   1  ]

        # Mamba 在純時序上建模（輸入完整，不中斷狀態傳遞）
        x_1d = x.squeeze(-1).squeeze(-1)           # [B, 64, T]
        x_1d = self.mamba(x_1d)                    # [B, 64, T]
        x_1d = self.dropout(x_1d)                  # Mamba 之後才 dropout
        x = x_1d.unsqueeze(-1).unsqueeze(-1)       # [B, 64, T, 1, 1]

        return self.output_conv(x)                  # [B, 1,  T, 1, 1]


# ── SpO2 分支 ─────────────────────────────────────────────────────────────────

class SpO2Branch(nn.Module):
    """SpO2 特徵提取：spatial 3×3 → temporal 3×1×1 → 空間聚合。"""

    def __init__(self, frames: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv2(self.conv1(x)))  # [B, 16, T, 1, 1]


# ── 主模型 ────────────────────────────────────────────────────────────────────

class FCAtt_v3(nn.Module):
    """
    FCAtt v3.1 — TSM + ChannelAttention + Mamba 增強版。

    Input:  [B, 3, T, H, W]  raw uint8.float() 或 float 0-255
    Output: (rppg_wave: [B, T], spo2_pred: [B, 1])
    """

    def __init__(self, cfg=None, frames: int = 128, fps: int = 30,
                 num_freq_blocks: int = 3, pool_size: int = 8,
                 temporal_kernel_size: int = 3, M: int = 16,
                 freq_low: float = 0.7, freq_high: float = 2.5,
                 sharpness: float = 10.0,
                 spo2_min: float = 85.0, spo2_range: float = 15.0):
        super().__init__()

        if cfg is not None:
            frames               = cfg.RPPG_MODEL.FRAMES
            fps                  = cfg.RPPG_MODEL.FPS
            pool_size            = getattr(cfg.RPPG_MODEL, 'FREQ_ATT_POOL_SIZE', 8)
            temporal_kernel_size = cfg.RPPG_MODEL.FREQ_ATT_TEMPORAL_KERNEL
            freq_low             = cfg.RPPG_MODEL.FREQ_LOW
            freq_high            = cfg.RPPG_MODEL.FREQ_HIGH
            sharpness            = getattr(cfg.RPPG_MODEL, 'FREQ_ATT_SHARPNESS', 10.0)
            num_freq_blocks      = getattr(cfg.RPPG_MODEL, 'NUM_FREQ_BLOCKS', 3)
            spo2_min             = cfg.RPPG_MODEL.SPO2_MIN
            spo2_range           = cfg.RPPG_MODEL.SPO2_RANGE

        self.frames     = frames
        self.spo2_min   = spo2_min
        self.spo2_range = spo2_range

        # ── 0. 差分正規化 ──
        self.diff_norm = DiffNormalizeLayer()

        # ── 1. 空間升維（含 TSM） ──
        self.stem = SpatialStemEncoder(fold_div=8)

        # ── 2. 頻域注意力主幹（含 ChannelAttention3D + Stochastic Depth） ──
        # drop_rate 隨深度線性遞增（淺層保持穩定特徵，深層容忍更多跳過）
        max_drop = 0.15
        self.freq_blocks = nn.ModuleList([
            FrequencyAttentionBlock(
                channels=64, frames=frames, fps=fps,
                pool_size=pool_size,
                temporal_kernel_size=temporal_kernel_size,
                M=M, freq_low=freq_low, freq_high=freq_high,
                sharpness=sharpness,
                drop_rate=max_drop * (i + 1) / num_freq_blocks,
            )
            for i in range(num_freq_blocks)
        ])

        # ── 3. HR 分支（含 MambaBlock） ──
        self.hr_branch = HRBranch(frames=frames, channels=64)

        # ── 4. SpO2 分支 ──
        self.spo2_branch = SpO2Branch(frames=frames)

        # ── 5. SpO2 融合頭：1(rPPG) + 1(VPG) + 1(APG) + 16(SpO2) = 19 ──
        self.spo2_fusion_head = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            ChannelAttention1D(in_channels=32, reduction=8),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                x2: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x.shape

        # 0. 差分正規化
        x = self.diff_norm(x)                       # [B, 3, T, H, W]

        # 1. 空間升維
        x = self.stem(x)                            # [B, 64, T, H/4, W/4]

        # 2. 頻域注意力主幹
        for block in self.freq_blocks:
            x = block(x)                            # [B, 64, T, H/4, W/4]

        # 3. HR 分支
        rppg_feat = self.hr_branch(x)               # [B, 1, T, 1, 1]
        rppg_wave = rppg_feat.view(B, T)            # [B, T]

        # 4. SpO2 分支
        spo2_feat = self.spo2_branch(x)             # [B, 16, T, 1, 1]

        # 5. VPG / APG
        rppg_1d = rppg_wave.unsqueeze(1)
        vpg = F.pad(torch.diff(rppg_1d, n=1, dim=-1), (0, 1), mode='replicate')
        apg = F.pad(torch.diff(vpg,     n=1, dim=-1), (0, 1), mode='replicate')

        # 6. SpO2 融合
        spo2_1d   = spo2_feat.view(B, 16, T)
        fused     = torch.cat([rppg_1d, vpg, apg, spo2_1d], dim=1)  # [B, 19, T]
        spo2_pred = self.spo2_fusion_head(fused)                     # [B, 1]
        spo2_pred = spo2_pred * self.spo2_range + self.spo2_min

        return rppg_wave, spo2_pred
