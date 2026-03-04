# FCATT v2 — 新架構設計規格
> 脫離 PhysNet 框架，以 FrequencyAttention 為核心主幹

---

## 一、設計哲學（必讀）

### 輸入性質
- 輸入是**差分影像**（`frame[t] - frame[t-1]`）
- 每幀代表的是像素值的**時間變化量**，不是原始 RGB

### 核心假設
> 「觀察每個空間點沿時間方向的像素值變化量，進而判斷哪些空間位置應該被關注」

### 由此產生的架構約束
- 空間 kernel **必須是 1×1**（不能讓鄰近像素污染彼此）
- **不能用 2D CNN** 做特徵提取（會在空間上混合訊號）
- **空間聚合放在最後**，特徵提取過程中盡量維持空間獨立性
- `FrequencyAttention` 是**主幹**，不是插件

---

## 二、可用模組（來自 rppg_modules_v2.py）

以下模組已實作完成，直接 import 使用：

```python
from rppg_modules_v2 import (
    FrequencyAttention,    # 核心頻域注意力模組
    FusionNet,             # 雙流融合（暫不使用）
    ChannelAttention1D,    # 1D 通道注意力（用於 SpO2 融合頭）
    ChannelAttention3D,    # 3D 通道注意力
)
```

### FrequencyAttention v2 的關鍵改進
1. **可學習軟頻率遮罩**：`freq_center` 和 `freq_width` 是可訓練參數，初始化在 0.7-2.5 Hz
2. **Residual attention**：`1 + tanh(pattern + energy - 1)`，輸出範圍 [0,2]，中心為 1
3. **Power ratio**：心率頻段能量 / 總能量，跨場景更穩健
4. **Time-varying**：每個時間步都有獨立的空間注意力圖
5. **pool_size=8**：比 v1 更輕量

---

## 三、新架構：FCATT_v2

### 整體結構

```
輸入差分影像 [B, 3, T, H, W]
      ↓
SpatialStemEncoder          # 通道升維，空間適度降採樣
[B, 64, T, H/4, W/4]
      ↓
FrequencyAttentionBlock × N  # 主幹，重複堆疊
      ↓
      ├──────────────────────────────┐
   HR 分支                     SpO2 分支
   TemporalRefiner             spo2_conv1-2
   SpatialPool                 spo2_pool
   ConvBlock10                     │
   rPPG wave [B, T]           [融合頭]
      │                        rPPG+VPG+APG+SpO2
      └──────────────────────→ SpO2 pred [B, 1]
```

---

## 四、各模組實作細節

### 4.1 SpatialStemEncoder

目的：把 3 通道升維到 64 通道，同時做適度空間降採樣。
約束：空間 kernel 盡量小（3×3 可以，但要搭配 stride 降採樣而非混合時序）。

```python
class SpatialStemEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 注意：kernel 的時序維度必須是 1（不混時序）
        self.stem = nn.Sequential(
            # 第一層：通道升維，空間小 kernel
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 空間 /2

            # 第二層：繼續升維
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),

            # 第三層：升到 64
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 空間 /2
        )
        # 最終空間：H/4, W/4；時序：T（不動）

    def forward(self, x):
        return self.stem(x)
```

**為什麼時序 kernel=1？**
因為 SpatialStemEncoder 只負責空間特徵升維，時序建模的責任交給 FrequencyAttention，分工明確。

---

### 4.2 FrequencyAttentionBlock（可堆疊的單元）

把 FrequencyAttention + Residual connection 包成一個可重複堆疊的 Block：

```python
class FrequencyAttentionBlock(nn.Module):
    def __init__(self, channels, frames=128, fps=30, pool_size=8,
                 temporal_kernel_size=3, M=16):
        super().__init__()
        self.freq_att = FrequencyAttention(
            num_input_channels=channels,
            frames=frames,
            fps=fps,
            pool_size=pool_size,
            temporal_kernel_size=temporal_kernel_size,
            M_intermediate_channels=M,
        )
        # 1×1×1 Conv 做通道混合（空間不動）
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        # FrequencyAttention 輸出加權後的 x
        att_out = self.freq_att(x)
        # pointwise 做通道混合
        out = self.pointwise(att_out)
        # Residual connection
        return out + x
```

堆疊 3 個 Block：
```python
self.freq_blocks = nn.ModuleList([
    FrequencyAttentionBlock(channels=64, frames=frames, fps=fps)
    for _ in range(3)
])
```

---

### 4.3 HR 分支（TemporalRefiner）

FrequencyAttention 已經做了頻域過濾，HR 分支只需要做時序精煉和空間聚合：

```python
class HRBranch(nn.Module):
    def __init__(self, frames):
        super().__init__()
        # 時序精煉（空間 kernel=1，只動時序）
        self.temporal_refiner = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        # Upsample 恢復原始時序長度（如果有被降採樣）
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(64, 64, kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        # 最後才做空間聚合
        self.spatial_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.output_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.temporal_refiner(x)
        x = self.upsample(x)
        x = self.spatial_pool(x)        # 這才是空間聚合的時機
        x = self.output_conv(x)
        return x  # [B, 1, T, 1, 1]
```

---

### 4.4 SpO2 分支（維持 v1 設計）

```python
class SpO2Branch(nn.Module):
    def __init__(self, frames):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(32), nn.ELU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(16), nn.ELU(inplace=True))
        self.pool = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x  # [B, 16, T, 1, 1]
```

---

### 4.5 SpO2 融合頭（維持 v1 設計）

VPG 和 APG 從 rPPG 差分計算，不需要額外分支：

```python
# 在 forward 裡計算
vpg = F.pad(torch.diff(rppg.unsqueeze(1), n=1, dim=-1), (0,1), mode='replicate')
apg = F.pad(torch.diff(vpg, n=1, dim=-1), (0,1), mode='replicate')

# 融合：1(rPPG) + 1(VPG) + 1(APG) + 16(SpO2分支) = 19 通道
fused = torch.cat([rppg.unsqueeze(1), vpg, apg,
                   spo2_feat.view(B, 16, T)], dim=1)  # [B, 19, T]
```

融合頭：
```python
self.spo2_fusion_head = nn.Sequential(
    nn.Conv1d(19, 32, kernel_size=3, padding=1),
    nn.BatchNorm1d(32), nn.ELU(),
    ChannelAttention1D(in_channels=32, reduction=8),
    nn.Conv1d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm1d(32), nn.ELU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
# 輸出映射到生理範圍
spo2_pred = spo2_pred * 15.0 + 85.0  # [85, 100]
```

---

## 五、完整模型類別

```python
class FCATT_v2(nn.Module):
    def __init__(self, frames=128, fps=30):
        super().__init__()
        self.frames = frames

        # 空間升維（只動空間，不動時序）
        self.stem = SpatialStemEncoder()

        # 主幹：頻域注意力堆疊（這是 FCATT 的核心貢獻）
        self.freq_blocks = nn.ModuleList([
            FrequencyAttentionBlock(channels=64, frames=frames, fps=fps)
            for _ in range(3)
        ])

        # HR 分支
        self.hr_branch = HRBranch(frames=frames)

        # SpO2 分支
        self.spo2_branch = SpO2Branch(frames=frames)

        # SpO2 融合頭
        self.spo2_fusion_head = nn.Sequential(...)  # 見 4.5

    def forward(self, x):
        B, C, T, H, W = x.shape

        # 1. 空間升維
        x = self.stem(x)                    # [B, 64, T, H/4, W/4]

        # 2. 頻域注意力主幹
        for block in self.freq_blocks:
            x = block(x)                    # [B, 64, T, H/4, W/4]

        # 3. 分支
        rppg_feat = self.hr_branch(x)       # [B, 1, T, 1, 1]
        spo2_feat = self.spo2_branch(x)     # [B, 16, T, 1, 1]

        # 4. rPPG 輸出
        rppg_wave = rppg_feat.view(B, T)    # [B, T]

        # 5. VPG / APG
        vpg = F.pad(torch.diff(rppg_wave.unsqueeze(1), n=1, dim=-1), (0,1), mode='replicate')
        apg = F.pad(torch.diff(vpg, n=1, dim=-1), (0,1), mode='replicate')

        # 6. SpO2 融合
        fused = torch.cat([
            rppg_wave.unsqueeze(1), vpg, apg,
            spo2_feat.view(B, 16, T)
        ], dim=1)                           # [B, 19, T]
        spo2_pred = self.spo2_fusion_head(fused)
        spo2_pred = spo2_pred * 15.0 + 85.0

        return rppg_wave, spo2_pred
```

---

## 六、與 PhysNet 的差異對照

| 面向 | PhysNet | FCATT v2 |
|------|---------|----------|
| 核心模組 | 3D Conv 堆疊 | FrequencyAttentionBlock 堆疊 |
| 頻域利用 | 無 | 可學習軟頻率遮罩 |
| 空間處理 | 早期空間混合 | 最後才聚合 |
| 時序建模 | 3D Conv 隱含 | FFT 顯式頻域分析 |
| 空間注意力 | 無 | 每幀獨立注意力圖 |
| 參數量 | ~0.77M | ~2.1M |
| SpO2 輸出 | 無 | 有（VPG+APG 融合） |

---

## 七、參數設定

| 參數 | 值 |
|------|---|
| frames (T) | 128 |
| fps | 30 |
| pool_size | 8 |
| M_intermediate_channels | 16 |
| temporal_kernel_size | 3 |
| FrequencyAttentionBlock 數量 | 3 |
| freq_center 初始值 | 1.6 Hz（可學習）|
| freq_width 初始值 | 0.9 Hz（可學習）|
| sharpness | 10.0 |

---

## 八、實作注意事項

1. `rppg_modules_v2.py` 已有 `FrequencyAttention`, `ChannelAttention1D`, `ChannelAttention3D`, `FusionNet`，直接 import
2. 需要新實作的類別：`SpatialStemEncoder`, `FrequencyAttentionBlock`, `HRBranch`, `SpO2Branch`, `FCATT_v2`
3. **空間 kernel 1×1 的約束是最重要的**，實作時請特別注意 `SpatialStemEncoder` 以外的所有模組
4. SpO2 分支的輸入來自 `freq_blocks` 之後，與 HR 分支共享同一個特徵圖
5. Upsample 的倍率要跟 `SpatialStemEncoder` 的時序降採樣倍率對應（如果有的話）

---

*使用此文件時請同時提供 `rppg_modules_v2.py`，Claude Code 需要讀取其中的模組實作。*
