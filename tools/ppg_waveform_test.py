"""
PPG 波形品質測試 — 比較微調前後的 V2 模型
對 UBFC 測試集隨機抽樣，計算 Pearson r 並繪製波形對比圖。

用法：
    python tools/ppg_waveform_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt

# ── Config ────────────────────────────────────────────────────────────────────
UBFC_DIR       = "D:/PreprocessedData_UBFC"
CKPT_PRETRAIN  = "checkpoints/rppg_semi_V3/rppg_fcatt_v3_128f_72px_best.pth"
CKPT_FINETUNE  = "checkpoints/rppg_finetune_v3/rppg_fcatt_v3_128f_72px_finetune_best.pth"
OUT_IMAGE      = "ppg_waveform_test.png"
N_SAMPLES      = 6        # 對比樣本數
FPS            = 30
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def bandpass(signal: np.ndarray, fs: float = 30.0,
             low: float = 0.7, high: float = 2.5, order: int = 3) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def diff_normalize_bvp(bvp: np.ndarray) -> np.ndarray:
    """Raw BVP → DiffNormalized（與 DataLoader 一致）。"""
    bvp = bvp.astype(np.float32)
    diff = bvp[1:] - bvp[:-1]
    denom = np.abs(bvp[1:]) + np.abs(bvp[:-1]) + 1e-7
    norm = diff / denom
    norm = norm / (norm.std() + 1e-7)
    return np.concatenate([[0.0], np.clip(norm, -5., 5.)]).astype(np.float32)


def integrate(pred_dn: np.ndarray) -> np.ndarray:
    """DiffNorm 預測 → 近似 raw BVP（cumsum + zscore 消除 DC drift）。"""
    return zscore(np.cumsum(pred_dn))


def load_model(ckpt_path: str, cfg_obj) -> torch.nn.Module:
    from models.rppg_model_v3 import FCAtt_v3
    model = FCAtt_v3(cfg=cfg_obj).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict(model: torch.nn.Module, inp: np.ndarray) -> np.ndarray:
    """inp: (T, H, W, C) uint8 → pred: (T,) float"""
    # T×H×W×C → C×T×H×W, add batch
    x = torch.from_numpy(inp).permute(3, 0, 1, 2).float()  # (C,T,H,W)
    x = x.unsqueeze(0).to(DEVICE)                           # (1,C,T,H,W)
    with torch.no_grad():
        wave, _ = model(x)
    return wave.squeeze(0).cpu().numpy()  # (T,)


# ── Build minimal cfg ─────────────────────────────────────────────────────────
from configs.defaults import get_config

# Load config using defaults + override (no yaml needed, just defaults)
cfg = get_config()
# Unfreeze to patch manually
cfg.defrost()
cfg.RPPG_MODEL.NAME                  = "FCAtt_v3"
cfg.RPPG_MODEL.FRAMES                = 128
cfg.RPPG_MODEL.FPS                   = FPS
cfg.RPPG_MODEL.INPUT_SIZE            = 72
cfg.RPPG_MODEL.FREQ_ATT_POOL_SIZE    = 8
cfg.RPPG_MODEL.FREQ_ATT_TEMPORAL_KERNEL = 5
cfg.RPPG_MODEL.FREQ_ATT_SHARPNESS   = 10.0
cfg.RPPG_MODEL.NUM_FREQ_BLOCKS       = 3
cfg.RPPG_MODEL.FREQ_LOW              = 0.7
cfg.RPPG_MODEL.FREQ_HIGH             = 2.5
cfg.RPPG_MODEL.SPO2_MIN              = 85.0
cfg.RPPG_MODEL.SPO2_RANGE            = 15.0
cfg.freeze()

# ── Load models ───────────────────────────────────────────────────────────────
print(f"[Device] {DEVICE}")
print(f"[Load] Pre-train: {CKPT_PRETRAIN}")
model_pre = load_model(CKPT_PRETRAIN, cfg)
print(f"[Load] Fine-tune: {CKPT_FINETUNE}")
model_ft  = load_model(CKPT_FINETUNE, cfg)

# ── Sample UBFC chunks ────────────────────────────────────────────────────────
all_inputs = sorted(glob.glob(os.path.join(UBFC_DIR, "*_input.npy")))
samples    = random.sample(all_inputs, min(N_SAMPLES, len(all_inputs)))
print(f"[Data] Sampled {len(samples)} chunks from {UBFC_DIR}")

# ── Inference + Stats ─────────────────────────────────────────────────────────
results = []
rs_pre  = []
rs_ft   = []

for inp_path in samples:
    bvp_path = inp_path.replace("_input.npy", "_bvp.npy")
    inp     = np.load(inp_path)   # (128,72,72,3) uint8
    bvp_raw = np.load(bvp_path)   # (128,) float32  raw BVP

    # Ground truth — two representations
    bvp_dn  = diff_normalize_bvp(bvp_raw)   # DiffNorm GT（用於計算 r）
    bvp_disp = zscore(bvp_raw)              # raw GT（用於視覺展示）

    # Predictions（DiffNorm domain）
    pred_pre_dn = predict(model_pre, inp)
    pred_ft_dn  = predict(model_ft,  inp)

    # Pearson r：在相同的 DiffNorm 域比較（正確比較）
    r_pre, _ = pearsonr(zscore(pred_pre_dn), zscore(bvp_dn))
    r_ft,  _ = pearsonr(zscore(pred_ft_dn),  zscore(bvp_dn))

    # Display：積分回 raw BVP domain，便於和 GT 比較（視覺直觀）
    pre_disp    = integrate(pred_pre_dn)
    ft_disp     = integrate(pred_ft_dn)
    pre_disp_bp = zscore(bandpass(pre_disp))
    ft_disp_bp  = zscore(bandpass(ft_disp))

    rs_pre.append(r_pre)
    rs_ft.append(r_ft)

    results.append({
        'name':       os.path.basename(inp_path).replace("_input.npy", ""),
        'bvp':        bvp_disp,      # raw GT for display
        'pre':        pre_disp,      # integrated prediction
        'ft':         ft_disp,
        'pre_bp':     pre_disp_bp,   # bandpass of integrated
        'ft_bp':      ft_disp_bp,
        'r_pre':      r_pre,         # correct DiffNorm-domain r
        'r_ft':       r_ft,
    })

print(f"\n{'':=<60}")
print(f"  Pre-train  Pearson r : mean={np.mean(rs_pre):+.4f}  std={np.std(rs_pre):.4f}")
print(f"  Fine-tune  Pearson r : mean={np.mean(rs_ft):+.4f}  std={np.std(rs_ft):.4f}")
print(f"{'':=<60}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
t = np.arange(128) / FPS  # seconds

fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(16, N_SAMPLES * 2.2),
                          facecolor='#0a0a0a')
fig.suptitle(
    f"PPG Waveform Quality: Pre-train vs Fine-tune (UBFC, n={N_SAMPLES})\n"
    f"Pre-train mean r={np.mean(rs_pre):+.4f}  |  "
    f"Fine-tune mean r={np.mean(rs_ft):+.4f}",
    color='white', fontsize=13, y=1.01,
)

COL_GT   = '#888888'
COL_PRE  = '#ff5555'
COL_FT   = '#00ff88'

for i, res in enumerate(results):
    for j, (pred, pred_bp, r, label, col) in enumerate([
        (res['pre'],  res['pre_bp'],  res['r_pre'], 'Pre-train', COL_PRE),
        (res['ft'],   res['ft_bp'],   res['r_ft'],  'Fine-tune', COL_FT),
    ]):
        ax = axes[i, j]
        ax.set_facecolor('#050505')
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#222')

        ax.plot(t, res['bvp'], color=COL_GT,  lw=0.9, alpha=0.8, label='BVP GT (raw)')
        ax.plot(t, pred,       color=col,     lw=0.8, alpha=0.5, label=f'Pred (integrated)')
        ax.plot(t, pred_bp,    color=col,     lw=1.4,             label=f'Pred (bp filtered)')

        ax.set_title(
            f"{res['name']}  [{label}]  r={r:+.4f}",
            color='white', fontsize=8, pad=3,
        )
        ax.set_xlim(0, 128 / FPS)
        ax.set_xlabel("Time (s)", color='#555', fontsize=7)
        ax.set_ylabel("Norm.", color='#555', fontsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right',
                      facecolor='#111', edgecolor='#333',
                      labelcolor='white')

plt.tight_layout()
plt.savefig(OUT_IMAGE, dpi=130, bbox_inches='tight',
            facecolor='#0a0a0a')
print(f"[Saved] {OUT_IMAGE}")
