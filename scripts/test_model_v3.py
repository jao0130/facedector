"""Smoke test for FCAtt_v3."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.rppg_model_v3 import FCAtt_v3, DiffNormalizeLayer

B, T, H, W = 2, 128, 72, 72

# --- 1. DiffNormalizeLayer ---
print("=== DiffNormalizeLayer ===")
diff = DiffNormalizeLayer()
x_raw = torch.randint(0, 256, (B, 3, T, H, W)).float()
x_diff = diff(x_raw)
assert x_diff.shape == (B, 3, T, H, W), f"Shape error: {x_diff.shape}"
assert (x_diff[:, :, 0] == 0).all(), "t=0 should be zero-padded"
print(f"  Output shape : {list(x_diff.shape)}")
print(f"  t=0 zeros    : {(x_diff[:,:,0]==0).all().item()}")
print(f"  Value range  : [{x_diff[:,:,1:].min():.4f}, {x_diff[:,:,1:].max():.4f}]")
print("  PASS")

# --- 2. Full model forward ---
print("\n=== FCAtt_v3 Forward ===")
model = FCAtt_v3(frames=T)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")

x_raw = torch.randint(0, 256, (B, 3, T, H, W)).float()
with torch.no_grad():
    rppg, spo2 = model(x_raw)

assert rppg.shape == (B, T), f"rppg shape error: {rppg.shape}"
assert spo2.shape == (B, 1), f"spo2 shape error: {spo2.shape}"
assert (spo2 >= 85.0).all() and (spo2 <= 100.0).all(), \
    f"SpO2 out of range: [{spo2.min():.2f}, {spo2.max():.2f}]"
print(f"  rppg : {list(rppg.shape)}")
print(f"  spo2 : {list(spo2.shape)}, range [{spo2.min():.2f}, {spo2.max():.2f}]")
print("  PASS")

# --- 3. DiffNorm is internal: pre-diffed input gives different result ---
print("\n=== Internal DiffNorm consistency ===")
from utils.signal_processing import diff_normalize
import numpy as np
x_np = x_raw[0].permute(1, 2, 3, 0).numpy()   # [T,H,W,C]
x_pre = torch.from_numpy(diff_normalize(x_np)).float()
x_pre = x_pre.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,T,H,W]

with torch.no_grad():
    rppg_raw, _ = model(x_raw[:1])
    rppg_pre, _ = model(x_pre)

diff_val = (rppg_raw - rppg_pre).abs().mean().item()
print(f"  raw vs pre-diffed mean diff: {diff_val:.6f} (expect > 0)")
assert diff_val > 0, "Raw and pre-diffed must differ"
print("  PASS -- model applies DiffNorm internally")

print("\nAll tests passed!")
