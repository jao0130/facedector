"""Smoke test for FCAtt_v2 model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.rppg_model_v2 import FCAtt_v2

model = FCAtt_v2(frames=128)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Single input test
x1 = torch.randn(2, 3, 128, 72, 72)
with torch.no_grad():
    rppg, spo2 = model(x1)
print(f"rppg shape: {rppg.shape}")
print(f"spo2 shape: {spo2.shape}")
print(f"spo2 range: [{spo2.min().item():.2f}, {spo2.max().item():.2f}]")
assert rppg.shape == (2, 128), f"Expected (2,128), got {rppg.shape}"
assert spo2.shape == (2, 1), f"Expected (2,1), got {spo2.shape}"
assert spo2.min().item() >= 85.0, f"SpO2 min {spo2.min().item()} < 85"
assert spo2.max().item() <= 100.0, f"SpO2 max {spo2.max().item()} > 100"
print("Single input: PASS")

# Dual input test (FusionNet)
x2 = torch.randn(2, 3, 128, 72, 72)
with torch.no_grad():
    rppg2, spo2_2 = model(x1, x2)
print(f"fusion rppg shape: {rppg2.shape}")
print(f"fusion spo2 shape: {spo2_2.shape}")
assert rppg2.shape == (2, 128)
assert spo2_2.shape == (2, 1)
print("Dual input (FusionNet): PASS")

# Test with dropout + channel attn flags
model_full = FCAtt_v2(frames=128, use_dropout=True, use_channel_attn=True)
model_full.eval()
with torch.no_grad():
    rppg3, spo2_3 = model_full(x1)
assert rppg3.shape == (2, 128)
print("Dropout + ChannelAttn flags: PASS")

model_minimal = FCAtt_v2(frames=128, use_dropout=False, use_channel_attn=False)
model_minimal.eval()
with torch.no_grad():
    rppg4, spo2_4 = model_minimal(x1)
assert rppg4.shape == (2, 128)
print("No dropout + No ChannelAttn: PASS")

print("\nAll tests passed!")
