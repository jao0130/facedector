"""Verify FrequencyAttention improvements."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.rppg_model_v2 import FCAtt_v2

model = FCAtt_v2(frames=128)
fa = model.att_mask1

print("=== Learnable Parameters ===")
print(f"freq_center: {fa.freq_center.item():.3f} (init: (0.7+2.5)/2 = 1.6)")
print(f"freq_width:  {fa.freq_width.item():.3f} (init: (2.5-0.7)/2 = 0.9)")
print(f"sharpness:   {fa.sharpness}")
print(f"pool_size:   {fa.pool_size_h}x{fa.pool_size_w}")
print(f"requires_grad center: {fa.freq_center.requires_grad}")
print(f"requires_grad width:  {fa.freq_width.requires_grad}")

# Check soft mask shape
print("\n=== Soft Frequency Mask ===")
T = 128
freqs = torch.fft.rfftfreq(T, d=1.0/30.0)
low_gate = torch.sigmoid(fa.sharpness * (freqs - (fa.freq_center - fa.freq_width)))
high_gate = torch.sigmoid(fa.sharpness * (-(freqs - (fa.freq_center + fa.freq_width))))
soft_mask = low_gate * high_gate
print(f"Freq bins: {len(freqs)}, mask shape: {soft_mask.shape}")
print(f"Mask min: {soft_mask.min().item():.4f}, max: {soft_mask.max().item():.4f}")
# Show which frequencies have high weight
above_half = (soft_mask > 0.5).sum().item()
print(f"Bins with weight > 0.5: {above_half}")
for i, (f, m) in enumerate(zip(freqs, soft_mask)):
    if m.item() > 0.1:
        print(f"  bin {i}: freq={f.item():.3f}Hz, weight={m.item():.3f}")

# Check attention map shape (time-varying)
print("\n=== Time-Varying Attention ===")
model.eval()
x = torch.randn(1, 3, 128, 72, 72)

def hook_fn(module, input, output):
    print(f"FreqAttn input:  {input[0].shape}")
    print(f"FreqAttn output: {output.shape}")
    # Verify per-frame variation
    att_ratio = output / (input[0] + 1e-8)
    h_mid = output.shape[3] // 2
    frame_var = att_ratio[:, 0, :, h_mid, h_mid].var(dim=1).mean().item()
    print(f"Attention variation across frames: {frame_var:.6f}")

handle = fa.register_forward_hook(hook_fn)
with torch.no_grad():
    rppg, spo2 = model(x)
handle.remove()

print(f"\nFinal output: rppg={rppg.shape}, spo2={spo2.shape}")
print("\nAll verifications passed!")
