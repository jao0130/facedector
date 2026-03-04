"""
Export face detection and rPPG models to ONNX format.

Usage:
    python export/export_onnx.py --config configs/face_detection.yaml --model face --output export/face.onnx
    python export/export_onnx.py --config configs/rppg_semisupervised.yaml --model rppg --output export/rppg_v2.onnx
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.defaults import get_config
from trainers.base_trainer import make_face_ckpt_name, make_rppg_ckpt_name


def export_face_onnx(cfg, output_path: str):
    """Export face detector to ONNX."""
    from models.face_detector import FaceDetector

    model = FaceDetector(cfg=cfg)
    if cfg.FACE_MODEL.WEIGHTS:
        checkpoint = torch.load(cfg.FACE_MODEL.WEIGHTS, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
    model.eval()

    input_size = cfg.FACE_MODEL.INPUT_SIZE
    dummy = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=['image'],
        output_names=['bbox', 'landmarks', 'confidence'],
        dynamic_axes={'image': {0: 'batch'}, 'bbox': {0: 'batch'},
                      'landmarks': {0: 'batch'}, 'confidence': {0: 'batch'}},
        opset_version=17,
    )
    print(f"[ONNX] Face detector exported to {output_path}")
    print(f"[ONNX] Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def patch_frequency_attention(model):
    """
    Monkey-patch FrequencyAttention modules in-place for ONNX export.

    Removes:
      - torch.fft.rfft / irfft           (not supported in ONNX opset 17)
      - AdaptiveAvgPool3d((None, ps, ps)) (dynamic output size unsupported)

    Keeps:
      - score_refiner weights  (temporal attention, trained weights reused)
      - spatial_score_refiner weights (spatial attention, trained weights reused)
      - Same residual attention formula: 1 + tanh(pattern + energy - 1)
      - F.interpolate back to original H, W (fixed size, ONNX-friendly)

    The original model source files are NOT modified.
    """
    from models.rppg_modules_v2 import FrequencyAttention

    class FrequencyAttentionONNX(nn.Module):
        def __init__(self, orig: FrequencyAttention):
            super().__init__()
            self.ph = orig.pool_size_h  # 8
            self.pw = orig.pool_size_w  # 8

            # Reuse trained weights from original module
            self.score_refiner = orig.score_refiner
            self.spatial_score_refiner = orig.spatial_score_refiner

            # AdaptiveAvgPool2d on spatial dims only — ONNX supports this
            self.pool_spatial = nn.AdaptiveAvgPool2d((self.ph, self.pw))

        def forward(self, x):
            B, C, T, H, W = x.shape

            # ── Step 1: Spatial pool [B,C,T,H,W] → [B,C,T,ph,pw] ──────────
            # Merge B*T so we can use 2D pooling (ONNX supports AdaptiveAvgPool2d)
            x_bt = x.reshape(B * T, C, H, W)
            x_pooled_bt = self.pool_spatial(x_bt)          # [B*T, C, ph, pw]
            x_pooled = x_pooled_bt.reshape(B, C, T, self.ph, self.pw)

            # ── Step 2: Temporal attention (score_refiner reuses trained weights) ──
            temporal_scores = self.score_refiner(x_pooled)  # [B, 1, T, ph, pw]
            pattern_map_probs = torch.sigmoid(temporal_scores)

            # ── Step 3: Spatial attention proxy (mean over T replaces FFT power ratio) ──
            ch_mean = x_pooled.mean(dim=2)                  # [B, C, ph, pw]
            spatial_score = self.spatial_score_refiner(ch_mean)   # [B, 1, ph, pw]
            energy_map_probs = torch.sigmoid(spatial_score).unsqueeze(2)  # [B,1,1,ph,pw]

            # ── Step 4: Residual attention fusion (same formula as original) ──
            att_map = 1.0 + torch.tanh(pattern_map_probs + energy_map_probs - 1.0)
            # [B, 1, T, ph, pw]

            # ── Step 5: Upsample to original H, W ───────────────────────────
            att_bt = att_map.reshape(B * T, 1, self.ph, self.pw)
            att_up = F.interpolate(att_bt, size=(H, W), mode='bilinear', align_corners=False)
            att_up = att_up.reshape(B, 1, T, H, W)

            return x * att_up

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, FrequencyAttention):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], FrequencyAttentionONNX(module))
            replaced += 1
            print(f"[ONNX] Patched FrequencyAttention '{name}' — FFT removed, weights reused")

    print(f"[ONNX] Total patched: {replaced} FrequencyAttention module(s)")
    return model


def export_rppg_onnx(cfg, output_path: str):
    """Export FCAtt_v2 rPPG model to ONNX."""
    from models.rppg_model_v2 import FCAtt_v2

    model = FCAtt_v2(cfg=cfg)
    if cfg.RPPG_MODEL.WEIGHTS:
        checkpoint = torch.load(cfg.RPPG_MODEL.WEIGHTS, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        # strict=False because FrequencyAttention pool/fft params are removed
        model.load_state_dict(state, strict=False)
        print(f"[ONNX] Loaded weights from {cfg.RPPG_MODEL.WEIGHTS}")
    model.eval()

    # Patch FFT modules for ONNX export (model source files are NOT changed)
    model = patch_frequency_attention(model)

    frames = cfg.RPPG_MODEL.FRAMES
    size   = cfg.RPPG_MODEL.INPUT_SIZE
    dummy  = torch.randn(1, 3, frames, size, size)

    print(f"[ONNX] Exporting with input shape: {list(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=['video'],
            output_names=['rppg_wave', 'spo2'],
            opset_version=17,
        )

    print(f"[ONNX] rPPG model exported to {output_path}")
    print(f"[ONNX] Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['face', 'rppg'], required=True)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path. Auto-generated from config if not specified.')
    args = parser.parse_args()

    cfg = get_config(args.config)

    if args.output is None:
        if args.model == 'face':
            args.output = os.path.join('export', make_face_ckpt_name(cfg, tag="best").replace('.pth', '.onnx'))
        else:
            args.output = os.path.join('export', make_rppg_ckpt_name(cfg, tag="best").replace('.pth', '.onnx'))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.model == 'face':
        export_face_onnx(cfg, args.output)
    else:
        export_rppg_onnx(cfg, args.output)


if __name__ == '__main__':
    main()