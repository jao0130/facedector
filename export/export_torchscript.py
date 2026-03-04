"""
Export models to TorchScript format for deployment without Python.

Usage:
    python export/export_torchscript.py --config configs/face_detection.yaml --model face --output export/face.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.defaults import get_config
from trainers.base_trainer import make_face_ckpt_name, make_rppg_ckpt_name


def export_face_torchscript(cfg, output_path: str):
    """Export face detector to TorchScript via tracing."""
    from models.face_detector import FaceDetector

    model = FaceDetector(cfg=cfg)
    if cfg.FACE_MODEL.WEIGHTS:
        checkpoint = torch.load(cfg.FACE_MODEL.WEIGHTS, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
    model.eval()

    input_size = cfg.FACE_MODEL.INPUT_SIZE
    dummy = torch.randn(1, 3, input_size, input_size)

    # Use scripting for dict output
    scripted = torch.jit.trace(model, dummy, strict=False)
    scripted.save(output_path)

    print(f"[TorchScript] Face detector exported to {output_path}")
    print(f"[TorchScript] Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


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
            args.output = os.path.join('export', make_face_ckpt_name(cfg, tag="best").replace('.pth', '.pt'))
        else:
            args.output = os.path.join('export', make_rppg_ckpt_name(cfg, tag="best").replace('.pth', '.pt'))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.model == 'face':
        export_face_torchscript(cfg, args.output)
    else:
        print("[TorchScript] rPPG model export uses same pattern. Not yet implemented.")


if __name__ == '__main__':
    main()
