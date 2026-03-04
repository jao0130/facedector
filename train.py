"""
Unified training entry point for face detection and rPPG.

Usage:
    python train.py --config configs/face_detection.yaml
    python train.py --config configs/rppg_training.yaml
    python train.py --config configs/face_detection.yaml --opts FACE_TRAIN.EPOCHS 10
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.defaults import get_config


def _run_all_tests(trainer, data_loaders: dict):
    """Run test() on every loader whose key starts with 'test_'."""
    test_keys = sorted(k for k in data_loaders if k.startswith('test_'))
    if not test_keys:
        if 'test' in data_loaders:
            # Legacy single test set with key 'test'
            print("\n[Test] Evaluating on test set...")
            trainer.test(data_loaders['test'])
        else:
            print("[Info] No test set configured.")
        return
    for key in test_keys:
        dataset_name = key[len('test_'):]  # strip "test_" prefix
        print(f"\n[Test] Evaluating on {dataset_name}...")
        trainer.test(data_loaders[key], dataset_name=dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Unified training: face detection + rPPG")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true', help='Run test evaluation only (skip training)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for --test mode')
    parser.add_argument('--opts', nargs='+', default=[], help='Override config options (KEY VALUE pairs)')
    args = parser.parse_args()

    cfg = get_config(args.config, args.opts if args.opts else None)

    print(f"[Config] Task: {cfg.TASK}")
    print(f"[Config] Device: {cfg.DEVICE}")

    if cfg.TASK == "face_detection":
        from data.dataset_face import create_face_dataloaders
        from trainers.face_trainer import FaceDetectorTrainer

        data_loaders = create_face_dataloaders(cfg)
        trainer = FaceDetectorTrainer(cfg)

        if args.resume:
            epoch = trainer.load_checkpoint(trainer.model, None, args.resume)
            print(f"[Resume] Loaded checkpoint from epoch {epoch}")

        history = trainer.train({'train': data_loaders['train'], 'val': data_loaders['val']})
        print(f"\n[Done] Training complete. Best val_loss in history.")

        # Run test if test set available
        if 'test' in data_loaders:
            print("\n[Test] Evaluating on 300W test set...")
            test_metrics = trainer.test(data_loaders['test'])
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")

    elif cfg.TASK == "rppg":
        from data.dataset_rppg import create_rppg_dataloaders
        from trainers.rppg_trainer import rPPGTrainer

        data_loaders = create_rppg_dataloaders(cfg)
        trainer = rPPGTrainer(cfg)

        history = trainer.train(data_loaders)
        print(f"\n[Done] rPPG training complete.")

    elif cfg.TASK == "rppg_semi":
        from data.dataset_rppg import create_rppg_semi_dataloaders
        from trainers.rppg_semi_trainer import rPPGSemiTrainer
        import torch

        data_loaders = create_rppg_semi_dataloaders(cfg)
        trainer = rPPGSemiTrainer(cfg)

        if args.test:
            # Test-only mode: load checkpoint and evaluate
            ckpt_path = args.checkpoint
            if ckpt_path is None:
                ckpt_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "rppg_fcatt_128f_72px_best.pth")
            if not os.path.isfile(ckpt_path):
                print(f"[Error] Checkpoint not found: {ckpt_path}")
                sys.exit(1)
            ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
            trainer.student.load_state_dict(ckpt['model_state_dict'])
            print(f"[Test] Loaded checkpoint: {ckpt_path}")

            _run_all_tests(trainer, data_loaders)
        else:
            history = trainer.train(data_loaders)
            print(f"\n[Done] Semi-supervised rPPG training complete.")
            _run_all_tests(trainer, data_loaders)

    elif cfg.TASK == "rppg_finetune":
        from data.dataset_rppg import create_rppg_dataloaders
        from trainers.rppg_waveform_finetune_trainer import rPPGWaveformFinetuneTrainer

        data_loaders = create_rppg_dataloaders(cfg)
        trainer = rPPGWaveformFinetuneTrainer(cfg)

        history = trainer.train(data_loaders)
        print(f"\n[Done] Waveform fine-tune complete.")
        _run_all_tests(trainer, data_loaders)

    elif cfg.TASK == "rppg_spo2_finetune":
        from data.dataset_rppg import create_rppg_spo2_dataloaders
        from trainers.rppg_spo2_finetune_trainer import rPPGSpO2FinetuneTrainer

        # MCD-rPPG (BVP + SpO2) → train (0.0-0.8) / val (0.8-1.0)
        # PURE subjects 01-10   → test only
        data_loaders = create_rppg_spo2_dataloaders(cfg)
        trainer = rPPGSpO2FinetuneTrainer(cfg)

        history = trainer.train(data_loaders)
        print(f"\n[Done] SpO2 fine-tune complete.")
        _run_all_tests(trainer, data_loaders)

    elif cfg.TASK == "rppg_joint":
        from data.dataset_rppg import create_rppg_joint_dataloaders
        from trainers.rppg_joint_trainer import rPPGJointTrainer
        import torch

        data_loaders = create_rppg_joint_dataloaders(cfg)
        trainer = rPPGJointTrainer(cfg)

        if args.test:
            ckpt_path = args.checkpoint
            if ckpt_path is None:
                # Default to best checkpoint in configured output dir
                import glob as _glob
                pattern = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "*joint_best*.pth")
                found   = sorted(_glob.glob(pattern))
                ckpt_path = found[-1] if found else None
            if not ckpt_path or not os.path.isfile(ckpt_path):
                print(f"[Error] Checkpoint not found: {ckpt_path}")
                sys.exit(1)
            ckpt  = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            trainer.student.load_state_dict(state, strict=False)
            print(f"[Test] Loaded: {ckpt_path}")
            _run_all_tests(trainer, data_loaders)
        else:
            history = trainer.train(data_loaders)
            print(f"\n[Done] Joint multi-task semi-supervised training complete.")
            _run_all_tests(trainer, data_loaders)

    else:
        print(f"[Error] Unknown task: {cfg.TASK}. "
              f"Use 'face_detection', 'rppg', 'rppg_semi', 'rppg_finetune', "
              f"'rppg_spo2_finetune', or 'rppg_joint'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
