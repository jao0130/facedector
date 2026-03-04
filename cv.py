"""
Cross-validation runner for rppg_joint task.

每個 fold 完整訓練一次，使用不同的 20% validation set（PURE 始終只當測試集）。
訓練結束後自動彙整所有 fold 的 HR + SpO2 指標，輸出 mean ± std。

Usage:
    # 全部 5 fold 依序訓練（最常用）
    python cv.py --config configs/rppg_joint.yaml

    # 只測試（各 fold 已有 checkpoint）
    python cv.py --config configs/rppg_joint.yaml --test-only

    # 只跑指定 fold（例如繼續中斷的 fold）
    python cv.py --config configs/rppg_joint.yaml --fold 2

    # 覆蓋任意 config 參數
    python cv.py --config configs/rppg_joint.yaml --opts RPPG_JOINT.EPOCHS 50
"""

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── 單一 fold 訓練 + 測試 ──────────────────────────────────────────────────────

def run_fold(config_file: str, base_opts: list, fold_idx: int, n_folds: int,
             test_only: bool = False) -> dict:
    """
    訓練（或載入）單一 fold，回傳 {metric_name: value} 字典。

    Output 目錄按 fold 分層，避免不同 fold checkpoint 互蓋：
        checkpoints/rppg_joint/fold_0/
        logs/rppg_joint/fold_0/
    """
    from configs.defaults import get_config
    from data.dataset_rppg import create_rppg_joint_dataloaders
    from trainers.rppg_joint_trainer import rPPGJointTrainer

    # fold-specific config overrides（在 base_opts 之後，優先覆蓋）
    fold_opts = list(base_opts) + [
        'RPPG_JOINT.FOLD_IDX',  str(fold_idx),
        'RPPG_JOINT.N_FOLDS',   str(n_folds),
        'OUTPUT.CHECKPOINT_DIR', f"checkpoints/rppg_joint/fold_{fold_idx}",
        'OUTPUT.LOG_DIR',        f"logs/rppg_joint/fold_{fold_idx}",
    ]
    cfg = get_config(config_file, fold_opts)

    data_loaders = create_rppg_joint_dataloaders(cfg)
    trainer      = rPPGJointTrainer(cfg)

    if test_only:
        # 尋找此 fold 的 best checkpoint
        pattern = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "*joint_best*.pth")
        found   = sorted(glob.glob(pattern))
        if not found:
            print(f"[CV] Fold {fold_idx}: 找不到 checkpoint，跳過測試")
            return {}
        ckpt_path = found[-1]
        ckpt  = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        trainer.student.load_state_dict(state, strict=False)
        print(f"[CV] Fold {fold_idx}: 載入 {ckpt_path}")
    else:
        trainer.train(data_loaders)

    # 對所有 test_* loader 評估（通常只有 test_PURE / test）
    fold_metrics: dict = {}
    test_keys = sorted(k for k in data_loaders if k.startswith('test_'))
    if not test_keys and 'test' in data_loaders:
        test_keys = ['test']

    for key in test_keys:
        dataset_name = key[len('test_'):] if key.startswith('test_') else 'default'
        metrics = trainer.test(data_loaders[key], dataset_name=dataset_name)
        for metric_name, value in metrics.items():
            fold_metrics[f"{dataset_name}/{metric_name}"] = value

    return fold_metrics


# ── 彙整報告 ──────────────────────────────────────────────────────────────────

def _aggregate(all_fold_metrics: list) -> dict:
    """
    每個 metric 計算 mean ± std，跳過 NaN。

    Returns:
        {metric_key: {'mean': float, 'std': float, 'values': list}}
    """
    from collections import defaultdict

    collected = defaultdict(list)
    for fold_m in all_fold_metrics:
        for k, v in fold_m.items():
            if v == v:    # skip NaN
                collected[k].append(v)

    summary = {}
    for k, vals in collected.items():
        summary[k] = {
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals)),
            'values': [float(v) for v in vals],
        }
    return summary


def _print_summary(summary: dict, n_folds: int):
    """Pretty-print the CV summary table."""
    print(f"\n{'='*65}")
    print(f"  CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    print(f"{'='*65}")
    print(f"  {'Metric':<40}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}")
    for key in sorted(summary.keys()):
        s = summary[key]
        print(f"  {key:<40}  {s['mean']:8.4f}  {s['std']:8.4f}")
    print(f"{'='*65}\n")


def _save_results(all_fold_metrics: list, summary: dict, output_dir: str):
    """
    Saves:
        {output_dir}/cv_per_fold.json   — per-fold raw values
        {output_dir}/cv_summary.json    — mean ± std per metric
        {output_dir}/cv_summary.csv     — human-readable table
    """
    os.makedirs(output_dir, exist_ok=True)

    # per-fold JSON
    per_fold_path = os.path.join(output_dir, "cv_per_fold.json")
    with open(per_fold_path, 'w', encoding='utf-8') as f:
        json.dump(all_fold_metrics, f, indent=2)
    print(f"[CV] Per-fold results → {per_fold_path}")

    # summary JSON
    summary_json_path = os.path.join(output_dir, "cv_summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[CV] Summary JSON     → {summary_json_path}")

    # summary CSV
    summary_csv_path = os.path.join(output_dir, "cv_summary.csv")
    with open(summary_csv_path, 'w', encoding='utf-8') as f:
        # Header: metric, mean, std, fold_0, fold_1, ...
        n = len(all_fold_metrics)
        fold_cols = ','.join(f"fold_{i}" for i in range(n))
        f.write(f"metric,mean,std,{fold_cols}\n")
        for key in sorted(summary.keys()):
            s = summary[key]
            vals = ','.join(f"{v:.4f}" for v in s['values'])
            f.write(f"{key},{s['mean']:.4f},{s['std']:.4f},{vals}\n")
    print(f"[CV] Summary CSV      → {summary_csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated K-fold cross-validation for rppg_joint")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config (rppg_joint.yaml)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Override N_FOLDS (default: read from config)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run only this specific fold index (0-based)')
    parser.add_argument('--test-only', action='store_true',
                        help='Skip training, load existing checkpoints and test')
    parser.add_argument('--opts', nargs='+', default=[],
                        help='Override config options (KEY VALUE pairs)')
    args = parser.parse_args()

    # Read N_FOLDS from config (before fold-specific overrides)
    from configs.defaults import get_config
    base_cfg = get_config(args.config, args.opts if args.opts else None)
    n_folds  = args.folds or getattr(base_cfg.RPPG_JOINT, 'N_FOLDS', 5)

    # Determine which folds to run
    fold_range = [args.fold] if args.fold is not None else list(range(n_folds))

    all_fold_metrics: list = []

    for fold_idx in fold_range:
        print(f"\n{'='*65}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}"
              + ("  [TEST ONLY]" if args.test_only else ""))
        print(f"{'='*65}")

        metrics = run_fold(
            config_file=args.config,
            base_opts=args.opts,
            fold_idx=fold_idx,
            n_folds=n_folds,
            test_only=args.test_only,
        )
        all_fold_metrics.append(metrics)
        print(f"\n[CV] Fold {fold_idx} metrics: "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # Free GPU memory between folds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate and report (only when multiple folds were run)
    if len(all_fold_metrics) > 1:
        summary    = _aggregate(all_fold_metrics)
        output_dir = "logs/rppg_joint"
        _print_summary(summary, n_folds=len(all_fold_metrics))
        _save_results(all_fold_metrics, summary, output_dir)
    else:
        print(f"\n[CV] Single fold run — skipping aggregate summary")


if __name__ == '__main__':
    main()
