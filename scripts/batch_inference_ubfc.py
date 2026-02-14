"""
UBFC 批次推理腳本。

對預處理後的 UBFC 測試集逐一執行推理，輸出標註影片與統計數據。

用法:
    # 全部 subjects
    python scripts/batch_inference_ubfc.py \
        --ubfc_dir /mnt/d/UBFC_processed \
        --model checkpoints/best_model.weights.h5 \
        --config configs/config_ubuntu.yaml \
        --output_dir results/ubfc

    # 指定 subjects
    python scripts/batch_inference_ubfc.py \
        --ubfc_dir /mnt/d/UBFC_processed \
        --model checkpoints/best_model.weights.h5 \
        --config configs/config_ubuntu.yaml \
        --output_dir results/ubfc \
        --subjects subject1 subject5

    # 只輸出統計（不存影片）
    python scripts/batch_inference_ubfc.py \
        --ubfc_dir /mnt/d/UBFC_processed \
        --model checkpoints/best_model.weights.h5 \
        --config configs/config_ubuntu.yaml \
        --output_dir results/ubfc \
        --stats_only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 確保 project root 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="UBFC 批次推理")
    parser.add_argument("--ubfc_dir", type=str, required=True, help="預處理後的 UBFC 目錄")
    parser.add_argument("--model", type=str, required=True, help="模型權重路徑")
    parser.add_argument("--config", type=str, required=True, help="配置檔路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="結果輸出目錄")
    parser.add_argument("--subjects", nargs="*", default=None, help="指定 subjects")
    parser.add_argument("--stats_only", action="store_true", help="只輸出統計，不存標註影片")
    args = parser.parse_args()

    ubfc_dir = Path(args.ubfc_dir)
    output_dir = Path(args.output_dir)

    if not ubfc_dir.exists():
        print(f"錯誤: UBFC 目錄不存在: {ubfc_dir}")
        sys.exit(1)

    # 延遲匯入 TF（避免不必要的 GPU 初始化）
    import tensorflow as tf
    from models.face_detector import create_face_detector

    # 載入配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_size = config.get('model', {}).get('input_size', 224)

    # 建立模型
    print("載入模型...")
    model = create_face_detector(config, pretrained=False)
    dummy = tf.zeros((1, input_size, input_size, 3))
    model(dummy, training=False)
    model.load_weights(args.model)
    print(f"模型已載入: {args.model}")

    # 找出 subjects
    if args.subjects:
        subjects = [ubfc_dir / s for s in args.subjects]
    else:
        subjects = sorted(
            [d for d in ubfc_dir.iterdir() if d.is_dir()],
            key=lambda p: int(p.name.replace("subject", "")),
        )

    if not subjects:
        print("錯誤: 找不到任何 subject 目錄")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n批次推理")
    print(f"  來源: {ubfc_dir}")
    print(f"  輸出: {output_dir}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  模式: {'統計' if args.stats_only else '標註影片 + 統計'}")
    print()

    all_stats = {}

    for idx, subject_dir in enumerate(subjects, 1):
        subject_name = subject_dir.name
        video_path = subject_dir / "vid.avi"

        if not video_path.exists():
            print(f"[{idx}/{len(subjects)}] {subject_name}: 跳過（無 vid.avi）")
            continue

        print(f"[{idx}/{len(subjects)}] {subject_name}: ", end="", flush=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("無法開啟影片")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 輸出影片
        writer = None
        if not args.stats_only:
            out_video = output_dir / f"{subject_name}_result.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        # 逐幀推理
        confidences = []
        bboxes = []
        frame_count = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (input_size, input_size))
            batch = np.expand_dims(resized.astype(np.float32) / 255.0, 0)

            preds = model(batch, training=False)
            bbox = preds['bbox'][0].numpy()
            landmarks = preds['landmarks'][0].numpy()
            conf = float(preds['confidence'][0].numpy().item())

            confidences.append(conf)
            bboxes.append(bbox.tolist())

            if writer:
                # 還原座標到原始尺寸
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                lm_orig = landmarks * np.array([[width, height]])
                colors = [
                    (255, 0, 0), (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255), (0, 0, 255),
                ]
                for i, (lx, ly) in enumerate(lm_orig.astype(int)):
                    cv2.circle(frame, (lx, ly), 3, colors[i], -1)

                cv2.putText(
                    frame, f"{conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                writer.write(frame)

            frame_count += 1

        elapsed = time.time() - t_start
        cap.release()
        if writer:
            writer.release()

        # 統計
        conf_arr = np.array(confidences)
        stats = {
            "frames": frame_count,
            "fps_processing": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
            "confidence_mean": round(float(conf_arr.mean()), 4),
            "confidence_std": round(float(conf_arr.std()), 4),
            "confidence_min": round(float(conf_arr.min()), 4),
            "confidence_max": round(float(conf_arr.max()), 4),
        }
        all_stats[subject_name] = stats

        print(
            f"{frame_count} frames, "
            f"{stats['fps_processing']} fps, "
            f"conf={stats['confidence_mean']:.3f}±{stats['confidence_std']:.3f}"
        )

    # 儲存統計
    stats_path = output_dir / "inference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # 總結
    print()
    print("=" * 60)
    print(f"推理完成: {len(all_stats)}/{len(subjects)} subjects")
    print(f"統計已存: {stats_path}")

    if all_stats:
        all_conf = [s["confidence_mean"] for s in all_stats.values()]
        all_fps = [s["fps_processing"] for s in all_stats.values()]
        print(f"整體信心分數: {np.mean(all_conf):.4f} ± {np.std(all_conf):.4f}")
        print(f"平均處理速度: {np.mean(all_fps):.1f} fps")


if __name__ == "__main__":
    main()
