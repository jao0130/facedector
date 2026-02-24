"""
Unified inference entry point: face detection, rPPG, or real-time pipeline.

Usage:
    # MediaPipe face detection on webcam (default, recommended)
    python inference.py --config configs/face_detection.yaml --mode face --input 0

    # MediaPipe face detection on image
    python inference.py --config configs/face_detection.yaml --mode face --input test.jpg

    # Real-time rPPG pipeline (MediaPipe + FCAtt)
    python inference.py --config configs/face_detection.yaml --mode realtime

    # Custom trained detector (for comparison/debug)
    python inference.py --config configs/face_detection.yaml --mode face --detector custom --input 0
"""

import argparse
import sys
import os
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.defaults import get_config


def run_face_inference(cfg, input_source, output_path=None, use_mediapipe=True):
    """Run face detection on image or webcam."""
    from inference.face_inference import AsyncCapture

    is_image = os.path.isfile(input_source)

    if use_mediapipe:
        from inference.mediapipe_face import MediaPipeFaceDetector
        # IMAGE mode for single images, VIDEO mode for webcam/video
        detector = MediaPipeFaceDetector(cfg=cfg, video_mode=not is_image)
        draw_fn = MediaPipeFaceDetector.draw_detection
        print("[Mode] MediaPipe face detection")
    else:
        from inference.face_inference import FaceDetectorInference
        detector = FaceDetectorInference(cfg)
        draw_fn = FaceDetectorInference.draw_detection
        print("[Mode] Custom trained face detection")

    # Image file
    if os.path.isfile(input_source):
        image = cv2.imread(input_source)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = detector.predict(image_rgb)
        result = draw_fn(image, pred)

        if output_path:
            cv2.imwrite(output_path, result)
            print(f"[Face] Saved to {output_path}")
        else:
            cv2.imshow("Face Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if use_mediapipe:
            detector.close()
        return

    # Video / webcam
    source = int(input_source) if input_source.isdigit() else input_source
    cap = AsyncCapture(source, resolution=(cfg.INFERENCE.WEBCAM_W, cfg.INFERENCE.WEBCAM_H))

    fps_ema = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = detector.predict(frame_rgb)
            draw_fn(frame, pred)

            elapsed = time.perf_counter() - t0
            fps_ema = 0.1 * (1.0 / max(elapsed, 1e-6)) + 0.9 * fps_ema
            cv2.putText(frame, f"FPS: {fps_ema:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show detection info
            conf = pred['confidence']
            if conf > 0:
                bbox = pred['bbox']
                iou_w = bbox[2] - bbox[0]
                iou_h = bbox[3] - bbox[1]
                cv2.putText(frame, f"Face: {iou_w:.0%}x{iou_h:.0%}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "No face", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if use_mediapipe:
            detector.close()
        cv2.destroyAllWindows()


def run_realtime(cfg):
    """Run unified real-time pipeline (MediaPipe face + rPPG)."""
    from inference.realtime_pipeline import RealtimePipeline
    pipeline = RealtimePipeline(cfg)
    pipeline.run()


def run_benchmark(cfg, input_source: str, num_frames: int = 200, use_mediapipe: bool = True):
    """Run FPS benchmark."""
    from inference.face_inference import AsyncCapture

    if use_mediapipe:
        from inference.mediapipe_face import MediaPipeFaceDetector
        detector = MediaPipeFaceDetector(cfg=cfg)
        name = "MediaPipe"
    else:
        from inference.face_inference import FaceDetectorInference
        detector = FaceDetectorInference(cfg)
        name = "Custom"

    source = int(input_source) if input_source.isdigit() else input_source
    cap = AsyncCapture(source, resolution=(cfg.INFERENCE.WEBCAM_W, cfg.INFERENCE.WEBCAM_H))

    # Wait for first frame
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    times = []
    detected = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        pred = detector.predict(frame_rgb)
        times.append(time.perf_counter() - t0)

        if pred['confidence'] > 0:
            detected += 1

        if (i + 1) % 50 == 0:
            avg = sum(times[-50:]) / 50
            print(f"  [{i+1}/{num_frames}] Avg: {avg*1000:.1f}ms ({1/avg:.0f} FPS)")

    cap.release()
    if use_mediapipe:
        detector.close()

    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    p50 = sorted(times_ms)[len(times_ms) // 2]
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]

    print(f"\n[Benchmark] {name} Face Detection | {num_frames} frames")
    print(f"  Avg: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")
    print(f"  P50: {p50:.1f}ms | P95: {p95:.1f}ms")
    print(f"  Detection rate: {detected}/{len(times)} ({detected/len(times)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Unified inference: face / rppg / realtime")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--mode', type=str, choices=['face', 'rppg', 'realtime'],
                        default='face', help='Inference mode')
    parser.add_argument('--detector', type=str, choices=['mediapipe', 'custom'],
                        default='mediapipe', help='Face detector backend')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (overrides config WEIGHTS)')
    parser.add_argument('--input', type=str, default='0', help='Input source (webcam id, image, video)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--benchmark', action='store_true', help='Run FPS benchmark')
    parser.add_argument('--benchmark-frames', type=int, default=200, help='Frames for benchmark')
    parser.add_argument('--opts', nargs='+', default=[], help='Override config options')
    args = parser.parse_args()

    # Inject --model into opts
    if args.model:
        if args.mode in ('face', 'realtime'):
            args.opts.extend(['FACE_MODEL.WEIGHTS', args.model])
        elif args.mode == 'rppg':
            args.opts.extend(['RPPG_MODEL.WEIGHTS', args.model])

    cfg = get_config(args.config, args.opts if args.opts else None)
    use_mp = (args.detector == 'mediapipe')

    if args.benchmark:
        run_benchmark(cfg, args.input, args.benchmark_frames, use_mp)
    elif args.mode == 'face':
        run_face_inference(cfg, args.input, args.output, use_mp)
    elif args.mode == 'realtime':
        run_realtime(cfg)
    else:
        print(f"[Error] Mode '{args.mode}' not yet implemented.")
        sys.exit(1)


if __name__ == '__main__':
    main()
