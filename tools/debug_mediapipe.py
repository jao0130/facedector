"""
Standalone MediaPipe debug tool — shows ALL 478 landmarks + derived 5-point + bbox.
Uses VIDEO mode with temporal smoothing for stable tracking.

Usage:
    python tools/debug_mediapipe.py                # webcam (default)
    python tools/debug_mediapipe.py --input test.jpg  # image file
    python tools/debug_mediapipe.py --input 0 --no-mesh  # webcam, 5-point only

Controls:
    q: quit
    m: toggle mesh (all 478 landmarks)
    b: toggle bbox
    5: toggle 5-point landmarks
"""

import argparse
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# 5-point landmark indices
_LEFT_IRIS = 468
_RIGHT_IRIS = 473
_NOSE_TIP = 1
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

# Face oval contour indices
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

_GRACE_FRAMES = 10


def find_model():
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task'),
        'models/face_landmarker.task',
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("face_landmarker.task not found in models/")


def extract_5_landmarks(face_lm):
    num_lm = len(face_lm)
    if num_lm > _LEFT_IRIS:
        le_x, le_y = face_lm[_LEFT_IRIS].x, face_lm[_LEFT_IRIS].y
    else:
        li, lo = face_lm[_LEFT_EYE_INNER], face_lm[_LEFT_EYE_OUTER]
        le_x, le_y = (li.x + lo.x) / 2, (li.y + lo.y) / 2
    if num_lm > _RIGHT_IRIS:
        re_x, re_y = face_lm[_RIGHT_IRIS].x, face_lm[_RIGHT_IRIS].y
    else:
        ri, ro = face_lm[_RIGHT_EYE_INNER], face_lm[_RIGHT_EYE_OUTER]
        re_x, re_y = (ri.x + ro.x) / 2, (ri.y + ro.y) / 2
    nose = face_lm[_NOSE_TIP]
    lm = face_lm[_LEFT_MOUTH]
    rm = face_lm[_RIGHT_MOUTH]
    return [
        ('LE', le_x, le_y), ('RE', re_x, re_y),
        ('N', nose.x, nose.y), ('LM', lm.x, lm.y), ('RM', rm.x, rm.y),
    ]


def landmarks_to_bbox(face_lm, padding=0.2):
    xs = [lm.x for lm in face_lm]
    ys = [lm.y for lm in face_lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    return (max(0.0, x_min - w * padding), max(0.0, y_min - h * padding),
            min(1.0, x_max + w * padding), min(1.0, y_max + h * padding))


def draw_debug(frame, face_lm, show_mesh=True, show_bbox=True, show_5pt=True):
    """Draw full debug visualization on BGR frame. face_lm can be None."""
    h, w = frame.shape[:2]

    if face_lm is None:
        cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return

    num_lm = len(face_lm)
    cv2.putText(frame, f"Landmarks: {num_lm}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if show_mesh:
        for i, lm in enumerate(face_lm):
            px, py = int(lm.x * w), int(lm.y * h)
            if i >= 468:
                color, radius = (255, 255, 0), 2
            elif i in _FACE_OVAL:
                color, radius = (0, 200, 0), 2
            else:
                color, radius = (100, 100, 100), 1
            cv2.circle(frame, (px, py), radius, color, -1)
        pts = [(int(face_lm[idx].x * w), int(face_lm[idx].y * h)) for idx in _FACE_OVAL]
        pts.append(pts[0])
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 200, 0), 1)

    if show_bbox:
        x1, y1, x2, y2 = landmarks_to_bbox(face_lm)
        bx1, by1, bx2, by2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    if show_5pt:
        five = extract_5_landmarks(face_lm)
        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
        for i, (name, lx, ly) in enumerate(five):
            px, py = int(lx * w), int(ly * h)
            cv2.circle(frame, (px, py), 5, colors[i], -1)
            cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
            cv2.putText(frame, name, (px + 8, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)


def run_image(image_path, model_path):
    """Run on a single image (IMAGE mode)."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot read image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    face_lm = result.face_landmarks[0] if result.face_landmarks else None
    draw_debug(image, face_lm)

    if face_lm:
        h, w = image.shape[:2]
        five = extract_5_landmarks(face_lm)
        print(f"\n[Debug] Image: {image_path} ({w}x{h}), landmarks: {len(face_lm)}")
        for name, lx, ly in five:
            print(f"  {name}: ({lx:.4f}, {ly:.4f}) -> pixel ({int(lx * w)}, {int(ly * h)})")
    else:
        print("[Debug] No face detected!")

    cv2.imshow("MediaPipe Debug", image)
    cv2.waitKey(0)
    landmarker.close()
    cv2.destroyAllWindows()


def run_webcam(source, model_path, show_mesh=True):
    """Run on webcam / video (VIDEO mode + grace period)."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    show_bbox = True
    show_5pt = True
    fps_ema = 0.0
    timestamp_ms = 0
    last_face_lm = None
    miss_count = 0

    print("[Debug] MediaPipe face debug started (VIDEO mode + grace period).")
    print("  Controls: q=quit, m=mesh, b=bbox, 5=5-point landmarks")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms += 33

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            cv2.putText(frame, f"Error: {e}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            result = type('obj', (object,), {'face_landmarks': []})()

        if result.face_landmarks:
            last_face_lm = result.face_landmarks[0]
            miss_count = 0
            status = "LIVE"
        else:
            miss_count += 1
            if miss_count <= _GRACE_FRAMES and last_face_lm is not None:
                status = f"GRACE ({miss_count}/{_GRACE_FRAMES})"
            else:
                last_face_lm = None
                status = "LOST"

        draw_debug(frame, last_face_lm, show_mesh, show_bbox, show_5pt)

        # FPS
        elapsed = time.perf_counter() - t0
        fps_ema = 0.1 * (1.0 / max(elapsed, 1e-6)) + 0.9 * fps_ema
        cv2.putText(frame, f"FPS: {fps_ema:.0f} ({elapsed*1000:.1f}ms)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Status + toggles
        status_color = (0, 255, 0) if status == "LIVE" else (0, 165, 255) if "GRACE" in status else (0, 0, 255)
        cv2.putText(frame, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        cv2.imshow("MediaPipe Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mesh = not show_mesh
        elif key == ord('b'):
            show_bbox = not show_bbox
        elif key == ord('5'):
            show_5pt = not show_5pt

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe face debug visualizer")
    parser.add_argument('--input', type=str, default='0', help='Webcam ID or image path')
    parser.add_argument('--no-mesh', action='store_true', help='Start with mesh off')
    args = parser.parse_args()

    model_path = find_model()
    print(f"[Debug] Model: {model_path}")

    input_source = args.input
    if os.path.isfile(input_source):
        run_image(input_source, model_path)
    else:
        source = int(input_source) if input_source.isdigit() else input_source
        run_webcam(source, model_path, show_mesh=not args.no_mesh)


if __name__ == '__main__':
    main()
