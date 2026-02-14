"""
UBFC-rPPG DATASET_2 預處理腳本。

將 raw AVI 轉換為 MJPEG 格式（OpenCV 4.x 無法直接讀取 raw AVI），
並驗證轉換結果。

用法:
    # 預處理全部 subjects
    python scripts/preprocess_ubfc.py --ubfc_dir /mnt/d/UBFC/DATASET_2 --output_dir /mnt/d/UBFC_processed

    # 只處理指定 subjects
    python scripts/preprocess_ubfc.py --ubfc_dir /mnt/d/UBFC/DATASET_2 --output_dir /mnt/d/UBFC_processed --subjects subject1 subject5 subject10

    # 跳過已轉換的（續跑）
    python scripts/preprocess_ubfc.py --ubfc_dir /mnt/d/UBFC/DATASET_2 --output_dir /mnt/d/UBFC_processed --skip_existing
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg():
    """確認 ffmpeg 可用。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> dict:
    """用 ffprobe 取得影片資訊。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {}

    parts = result.stdout.strip().split(",")
    if len(parts) < 5:
        return {}

    # r_frame_rate 是分數格式 "num/den"
    fps_parts = parts[3].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return {
        "codec": parts[0],
        "width": int(parts[1]),
        "height": int(parts[2]),
        "fps": round(fps, 2),
        "frames": int(parts[4]) if parts[4].isdigit() else 0,
    }


def convert_video(src: Path, dst: Path, quality: int = 2) -> bool:
    """
    將 raw AVI 轉換為 MJPEG AVI。

    Args:
        src: 來源影片路徑
        dst: 目標影片路徑
        quality: MJPEG 品質 (2=高品質, 10=低品質)

    Returns:
        是否成功
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "mjpeg",
        "-q:v", str(quality),
        str(dst),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0


def verify_conversion(src: Path, dst: Path) -> tuple:
    """
    驗證轉換結果：比對幀數。

    Returns:
        (success, message)
    """
    src_info = get_video_info(src)
    dst_info = get_video_info(dst)

    if not dst_info:
        return False, "無法讀取輸出檔案"

    if dst_info["codec"] != "mjpeg":
        return False, f"編碼器錯誤: {dst_info['codec']}"

    src_frames = src_info.get("frames", 0)
    dst_frames = dst_info.get("frames", 0)

    if src_frames > 0 and dst_frames > 0 and abs(src_frames - dst_frames) > 1:
        return False, f"幀數不符: {src_frames} → {dst_frames}"

    return True, f"OK ({dst_frames} frames, {dst_info['fps']}fps)"


def main():
    parser = argparse.ArgumentParser(description="UBFC DATASET_2 預處理：raw AVI → MJPEG AVI")
    parser.add_argument("--ubfc_dir", type=str, required=True, help="UBFC DATASET_2 根目錄")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出目錄")
    parser.add_argument("--subjects", nargs="*", default=None, help="指定處理的 subjects（預設全部）")
    parser.add_argument("--skip_existing", action="store_true", help="跳過已存在的轉換檔案")
    parser.add_argument("--quality", type=int, default=2, help="MJPEG 品質 (2=高, 10=低, 預設 2)")
    args = parser.parse_args()

    ubfc_dir = Path(args.ubfc_dir)
    output_dir = Path(args.output_dir)

    if not ubfc_dir.exists():
        print(f"錯誤: UBFC 目錄不存在: {ubfc_dir}")
        sys.exit(1)

    if not check_ffmpeg():
        print("錯誤: ffmpeg 未安裝，請執行: sudo apt install ffmpeg")
        sys.exit(1)

    # 找出所有 subjects
    if args.subjects:
        subjects = [ubfc_dir / s for s in args.subjects]
    else:
        subjects = sorted(ubfc_dir.iterdir(), key=lambda p: int(p.name.replace("subject", "")))
    subjects = [s for s in subjects if s.is_dir()]

    if not subjects:
        print("錯誤: 找不到任何 subject 目錄")
        sys.exit(1)

    print(f"UBFC 預處理")
    print(f"  來源: {ubfc_dir}")
    print(f"  輸出: {output_dir}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  品質: {args.quality}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0
    results = []

    for i, subject_dir in enumerate(subjects, 1):
        subject_name = subject_dir.name
        src_video = subject_dir / "vid.avi"
        src_gt = subject_dir / "ground_truth.txt"

        dst_subject_dir = output_dir / subject_name
        dst_video = dst_subject_dir / "vid.avi"
        dst_gt = dst_subject_dir / "ground_truth.txt"

        prefix = f"[{i}/{len(subjects)}] {subject_name}"

        if not src_video.exists():
            print(f"{prefix}: 跳過（無 vid.avi）")
            fail_count += 1
            results.append((subject_name, "SKIP", "no vid.avi"))
            continue

        if args.skip_existing and dst_video.exists():
            print(f"{prefix}: 已存在，跳過")
            skip_count += 1
            results.append((subject_name, "SKIP", "already exists"))
            continue

        dst_subject_dir.mkdir(parents=True, exist_ok=True)

        # 轉換影片
        src_info = get_video_info(src_video)
        frames_str = f"{src_info.get('frames', '?')} frames" if src_info else "?"
        print(f"{prefix}: 轉換中 ({frames_str})...", end=" ", flush=True)

        if convert_video(src_video, dst_video, args.quality):
            ok, msg = verify_conversion(src_video, dst_video)
            if ok:
                print(f"完成 - {msg}")
                success_count += 1
                results.append((subject_name, "OK", msg))
            else:
                print(f"驗證失敗 - {msg}")
                fail_count += 1
                results.append((subject_name, "FAIL", msg))
        else:
            print("轉換失敗")
            fail_count += 1
            results.append((subject_name, "FAIL", "ffmpeg error"))
            continue

        # 複製 ground_truth.txt
        if src_gt.exists():
            shutil.copy2(src_gt, dst_gt)

    # 摘要
    print()
    print("=" * 50)
    print(f"預處理完成")
    print(f"  成功: {success_count}")
    print(f"  跳過: {skip_count}")
    print(f"  失敗: {fail_count}")
    print(f"  總計: {len(subjects)}")

    if fail_count > 0:
        print()
        print("失敗項目:")
        for name, status, msg in results:
            if status == "FAIL":
                print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
