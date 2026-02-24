"""Verify preprocessed NPY labels for UBFC, PURE, and MCD-rPPG."""

import os
import glob
import numpy as np
from numpy.fft import rfft, rfftfreq


def verify_dataset(name, data_dir, has_spo2=False, fps=30):
    print(f"\n{'='*60}")
    print(f"  {name}: {data_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(data_dir):
        print(f"  Directory not found!")
        return

    bvp_files = sorted(glob.glob(os.path.join(data_dir, "*_bvp.npy")))
    input_files = sorted(glob.glob(os.path.join(data_dir, "*_input.npy")))
    print(f"  BVP files: {len(bvp_files)}")
    print(f"  Input files: {len(input_files)}")

    if not bvp_files:
        return

    hr_estimates = []
    bad_chunks = 0

    for bvp_path in bvp_files:
        bvp = np.load(bvp_path)

        # FFT-based HR estimation
        fft_vals = np.abs(rfft(bvp - bvp.mean()))
        freqs = rfftfreq(len(bvp), d=1.0 / fps)
        mask = (freqs >= 0.7) & (freqs <= 3.5)
        if mask.any() and fft_vals[mask].max() > 0:
            peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
            hr = peak_freq * 60
            hr_estimates.append(hr)
            if hr < 40 or hr > 200:
                bad_chunks += 1
        else:
            bad_chunks += 1

    hr_arr = np.array(hr_estimates)
    print(f"\n  HR Statistics ({len(hr_arr)} chunks):")
    print(f"    Mean: {hr_arr.mean():.1f} BPM")
    print(f"    Std:  {hr_arr.std():.1f} BPM")
    print(f"    Min:  {hr_arr.min():.1f} BPM")
    print(f"    Max:  {hr_arr.max():.1f} BPM")
    print(f"    Bad chunks (HR<40 or >200): {bad_chunks}")

    # Sample 5 chunks for detail
    samples = bvp_files[:3] + bvp_files[len(bvp_files)//2:len(bvp_files)//2+1] + bvp_files[-1:]
    print(f"\n  Sample chunks:")
    for path in samples:
        name_str = os.path.basename(path).replace("_bvp.npy", "")
        bvp = np.load(path)
        inp_path = path.replace("_bvp.npy", "_input.npy")
        inp = np.load(inp_path)

        fft_vals = np.abs(rfft(bvp - bvp.mean()))
        freqs = rfftfreq(len(bvp), d=1.0 / fps)
        mask = (freqs >= 0.7) & (freqs <= 3.5)
        peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
        hr = peak_freq * 60

        spo2_str = ""
        spo2_path = path.replace("_bvp.npy", "_spo2.npy")
        if os.path.exists(spo2_path):
            spo2 = float(np.load(spo2_path))
            spo2_str = f", SpO2={spo2:.1f}"

        print(f"    {name_str}: input{inp.shape} bvp[{bvp.min():.2f}..{bvp.max():.2f}] "
              f"HR={hr:.0f}BPM{spo2_str}")


if __name__ == "__main__":
    verify_dataset("UBFC (Test)", "D:/PreprocessedData_UBFC", has_spo2=False)
    verify_dataset("PURE (Labeled)", "D:/PreprocessedData", has_spo2=True)
    verify_dataset("MCD-rPPG (Labeled)", "D:/PreprocessedData_MCD", has_spo2=True)
