"""rPPG evaluation metrics: HR and SpO2 estimation and comparison."""

import numpy as np
from scipy.stats import pearsonr
from utils.signal_processing import estimate_hr_fft


def calculate_rppg_metrics(pred_signals: list, gt_signals: list,
                           pred_spo2_list: list, gt_spo2_list: list,
                           fs: float = 30.0) -> dict:
    """
    Calculate HR and SpO2 metrics.

    Args:
        pred_signals: list of predicted rPPG signals (1D numpy arrays)
        gt_signals: list of ground truth BVP signals
        pred_spo2_list: list of predicted SpO2 values
        gt_spo2_list: list of ground truth SpO2 values
        fs: sampling frequency

    Returns:
        dict with metric names -> (mean, std) tuples.
    """
    pred_hrs = []
    gt_hrs = []
    for pred, gt in zip(pred_signals, gt_signals):
        pred_hrs.append(estimate_hr_fft(pred, fs))
        gt_hrs.append(estimate_hr_fft(gt, fs))

    pred_hrs = np.array(pred_hrs)
    gt_hrs = np.array(gt_hrs)
    hr_errors = pred_hrs - gt_hrs

    pred_spo2 = np.array(pred_spo2_list)
    gt_spo2 = np.array(gt_spo2_list)
    spo2_errors = pred_spo2 - gt_spo2

    metrics = {}

    # HR metrics
    metrics['HR_MAE'] = (np.mean(np.abs(hr_errors)), np.std(np.abs(hr_errors)))
    metrics['HR_RMSE'] = (np.sqrt(np.mean(hr_errors ** 2)), 0.0)
    if len(pred_hrs) > 2:
        r, _ = pearsonr(pred_hrs, gt_hrs)
        metrics['HR_Pearson'] = (r, 0.0)
    else:
        metrics['HR_Pearson'] = (0.0, 0.0)

    # SpO2 metrics
    if len(pred_spo2) > 0 and len(gt_spo2) > 0:
        metrics['SpO2_MAE'] = (np.mean(np.abs(spo2_errors)), np.std(np.abs(spo2_errors)))
        metrics['SpO2_RMSE'] = (np.sqrt(np.mean(spo2_errors ** 2)), 0.0)
        if len(pred_spo2) > 2:
            r, _ = pearsonr(pred_spo2, gt_spo2)
            metrics['SpO2_Pearson'] = (r, 0.0)
        else:
            metrics['SpO2_Pearson'] = (0.0, 0.0)

    return metrics
