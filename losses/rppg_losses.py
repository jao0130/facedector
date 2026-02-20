"""rPPG losses: Negative Pearson, Weighted MSE, and semi-supervised losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegPearsonLoss(nn.Module):
    """
    Negative Pearson correlation loss for BVP signals.
    Ported from rPPG-Toolbox PhysNetNegPearsonLoss.
    """

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [B, T] predicted rPPG signal
            labels: [B, T] ground truth BVP signal

        Returns:
            Scalar loss: mean(1 - pearson_r) across batch.
        """
        loss = torch.tensor(0.0, device=preds.device)
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i] * labels[i])
            sum_x2 = torch.sum(preds[i] ** 2)
            sum_y2 = torch.sum(labels[i] ** 2)
            N = preds.shape[1]

            numerator = N * sum_xy - sum_x * sum_y
            denominator = torch.sqrt(
                (N * sum_x2 - sum_x ** 2 + 1e-8) * (N * sum_y2 - sum_y ** 2 + 1e-8)
            )
            pearson = numerator / denominator
            loss = loss + (1 - pearson)

        return loss / preds.shape[0]


class WeightedMSELoss(nn.Module):
    """
    SpO2 loss: MSE weighted by (100 - mean_label)^2.
    Higher weight for lower SpO2 values (clinically critical).
    """

    def forward(self, pred_spo2: torch.Tensor, label_spo2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_spo2: [B] or [B, 1]
            label_spo2: [B] or [B, 1]

        Returns:
            Weighted MSE scalar.
        """
        pred_spo2 = pred_spo2.view(-1)
        label_spo2 = label_spo2.view(-1)
        mse = torch.mean((pred_spo2 - label_spo2) ** 2)
        weight = (100.0 - label_spo2.mean()) ** 2
        return mse * weight


class FrequencyConstraintLoss(nn.Module):
    """
    Self-supervised loss: encourages rPPG FFT energy to concentrate
    in the cardiac frequency band [freq_low, freq_high] Hz.

    Loss = 1 - (cardiac_band_power / total_power)
    """

    def __init__(self, fs: float = 30.0, freq_low: float = 0.75, freq_high: float = 2.5):
        super().__init__()
        self.fs = fs
        self.freq_low = freq_low
        self.freq_high = freq_high

    def forward(self, rppg_wave: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rppg_wave: [B, T] predicted rPPG signal

        Returns:
            Scalar loss.
        """
        T = rppg_wave.shape[1]
        fft = torch.fft.rfft(rppg_wave, dim=1)
        power = fft.real ** 2 + fft.imag ** 2  # [B, T//2+1]

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(rppg_wave.device)
        cardiac_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)

        cardiac_power = power[:, cardiac_mask].sum(dim=1)
        total_power = power.sum(dim=1) + 1e-8

        cardiac_ratio = cardiac_power / total_power
        return (1.0 - cardiac_ratio).mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Mean Teacher consistency loss: MSE between z-scored student
    and teacher rPPG predictions on the same unlabeled input.
    """

    def forward(self, student_wave: torch.Tensor, teacher_wave: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_wave: [B, T] student model output
            teacher_wave: [B, T] teacher model output (detached)

        Returns:
            Scalar MSE loss on z-scored signals.
        """
        s_norm = (student_wave - student_wave.mean(dim=1, keepdim=True)) / (student_wave.std(dim=1, keepdim=True).clamp(min=1e-6))
        t_norm = (teacher_wave - teacher_wave.mean(dim=1, keepdim=True)) / (teacher_wave.std(dim=1, keepdim=True).clamp(min=1e-6))
        return F.mse_loss(s_norm, t_norm)
