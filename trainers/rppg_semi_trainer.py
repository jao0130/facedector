"""
Semi-supervised rPPG trainer using Mean Teacher framework.

Student model trains with:
  - Supervised NegPearson loss on labeled data (PURE/UBFC)
  - Unsupervised consistency + frequency losses on unlabeled data (VoxCeleb2)

Teacher model is an EMA copy of Student (no gradient updates).
"""

import os
import copy
import math
from typing import Iterator
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_rppg_ckpt_name
from losses.rppg_losses import NegPearsonLoss, FrequencyConstraintLoss, TemporalConsistencyLoss


class rPPGSemiTrainer(BaseTrainer):
    """
    Mean Teacher semi-supervised trainer for rPPG (HR only).

    Architecture:
        Student (FCAtt) — trained with backprop
        Teacher (FCAtt) — EMA of Student, no backprop

    Loss:
        L = L_supervised + lambda(epoch) * (L_consistency + alpha * L_frequency)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Student model (select by config NAME)
        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.student = FCAtt_v3(cfg=cfg).to(self.device)
        elif model_name == 'FCAtt_v2':
            from models.rppg_model_v2 import FCAtt_v2
            self.student = FCAtt_v2(cfg=cfg).to(self.device)
        else:
            from models.rppg_model import FCAtt
            self.student = FCAtt(cfg=cfg).to(self.device)

        # Load pretrained weights if available
        weights_path = cfg.RPPG_MODEL.WEIGHTS
        if weights_path and os.path.isfile(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.student.load_state_dict(state_dict, strict=False)
            print(f"[Semi] Loaded pretrained weights: {weights_path}")

        # Teacher model (EMA copy, no gradients, always eval mode)
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Losses
        self.bvp_criterion = NegPearsonLoss()
        self.freq_criterion = FrequencyConstraintLoss(
            fs=cfg.RPPG_MODEL.FPS,
            freq_low=cfg.RPPG_MODEL.FREQ_LOW,
            freq_high=cfg.RPPG_MODEL.FREQ_HIGH,
        )
        self.consistency_criterion = TemporalConsistencyLoss()

        # Hyperparams
        self.ema_decay = cfg.RPPG_SEMI.EMA_DECAY
        self.lambda_max = cfg.RPPG_SEMI.LAMBDA_UNSUP
        self.ramp_up_epochs = cfg.RPPG_SEMI.RAMP_UP_EPOCHS
        self.freq_weight = cfg.RPPG_SEMI.FREQ_WEIGHT
        self.grad_accum = cfg.RPPG_TRAIN.GRAD_ACCUMULATION
        self.grad_clip = cfg.RPPG_TRAIN.GRAD_CLIP

    def train(self, data_loaders: dict) -> dict:
        train_loader = data_loaders['train']
        valid_loader = data_loaders.get('valid')
        unlabeled_loader = data_loaders.get('unlabeled')

        if unlabeled_loader is None:
            print("[Semi] WARNING: No unlabeled data. Falling back to supervised-only training.")

        total_epochs = self.cfg.RPPG_SEMI.EPOCHS
        patience = self.cfg.RPPG_SEMI.PATIENCE

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.cfg.RPPG_SEMI.LR)
        steps_per_epoch = len(train_loader)
        effective_steps_per_epoch = math.ceil(steps_per_epoch / self.grad_accum)
        scheduler = OneCycleLR(
            optimizer, max_lr=self.cfg.RPPG_SEMI.LR,
            epochs=total_epochs, steps_per_epoch=effective_steps_per_epoch,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_sup': [], 'train_unsup': [], 'train_total': [], 'val_loss': [],
        }

        for epoch in range(total_epochs):
            self.student.train()
            # Teacher stays in eval() mode — BN uses frozen running stats

            epoch_sup = 0.0
            epoch_unsup = 0.0
            epoch_total = 0.0

            labeled_iter = iter(train_loader)
            unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

            lambda_w = self._ramp_up_weight(epoch, self.ramp_up_epochs) * self.lambda_max

            optimizer.zero_grad()
            pbar = tqdm(range(steps_per_epoch), desc=f"Semi Epoch {epoch+1}/{total_epochs}", leave=False)

            for step in pbar:
                # ── Supervised branch: forward + backward ──
                # (backward immediately to free activations before unsupervised branch)
                try:
                    video_l, bvp_label, _ = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train_loader)
                    video_l, bvp_label, _ = next(labeled_iter)

                video_l = video_l.to(self.device)
                bvp_label = bvp_label.to(self.device)

                pred_ppg_l, _ = self.student(video_l)

                # Z-score normalize for NegPearson
                pred_norm = (pred_ppg_l - pred_ppg_l.mean(dim=1, keepdim=True)) / (pred_ppg_l.std(dim=1, keepdim=True) + 1e-8)
                label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)
                loss_sup = self.bvp_criterion(pred_norm, label_norm) * 100.0

                (loss_sup / self.grad_accum).backward()
                del video_l, bvp_label, pred_ppg_l, pred_norm, label_norm

                # ── Unsupervised branch: forward + backward ──
                loss_unsup = torch.tensor(0.0, device=self.device)
                if unlabeled_iter is not None and lambda_w > 0:
                    try:
                        video_u = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        video_u = next(unlabeled_iter)
                    video_u = video_u.to(self.device)

                    # Student forward
                    pred_ppg_s, _ = self.student(video_u)

                    # Teacher forward (no gradient)
                    with torch.no_grad():
                        pred_ppg_t, _ = self.teacher(video_u)

                    # Consistency loss (z-score MSE)
                    loss_consist = self.consistency_criterion(pred_ppg_s, pred_ppg_t)

                    # Frequency constraint on student output
                    loss_freq = self.freq_criterion(pred_ppg_s)

                    loss_unsup = loss_consist + self.freq_weight * loss_freq

                    (lambda_w * loss_unsup / self.grad_accum).backward()
                    del video_u, pred_ppg_s, pred_ppg_t

                if (step + 1) % self.grad_accum == 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # EMA update teacher
                    self._update_teacher(self.ema_decay)

                epoch_sup += loss_sup.item()
                epoch_unsup += loss_unsup.item()
                epoch_total += loss_sup.item() + lambda_w * loss_unsup.item()

                pbar.set_postfix(
                    sup=f"{loss_sup.item():.2f}",
                    unsup=f"{loss_unsup.item():.2f}",
                    lam=f"{lambda_w:.3f}",
                )

            # Flush remaining accumulated gradients at epoch end
            if steps_per_epoch % self.grad_accum != 0:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self._update_teacher(self.ema_decay)

            avg_sup = epoch_sup / steps_per_epoch
            avg_unsup = epoch_unsup / steps_per_epoch
            avg_total = epoch_total / steps_per_epoch
            history['train_sup'].append(avg_sup)
            history['train_unsup'].append(avg_unsup)
            history['train_total'].append(avg_total)

            # Validation (supervised metric only)
            if valid_loader is not None:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(f"  Epoch {epoch+1} | Sup: {avg_sup:.4f} | Unsup: {avg_unsup:.4f} | "
                      f"Total: {avg_total:.4f} | Val: {val_loss:.4f} | λ: {lambda_w:.3f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    self.save_checkpoint(self.student, optimizer, epoch, path, val_loss=val_loss)
                    print(f"  -> Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"  Epoch {epoch+1} | Sup: {avg_sup:.4f} | Unsup: {avg_unsup:.4f} | "
                      f"Total: {avg_total:.4f} | λ: {lambda_w:.3f}")

        # Save final history
        ckpt_name = make_rppg_ckpt_name(self.cfg, tag="semi")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    @torch.no_grad()
    def _update_teacher(self, alpha: float):
        """EMA: teacher = alpha * teacher + (1 - alpha) * student."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
        # EMA for BN running stats (float buffers only); copy integer buffers directly
        for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
            if t_buf.is_floating_point():
                t_buf.data.mul_(alpha).add_(s_buf.data, alpha=1 - alpha)
            else:
                t_buf.data.copy_(s_buf.data)

    @staticmethod
    def _ramp_up_weight(epoch: int, ramp_up_epochs: int) -> float:
        """Sigmoid ramp-up schedule for unsupervised loss weight."""
        if ramp_up_epochs <= 0 or epoch >= ramp_up_epochs:
            return 1.0
        return math.exp(-5.0 * (1.0 - epoch / ramp_up_epochs) ** 2)

    @torch.no_grad()
    def _validate(self, loader) -> float:
        """Validate using NegPearson on labeled data (student model)."""
        self.student.eval()
        total_loss = 0.0
        for video, bvp_label, _ in loader:
            video = video.to(self.device)
            bvp_label = bvp_label.to(self.device)
            pred_ppg, _ = self.student(video)
            pred_norm = (pred_ppg - pred_ppg.mean(dim=1, keepdim=True)) / (pred_ppg.std(dim=1, keepdim=True) + 1e-8)
            label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)
            loss = self.bvp_criterion(pred_norm, label_norm) * 100.0
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    def test(self, data_loader) -> dict:
        """Evaluate HR metrics on test set."""
        from utils.signal_processing import estimate_hr_fft

        self.student.eval()
        hr_preds = []
        hr_gts = []

        with torch.no_grad():
            for video, bvp_label, _ in data_loader:
                video = video.to(self.device)
                pred_ppg, _ = self.student(video)

                for i in range(pred_ppg.shape[0]):
                    pred_hr = estimate_hr_fft(pred_ppg[i].cpu().numpy(), self.cfg.RPPG_MODEL.FPS)
                    gt_hr = estimate_hr_fft(bvp_label[i].numpy(), self.cfg.RPPG_MODEL.FPS)
                    hr_preds.append(pred_hr)
                    hr_gts.append(gt_hr)

        import numpy as np
        hr_preds = np.array(hr_preds)
        hr_gts = np.array(hr_gts)
        mae = np.mean(np.abs(hr_preds - hr_gts))
        rmse = np.sqrt(np.mean((hr_preds - hr_gts) ** 2))

        # Pearson correlation
        if len(hr_preds) > 1:
            pearson_r = np.corrcoef(hr_preds, hr_gts)[0, 1]
        else:
            pearson_r = 0.0

        print(f"  [Test] HR MAE: {mae:.2f} BPM | RMSE: {rmse:.2f} BPM | Pearson r: {pearson_r:.4f}")
        return {'hr_mae': mae, 'hr_rmse': rmse, 'hr_pearson_r': pearson_r}
