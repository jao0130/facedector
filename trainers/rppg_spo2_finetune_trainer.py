"""
rPPG SpO2 Fine-tune Trainer (Stage-3 Training)

策略：多任務學習，同時改善 SpO2 估計並維持 rPPG 品質。

訓練資料：
  MCD-rPPG  — 有 BVP + SpO2 標籤（spo2 > 50）
  UBFC-Phys — 只有 BVP（spo2 = 0.0，SpO2 loss 跳過）

測試資料：
  PURE — 有 BVP + SpO2（o2saturation from JSON）

凍結策略：
  凍結：stem, freq_blocks（保留空間特徵萃取，防止特徵破壞）
  微調：hr_branch, spo2_branch, spo2_fusion_head

Loss（每個 batch）：
  L_total = L_rppg + λ_spo2 × L_spo2

  L_rppg  = NegPearson + λ_spec × SpectralMSE   (全部樣本)
  L_spo2  = MSE(pred, gt) + 0.5 × MAE(pred, gt) (spo2 > 50 的樣本)

SpO2 遮罩：
  spo2_label == 0.0 → 無標籤 → 不計算 SpO2 loss
  spo2_label > 50   → 有效標籤（生理上 SpO2 不可能 < 50%）

Test metrics：
  HR  : MAE, RMSE
  SpO2: MAE, RMSE, Pearson r（僅 spo2 > 50 樣本）
  Wave: Pearson r
"""

import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_rppg_ckpt_name
from losses.rppg_losses import NegPearsonLoss


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True).clamp(min=1e-8))


def _spectral_mse(pred: torch.Tensor, target: torch.Tensor,
                  fs: float = 30.0,
                  freq_low: float = 0.7, freq_high: float = 2.5) -> torch.Tensor:
    N = pred.shape[-1]
    pred_mag   = torch.abs(torch.fft.rfft(pred,   dim=-1))
    target_mag = torch.abs(torch.fft.rfft(target, dim=-1))
    freqs = torch.linspace(0, fs / 2, pred_mag.shape[-1], device=pred.device)
    mask  = (freqs >= freq_low) & (freqs <= freq_high)
    return F.mse_loss(pred_mag[:, mask], target_mag[:, mask])


# ── SpO2 Fine-tune Trainer ────────────────────────────────────────────────────

class rPPGSpO2FinetuneTrainer(BaseTrainer):
    """
    Stage-3：從 Stage-2 waveform fine-tune checkpoint 出發，
    針對 SpO2 branch 進行多任務微調。
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── 建立並載入基礎模型 ──────────────────────────────────────────────
        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt_v3')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.model = FCAtt_v3(cfg=cfg).to(self.device)
        else:
            raise ValueError(f"[SpO2Finetune] Unsupported model: {model_name}")

        weights_path = cfg.RPPG_MODEL.WEIGHTS
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"[SpO2Finetune] Weights required. Not found: {weights_path}")
        ckpt  = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, _ = self.model.load_state_dict(state, strict=False)
        print(f"[SpO2Finetune] Loaded: {weights_path}")
        if missing:
            print(f"[SpO2Finetune] Missing keys ({len(missing)}): {missing[:3]} ...")

        # ── 凍結空間特徵萃取，微調時序 + SpO2 分支 ─────────────────────────
        # 保留 stem + freq_blocks（頻率特徵不破壞）
        # 微調 hr_branch + spo2_branch + spo2_fusion_head
        FROZEN_PREFIXES = ('stem', 'freq_blocks')
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in FROZEN_PREFIXES):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"[SpO2Finetune] Trainable: {trainable:,} / {total:,} "
              f"(frozen: {total - trainable:,})")

        # ── Losses ────────────────────────────────────────────────────────────
        self.neg_pearson = NegPearsonLoss()

        # ── Hyperparams ───────────────────────────────────────────────────────
        ft = cfg.RPPG_SPO2_FINETUNE
        self.lambda_spectral = ft.LAMBDA_SPECTRAL
        self.lambda_spo2     = ft.LAMBDA_SPO2
        self.grad_clip       = cfg.RPPG_TRAIN.GRAD_CLIP
        self.fps             = float(cfg.RPPG_MODEL.FPS)
        self.freq_low        = float(cfg.RPPG_MODEL.FREQ_LOW)
        self.freq_high       = float(cfg.RPPG_MODEL.FREQ_HIGH)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, data_loaders: dict) -> dict:
        train_loader = data_loaders['train']
        valid_loader = data_loaders.get('valid')

        ft           = self.cfg.RPPG_SPO2_FINETUNE
        total_epochs = ft.EPOCHS
        patience     = ft.PATIENCE

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=ft.LR, weight_decay=1e-4,
        )
        sched = CosineAnnealingLR(opt, T_max=total_epochs, eta_min=ft.LR * 0.1)

        best_val         = float('inf')
        patience_counter = 0
        history = {
            'loss_total': [], 'loss_rppg': [], 'loss_spectral': [],
            'loss_spo2': [], 'n_spo2': [], 'val_loss': [],
        }

        csv_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        os.makedirs(self.cfg.OUTPUT.LOG_DIR, exist_ok=True)
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'loss_total', 'loss_rppg', 'loss_spectral',
            'loss_spo2', 'n_spo2_batches', 'val_loss', 'best'
        ])
        csv_file.flush()

        for epoch in range(total_epochs):
            self.model.train()

            sum_total = sum_rppg = sum_spec = sum_spo2 = 0.0
            n_steps = n_spo2_steps = 0

            pbar = tqdm(train_loader,
                        desc=f"SpO2Finetune {epoch+1}/{total_epochs}",
                        leave=False)

            for video, bvp_label, spo2_label in pbar:
                video      = video.to(self.device)
                bvp_label  = bvp_label.to(self.device)
                spo2_label = spo2_label.to(self.device)   # [B, 1]

                pred_wave, pred_spo2 = self.model(video)

                # ── rPPG loss（全部樣本）──────────────────────────────────────
                pred_norm  = _zscore(pred_wave)
                label_norm = _zscore(bvp_label)
                loss_rppg  = self.neg_pearson(pred_norm, label_norm) * 100
                loss_spec  = _spectral_mse(pred_wave, bvp_label,
                                           fs=self.fps,
                                           freq_low=self.freq_low,
                                           freq_high=self.freq_high)

                # ── SpO2 loss（有標籤樣本）────────────────────────────────────
                # spo2_label == 0.0 → UBFC-Phys，無 SpO2，跳過
                valid_mask = (spo2_label.squeeze(-1) > 50.0)
                if valid_mask.any():
                    pred_v  = pred_spo2[valid_mask]
                    label_v = spo2_label[valid_mask]
                    loss_spo2 = (F.mse_loss(pred_v, label_v)
                                 + 0.5 * F.l1_loss(pred_v, label_v))
                    n_spo2_steps += 1
                else:
                    loss_spo2 = torch.zeros(1, device=self.device)

                loss_total = (loss_rppg
                              + self.lambda_spectral * loss_spec
                              + self.lambda_spo2    * loss_spo2)

                opt.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                opt.step()

                sum_total += loss_total.item()
                sum_rppg  += loss_rppg.item()
                sum_spec  += loss_spec.item()
                sum_spo2  += loss_spo2.item()
                n_steps   += 1

                pbar.set_postfix(
                    tot=f"{loss_total.item():.3f}",
                    rppg=f"{loss_rppg.item():.2f}",
                    spo2=f"{loss_spo2.item():.3f}" if valid_mask.any() else "skip",
                )

            sched.step()

            avg_total = sum_total / max(n_steps, 1)
            avg_rppg  = sum_rppg  / max(n_steps, 1)
            avg_spec  = sum_spec  / max(n_steps, 1)
            avg_spo2  = sum_spo2  / max(n_spo2_steps, 1)

            history['loss_total'].append(avg_total)
            history['loss_rppg'].append(avg_rppg)
            history['loss_spectral'].append(avg_spec)
            history['loss_spo2'].append(avg_spo2)
            history['n_spo2'].append(n_spo2_steps)

            # ── Validation ────────────────────────────────────────────────────
            is_best  = False
            val_loss = float('nan')
            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"tot: {avg_total:.4f}  rppg: {avg_rppg:.4f}  "
                    f"spec: {avg_spec:.4f}  spo2: {avg_spo2:.4f} "
                    f"(n={n_spo2_steps}) | Val: {val_loss:.4f}"
                )

                if val_loss < best_val:
                    best_val         = val_loss
                    patience_counter = 0
                    is_best          = True
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    os.makedirs(self.cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)
                    self.save_checkpoint(self.model, opt, epoch, path,
                                         val_loss=val_loss)
                    print(f"  -> Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        csv_writer.writerow([
                            epoch+1, f"{avg_total:.4f}", f"{avg_rppg:.4f}",
                            f"{avg_spec:.4f}", f"{avg_spo2:.4f}",
                            n_spo2_steps, f"{val_loss:.4f}", '',
                        ])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"tot: {avg_total:.4f}  rppg: {avg_rppg:.4f}  "
                    f"spec: {avg_spec:.4f}  spo2: {avg_spo2:.4f} (n={n_spo2_steps})"
                )

            csv_writer.writerow([
                epoch+1, f"{avg_total:.4f}", f"{avg_rppg:.4f}",
                f"{avg_spec:.4f}", f"{avg_spo2:.4f}",
                n_spo2_steps, f"{val_loss:.4f}", '*' if is_best else '',
            ])
            csv_file.flush()

        csv_file.close()
        ckpt_name    = make_rppg_ckpt_name(self.cfg, tag="spo2ft")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> float:
        """Val loss = rPPG Pearson + SpO2 MAE (when available)."""
        self.model.eval()
        total = 0.0
        ft = self.cfg.RPPG_SPO2_FINETUNE
        for video, bvp_label, spo2_label in loader:
            video      = video.to(self.device)
            bvp_label  = bvp_label.to(self.device)
            spo2_label = spo2_label.to(self.device)
            pred_wave, pred_spo2 = self.model(video)

            pred_norm  = _zscore(pred_wave)
            label_norm = _zscore(bvp_label)
            loss = self.neg_pearson(pred_norm, label_norm) * 100

            valid_mask = (spo2_label.squeeze(-1) > 50.0)
            if valid_mask.any():
                loss = loss + ft.LAMBDA_SPO2 * F.l1_loss(
                    pred_spo2[valid_mask], spo2_label[valid_mask])
            total += loss.item()
        self.model.train()
        return total / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    # ── Test ──────────────────────────────────────────────────────────────────

    def test(self, data_loader, dataset_name: str = "") -> dict:
        """Evaluate HR metrics + SpO2 metrics + waveform Pearson r."""
        from utils.signal_processing import estimate_hr_fft
        import numpy as np
        from scipy.stats import pearsonr

        self.model.eval()
        hr_preds   = []
        hr_gts     = []
        spo2_preds = []
        spo2_gts   = []
        wave_rs    = []

        with torch.no_grad():
            for video, bvp_label, spo2_label in data_loader:
                video      = video.to(self.device)
                pred_wave, pred_spo2 = self.model(video)
                pred_norm  = _zscore(pred_wave)
                label_norm = _zscore(bvp_label.to(self.device))

                for i in range(pred_wave.shape[0]):
                    p = pred_wave[i].cpu().numpy()
                    g = bvp_label[i].numpy()
                    r, _ = pearsonr(pred_norm[i].cpu().numpy(),
                                    label_norm[i].cpu().numpy())
                    wave_rs.append(r)
                    hr_preds.append(estimate_hr_fft(p, self.cfg.RPPG_MODEL.FPS))
                    hr_gts.append(estimate_hr_fft(g, self.cfg.RPPG_MODEL.FPS))

                    spo2_val = float(spo2_label[i].item())
                    if spo2_val > 50.0:
                        spo2_preds.append(float(pred_spo2[i].item()))
                        spo2_gts.append(spo2_val)

        hr_preds = np.array(hr_preds)
        hr_gts   = np.array(hr_gts)
        hr_mae   = float(np.mean(np.abs(hr_preds - hr_gts)))
        hr_rmse  = float(np.sqrt(np.mean((hr_preds - hr_gts) ** 2)))
        mean_r   = float(np.mean(wave_rs))

        tag = f" [{dataset_name}]" if dataset_name else ""
        print(f"  [Test]{tag} HR  MAE: {hr_mae:.2f} BPM | "
              f"RMSE: {hr_rmse:.2f} | Wave r: {mean_r:.4f}")

        result = {'hr_mae': hr_mae, 'hr_rmse': hr_rmse, 'waveform_pearson_r': mean_r}

        if spo2_preds:
            spo2_preds = np.array(spo2_preds)
            spo2_gts   = np.array(spo2_gts)
            spo2_mae   = float(np.mean(np.abs(spo2_preds - spo2_gts)))
            spo2_rmse  = float(np.sqrt(np.mean((spo2_preds - spo2_gts) ** 2)))
            spo2_r, _  = pearsonr(spo2_preds, spo2_gts) if len(spo2_preds) > 1 else (float('nan'), None)
            print(f"  [Test]{tag} SpO2 MAE: {spo2_mae:.3f}% | "
                  f"RMSE: {spo2_rmse:.3f}% | r: {spo2_r:.4f} "
                  f"(n={len(spo2_preds)})")
            result.update({
                'spo2_mae': spo2_mae, 'spo2_rmse': spo2_rmse, 'spo2_r': spo2_r,
            })
        else:
            print(f"  [Test]{tag} SpO2: no labeled samples in this set")

        return result
