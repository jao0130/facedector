"""
rPPG SpO2 Fine-tune Trainer (Stage-3 Training)

資料架構：
  ┌─ 有標籤 (labeled)  ─┐  MCD-rPPG：BVP + SpO2
  │  supervised loss      │  → L_rppg + λ_spo2 × L_spo2
  └────────────────────────┘

  ┌─ 無 SpO2 標籤 (unlabeled) ─┐  UBFC-Phys：只有 BVP
  │  semi-supervised loss        │  → λ_unsup × L_rppg  (無 SpO2 loss)
  └──────────────────────────────┘

  測試集：PURE（BVP + SpO2）→ 評估 HR + SpO2 兩者

凍結策略：
  凍結：stem, freq_blocks（保留頻率特徵萃取）
  微調：hr_branch, spo2_branch, spo2_fusion_head

每個 step 處理：
  1. 取一個 labeled batch（MCD-rPPG）→ L_labeled = L_rppg + λ_spo2×L_spo2
  2. 取一個 unlabeled batch（UBFC-Phys）→ L_unsup = λ_unsup×L_rppg
  3. L_total = L_labeled + L_unsup，一次 backward

其中 L_rppg = NegPearson×100 + λ_spec×SpectralMSE
     L_spo2  = MSE(pred, gt) + 0.5×MAE(pred, gt)
"""

import csv
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_rppg_ckpt_name
from losses.rppg_losses import NegPearsonLoss


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True).clamp(min=1e-8))


def _spectral_mse(pred: torch.Tensor, target: torch.Tensor,
                  fs: float = 30.0,
                  freq_low: float = 0.7, freq_high: float = 2.5) -> torch.Tensor:
    pred_mag   = torch.abs(torch.fft.rfft(pred,   dim=-1))
    target_mag = torch.abs(torch.fft.rfft(target, dim=-1))
    freqs = torch.linspace(0, fs / 2, pred_mag.shape[-1], device=pred.device)
    mask  = (freqs >= freq_low) & (freqs <= freq_high)
    return F.mse_loss(pred_mag[:, mask], target_mag[:, mask])


def _rppg_loss(pred_wave, bvp_label, neg_pearson, fs, freq_low, freq_high, lambda_spectral):
    """共用 rPPG loss：NegPearson + λ_spec × SpectralMSE。"""
    pred_norm  = _zscore(pred_wave)
    label_norm = _zscore(bvp_label)
    loss_pear  = neg_pearson(pred_norm, label_norm) * 100
    loss_spec  = _spectral_mse(pred_wave, bvp_label, fs=fs,
                               freq_low=freq_low, freq_high=freq_high)
    return loss_pear + lambda_spectral * loss_spec, loss_pear, loss_spec


# ── SpO2 Fine-tune Trainer ────────────────────────────────────────────────────

class rPPGSpO2FinetuneTrainer(BaseTrainer):
    """
    Stage-3：MCD-rPPG（有標籤）+ UBFC-Phys（BVP-only 半監督）聯合訓練。
    測試集：PURE（BVP + SpO2 均有標籤）。
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
        FROZEN_PREFIXES = ('stem', 'freq_blocks')
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in FROZEN_PREFIXES):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"[SpO2Finetune] Trainable: {trainable:,} / {total:,} "
              f"(frozen: {total - trainable:,})")

        # ── Losses & Hyperparams ──────────────────────────────────────────────
        self.neg_pearson     = NegPearsonLoss()
        ft                   = cfg.RPPG_SPO2_FINETUNE
        self.lambda_spectral = ft.LAMBDA_SPECTRAL
        self.lambda_spo2     = ft.LAMBDA_SPO2
        self.lambda_unsup    = ft.LAMBDA_UNSUP
        self.grad_clip       = cfg.RPPG_TRAIN.GRAD_CLIP
        self.fps             = float(cfg.RPPG_MODEL.FPS)
        self.freq_low        = float(cfg.RPPG_MODEL.FREQ_LOW)
        self.freq_high       = float(cfg.RPPG_MODEL.FREQ_HIGH)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, data_loaders: dict) -> dict:
        labeled_loader   = data_loaders['train']        # MCD-rPPG: BVP + SpO2
        valid_loader     = data_loaders.get('valid')
        unlabeled_loader = data_loaders.get('unlabeled')  # UBFC-Phys: BVP only

        has_unsup = unlabeled_loader is not None
        if has_unsup:
            print(f"[SpO2Finetune] Semi-supervised: "
                  f"{len(labeled_loader)} labeled + {len(unlabeled_loader)} unlabeled batches/epoch")
        else:
            print("[SpO2Finetune] Supervised only (no unlabeled loader found)")

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
            'loss_total':   [], 'loss_labeled': [], 'loss_rppg_l': [],
            'loss_spo2':    [], 'loss_unsup':   [], 'val_loss':    [],
        }

        csv_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        os.makedirs(self.cfg.OUTPUT.LOG_DIR, exist_ok=True)
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'loss_total', 'loss_labeled', 'loss_rppg_l',
            'loss_spo2', 'loss_unsup', 'val_loss', 'best',
        ])
        csv_file.flush()

        for epoch in range(total_epochs):
            self.model.train()

            # 無限迴圈 unlabeled iterator（與 labeled 對齊，labeled 走完一圈即為一個 epoch）
            unlabeled_iter = itertools.cycle(unlabeled_loader) if has_unsup else None

            sum_total = sum_lab = sum_rppg_l = sum_spo2 = sum_unsup = 0.0
            n_steps   = 0

            pbar = tqdm(labeled_loader,
                        desc=f"SpO2Ft {epoch+1}/{total_epochs}",
                        leave=False)

            for video_l, bvp_l, spo2_l in pbar:
                video_l = video_l.to(self.device)
                bvp_l   = bvp_l.to(self.device)
                spo2_l  = spo2_l.to(self.device)   # [B, 1]

                pred_wave_l, pred_spo2_l = self.model(video_l)

                # ── Labeled loss（MCD-rPPG）───────────────────────────────────
                loss_rppg_l, loss_pear_l, _ = _rppg_loss(
                    pred_wave_l, bvp_l, self.neg_pearson,
                    self.fps, self.freq_low, self.freq_high, self.lambda_spectral,
                )
                loss_spo2 = (F.mse_loss(pred_spo2_l, spo2_l)
                             + 0.5 * F.l1_loss(pred_spo2_l, spo2_l))
                loss_labeled = loss_rppg_l + self.lambda_spo2 * loss_spo2

                # ── Unlabeled loss（UBFC-Phys：只有 BVP，無 SpO2）─────────────
                loss_unsup = torch.zeros(1, device=self.device)
                if unlabeled_iter is not None:
                    video_u, bvp_u, _ = next(unlabeled_iter)
                    video_u = video_u.to(self.device)
                    bvp_u   = bvp_u.to(self.device)
                    pred_wave_u, _ = self.model(video_u)
                    loss_unsup_raw, _, _ = _rppg_loss(
                        pred_wave_u, bvp_u, self.neg_pearson,
                        self.fps, self.freq_low, self.freq_high, self.lambda_spectral,
                    )
                    loss_unsup = self.lambda_unsup * loss_unsup_raw

                loss_total = loss_labeled + loss_unsup

                opt.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                opt.step()

                sum_total   += loss_total.item()
                sum_lab     += loss_labeled.item()
                sum_rppg_l  += loss_rppg_l.item()
                sum_spo2    += loss_spo2.item()
                sum_unsup   += loss_unsup.item()
                n_steps     += 1

                pbar.set_postfix(
                    tot=f"{loss_total.item():.3f}",
                    rppg=f"{loss_pear_l.item():.2f}",
                    spo2=f"{loss_spo2.item():.3f}",
                    u=f"{loss_unsup.item():.3f}" if has_unsup else "-",
                )

            sched.step()

            avg_total  = sum_total  / max(n_steps, 1)
            avg_lab    = sum_lab    / max(n_steps, 1)
            avg_rppg_l = sum_rppg_l / max(n_steps, 1)
            avg_spo2   = sum_spo2   / max(n_steps, 1)
            avg_unsup  = sum_unsup  / max(n_steps, 1)

            history['loss_total'].append(avg_total)
            history['loss_labeled'].append(avg_lab)
            history['loss_rppg_l'].append(avg_rppg_l)
            history['loss_spo2'].append(avg_spo2)
            history['loss_unsup'].append(avg_unsup)

            # ── Validation ────────────────────────────────────────────────────
            is_best  = False
            val_loss = float('nan')
            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"lab: {avg_lab:.4f}  rppg: {avg_rppg_l:.4f}  "
                    f"spo2: {avg_spo2:.4f}  unsup: {avg_unsup:.4f} | "
                    f"Val: {val_loss:.4f}"
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
                            epoch+1, f"{avg_total:.4f}", f"{avg_lab:.4f}",
                            f"{avg_rppg_l:.4f}", f"{avg_spo2:.4f}",
                            f"{avg_unsup:.4f}", f"{val_loss:.4f}", '',
                        ])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"lab: {avg_lab:.4f}  rppg: {avg_rppg_l:.4f}  "
                    f"spo2: {avg_spo2:.4f}  unsup: {avg_unsup:.4f}"
                )

            csv_writer.writerow([
                epoch+1, f"{avg_total:.4f}", f"{avg_lab:.4f}",
                f"{avg_rppg_l:.4f}", f"{avg_spo2:.4f}",
                f"{avg_unsup:.4f}", f"{val_loss:.4f}", '*' if is_best else '',
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
        """Val loss = rPPG loss + λ_spo2 × SpO2 MAE（在 labeled val set 上）。"""
        self.model.eval()
        total = 0.0
        ft = self.cfg.RPPG_SPO2_FINETUNE
        for video, bvp_label, spo2_label in loader:
            video      = video.to(self.device)
            bvp_label  = bvp_label.to(self.device)
            spo2_label = spo2_label.to(self.device)
            pred_wave, pred_spo2 = self.model(video)
            loss_rppg, _, _ = _rppg_loss(
                pred_wave, bvp_label, self.neg_pearson,
                self.fps, self.freq_low, self.freq_high, self.lambda_spectral,
            )
            loss_spo2 = F.l1_loss(pred_spo2, spo2_label)
            total += (loss_rppg + ft.LAMBDA_SPO2 * loss_spo2).item()
        self.model.train()
        return total / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    # ── Test ──────────────────────────────────────────────────────────────────

    def test(self, data_loader, dataset_name: str = "") -> dict:
        """評估 HR 指標 + SpO2 指標 + 波形 Pearson r（PURE 有兩者標籤）。"""
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
                    p_wave = pred_wave[i].cpu().numpy()
                    g_wave = bvp_label[i].numpy()
                    r, _   = pearsonr(pred_norm[i].cpu().numpy(),
                                      label_norm[i].cpu().numpy())
                    wave_rs.append(r)
                    hr_preds.append(estimate_hr_fft(p_wave, self.cfg.RPPG_MODEL.FPS))
                    hr_gts.append(estimate_hr_fft(g_wave,  self.cfg.RPPG_MODEL.FPS))

                    spo2_val = float(spo2_label[i].item())
                    # spo2 > 50 → 有效標籤（PURE 範圍 ~97-100%）
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
            if len(spo2_preds) > 1:
                spo2_r, _ = pearsonr(spo2_preds, spo2_gts)
            else:
                spo2_r = float('nan')
            print(f"  [Test]{tag} SpO2 MAE: {spo2_mae:.3f}% | "
                  f"RMSE: {spo2_rmse:.3f}% | r: {spo2_r:.4f} "
                  f"(n={len(spo2_preds)} / {len(hr_preds)})")
            result.update({
                'spo2_mae': spo2_mae, 'spo2_rmse': spo2_rmse, 'spo2_r': spo2_r,
            })
        else:
            print(f"  [Test]{tag} SpO2: 此資料集無 SpO2 標籤（spo2=0）")

        return result
