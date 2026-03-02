"""
rPPG Waveform Fine-tune Trainer (Stage-2 Training)

策略：凍結 Encoder，只微調 HRBranch，搭配 LSGAN Discriminator。

為何選擇 GAN 而非對比學習：
  - 對比學習 (ContrastPhys/SimCLR) → 學習特徵相似性，適合無標籤 pre-train
    對有 GT label 的 waveform 重建效果有限
  - LSGAN Discriminator → 直接學習「真實 PPG 形狀先驗」
    complementary to supervised loss，強制 HRBranch 輸出生理合理波形

訓練架構：
  凍結 (Encoder 保留頻率特徵)     微調 (Generator)
  ─────────────────────────────  ─────────────────────────
  spatial_stem                    hr_branch.temporal_refiner
  freq_att_blocks × N             hr_branch.mamba
  spo2_branch                     hr_branch.hr_head
                                        ↓ pred_wave [B, T]
                             ┌─── Discriminator (新)
                             │    1D CNN + Spectral Norm
                             │    Real BVP vs Predicted PPG
                             └─── LSGAN loss → 波形形狀約束

Loss：
  L_G = NegPearson(×100) + λ_mse × Z-MSE + λ_adv × LSGAN_G   (warmup 後)
  L_D = 0.5 × [MSE(D(real), 0.9) + MSE(D(fake), 0)]

DiffNorm → Discriminator 修正（cumsum）：
  標籤是 DiffNormalized（BVP 一階差分），形狀為正負脈衝對，
  不含平滑波峰的 shape prior，Discriminator 無法有效學習 PPG 形狀。
  修正：餵給 D 之前先做 cumsum 積分還原成近似 raw BVP，
       zscore 消除 DC drift 後 D 才能判斷「是否有合理 PPG 波形」。
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
    """Z-score normalize along time axis [B, T]."""
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True).clamp(min=1e-8))


def _integrate(x: torch.Tensor) -> torch.Tensor:
    """DiffNorm → 近似 raw BVP：cumsum 積分後 zscore 消除 DC drift。[B, T]"""
    return _zscore(torch.cumsum(x, dim=-1))


def _spectral_mse(pred: torch.Tensor, target: torch.Tensor,
                  fs: float = 30.0,
                  freq_low: float = 0.7, freq_high: float = 2.5) -> torch.Tensor:
    """
    頻域能量分佈 MSE（cardiac band）。

    直接約束預測訊號在心跳頻率段的頻譜形狀，防止 Z-MSE 的時域對齊壓力
    造成主頻漂移→心率估計下滑。

    Input:  [B, T]（DiffNorm 訊號，zscore 前後皆可）
    Output: scalar loss
    """
    N = pred.shape[-1]
    pred_mag   = torch.abs(torch.fft.rfft(pred,   dim=-1))   # [B, N//2+1]
    target_mag = torch.abs(torch.fft.rfft(target, dim=-1))

    # 建立心跳頻率遮罩（不可微，但只影響哪些 bin 參與計算，梯度仍正確流通）
    freqs = torch.linspace(0, fs / 2, pred_mag.shape[-1], device=pred.device)
    mask  = (freqs >= freq_low) & (freqs <= freq_high)        # [N//2+1] bool

    return F.mse_loss(pred_mag[:, mask], target_mag[:, mask])


# ── LSGAN Discriminator ───────────────────────────────────────────────────────

class PPGDiscriminator(nn.Module):
    """
    1D temporal discriminator for PPG waveform realism (LSGAN).

    使用 Spectral Normalization 確保 Lipschitz 約束，防止梯度爆炸。
    Input:  z-score normalized waveform [B, T]
    Output: realness score [B, 1]

    架構故意設計較小（~12K params）：
      - 避免 Discriminator 過強導致 Generator 梯度消失
      - 只需判別「是否有合理 PPG 形狀」，不需要高容量
    """

    def __init__(self):
        super().__init__()
        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            SN(nn.Conv1d(1, 16,  kernel_size=7, padding=3)),            # [B, 16, T]
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2)),  # [B, 32, T/2]
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)),  # [B, 64, T/4]
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)), # [B, 128, T/8]
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))  # [B, T] → [B, 1, T] → [B, 1]


# ── Fine-tune Trainer ─────────────────────────────────────────────────────────

class rPPGWaveformFinetuneTrainer(BaseTrainer):
    """
    Stage-2 fine-tune trainer：
      1. 載入預訓練 checkpoint（必須提供 RPPG_MODEL.WEIGHTS）
      2. 凍結 Encoder（空間特徵 + 頻率注意力），只訓練 HRBranch
      3. 前 WARMUP_EPOCHS 只用 supervised loss
      4. WARMUP_EPOCHS 之後加入 LSGAN adversarial loss
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── 建立並載入基礎模型 ──────────────────────────────────────────────
        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt_v3')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.model = FCAtt_v3(cfg=cfg).to(self.device)
        elif model_name == 'FCAtt_v2':
            from models.rppg_model_v2 import FCAtt_v2
            self.model = FCAtt_v2(cfg=cfg).to(self.device)
        else:
            raise ValueError(f"[Finetune] Unsupported model: {model_name}")

        weights_path = cfg.RPPG_MODEL.WEIGHTS
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"[Finetune] Pre-trained weights required. Not found: {weights_path}"
            )
        ckpt  = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[Finetune] Loaded: {weights_path}")
        if missing:
            print(f"[Finetune] Missing keys ({len(missing)}): {missing[:3]} ...")

        # ── 凍結 Encoder ─────────────────────────────────────────────────────
        FROZEN_PREFIXES = ('spatial_stem', 'freq_att_blocks', 'spo2_branch')
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in FROZEN_PREFIXES):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"[Finetune] Trainable: {trainable:,} / {total:,} params "
              f"(frozen: {total - trainable:,})")

        # ── Discriminator ────────────────────────────────────────────────────
        self.disc = PPGDiscriminator().to(self.device)
        disc_params = sum(p.numel() for p in self.disc.parameters())
        print(f"[Finetune] Discriminator params: {disc_params:,}")

        # ── Losses ────────────────────────────────────────────────────────────
        self.neg_pearson = NegPearsonLoss()

        # ── Hyperparams ───────────────────────────────────────────────────────
        ft = cfg.RPPG_FINETUNE
        self.lambda_mse      = ft.LAMBDA_MSE
        self.lambda_spectral = ft.LAMBDA_SPECTRAL
        self.lambda_adv      = ft.LAMBDA_ADV
        self.warmup_epochs   = ft.WARMUP_EPOCHS
        self.grad_clip       = cfg.RPPG_TRAIN.GRAD_CLIP
        self.fps             = float(cfg.RPPG_MODEL.FPS)
        self.freq_low        = float(cfg.RPPG_MODEL.FREQ_LOW)
        self.freq_high       = float(cfg.RPPG_MODEL.FREQ_HIGH)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, data_loaders: dict) -> dict:
        train_loader = data_loaders['train']
        valid_loader = data_loaders.get('valid')

        ft           = self.cfg.RPPG_FINETUNE
        total_epochs = ft.EPOCHS
        patience     = ft.PATIENCE

        # Two separate optimizers
        opt_g = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=ft.LR_G, weight_decay=1e-4,
        )
        opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=ft.LR_D, weight_decay=1e-4,
        )
        sched_g = CosineAnnealingLR(opt_g, T_max=total_epochs, eta_min=ft.LR_G * 0.1)
        sched_d = CosineAnnealingLR(opt_d, T_max=total_epochs, eta_min=ft.LR_D * 0.1)

        best_val         = float('inf')
        patience_counter = 0
        history = {'loss_g': [], 'loss_pearson': [], 'loss_mse': [],
                   'loss_spectral': [], 'loss_adv': [], 'loss_d': [], 'val_loss': []}

        # Per-epoch CSV log
        csv_name = make_rppg_ckpt_name(self.cfg, tag="finetune_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'loss_g', 'loss_pearson', 'loss_mse',
            'loss_spectral', 'loss_adv', 'loss_d', 'val_loss', 'best'
        ])
        csv_file.flush()

        for epoch in range(total_epochs):
            self.model.train()
            self.disc.train()

            use_adv = epoch >= self.warmup_epochs

            sum_g = sum_pear = sum_mse = sum_spec = sum_adv = sum_d = 0.0
            n_steps = len(train_loader)

            pbar = tqdm(train_loader,
                        desc=f"Finetune {epoch+1}/{total_epochs} "
                             f"({'adv' if use_adv else 'warmup'})",
                        leave=False)

            for video, bvp_label, _ in pbar:
                video     = video.to(self.device)
                bvp_label = bvp_label.to(self.device)

                # ── Generator step ────────────────────────────────────────────
                pred_wave, _ = self.model(video)
                pred_norm    = _zscore(pred_wave)
                label_norm   = _zscore(bvp_label)

                loss_pearson  = self.neg_pearson(pred_norm, label_norm) * 100
                loss_mse      = F.mse_loss(pred_norm, label_norm)
                # 頻域能量約束：直接防止主頻漂移（Z-MSE 時域壓力的補充）
                loss_spectral = _spectral_mse(pred_wave, bvp_label,
                                              fs=self.fps,
                                              freq_low=self.freq_low,
                                              freq_high=self.freq_high)

                loss_adv = torch.zeros(1, device=self.device)
                if use_adv:
                    # cumsum 積分：DiffNorm → 近似 raw BVP，D 才能學 PPG shape prior
                    pred_integrated = _integrate(pred_wave)
                    d_pred   = self.disc(pred_integrated)
                    loss_adv = F.mse_loss(d_pred, torch.ones_like(d_pred))

                loss_g = (loss_pearson
                          + self.lambda_mse      * loss_mse
                          + self.lambda_spectral * loss_spectral
                          + self.lambda_adv      * loss_adv)

                opt_g.zero_grad()
                loss_g.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                opt_g.step()

                # ── Discriminator step ─────────────────────────────────────────
                loss_d = torch.zeros(1, device=self.device)
                if use_adv:
                    with torch.no_grad():
                        pred_wave_d, _ = self.model(video)
                        pred_integrated_d = _integrate(pred_wave_d)
                    label_integrated = _integrate(bvp_label)
                    d_real = self.disc(label_integrated.detach())
                    d_fake = self.disc(pred_integrated_d)
                    # LSGAN: real→0.9 (label smoothing), fake→0
                    loss_d = 0.5 * (
                        F.mse_loss(d_real, torch.full_like(d_real, 0.9)) +
                        F.mse_loss(d_fake, torch.zeros_like(d_fake))
                    )
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()

                sum_g    += loss_g.item()
                sum_pear += loss_pearson.item()
                sum_mse  += loss_mse.item()
                sum_spec += loss_spectral.item()
                sum_adv  += loss_adv.item()
                sum_d    += loss_d.item()

                pbar.set_postfix(
                    G=f"{loss_g.item():.3f}",
                    pear=f"{loss_pearson.item():.2f}",
                    spec=f"{loss_spectral.item():.3f}",
                    adv=f"{loss_adv.item():.3f}" if use_adv else "warm",
                    D=f"{loss_d.item():.3f}" if use_adv else "-",
                )

            sched_g.step()
            sched_d.step()

            avg_g    = sum_g    / n_steps
            avg_pear = sum_pear / n_steps
            avg_mse  = sum_mse  / n_steps
            avg_spec = sum_spec / n_steps
            avg_adv  = sum_adv  / n_steps
            avg_d    = sum_d    / n_steps

            history['loss_g'].append(avg_g)
            history['loss_pearson'].append(avg_pear)
            history['loss_mse'].append(avg_mse)
            history['loss_spectral'].append(avg_spec)
            history['loss_adv'].append(avg_adv)
            history['loss_d'].append(avg_d)

            # ── Validation ────────────────────────────────────────────────────
            is_best  = False
            val_loss = float('nan')
            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"G: {avg_g:.4f}  pear: {avg_pear:.4f}  "
                    f"mse: {avg_mse:.4f}  adv: {avg_adv:.4f} | "
                    f"D: {avg_d:.4f} | Val: {val_loss:.4f}"
                )

                if val_loss < best_val:
                    best_val         = val_loss
                    patience_counter = 0
                    is_best          = True
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="finetune_best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    self.save_checkpoint(self.model, opt_g, epoch, path,
                                         val_loss=val_loss)
                    print(f"  -> Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        csv_writer.writerow([
                            epoch + 1, f"{avg_g:.4f}", f"{avg_pear:.4f}",
                            f"{avg_mse:.4f}", f"{avg_spec:.4f}",
                            f"{avg_adv:.4f}", f"{avg_d:.4f}",
                            f"{val_loss:.4f}", '',
                        ])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"G: {avg_g:.4f}  pear: {avg_pear:.4f}  "
                    f"mse: {avg_mse:.4f}  adv: {avg_adv:.4f} | "
                    f"D: {avg_d:.4f}"
                )

            csv_writer.writerow([
                epoch + 1, f"{avg_g:.4f}", f"{avg_pear:.4f}",
                f"{avg_mse:.4f}", f"{avg_spec:.4f}",
                f"{avg_adv:.4f}", f"{avg_d:.4f}",
                f"{val_loss:.4f}", '*' if is_best else '',
            ])
            csv_file.flush()

        csv_file.close()

        ckpt_name    = make_rppg_ckpt_name(self.cfg, tag="finetune")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for video, bvp_label, _ in loader:
            video     = video.to(self.device)
            bvp_label = bvp_label.to(self.device)
            pred, _   = self.model(video)
            pred_norm  = _zscore(pred)
            label_norm = _zscore(bvp_label)
            loss  = self.neg_pearson(pred_norm, label_norm) * 100
            loss += self.lambda_mse * F.mse_loss(pred_norm, label_norm)
            total += loss.item()
        self.model.train()
        return total / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    # ── Test ──────────────────────────────────────────────────────────────────

    def test(self, data_loader, dataset_name: str = "") -> dict:
        """Evaluate waveform quality (Pearson r) + HR metrics."""
        from utils.signal_processing import estimate_hr_fft
        import numpy as np
        from scipy.stats import pearsonr

        self.model.eval()
        hr_preds = []
        hr_gts   = []
        pearson_rs = []

        with torch.no_grad():
            for video, bvp_label, _ in data_loader:
                video     = video.to(self.device)
                pred, _   = self.model(video)
                pred_norm  = _zscore(pred)
                label_norm = _zscore(bvp_label.to(self.device))

                for i in range(pred.shape[0]):
                    p = pred[i].cpu().numpy()
                    g = bvp_label[i].numpy()
                    r, _ = pearsonr(
                        pred_norm[i].cpu().numpy(),
                        label_norm[i].cpu().numpy(),
                    )
                    pearson_rs.append(r)
                    hr_preds.append(estimate_hr_fft(p, self.cfg.RPPG_MODEL.FPS))
                    hr_gts.append(estimate_hr_fft(g, self.cfg.RPPG_MODEL.FPS))

        hr_preds   = np.array(hr_preds)
        hr_gts     = np.array(hr_gts)
        mae        = np.mean(np.abs(hr_preds - hr_gts))
        rmse       = np.sqrt(np.mean((hr_preds - hr_gts) ** 2))
        mean_r     = float(np.mean(pearson_rs))

        tag = f" [{dataset_name}]" if dataset_name else ""
        print(f"  [Test]{tag} HR MAE: {mae:.2f} | RMSE: {rmse:.2f} | "
              f"Waveform Pearson r: {mean_r:.4f}")
        return {'hr_mae': mae, 'hr_rmse': rmse, 'waveform_pearson_r': mean_r}
