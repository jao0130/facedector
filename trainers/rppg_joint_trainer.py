"""
rPPG Joint Multi-task Semi-supervised Trainer

同時訓練 BVP + SpO2，並用 Mean Teacher 半監督 CelebV-HQ 無標籤資料。

訓練資料：
  Labeled  : MCD-rPPG + UBFC  (BVP + SpO2，其中 UBFC SpO2=None → 濾除)
  Unlabeled: CelebV-HQ        (無標籤，用於 Mean Teacher 一致性)
測試資料：
  PURE (全集，BVP + SpO2，不參與訓練)

Loss 設計：
  L_bvp       = NegPearson(pred_wave, bvp_label)          [labeled, 每步]
  L_spo2      = MSE + 0.5*MAE(pred_spo2, spo2_label)      [labeled, 有效標籤]
  L_hr_semi   = MSE(znorm(s_wave), znorm(t_wave))          [unlabeled, rampup 後]
  L_spo2_semi = MSE(s_spo2, t_spo2)                       [unlabeled, warmup 後]

  L_total = L_bvp + λ_spo2*L_spo2
          + λ_unsup * rampup(epoch) * (λ_hr*L_hr_semi + λ_spo2*L_spo2_semi)

SpO2 Semi 啟動：
  前 SPO2_WARMUP_EPOCHS epoch 只用監督 SpO2 loss，等分支穩定後
  才加入 teacher-student SpO2 一致性（避免 teacher 偏移傳播）。

SpO2 標籤約定（label = SpO2% − 80）：
  有效範圍 : [0, 20]  (SpO2 80%~100%)
  無 SpO2  : −80.0   (原始 0.0 − 80.0)
  過濾閾值 : > −30.0
"""

import copy
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import NegPearsonLoss
from .base_trainer import BaseTrainer, make_rppg_ckpt_name

# SpO2 label filter (labels stored as spo2% - 80; fallback = 0 - 80 = -80)
_SPO2_VALID_THRESH = -30.0


def _infinite(loader):
    """每步從 unlabeled 隨機取一批，耗盡後重建 iterator（新一輪 shuffle）。"""
    while True:
        for batch in loader:
            yield batch


def _sigmoid_rampup(current: int, rampup_length: int) -> float:
    """Sigmoid ramp-up weight: 0 → 1 over rampup_length epochs."""
    if rampup_length == 0:
        return 1.0
    current = max(0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(torch.exp(torch.tensor(-5.0 * phase * phase)).item())


def _znorm(x: torch.Tensor) -> torch.Tensor:
    """Z-normalize along last dimension (time)."""
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True).clamp(min=1e-6))


class rPPGJointTrainer(BaseTrainer):
    """
    Joint multi-task semi-supervised trainer.

    self.student : FCAtt_v3 (fully trainable — all branches)
    self.teacher : FCAtt_v3 (EMA of student, no grad)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt_v3')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.student = FCAtt_v3(cfg=cfg).to(self.device)
        else:
            raise ValueError(f"[Joint] Unsupported model: {model_name}")

        # Teacher: deep-copy, no gradients, updated by EMA only
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Optionally load a pretrained checkpoint
        weights_path = cfg.RPPG_MODEL.WEIGHTS
        if weights_path and os.path.isfile(weights_path):
            ckpt  = torch.load(weights_path, map_location=self.device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            missing, _ = self.student.load_state_dict(state, strict=False)
            self.teacher.load_state_dict(state, strict=False)
            print(f"[Joint] Loaded checkpoint: {weights_path}")
            if missing:
                print(f"[Joint] Missing keys ({len(missing)}): {missing[:3]} ...")
        else:
            print("[Joint] Training from scratch (no checkpoint specified)")

        jt   = cfg.RPPG_JOINT
        semi = cfg.RPPG_SEMI

        self.ema_decay           = getattr(semi, 'EMA_DECAY',         0.999)
        self.ramp_up_epochs      = getattr(semi, 'RAMP_UP_EPOCHS',    30)
        self.lambda_unsup        = getattr(semi, 'LAMBDA_UNSUP',      1.0)
        self.lambda_spo2         = getattr(jt,   'LAMBDA_SPO2',       1.0)
        self.lambda_hr_semi      = getattr(jt,   'LAMBDA_HR_SEMI',    1.0)
        self.lambda_spo2_semi    = getattr(jt,   'LAMBDA_SPO2_SEMI',  0.5)
        self.spo2_warmup_epochs  = getattr(jt,   'SPO2_WARMUP_EPOCHS', 30)
        self.grad_clip           = getattr(jt,   'GRAD_CLIP',         cfg.RPPG_TRAIN.GRAD_CLIP)

        trainable = sum(p.numel() for p in self.student.parameters())
        print(f"[Joint] Student params: {trainable:,}")
        print(f"[Joint] EMA decay: {self.ema_decay}  "
              f"rampup: {self.ramp_up_epochs}ep  "
              f"SpO2 warmup: {self.spo2_warmup_epochs}ep")

        self.neg_pearson = NegPearsonLoss()

    # ── EMA ───────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_teacher(self):
        """EMA: teacher ← decay·teacher + (1−decay)·student."""
        d = self.ema_decay
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = d * p_t.data + (1.0 - d) * p_s.data

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, data_loaders: dict) -> dict:
        train_loader     = data_loaders['train']
        valid_loader     = data_loaders.get('valid')
        unlabeled_loader = data_loaders.get('unlabeled')

        jt           = self.cfg.RPPG_JOINT
        total_epochs = getattr(jt, 'EPOCHS',   100)
        patience     = getattr(jt, 'PATIENCE',  30)

        opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=getattr(jt, 'LR', 1e-3), weight_decay=1e-4,
        )
        sched = CosineAnnealingLR(
            opt, T_max=total_epochs,
            eta_min=getattr(jt, 'LR', 1e-3) * 0.01,
        )

        best_r           = -float('inf')
        patience_counter = 0
        history = {'loss_bvp': [], 'loss_spo2': [], 'loss_semi': [], 'val_r': []}

        csv_name = make_rppg_ckpt_name(self.cfg, tag="joint_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        os.makedirs(self.cfg.OUTPUT.LOG_DIR, exist_ok=True)
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss_bvp', 'loss_spo2', 'loss_semi', 'val_r', 'best'])
        csv_file.flush()

        unlabeled_iter = (_infinite(unlabeled_loader)
                          if unlabeled_loader else None)

        for epoch in range(total_epochs):
            self.student.train()
            self.teacher.eval()

            rampup        = _sigmoid_rampup(epoch, self.ramp_up_epochs)
            spo2_semi_on  = (epoch >= self.spo2_warmup_epochs)

            sum_bvp = sum_spo2 = sum_semi = 0.0
            n_steps = 0

            tag = "js" if spo2_semi_on else "h "
            pbar = tqdm(
                train_loader,
                desc=f"Joint {epoch+1}/{total_epochs} ramp={rampup:.2f} [{tag}]",
                leave=False,
            )

            for video, bvp_label, spo2_label in pbar:
                video      = video.to(self.device)
                bvp_label  = bvp_label.to(self.device)
                spo2_label = spo2_label.to(self.device)   # [B, 1], label = SpO2% − 80

                opt.zero_grad()

                # ── Pass 1: Supervised (labeled) ──────────────────────────────
                # backward() 立刻釋放此 pass 的 activation，不等 unlabeled pass
                pred_wave, pred_spo2 = self.student(video)
                L_bvp = self.neg_pearson(pred_wave, bvp_label)

                # UBFC has no SpO2 → loaded as 0 − 80 = −80 → filtered by threshold
                valid_mask = (spo2_label > _SPO2_VALID_THRESH).squeeze(1)  # [B]
                if valid_mask.any():
                    p_v = pred_spo2[valid_mask]
                    g_v = spo2_label[valid_mask]
                    L_spo2 = F.mse_loss(p_v, g_v) + 0.5 * F.l1_loss(p_v, g_v)
                else:
                    L_spo2 = torch.tensor(0.0, device=self.device)

                L_labeled = L_bvp + self.lambda_spo2 * L_spo2
                L_labeled.backward()   # ← 釋放 labeled activation；unlabeled 還未載入

                # ── Pass 2: Semi-supervised (unlabeled) ───────────────────────
                # 此時 GPU 只有 labeled 的梯度 + unlabeled 的 activation，peak 減半
                L_semi = torch.tensor(0.0, device=self.device)
                if unlabeled_iter is not None and rampup > 0.0:
                    unlab_video = next(unlabeled_iter).to(self.device)

                    s_wave, s_spo2 = self.student(unlab_video)
                    with torch.no_grad():
                        t_wave, t_spo2 = self.teacher(unlab_video)

                    L_hr_semi = F.mse_loss(_znorm(s_wave), _znorm(t_wave))

                    if spo2_semi_on:
                        L_spo2_semi = F.mse_loss(s_spo2, t_spo2)
                    else:
                        L_spo2_semi = torch.tensor(0.0, device=self.device)

                    L_semi = rampup * self.lambda_unsup * (
                        self.lambda_hr_semi   * L_hr_semi
                        + self.lambda_spo2_semi * L_spo2_semi
                    )
                    L_semi.backward()  # ← 釋放 unlabeled activation；梯度累加

                nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                opt.step()
                self._update_teacher()

                sum_bvp  += L_bvp.item()
                sum_spo2 += L_spo2.item()
                sum_semi += L_semi.item()
                n_steps  += 1

                pbar.set_postfix(
                    bvp=f"{L_bvp.item():.3f}",
                    spo2=f"{L_spo2.item():.3f}",
                    semi=f"{L_semi.item():.3f}",
                )

            sched.step()

            avg_bvp  = sum_bvp  / max(n_steps, 1)
            avg_spo2 = sum_spo2 / max(n_steps, 1)
            avg_semi = sum_semi / max(n_steps, 1)
            history['loss_bvp'].append(avg_bvp)
            history['loss_spo2'].append(avg_spo2)
            history['loss_semi'].append(avg_semi)

            # ── Validation ────────────────────────────────────────────────────
            is_best = False
            val_r   = float('nan')
            if valid_loader:
                val_r   = self._validate(valid_loader)
                r_valid = val_r == val_r   # True when not NaN
                r_disp  = f"{val_r:.4f}" if r_valid else "NaN"
                print(f"  Epoch {epoch+1:3d} | "
                      f"bvp: {avg_bvp:.4f}  spo2: {avg_spo2:.4f}  "
                      f"semi: {avg_semi:.4f}  val_r: {r_disp}")
                history['val_r'].append(val_r if r_valid else float('nan'))

                if r_valid and val_r > best_r:
                    best_r           = val_r
                    patience_counter = 0
                    is_best          = True
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="joint_best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    os.makedirs(self.cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)
                    self.save_checkpoint(self.student, opt, epoch, path,
                                         val_loss=1.0 - val_r)
                    print(f"  -> Best val_r={val_r:.4f}  Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        csv_writer.writerow([epoch + 1,
                                             f"{avg_bvp:.4f}", f"{avg_spo2:.4f}",
                                             f"{avg_semi:.4f}", r_disp, ''])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break
            else:
                print(f"  Epoch {epoch+1:3d} | "
                      f"bvp: {avg_bvp:.4f}  spo2: {avg_spo2:.4f}  semi: {avg_semi:.4f}")

            csv_writer.writerow([epoch + 1,
                                  f"{avg_bvp:.4f}", f"{avg_spo2:.4f}",
                                  f"{avg_semi:.4f}", f"{val_r:.4f}",
                                  '*' if is_best else ''])
            csv_file.flush()

        csv_file.close()
        ckpt_name    = make_rppg_ckpt_name(self.cfg, tag="joint")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    # ── Validation：BVP 波形 Pearson r ────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> float:
        """Returns mean Pearson r on z-normed BVP waveform."""
        from scipy.stats import pearsonr
        import numpy as np

        self.student.eval()
        rs = []

        for video, bvp_label, _ in loader:
            video     = video.to(self.device)
            pred_wave, _ = self.student(video)
            p_norm = _znorm(pred_wave).cpu().numpy()
            g_norm = _znorm(bvp_label.to(self.device)).cpu().numpy()
            for i in range(p_norm.shape[0]):
                try:
                    r, _ = pearsonr(p_norm[i], g_norm[i])
                    if r == r:   # not NaN
                        rs.append(r)
                except ValueError:
                    pass

        self.student.train()
        return float(np.mean(rs)) if rs else float('nan')

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    # ── Test：HR + SpO2（使用 student）──────────────────────────────────────────

    def test(self, data_loader, dataset_name: str = "") -> dict:
        from utils.signal_processing import estimate_hr_fft
        from scipy.stats import pearsonr
        import numpy as np

        self.student.eval()
        hr_preds   = []
        hr_gts     = []
        spo2_preds = []
        spo2_gts   = []
        wave_rs    = []

        with torch.no_grad():
            for video, bvp_label, spo2_label in data_loader:
                video = video.to(self.device)
                pred_wave, pred_spo2 = self.student(video)

                p_norm = _znorm(pred_wave)
                g_norm = _znorm(bvp_label.to(self.device))

                for i in range(pred_wave.shape[0]):
                    try:
                        r, _ = pearsonr(p_norm[i].cpu().numpy(),
                                        g_norm[i].cpu().numpy())
                        wave_rs.append(r if r == r else 0.0)
                    except ValueError:
                        wave_rs.append(0.0)

                    hr_preds.append(estimate_hr_fft(
                        pred_wave[i].cpu().numpy(), self.cfg.RPPG_MODEL.FPS))
                    hr_gts.append(estimate_hr_fft(
                        bvp_label[i].numpy(), self.cfg.RPPG_MODEL.FPS))

                    spo2_val = float(spo2_label[i].item())
                    if spo2_val > _SPO2_VALID_THRESH:
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
            spo2_r     = (float(pearsonr(spo2_preds, spo2_gts)[0])
                          if len(spo2_preds) > 1 else float('nan'))
            print(f"  [Test]{tag} SpO2 MAE: {spo2_mae:.3f}  | "
                  f"RMSE: {spo2_rmse:.3f}  | r: {spo2_r:.4f} "
                  f"(n={len(spo2_preds)})")
            result.update({'spo2_mae': spo2_mae,
                           'spo2_rmse': spo2_rmse,
                           'spo2_r': spo2_r})
        else:
            print(f"  [Test]{tag} SpO2: 此資料集無有效 SpO2 標籤")

        return result
