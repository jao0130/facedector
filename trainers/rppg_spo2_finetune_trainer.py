"""
rPPG SpO2 Fine-tune Trainer (Stage-3)

凍結策略（重要）：
  凍結：stem, freq_blocks, hr_branch
  微調：spo2_branch, spo2_fusion_head

  ┌──────────────────────────────────────────────────────────────┐
  │  video → stem(❄) → freq_blocks(❄) → x                      │
  │                                      ├→ hr_branch(❄) → PPG  │  梯度不流入
  │                                      └→ spo2_branch(✓)      │
  │  spo2_fusion_head(✓)([PPG_detached, vpg, apg, spo2_feat])   │
  │                              ↑ PPG 已凍結，不影響波形預測     │
  └──────────────────────────────────────────────────────────────┘

為何不加 L_rppg：
  hr_branch 凍結後，pred_wave.requires_grad = False（其輸入路徑
  stem/freq_blocks/hr_branch 均無可訓練參數），L_rppg 對所有
  trainable params 的梯度為 0，加入 loss 只是浪費運算，不加。

資料：
  Train: MCD-rPPG（BVP + SpO2 均有標籤）
  Test : PURE（BVP + SpO2 均有標籤）

  無標籤資料（UBFC-rPPG，只有 BVP）已從此 stage 移除：
  hr_branch 凍結後 L_rppg 無梯度，spo2 loss 也無法套用（spo2=0）
  → 對任何可訓練參數無貢獻，無需載入。
"""

import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_rppg_ckpt_name


# ── SpO2 Fine-tune Trainer ────────────────────────────────────────────────────

class rPPGSpO2FinetuneTrainer(BaseTrainer):
    """
    Stage-3：只訓練 spo2_branch + spo2_fusion_head，完整保留 PPG 預測品質。
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── 建立並載入 checkpoint ───────────────────────────────────────────
        model_name = getattr(cfg.RPPG_MODEL, 'NAME', 'FCAtt_v3')
        if model_name == 'FCAtt_v3':
            from models.rppg_model_v3 import FCAtt_v3
            self.model = FCAtt_v3(cfg=cfg).to(self.device)
        else:
            raise ValueError(f"[SpO2Finetune] Unsupported model: {model_name}")

        weights_path = cfg.RPPG_MODEL.WEIGHTS
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"[SpO2Finetune] Weights not found: {weights_path}")
        ckpt  = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, _ = self.model.load_state_dict(state, strict=False)
        print(f"[SpO2Finetune] Loaded: {weights_path}")
        if missing:
            print(f"[SpO2Finetune] Missing keys ({len(missing)}): {missing[:3]} ...")

        # ── 凍結 PPG 路徑，只訓練 SpO2 分支 ────────────────────────────────
        # stem + freq_blocks + hr_branch 全部凍結
        # → pred_wave.requires_grad = False（整條路徑無可訓練參數）
        # → SpO2 loss 梯度不流入 hr_branch → PPG 預測完全不受影響
        FROZEN_PREFIXES = ('stem', 'freq_blocks', 'hr_branch')
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in FROZEN_PREFIXES):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        frozen    = total - trainable
        print(f"[SpO2Finetune] Trainable : {trainable:,}  ({trainable/total*100:.1f}%)")
        print(f"[SpO2Finetune] Frozen    : {frozen:,}  (stem + freq_blocks + hr_branch)")
        print(f"[SpO2Finetune] Training  : spo2_branch + spo2_fusion_head")

        ft = cfg.RPPG_SPO2_FINETUNE
        self.lambda_spo2 = ft.LAMBDA_SPO2
        self.grad_clip   = cfg.RPPG_TRAIN.GRAD_CLIP

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
        history = {'loss_spo2': [], 'val_loss': []}

        csv_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        os.makedirs(self.cfg.OUTPUT.LOG_DIR, exist_ok=True)
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss_spo2', 'val_loss', 'best'])
        csv_file.flush()

        for epoch in range(total_epochs):
            self.model.train()

            sum_spo2 = 0.0
            n_steps  = 0

            pbar = tqdm(train_loader,
                        desc=f"SpO2Ft {epoch+1}/{total_epochs}",
                        leave=False)

            for video, bvp_label, spo2_label in pbar:
                video      = video.to(self.device)
                spo2_label = spo2_label.to(self.device)   # [B, 1]

                _, pred_spo2 = self.model(video)

                # SpO2 MSE + MAE（hr_branch 凍結，L_spo2 梯度只流向 spo2 分支）
                loss_spo2 = (F.mse_loss(pred_spo2, spo2_label)
                             + 0.5 * F.l1_loss(pred_spo2, spo2_label))

                opt.zero_grad()
                loss_spo2.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                opt.step()

                sum_spo2 += loss_spo2.item()
                n_steps  += 1

                pbar.set_postfix(spo2=f"{loss_spo2.item():.4f}")

            sched.step()

            avg_spo2 = sum_spo2 / max(n_steps, 1)
            history['loss_spo2'].append(avg_spo2)

            # ── Validation ────────────────────────────────────────────────────
            is_best  = False
            val_loss = float('nan')
            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(f"  Epoch {epoch+1:3d} | spo2: {avg_spo2:.4f} | Val: {val_loss:.4f}")

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
                        csv_writer.writerow([epoch+1, f"{avg_spo2:.4f}",
                                             f"{val_loss:.4f}", ''])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"  Epoch {epoch+1:3d} | spo2: {avg_spo2:.4f}")

            csv_writer.writerow([epoch+1, f"{avg_spo2:.4f}",
                                  f"{val_loss:.4f}", '*' if is_best else ''])
            csv_file.flush()

        csv_file.close()
        ckpt_name    = make_rppg_ckpt_name(self.cfg, tag="spo2ft")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    # ── Validation：SpO2 MAE ──────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for video, _, spo2_label in loader:
            video      = video.to(self.device)
            spo2_label = spo2_label.to(self.device)
            _, pred_spo2 = self.model(video)
            total += F.l1_loss(pred_spo2, spo2_label).item()
        self.model.train()
        return total / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        return self._validate(data_loader)

    # ── Test：HR（PPG 凍結，驗證不受影響）+ SpO2 ─────────────────────────────

    def test(self, data_loader, dataset_name: str = "") -> dict:
        from utils.signal_processing import estimate_hr_fft
        import numpy as np
        from scipy.stats import pearsonr

        def _zscore(x):
            return (x - x.mean(dim=1, keepdim=True)) / (
                x.std(dim=1, keepdim=True).clamp(min=1e-8))

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
            spo2_r     = float(pearsonr(spo2_preds, spo2_gts)[0]) \
                         if len(spo2_preds) > 1 else float('nan')
            print(f"  [Test]{tag} SpO2 MAE: {spo2_mae:.3f}% | "
                  f"RMSE: {spo2_rmse:.3f}% | r: {spo2_r:.4f} "
                  f"(n={len(spo2_preds)})")
            result.update({'spo2_mae': spo2_mae, 'spo2_rmse': spo2_rmse, 'spo2_r': spo2_r})
        else:
            print(f"  [Test]{tag} SpO2: 此資料集無有效 SpO2 標籤")

        return result
