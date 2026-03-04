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

from losses import NegPearsonScalarLoss
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
        print(f"[SpO2Finetune] Training  : spo2_branch + spo2_fusion_head + ratio_refiner")

        ft = cfg.RPPG_SPO2_FINETUNE
        self.lambda_consist  = getattr(ft, 'LAMBDA_CONSIST', 0.5)
        self.lambda_div      = getattr(ft, 'LAMBDA_DIV',     1.0)
        self.grad_clip       = cfg.RPPG_TRAIN.GRAD_CLIP
        self.neg_pearson     = NegPearsonScalarLoss()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pearson r over a batch [B, 1] → scalar tensor."""
        p = pred.view(-1) - pred.mean()
        t = target.view(-1) - target.mean()
        r = (p * t).sum() / (
            p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt() + 1e-8)
        return r

    @staticmethod
    def _augment_video(video: torch.Tensor) -> torch.Tensor:
        """
        SpO2-safe augmentation: 不改變血氧生理特徵，僅干擾像素細節。

        SpO2 由 R/IR 光吸收比估計，對輕微亮度/對比度變化不敏感。
        加入這些擾動後，模型若仍預測一致，代表學到真實特徵而非 artifact。

        Input/output: [B, C, T, H, W] float32, range [0, 255]
        """
        B = video.shape[0]
        # 亮度縮放 [0.90, 1.10]：per-sample 整體乘法，保留 R/G/B 通道比值
        brightness = 0.90 + 0.20 * torch.rand(B, 1, 1, 1, 1, device=video.device)
        aug = video * brightness

        # 對比度縮放：per-channel 均值為中心，不改變通道間相對強度
        # SpO2 依賴 R/G 比值，必須 per-channel 計算中心點
        contrast  = 0.90 + 0.20 * torch.rand(B, 1, 1, 1, 1, device=video.device)
        chan_mean  = aug.mean(dim=(2, 3, 4), keepdim=True)   # [B, C, 1, 1, 1]
        aug = (aug - chan_mean) * contrast + chan_mean

        # 高斯雜訊 σ=3（[0,255] 約 1.2%）
        aug = aug + torch.randn_like(aug) * 3.0
        return aug.clamp(0.0, 255.0)

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

        best_r            = -float('inf')   # maximize val_r（主要指標）
        best_mae          =  float('inf')   # minimize val_mae（r=NaN 時備用）
        patience_counter  = 0
        history = {'loss_spo2': [], 'val_r': []}

        csv_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_log").replace(".pth", ".csv")
        csv_path = os.path.join(self.cfg.OUTPUT.LOG_DIR, csv_name)
        os.makedirs(self.cfg.OUTPUT.LOG_DIR, exist_ok=True)
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss_spo2', 'val_mae', 'val_r', 'best'])
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

                # ── 監督 loss（原始影片）────────────────────────────────────
                _, pred_spo2 = self.model(video)

                # MSE（主要）+ MAE（抑制離群值）
                # 不用 NegPearson 作 training loss：batch=8 + SpO2 窄範圍 (σ≈0.95%)
                # → batch Pearson r 信賴區間極寬 → 梯度方向為隨機雜訊 → r 越訓練越低
                loss_mse = F.mse_loss(pred_spo2, spo2_label)
                loss_mae = F.l1_loss(pred_spo2, spo2_label)
                loss_sup = loss_mse + 0.5 * loss_mae

                # batch_r 僅監控（不加入梯度）
                batch_r = self._batch_pearson(pred_spo2.detach(), spo2_label)

                # ── 一致性：擾動影片仍對標籤負責（MSE 穩定）────────────
                _, pred_spo2_aug = self.model(self._augment_video(video))
                loss_consist = (F.mse_loss(pred_spo2_aug, spo2_label)
                                + 0.5 * F.l1_loss(pred_spo2_aug, spo2_label))

                # ── Diversity：防止 collapse（std(pred) < 1.0% 才觸發）─
                # 原閾值 0.3 過低，batch 均值擾動就能繞過；提高至 1.0
                loss_div = F.relu(1.0 - pred_spo2.std())

                loss_total = (loss_sup
                              + self.lambda_consist * loss_consist
                              + self.lambda_div     * loss_div)

                opt.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                opt.step()

                sum_spo2 += loss_total.item()
                n_steps  += 1

                pbar.set_postfix(
                    mse=f"{loss_mse.item():.3f}",
                    div=f"{loss_div.item():.3f}",
                    r=f"{batch_r.item():.3f}",
                )

            sched.step()

            avg_spo2 = sum_spo2 / max(n_steps, 1)
            history['loss_spo2'].append(avg_spo2)

            # ── Validation ────────────────────────────────────────────────────
            is_best  = False
            val_mae  = float('nan')
            val_r    = float('nan')
            if valid_loader:
                val_mae, val_r = self._validate(valid_loader)
                r_valid = not (val_r != val_r)   # True when val_r is not NaN

                history['val_r'].append(val_r if r_valid else float('nan'))
                r_disp = f"{val_r:.4f}" if r_valid else "NaN"
                print(f"  Epoch {epoch+1:3d} | loss: {avg_spo2:.4f} | "
                      f"Val MAE: {val_mae:.3f}%  r: {r_disp}")

                # r 有效時以 r 為主（越高越好），各自與各自的 best 比較
                # 避免 r 和 MAE 跨量級混比（例如 r=0.4 vs -MAE=-0.5 比較無意義）
                if r_valid:
                    improved = val_r > best_r
                    if improved:
                        best_r = val_r
                else:
                    improved = val_mae < best_mae
                    if improved:
                        best_mae = val_mae

                if improved:
                    patience_counter = 0
                    is_best          = True
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="spo2ft_best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    os.makedirs(self.cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)
                    self.save_checkpoint(self.model, opt, epoch, path,
                                         val_loss=val_mae)
                    metric_str = f"r={val_r:.4f}" if r_valid else f"MAE={val_mae:.3f}%"
                    print(f"  -> Best {metric_str}  Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        csv_writer.writerow([epoch+1, f"{avg_spo2:.4f}",
                                             f"{val_mae:.4f}",
                                             r_disp, ''])
                        csv_file.flush()
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"  Epoch {epoch+1:3d} | loss: {avg_spo2:.4f}")

            csv_writer.writerow([epoch+1, f"{avg_spo2:.4f}",
                                  f"{val_mae:.4f}", f"{val_r:.4f}",
                                  '*' if is_best else ''])
            csv_file.flush()

        csv_file.close()
        ckpt_name    = make_rppg_ckpt_name(self.cfg, tag="spo2ft")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    # ── Validation：SpO2 MAE + r ──────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader):
        """Returns (mae, pearson_r). Early stopping targets maximum r."""
        from scipy.stats import pearsonr as scipy_pearsonr
        import numpy as np

        self.model.eval()
        all_preds = []
        all_gts   = []

        for video, _, spo2_label in loader:
            video = video.to(self.device)
            _, pred_spo2 = self.model(video)
            all_preds.append(pred_spo2.cpu().numpy())
            all_gts.append(spo2_label.numpy())

        self.model.train()

        preds = np.concatenate(all_preds).ravel()
        gts   = np.concatenate(all_gts).ravel()

        # filter spo2=0 fallback values (label is spo2 - 80, so threshold is -30)
        mask  = gts > -30.0
        preds, gts = preds[mask], gts[mask]

        mae = float(np.mean(np.abs(preds - gts))) if len(gts) else float('nan')
        try:
            r = float(scipy_pearsonr(preds, gts)[0]) if len(gts) > 1 else float('nan')
        except ValueError:
            # target 為常數時（MCD-rPPG SpO2 窄範圍可能發生），r 無定義
            r = float('nan')
        return mae, r

    def validate(self, data_loader):
        mae, r = self._validate(data_loader)
        return mae

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
                    if spo2_val > -30.0:
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
