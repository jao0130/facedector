"""
Two-stage rPPG trainer: BVP pre-train (60%) -> SpO2 fine-tune (40%).
Ported from rPPG-Toolbox BvpSpo2Trainer.
"""

import os
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_rppg_ckpt_name
from models.rppg_model import FCAtt
from losses.rppg_losses import NegPearsonLoss, WeightedMSELoss


class rPPGTrainer(BaseTrainer):
    """
    Two-stage rPPG trainer.

    Stage 1: BVP pre-training (PRETRAIN_RATIO of epochs)
        - NegPearson loss only
        - Train full model

    Stage 2: SpO2 fine-tuning (remaining epochs)
        - Transfer weights from Stage 1
        - Freeze ConvBlock1-9
        - NegPearson + WeightedMSE
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bvp_criterion = NegPearsonLoss()
        self.spo2_criterion = WeightedMSELoss()
        self.grad_accum = cfg.RPPG_TRAIN.GRAD_ACCUMULATION
        self.grad_clip = cfg.RPPG_TRAIN.GRAD_CLIP

    def train(self, data_loaders: dict) -> dict:
        train_loader = data_loaders['train']
        valid_loader = data_loaders.get('valid')

        total_epochs = self.cfg.RPPG_TRAIN.EPOCHS
        pretrain_epochs = int(total_epochs * self.cfg.RPPG_TRAIN.PRETRAIN_RATIO)
        finetune_epochs = total_epochs - pretrain_epochs

        history = {'stage1_loss': [], 'stage2_loss_bvp': [], 'stage2_loss_spo2': []}

        # Stage 1: BVP pre-training
        print(f"\n[rPPG] Stage 1: BVP Pre-training ({pretrain_epochs} epochs)")
        pre_model = FCAtt(cfg=self.cfg).to(self.device)
        history['stage1_loss'] = self._train_bvp(
            pre_model, train_loader, valid_loader, pretrain_epochs, stage_name="pretrain",
        )

        # Stage 2: SpO2 fine-tuning
        print(f"\n[rPPG] Stage 2: SpO2 Fine-tuning ({finetune_epochs} epochs)")
        ppg_model = FCAtt(cfg=self.cfg).to(self.device)
        self._transfer_weights(pre_model, ppg_model)
        self._freeze_encoder(ppg_model)
        del pre_model
        torch.cuda.empty_cache()

        s2_bvp, s2_spo2 = self._train_spo2(
            ppg_model, train_loader, valid_loader, finetune_epochs,
        )
        history['stage2_loss_bvp'] = s2_bvp
        history['stage2_loss_spo2'] = s2_spo2

        ckpt_name = make_rppg_ckpt_name(self.cfg, tag="best")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    def _train_bvp(self, model, train_loader, valid_loader, epochs, stage_name="pretrain"):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.RPPG_TRAIN.LR)
        scheduler = OneCycleLR(
            optimizer, max_lr=self.cfg.RPPG_TRAIN.LR,
            epochs=epochs, steps_per_epoch=len(train_loader),
        )

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.cfg.RPPG_TRAIN.PATIENCE_BVP
        losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"BVP Epoch {epoch+1}/{epochs}", leave=False)
            for step, (video, bvp_label, _) in enumerate(pbar):
                video = video.to(self.device)
                bvp_label = bvp_label.to(self.device)

                pred_ppg, _ = model(video)

                # Z-score normalize
                pred_norm = (pred_ppg - pred_ppg.mean(dim=1, keepdim=True)) / (pred_ppg.std(dim=1, keepdim=True) + 1e-8)
                label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)

                loss = self.bvp_criterion(pred_norm, label_norm) * 100.0

                (loss / self.grad_accum).backward()

                if (step + 1) % self.grad_accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)

            # Validation
            if valid_loader is not None:
                val_loss = self._validate_bvp(model, valid_loader)
                print(f"  BVP Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag=stage_name)
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    self.save_checkpoint(model, optimizer, epoch, path, val_loss=val_loss)
                    print(f"  -> Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"  BVP Epoch {epoch+1} | Train: {avg_loss:.4f}")

        return losses

    def _train_spo2(self, model, train_loader, valid_loader, epochs):
        # Only train unfrozen parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.RPPG_TRAIN.LR)
        scheduler = OneCycleLR(
            optimizer, max_lr=self.cfg.RPPG_TRAIN.LR,
            epochs=epochs, steps_per_epoch=len(train_loader),
        )

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.cfg.RPPG_TRAIN.PATIENCE_SPO2
        bvp_losses = []
        spo2_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_bvp = 0.0
            epoch_spo2 = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"SpO2 Epoch {epoch+1}/{epochs}", leave=False)
            for step, (video, bvp_label, spo2_label) in enumerate(pbar):
                video = video.to(self.device)
                bvp_label = bvp_label.to(self.device)
                spo2_label = spo2_label.to(self.device)

                pred_ppg, spo2_pred = model(video)

                # BVP loss
                pred_norm = (pred_ppg - pred_ppg.mean(dim=1, keepdim=True)) / (pred_ppg.std(dim=1, keepdim=True) + 1e-8)
                label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)
                loss_bvp = self.bvp_criterion(pred_norm, label_norm) * 100.0

                # SpO2 loss
                loss_spo2 = self.spo2_criterion(spo2_pred, spo2_label)

                total_loss = loss_bvp + loss_spo2

                (total_loss / self.grad_accum).backward()

                if (step + 1) % self.grad_accum == 0:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        self.grad_clip,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_bvp += loss_bvp.item()
                epoch_spo2 += loss_spo2.item()
                pbar.set_postfix(bvp=f"{loss_bvp.item():.2f}", spo2=f"{loss_spo2.item():.2f}")

            avg_bvp = epoch_bvp / len(train_loader)
            avg_spo2 = epoch_spo2 / len(train_loader)
            bvp_losses.append(avg_bvp)
            spo2_losses.append(avg_spo2)

            if valid_loader is not None:
                val_loss = self._validate_spo2(model, valid_loader)
                print(f"  SpO2 Epoch {epoch+1} | BVP: {avg_bvp:.4f} | SpO2: {avg_spo2:.4f} | Val: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_name = make_rppg_ckpt_name(self.cfg, tag="best")
                    path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                    self.save_checkpoint(model, optimizer, epoch, path, val_loss=val_loss)
                    print(f"  -> Saved: {ckpt_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"  SpO2 Epoch {epoch+1} | BVP: {avg_bvp:.4f} | SpO2: {avg_spo2:.4f}")

        return bvp_losses, spo2_losses

    @torch.no_grad()
    def _validate_bvp(self, model, loader) -> float:
        model.eval()
        total_loss = 0.0
        for video, bvp_label, _ in loader:
            video = video.to(self.device)
            bvp_label = bvp_label.to(self.device)
            pred_ppg, _ = model(video)
            pred_norm = (pred_ppg - pred_ppg.mean(dim=1, keepdim=True)) / (pred_ppg.std(dim=1, keepdim=True) + 1e-8)
            label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)
            loss = self.bvp_criterion(pred_norm, label_norm) * 100.0
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _validate_spo2(self, model, loader) -> float:
        model.eval()
        total_loss = 0.0
        for video, bvp_label, spo2_label in loader:
            video = video.to(self.device)
            bvp_label = bvp_label.to(self.device)
            spo2_label = spo2_label.to(self.device)
            pred_ppg, spo2_pred = model(video)
            pred_norm = (pred_ppg - pred_ppg.mean(dim=1, keepdim=True)) / (pred_ppg.std(dim=1, keepdim=True) + 1e-8)
            label_norm = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + 1e-8)
            loss_bvp = self.bvp_criterion(pred_norm, label_norm) * 100.0
            loss_spo2 = self.spo2_criterion(spo2_pred, spo2_label)
            total_loss += (loss_bvp + loss_spo2).item()
        return total_loss / max(len(loader), 1)

    def validate(self, data_loader) -> float:
        model = FCAtt(cfg=self.cfg).to(self.device)
        return self._validate_bvp(model, data_loader)

    def test(self, data_loader) -> dict:
        return {'test_loss': 0.0}

    @staticmethod
    def _transfer_weights(src_model, dst_model):
        """Transfer weights from pre-trained model to fine-tune model."""
        src_dict = src_model.state_dict()
        dst_dict = dst_model.state_dict()
        for key in src_dict:
            if key in dst_dict and src_dict[key].shape == dst_dict[key].shape:
                dst_dict[key] = src_dict[key]
        dst_model.load_state_dict(dst_dict)

    @staticmethod
    def _freeze_encoder(model):
        """Freeze encoder ConvBlocks 1-9 for fine-tuning."""
        frozen_prefixes = [
            'ConvBlock1', 'ConvBlock2', 'ConvBlock3', 'ConvBlock4',
            'ConvBlock5', 'ConvBlock6', 'ConvBlock7', 'ConvBlock8', 'ConvBlock9',
            'residual_conv4', 'residual_conv6', 'residual_conv8',
            'att_mask1',
        ]
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                param.requires_grad = False
                frozen_count += 1
        print(f"  [Freeze] Frozen {frozen_count} parameters in encoder")
