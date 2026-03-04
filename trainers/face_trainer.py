"""Face detection trainer with WarmupCosineDecay schedule."""

import os
import math
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from .base_trainer import BaseTrainer, make_face_ckpt_name
from models.face_detector import FaceDetector
from losses.face_losses import FaceDetectionLoss
from metrics.face_metrics import IoUTracker, NMETracker


class WarmupCosineDecay(_LRScheduler):
    """Linear warmup then cosine decay schedule."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.01, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


class FaceDetectorTrainer(BaseTrainer):
    """
    Trains face detection model.
    AdamW optimizer + WarmupCosineDecay + GIoU/SmoothL1/BCE loss + early stopping.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = FaceDetector(cfg=cfg).to(self.device)
        self.criterion = FaceDetectionLoss(cfg=cfg)

    def train(self, data_loaders: dict) -> dict:
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.FACE_TRAIN.LR,
            weight_decay=self.cfg.FACE_TRAIN.WEIGHT_DECAY,
        )

        steps_per_epoch = len(train_loader)
        total_steps = self.cfg.FACE_TRAIN.EPOCHS * steps_per_epoch
        warmup_steps = self.cfg.FACE_TRAIN.WARMUP_EPOCHS * steps_per_epoch

        scheduler = WarmupCosineDecay(optimizer, warmup_steps, total_steps,
                                      min_lr_ratio=self.cfg.FACE_TRAIN.MIN_LR / self.cfg.FACE_TRAIN.LR)

        history = {'train_loss': [], 'val_loss': [], 'val_1miou': [], 'val_nme': [], 'lr': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.cfg.FACE_TRAIN.EARLY_STOPPING_PATIENCE

        print(f"[Train] Face Detection | Epochs: {self.cfg.FACE_TRAIN.EPOCHS} | "
              f"Batch: {self.cfg.FACE_TRAIN.BATCH_SIZE} | LR: {self.cfg.FACE_TRAIN.LR}")
        print(f"[Train] All metrics: lower = better (loss, 1-IoU, NME)")

        for epoch in range(self.cfg.FACE_TRAIN.EPOCHS):
            # Train
            train_loss = self._train_one_epoch(train_loader, optimizer, scheduler, epoch)

            # Validate
            val_loss, val_1miou, val_nme = self._validate(val_loader)

            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_1miou'].append(val_1miou)
            history['val_nme'].append(val_nme)
            history['lr'].append(current_lr)

            print(f"  Epoch {epoch+1}/{self.cfg.FACE_TRAIN.EPOCHS} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"1-IoU: {val_1miou:.4f} | NME: {val_nme:.4f} | LR: {current_lr:.6f}")

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_name = make_face_ckpt_name(self.cfg, tag="best")
                best_path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, ckpt_name)
                self.save_checkpoint(self.model, optimizer, epoch, best_path,
                                     val_loss=val_loss, val_1miou=val_1miou)
                print(f"  -> Saved best model: {ckpt_name} (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Save last epoch checkpoint
        last_name = make_face_ckpt_name(self.cfg, tag="last")
        last_path = os.path.join(self.cfg.OUTPUT.CHECKPOINT_DIR, last_name)
        self.save_checkpoint(self.model, optimizer, epoch, last_path,
                             val_loss=val_loss, val_1miou=val_1miou)
        print(f"  -> Saved last model: {last_name}")

        # Save history
        ckpt_name = make_face_ckpt_name(self.cfg, tag="best")
        history_name = ckpt_name.replace(".pth", "_history.json")
        self.save_history(history, os.path.join(self.cfg.OUTPUT.LOG_DIR, history_name))
        return history

    def _train_one_epoch(self, loader, optimizer, scheduler, epoch) -> float:
        self.model.train()
        total_loss = 0.0
        grad_clip = self.cfg.FACE_TRAIN.GRAD_CLIP

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for images, gt_bbox, gt_landmarks in pbar:
            images = images.to(self.device)
            gt_bbox = gt_bbox.to(self.device)
            gt_landmarks = gt_landmarks.to(self.device)

            predictions = self.model(images)
            loss, loss_dict = self.criterion(predictions, gt_bbox, gt_landmarks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, data_loader) -> float:
        val_loss, _, _ = self._validate(data_loader)
        return val_loss

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        iou_tracker = IoUTracker()
        nme_tracker = NMETracker()

        for images, gt_bbox, gt_landmarks in loader:
            images = images.to(self.device)
            gt_bbox = gt_bbox.to(self.device)
            gt_landmarks = gt_landmarks.to(self.device)

            predictions = self.model(images)
            loss, _ = self.criterion(predictions, gt_bbox, gt_landmarks)
            total_loss += loss.item()

            iou_tracker.update(predictions['bbox'], gt_bbox)
            nme_tracker.update(predictions['landmarks'], gt_landmarks, gt_bbox)

        val_loss = total_loss / max(len(loader), 1)
        val_1miou = 1.0 - iou_tracker.compute()
        return val_loss, val_1miou, nme_tracker.compute()

    def test(self, data_loader) -> dict:
        val_loss, val_1miou, val_nme = self._validate(data_loader)
        return {'test_loss': val_loss, 'test_1-IoU': val_1miou, 'test_nme': val_nme}
