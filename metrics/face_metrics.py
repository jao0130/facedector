"""Face detection metrics: IoU, NME trackers."""

import torch
import numpy as np


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1, box2: [B, 4] (x_min, y_min, x_max, y_max)

    Returns:
        [B] IoU scores.
    """
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter_area + 1e-7

    return inter_area / union


def compute_nme(pred_lm: torch.Tensor, gt_lm: torch.Tensor,
                gt_bbox: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean Error for landmarks using inter-ocular distance.

    Args:
        pred_lm: [B, 5, 2]
        gt_lm: [B, 5, 2]
        gt_bbox: [B, 4]

    Returns:
        Scalar NME.
    """
    # Inter-ocular distance: left_eye (0) to right_eye (1)
    left_eye = gt_lm[:, 0, :]  # [B, 2]
    right_eye = gt_lm[:, 1, :]
    inter_ocular = torch.norm(left_eye - right_eye, dim=1, keepdim=True) + 1e-7  # [B, 1]

    # Mean error per landmark
    diff = torch.norm(pred_lm - gt_lm, dim=2)  # [B, 5]
    nme_per_sample = diff.mean(dim=1, keepdim=True) / inter_ocular  # [B, 1]

    return nme_per_sample.mean()


class IoUTracker:
    """Running mean IoU tracker."""

    def __init__(self):
        self.reset()

    def update(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor):
        ious = compute_iou(pred_bbox, gt_bbox)
        self._sum += ious.sum().item()
        self._count += ious.numel()

    def compute(self) -> float:
        return self._sum / max(self._count, 1)

    def reset(self):
        self._sum = 0.0
        self._count = 0


class NMETracker:
    """Running mean NME tracker."""

    def __init__(self):
        self.reset()

    def update(self, pred_lm: torch.Tensor, gt_lm: torch.Tensor, gt_bbox: torch.Tensor):
        nme = compute_nme(pred_lm, gt_lm, gt_bbox).item()
        batch_size = pred_lm.shape[0]
        self._sum += nme * batch_size
        self._count += batch_size

    def compute(self) -> float:
        return self._sum / max(self._count, 1)

    def reset(self):
        self._sum = 0.0
        self._count = 0
