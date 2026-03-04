"""
Face detection losses: GIoU, SmoothL1, BCE, combined loss.
All components are non-negative. Lower = better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU loss for bounding boxes.

    Args:
        pred_boxes: [B, 4] (x_min, y_min, x_max, y_max), normalized [0,1].
        gt_boxes: [B, 4] same format.

    Returns:
        Scalar loss = (1 - GIoU).mean(), range [0, 2]. Lower = better.
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas (clamp to prevent negative areas from malformed boxes)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    union = area_pred + area_gt - inter_area + 1e-7

    iou = inter_area / union

    # Enclosing box
    enclose_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * \
                   (enclose_y2 - enclose_y1).clamp(min=0) + 1e-7

    giou = iou - (enclose_area - union) / enclose_area
    return (1.0 - giou).mean()


def landmark_loss(pred: torch.Tensor, target: torch.Tensor,
                  beta: float = 0.1) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss for normalized landmark coordinates.

    Args:
        pred: [B, 5, 2] predicted landmarks, normalized [0,1].
        target: [B, 5, 2] ground truth landmarks, normalized [0,1].
        beta: Smooth L1 transition point. 0.1 works well for [0,1] range.

    Returns:
        Scalar loss >= 0. Lower = better.
    """
    return F.smooth_l1_loss(pred, target, beta=beta)


def confidence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss with numerical safety clamping.

    Args:
        pred: [B, 1] predicted confidence (sigmoid output, [0,1]).
        target: [B, 1] ground truth (1.0 for positive samples).

    Returns:
        Scalar loss >= 0. Lower = better.
    """
    pred_safe = pred.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(pred_safe, target)


class FaceDetectionLoss(nn.Module):
    """
    Combined face detection loss.

    total = bbox_weight * GIoU + landmark_weight * SmoothL1 + confidence_weight * BCE

    All components are non-negative. Lower = better.
    """

    def __init__(self, cfg=None, bbox_weight=1.0, landmark_weight=0.5, confidence_weight=1.0):
        super().__init__()
        if cfg is not None:
            bbox_weight = cfg.FACE_TRAIN.BBOX_WEIGHT
            landmark_weight = cfg.FACE_TRAIN.LANDMARK_WEIGHT
            confidence_weight = cfg.FACE_TRAIN.CONFIDENCE_WEIGHT
        self.bbox_weight = bbox_weight
        self.landmark_weight = landmark_weight
        self.confidence_weight = confidence_weight
        self._diagnostic_done = False

    def forward(self, predictions: dict, gt_bbox: torch.Tensor,
                gt_landmarks: torch.Tensor) -> tuple:
        """
        Args:
            predictions: dict with 'bbox' [B,4], 'landmarks' [B,5,2], 'confidence' [B,1]
            gt_bbox: [B, 4]
            gt_landmarks: [B, 5, 2]

        Returns:
            (total_loss, loss_dict)
        """
        # First-batch diagnostic
        if not self._diagnostic_done:
            self._run_diagnostic(predictions, gt_bbox, gt_landmarks)
            self._diagnostic_done = True

        bbox_l = giou_loss(predictions['bbox'], gt_bbox)
        lm_l = landmark_loss(predictions['landmarks'], gt_landmarks)

        conf_target = torch.ones_like(predictions['confidence'])
        conf_l = confidence_loss(predictions['confidence'], conf_target)

        total = (
            self.bbox_weight * bbox_l +
            self.landmark_weight * lm_l +
            self.confidence_weight * conf_l
        )

        # Safety: replace NaN/Inf with fallback
        if not torch.isfinite(total):
            print(f"[WARNING] Non-finite loss! bbox={bbox_l.item():.4f}, "
                  f"lm={lm_l.item():.4f}, conf={conf_l.item():.4f}")
            total = torch.tensor(10.0, device=total.device, requires_grad=True)

        loss_dict = {
            'bbox_loss': bbox_l.item(),
            'landmark_loss': lm_l.item(),
            'confidence_loss': conf_l.item(),
            'total_loss': total.item(),
        }
        return total, loss_dict

    @torch.no_grad()
    def _run_diagnostic(self, predictions, gt_bbox, gt_landmarks):
        """One-time diagnostic on first batch to detect data issues."""
        pred_bbox = predictions['bbox']
        pred_lm = predictions['landmarks']
        pred_conf = predictions['confidence']

        print("\n" + "=" * 60)
        print("[DIAGNOSTIC] First batch loss component analysis")
        print("=" * 60)

        # Value ranges
        print(f"  Pred bbox    : min={pred_bbox.min().item():.4f}, "
              f"max={pred_bbox.max().item():.4f}, shape={list(pred_bbox.shape)}")
        print(f"  Pred lm      : min={pred_lm.min().item():.4f}, "
              f"max={pred_lm.max().item():.4f}, shape={list(pred_lm.shape)}")
        print(f"  Pred conf    : min={pred_conf.min().item():.4f}, "
              f"max={pred_conf.max().item():.4f}")
        print(f"  GT bbox      : min={gt_bbox.min().item():.4f}, "
              f"max={gt_bbox.max().item():.4f}, shape={list(gt_bbox.shape)}")
        print(f"  GT landmarks : min={gt_landmarks.min().item():.4f}, "
              f"max={gt_landmarks.max().item():.4f}, shape={list(gt_landmarks.shape)}")

        # Data issue detection
        issues = []
        if gt_bbox.max().item() > 1.1:
            issues.append(f"GT bbox max={gt_bbox.max().item():.2f} > 1.0 (not normalized?)")
        if gt_landmarks.max().item() > 1.1:
            issues.append(f"GT landmarks max={gt_landmarks.max().item():.2f} > 1.0 (not normalized?)")
        if gt_bbox.min().item() < -0.1:
            issues.append(f"GT bbox min={gt_bbox.min().item():.2f} < 0 (invalid coords?)")
        if gt_landmarks.min().item() < -0.1:
            issues.append(f"GT landmarks min={gt_landmarks.min().item():.2f} < 0 (invalid coords?)")
        if torch.isnan(gt_bbox).any():
            issues.append("GT bbox contains NaN!")
        if torch.isnan(gt_landmarks).any():
            issues.append("GT landmarks contains NaN!")

        if issues:
            print("  [DATA ISSUES DETECTED]")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  [OK] All values in expected [0, 1] range.")

        # Per-component losses
        bbox_l = giou_loss(pred_bbox, gt_bbox)
        lm_l = landmark_loss(pred_lm, gt_landmarks)
        conf_target = torch.ones_like(pred_conf)
        conf_l = confidence_loss(pred_conf, conf_target)
        total = self.bbox_weight * bbox_l + self.landmark_weight * lm_l + self.confidence_weight * conf_l

        print(f"  GIoU loss    : {bbox_l.item():.6f} (weight={self.bbox_weight})")
        print(f"  Landmark loss: {lm_l.item():.6f} (weight={self.landmark_weight})")
        print(f"  Conf BCE loss: {conf_l.item():.6f} (weight={self.confidence_weight})")
        print(f"  Total loss   : {total.item():.6f}")
        print("=" * 60 + "\n")
