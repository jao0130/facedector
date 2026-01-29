"""
Loss functions for face detection training.
"""

import tensorflow as tf
from tensorflow import keras


def smooth_l1_loss(pred: tf.Tensor, target: tf.Tensor, beta: float = 1.0) -> tf.Tensor:
    """
    Smooth L1 loss (Huber loss).

    Args:
        pred: Predictions tensor
        target: Ground truth tensor
        beta: Threshold for switching between L1 and L2

    Returns:
        Loss scalar
    """
    diff = tf.abs(pred - target)
    loss = tf.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta,
    )
    return tf.reduce_mean(loss)


def wing_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    w: float = 10.0,
    epsilon: float = 2.0
) -> tf.Tensor:
    """
    Wing loss for landmark regression.
    Better gradient for small errors compared to L2.

    Reference: Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks (CVPR 2018)

    Args:
        pred: Predicted landmarks
        target: Ground truth landmarks
        w: Wing width parameter
        epsilon: Curvature parameter

    Returns:
        Loss scalar
    """
    diff = tf.abs(pred - target)
    # Add numerical stability with epsilon protection
    epsilon_safe = tf.maximum(epsilon, 1e-7)
    c = w * (1.0 - tf.math.log(tf.maximum(1.0 + w / epsilon_safe, 1e-7)))

    loss = tf.where(
        diff < w,
        w * tf.math.log(tf.maximum(1.0 + diff / epsilon_safe, 1e-7)),
        diff - c,
    )
    return tf.reduce_mean(loss)


def adaptive_wing_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    omega: float = 14.0,
    theta: float = 0.5,
    epsilon: float = 1.0,
    alpha: float = 2.1,
) -> tf.Tensor:
    """
    Adaptive Wing loss for heatmap-based landmark detection.

    Reference: Adaptive Wing Loss for Robust Face Alignment via
    Heatmap Regression (ICCV 2019)

    Args:
        pred: Predicted values
        target: Ground truth values
        omega: Wing width
        theta: Threshold
        epsilon: Small constant
        alpha: Exponent

    Returns:
        Loss scalar
    """
    delta = pred - target
    delta_abs = tf.abs(delta)

    A = omega * (1.0 / (1.0 + tf.pow(theta / epsilon, alpha - target))) * \
        (alpha - target) * tf.pow(theta / epsilon, alpha - target - 1) / epsilon
    C = theta * A - omega * tf.math.log(1.0 + tf.pow(theta / epsilon, alpha - target))

    loss = tf.where(
        delta_abs < theta,
        omega * tf.math.log(1.0 + tf.pow(delta_abs / epsilon, alpha - target)),
        A * delta_abs - C,
    )
    return tf.reduce_mean(loss)


def giou_loss(pred_boxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    Generalized IoU loss for bounding box regression.

    Args:
        pred_boxes: Predicted boxes [B, 4] (x_min, y_min, x_max, y_max)
        gt_boxes: Ground truth boxes [B, 4]

    Returns:
        GIoU loss (1 - GIoU)
    """
    # Compute intersection
    inter_x_min = tf.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y_min = tf.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x_max = tf.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y_max = tf.minimum(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = tf.maximum(0.0, inter_x_max - inter_x_min)
    inter_h = tf.maximum(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Compute union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = pred_area + gt_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)

    # Enclosing box
    enclose_x_min = tf.minimum(pred_boxes[:, 0], gt_boxes[:, 0])
    enclose_y_min = tf.minimum(pred_boxes[:, 1], gt_boxes[:, 1])
    enclose_x_max = tf.maximum(pred_boxes[:, 2], gt_boxes[:, 2])
    enclose_y_max = tf.maximum(pred_boxes[:, 3], gt_boxes[:, 3])

    enclose_area = (enclose_x_max - enclose_x_min) * \
                   (enclose_y_max - enclose_y_min)

    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

    return tf.reduce_mean(1.0 - giou)


def focal_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tf.Tensor:
    """
    Focal loss for classification (handles class imbalance).

    Args:
        pred: Predicted probabilities [B, N]
        target: Ground truth labels [B, N]
        alpha: Balancing factor
        gamma: Focusing parameter

    Returns:
        Focal loss scalar
    """
    pred = tf.clip_by_value(pred, 1e-7, 1.0 - 1e-7)

    # Binary cross entropy
    bce = -target * tf.math.log(pred) - (1 - target) * tf.math.log(1 - pred)

    # Focal weight
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = focal_weight * tf.pow(1 - p_t, gamma)

    return tf.reduce_mean(focal_weight * bce)


class CombinedLoss(keras.losses.Loss):
    """Combined loss for face detection training."""

    def __init__(
        self,
        bbox_weight: float = 1.0,
        landmark_weight: float = 0.5,
        confidence_weight: float = 1.0,
        use_giou: bool = True,
        use_wing_loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bbox_weight = bbox_weight
        self.landmark_weight = landmark_weight
        self.confidence_weight = confidence_weight
        self.use_giou = use_giou
        self.use_wing_loss = use_wing_loss

    def call(self, y_true, y_pred):
        """
        Compute combined loss.

        Args:
            y_true: Tuple of (gt_bbox, gt_landmarks)
            y_pred: Tuple of (pred_bbox, pred_landmarks, pred_confidence)

        Returns:
            Combined loss scalar
        """
        gt_bbox, gt_landmarks = y_true
        pred_bbox, pred_landmarks, pred_confidence = y_pred

        # Bounding box loss
        if self.use_giou:
            bbox_loss = giou_loss(pred_bbox, gt_bbox)
        else:
            bbox_loss = smooth_l1_loss(pred_bbox, gt_bbox)

        # Landmark loss
        if self.use_wing_loss:
            landmark_loss = wing_loss(pred_landmarks, gt_landmarks)
        else:
            landmark_loss = smooth_l1_loss(pred_landmarks, gt_landmarks)

        # Confidence loss (all samples have faces)
        confidence_target = tf.ones_like(pred_confidence)
        confidence_loss = keras.losses.binary_crossentropy(
            confidence_target, pred_confidence
        )
        confidence_loss = tf.reduce_mean(confidence_loss)

        # Combined
        total_loss = (
            self.bbox_weight * bbox_loss +
            self.landmark_weight * landmark_loss +
            self.confidence_weight * confidence_loss
        )

        return total_loss
