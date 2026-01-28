"""
Evaluation metrics for face detection.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def compute_iou(box1: tf.Tensor, box2: tf.Tensor) -> tf.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: Boxes [B, 4] (x_min, y_min, x_max, y_max)
        box2: Boxes [B, 4] (x_min, y_min, x_max, y_max)

    Returns:
        IoU scores [B]
    """
    # Intersection
    inter_x_min = tf.maximum(box1[:, 0], box2[:, 0])
    inter_y_min = tf.maximum(box1[:, 1], box2[:, 1])
    inter_x_max = tf.minimum(box1[:, 2], box2[:, 2])
    inter_y_max = tf.minimum(box1[:, 3], box2[:, 3])

    inter_w = tf.maximum(0.0, inter_x_max - inter_x_min)
    inter_h = tf.maximum(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-7)


def compute_nme(
    pred_landmarks: tf.Tensor,
    gt_landmarks: tf.Tensor,
    gt_bbox: tf.Tensor,
    norm_type: str = 'inter_ocular',
) -> tf.Tensor:
    """
    Compute Normalized Mean Error (NME) for landmarks.

    Args:
        pred_landmarks: Predicted landmarks [B, 5, 2]
        gt_landmarks: Ground truth landmarks [B, 5, 2]
        gt_bbox: Ground truth bounding boxes [B, 4]
        norm_type: Normalization type ('inter_ocular' or 'bbox')

    Returns:
        NME score (lower is better)
    """
    # Compute per-landmark error
    error = tf.sqrt(tf.reduce_sum((pred_landmarks - gt_landmarks) ** 2, axis=-1))

    # Normalization factor
    if norm_type == 'inter_ocular':
        # Distance between eye centers
        left_eye = gt_landmarks[:, 0, :]
        right_eye = gt_landmarks[:, 1, :]
        norm_factor = tf.sqrt(tf.reduce_sum((left_eye - right_eye) ** 2, axis=-1))
        norm_factor = tf.maximum(norm_factor, 1e-7)  # Avoid division by zero
    else:
        # Bounding box diagonal
        bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        norm_factor = tf.sqrt(bbox_w ** 2 + bbox_h ** 2)

    # Average error across landmarks
    mean_error = tf.reduce_mean(error, axis=-1)

    # Normalized error
    nme = mean_error / norm_factor

    return tf.reduce_mean(nme)


class IoUMetric(keras.metrics.Metric):
    """Keras metric for IoU computation."""

    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state."""
        gt_bbox = y_true[0] if isinstance(y_true, (list, tuple)) else y_true
        pred_bbox = y_pred['bbox'] if isinstance(y_pred, dict) else y_pred[0]

        iou = compute_iou(pred_bbox, gt_bbox)
        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(iou)[0], tf.float32))

    def result(self):
        return self.total_iou / (self.count + 1e-7)

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


class NMEMetric(keras.metrics.Metric):
    """Keras metric for NME (Normalized Mean Error) computation."""

    def __init__(self, name='nme', norm_type='inter_ocular', **kwargs):
        super().__init__(name=name, **kwargs)
        self.norm_type = norm_type
        self.total_nme = self.add_weight(name='total_nme', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state."""
        if isinstance(y_true, (list, tuple)):
            gt_bbox, gt_landmarks = y_true[0], y_true[1]
        else:
            gt_bbox, gt_landmarks = y_true, y_true

        if isinstance(y_pred, dict):
            pred_landmarks = y_pred['landmarks']
        else:
            pred_landmarks = y_pred[1]

        nme = compute_nme(pred_landmarks, gt_landmarks, gt_bbox, self.norm_type)
        batch_size = tf.cast(tf.shape(pred_landmarks)[0], tf.float32)

        self.total_nme.assign_add(nme * batch_size)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total_nme / (self.count + 1e-7)

    def reset_state(self):
        self.total_nme.assign(0.0)
        self.count.assign(0.0)


class MeanAveragePrecision(keras.metrics.Metric):
    """
    Mean Average Precision (mAP) metric for object detection.
    Simplified version for single-class face detection.
    """

    def __init__(self, iou_threshold: float = 0.5, name='mAP', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.total_gt = self.add_weight(name='total_gt', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state."""
        gt_bbox = y_true[0] if isinstance(y_true, (list, tuple)) else y_true

        if isinstance(y_pred, dict):
            pred_bbox = y_pred['bbox']
            pred_conf = y_pred.get('confidence', tf.ones_like(pred_bbox[:, :1]))
        else:
            pred_bbox, pred_conf = y_pred[0], y_pred[2]

        # Compute IoU
        iou = compute_iou(pred_bbox, gt_bbox)

        # Count TP and FP
        tp = tf.reduce_sum(tf.cast(iou >= self.iou_threshold, tf.float32))
        fp = tf.reduce_sum(tf.cast(iou < self.iou_threshold, tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.total_gt.assign_add(tf.cast(tf.shape(gt_bbox)[0], tf.float32))

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (self.total_gt + 1e-7)
        # Simplified AP as precision * recall
        return precision * recall

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.total_gt.assign(0.0)


def evaluate_detection(
    model,
    dataset,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate face detection model on a dataset.

    Args:
        model: Face detection model
        dataset: tf.data.Dataset yielding (image, bbox, landmarks)
        iou_threshold: IoU threshold for positive detection

    Returns:
        Dictionary of metrics
    """
    iou_metric = IoUMetric()
    nme_metric = NMEMetric()

    for batch in dataset:
        images, gt_bbox, gt_landmarks = batch
        predictions = model(images, training=False)

        iou_metric.update_state((gt_bbox, gt_landmarks), predictions)
        nme_metric.update_state((gt_bbox, gt_landmarks), predictions)

    return {
        'iou': float(iou_metric.result().numpy()),
        'nme': float(nme_metric.result().numpy()),
    }
