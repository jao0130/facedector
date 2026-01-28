"""
Complete face detection model combining backbone, detection head, and landmark head.
BlazeFace-inspired lightweight architecture for mobile deployment.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple

from .backbone import create_mobilenetv2_backbone, FeaturePyramidNeck
from .detector import DetectionHead, LandmarkHead, AnchorGenerator


class FaceDetector(keras.Model):
    """
    Lightweight face detector with bounding box and landmark prediction.

    Architecture:
    - MobileNetV2 backbone (width multiplier configurable)
    - Feature Pyramid Network for multi-scale features
    - Single detection head operating on fused features
    - Landmark regression head for 5 facial keypoints
    """

    def __init__(
        self,
        input_size: int = 256,
        backbone_alpha: float = 0.5,
        num_landmarks: int = 5,
        num_anchors: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.num_anchors = num_anchors

        # Backbone
        self.backbone = create_mobilenetv2_backbone(
            input_shape=(input_size, input_size, 3),
            alpha=backbone_alpha,
        )

        # Feature pyramid neck
        self.fpn = FeaturePyramidNeck(out_channels=64)

        # Detection heads
        self.detection_head = DetectionHead(num_anchors=num_anchors)
        self.landmark_head = LandmarkHead(
            num_landmarks=num_landmarks,
            num_anchors=num_anchors,
        )

        # Global feature aggregation for single face detection
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc_bbox = layers.Dense(4)  # Direct bbox regression
        self.fc_landmarks = layers.Dense(num_landmarks * 2)  # Direct landmark regression
        self.fc_confidence = layers.Dense(1)  # Face confidence

        # Anchor generator for SSD-style detection (backup)
        self.anchor_generator = AnchorGenerator(
            scales=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            aspect_ratios=[1.0],
        )

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Input images [B, H, W, 3]
            training: Training mode flag

        Returns:
            Dictionary containing:
            - bbox: Predicted bounding boxes [B, 4] (x_min, y_min, x_max, y_max)
            - landmarks: Predicted landmarks [B, 5, 2] (x, y per landmark)
            - confidence: Face confidence score [B, 1]
        """
        # Extract multi-scale features
        features = self.backbone(inputs, training=training)

        # Apply FPN
        fpn_features = self.fpn(features, training=training)

        # Use the second-to-last scale for best balance
        # (1/8 scale has good resolution and receptive field)
        main_feature = fpn_features[1]

        # Global pooling for single-face prediction
        global_feature = self.global_pool(main_feature)

        # Predict bbox
        bbox_raw = self.fc_bbox(global_feature)
        # Apply sigmoid to constrain to [0, 1]
        bbox = tf.sigmoid(bbox_raw)

        # Predict landmarks
        landmarks_raw = self.fc_landmarks(global_feature)
        landmarks = tf.sigmoid(landmarks_raw)
        landmarks = tf.reshape(landmarks, [-1, self.num_landmarks, 2])

        # Predict confidence
        confidence = tf.sigmoid(self.fc_confidence(global_feature))

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }

    def compute_loss(
        self,
        predictions: Dict[str, tf.Tensor],
        targets: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions dictionary
            targets: Tuple of (images, gt_bbox, gt_landmarks)
            loss_weights: Dictionary of loss weights

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual losses
        """
        if loss_weights is None:
            loss_weights = {
                'bbox': 1.0,
                'landmark': 0.5,
                'confidence': 1.0,
            }

        _, gt_bbox, gt_landmarks = targets
        pred_bbox = predictions['bbox']
        pred_landmarks = predictions['landmarks']
        pred_confidence = predictions['confidence']

        # Bounding box loss (Smooth L1)
        bbox_loss = smooth_l1_loss(pred_bbox, gt_bbox)

        # Landmark loss (Smooth L1)
        landmark_loss = smooth_l1_loss(
            pred_landmarks,
            gt_landmarks,
        )

        # Confidence loss (Binary cross-entropy)
        # All samples have faces, so target is 1
        confidence_target = tf.ones_like(pred_confidence)
        confidence_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(
                confidence_target,
                pred_confidence,
            )
        )

        # Combined loss
        total_loss = (
            loss_weights['bbox'] * bbox_loss +
            loss_weights['landmark'] * landmark_loss +
            loss_weights['confidence'] * confidence_loss
        )

        loss_dict = {
            'bbox_loss': bbox_loss,
            'landmark_loss': landmark_loss,
            'confidence_loss': confidence_loss,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'num_landmarks': self.num_landmarks,
            'num_anchors': self.num_anchors,
        })
        return config


def smooth_l1_loss(pred: tf.Tensor, target: tf.Tensor, beta: float = 1.0) -> tf.Tensor:
    """
    Smooth L1 loss (Huber loss).

    Args:
        pred: Predictions
        target: Ground truth
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


def wing_loss(pred: tf.Tensor, target: tf.Tensor, w: float = 10.0, epsilon: float = 2.0) -> tf.Tensor:
    """
    Wing loss for landmark regression (better for small errors).

    Args:
        pred: Predicted landmarks
        target: Ground truth landmarks
        w: Wing loss width parameter
        epsilon: Curvature parameter

    Returns:
        Loss scalar
    """
    diff = tf.abs(pred - target)
    c = w * (1.0 - tf.math.log(1.0 + w / epsilon))

    loss = tf.where(
        diff < w,
        w * tf.math.log(1.0 + diff / epsilon),
        diff - c,
    )
    return tf.reduce_mean(loss)


def create_face_detector(config: Dict) -> FaceDetector:
    """Create face detector model from config."""
    model_config = config.get('model', {})

    model = FaceDetector(
        input_size=model_config.get('input_size', 256),
        backbone_alpha=model_config.get('backbone_alpha', 0.5),
        num_landmarks=model_config.get('num_landmarks', 5),
        num_anchors=model_config.get('num_anchors', 6),
    )

    return model


class FaceDetectorLite(keras.Model):
    """
    Ultra-lightweight face detector for mobile inference.
    Simplified architecture for maximum speed.
    """

    def __init__(
        self,
        input_size: int = 128,
        num_landmarks: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.num_landmarks = num_landmarks

        # Lightweight backbone
        self.features = keras.Sequential([
            # Initial conv
            layers.Conv2D(16, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Depthwise separable blocks
            self._dsconv_block(32, stride=2),
            self._dsconv_block(64, stride=2),
            self._dsconv_block(64, stride=1),
            self._dsconv_block(128, stride=2),
            self._dsconv_block(128, stride=1),

            # Global pooling
            layers.GlobalAveragePooling2D(),
        ])

        # Output heads
        self.fc_bbox = layers.Dense(4)
        self.fc_landmarks = layers.Dense(num_landmarks * 2)
        self.fc_confidence = layers.Dense(1)

    def _dsconv_block(self, filters, stride=1):
        return keras.Sequential([
            layers.DepthwiseConv2D(3, strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 1),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, inputs, training=None):
        features = self.features(inputs, training=training)

        bbox = tf.sigmoid(self.fc_bbox(features))
        landmarks = tf.sigmoid(self.fc_landmarks(features))
        landmarks = tf.reshape(landmarks, [-1, self.num_landmarks, 2])
        confidence = tf.sigmoid(self.fc_confidence(features))

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }
