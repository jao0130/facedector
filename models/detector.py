"""
SSD detection head with landmark regression for face detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DetectionHead(layers.Layer):
    """
    Single-shot detection head for face detection.
    Outputs bounding box and confidence score.
    """

    def __init__(
        self,
        num_anchors: int = 6,
        num_classes: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Separate heads for classification and regression
        # Classification: face/no-face per anchor
        self.cls_head = None
        # Regression: [dx, dy, dw, dh] per anchor
        self.reg_head = None

    def build(self, input_shape):
        # Classification head
        self.cls_head = keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(
                self.num_anchors * self.num_classes,
                1,
                padding='same',
            ),
        ], name='cls_head')

        # Regression head
        self.reg_head = keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(
                self.num_anchors * 4,  # 4 bbox coordinates
                1,
                padding='same',
            ),
        ], name='reg_head')

        super().build(input_shape)

    def call(self, features, training=None):
        """
        Args:
            features: Feature map [B, H, W, C]

        Returns:
            cls_output: Classification logits [B, H*W*num_anchors, num_classes]
            reg_output: Regression output [B, H*W*num_anchors, 4]
        """
        batch_size = tf.shape(features)[0]
        h, w = tf.shape(features)[1], tf.shape(features)[2]

        # Classification
        cls_output = self.cls_head(features, training=training)
        cls_output = tf.reshape(
            cls_output,
            [batch_size, -1, self.num_classes]
        )

        # Regression
        reg_output = self.reg_head(features, training=training)
        reg_output = tf.reshape(
            reg_output,
            [batch_size, -1, 4]
        )

        return cls_output, reg_output


class LandmarkHead(layers.Layer):
    """
    Landmark regression head for 5 facial landmarks.
    Outputs normalized landmark coordinates relative to bounding box.
    """

    def __init__(
        self,
        num_landmarks: int = 5,
        num_anchors: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_landmarks = num_landmarks
        self.num_anchors = num_anchors
        self.landmark_head = None

    def build(self, input_shape):
        self.landmark_head = keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(
                self.num_anchors * self.num_landmarks * 2,  # x, y per landmark
                1,
                padding='same',
            ),
        ], name='landmark_head')

        super().build(input_shape)

    def call(self, features, training=None):
        """
        Args:
            features: Feature map [B, H, W, C]

        Returns:
            landmarks: Landmark predictions [B, H*W*num_anchors, num_landmarks, 2]
        """
        batch_size = tf.shape(features)[0]

        landmarks = self.landmark_head(features, training=training)
        landmarks = tf.reshape(
            landmarks,
            [batch_size, -1, self.num_landmarks, 2]
        )

        # Apply sigmoid to constrain to [0, 1]
        landmarks = tf.sigmoid(landmarks)

        return landmarks


class AnchorGenerator:
    """Generate anchor boxes for SSD-style detection."""

    def __init__(
        self,
        scales: list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        aspect_ratios: list = [1.0],
    ):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(scales) * len(aspect_ratios)

    def generate_anchors(
        self,
        feature_map_size: tuple,
        input_size: int,
    ) -> tf.Tensor:
        """
        Generate anchor boxes for a feature map.

        Args:
            feature_map_size: (height, width) of feature map
            input_size: Original input image size

        Returns:
            anchors: [H*W*num_anchors, 4] in (x_center, y_center, width, height)
        """
        h, w = feature_map_size
        anchors = []

        for i in range(h):
            for j in range(w):
                # Center coordinates (normalized)
                cx = (j + 0.5) / w
                cy = (i + 0.5) / h

                for scale in self.scales:
                    for ratio in self.aspect_ratios:
                        anchor_w = scale * np.sqrt(ratio)
                        anchor_h = scale / np.sqrt(ratio)
                        anchors.append([cx, cy, anchor_w, anchor_h])

        return tf.constant(anchors, dtype=tf.float32)

    def decode_boxes(
        self,
        anchors: tf.Tensor,
        deltas: tf.Tensor,
    ) -> tf.Tensor:
        """
        Decode bounding box deltas to actual boxes.

        Args:
            anchors: [N, 4] anchor boxes (cx, cy, w, h)
            deltas: [B, N, 4] predicted deltas (dx, dy, dw, dh)

        Returns:
            boxes: [B, N, 4] decoded boxes (x_min, y_min, x_max, y_max)
        """
        # Decode center
        cx = anchors[..., 0] + deltas[..., 0] * anchors[..., 2]
        cy = anchors[..., 1] + deltas[..., 1] * anchors[..., 3]

        # Decode size
        w = anchors[..., 2] * tf.exp(deltas[..., 2])
        h = anchors[..., 3] * tf.exp(deltas[..., 3])

        # Convert to corner format
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

        return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    def encode_boxes(
        self,
        anchors: tf.Tensor,
        gt_boxes: tf.Tensor,
    ) -> tf.Tensor:
        """
        Encode ground truth boxes as deltas from anchors.

        Args:
            anchors: [N, 4] anchor boxes (cx, cy, w, h)
            gt_boxes: [N, 4] ground truth boxes (x_min, y_min, x_max, y_max)

        Returns:
            deltas: [N, 4] encoded deltas
        """
        # Convert gt to center format
        gt_cx = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_cy = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        gt_w = gt_boxes[..., 2] - gt_boxes[..., 0]
        gt_h = gt_boxes[..., 3] - gt_boxes[..., 1]

        # Compute deltas
        dx = (gt_cx - anchors[..., 0]) / (anchors[..., 2] + 1e-8)
        dy = (gt_cy - anchors[..., 1]) / (anchors[..., 3] + 1e-8)
        dw = tf.math.log(gt_w / (anchors[..., 2] + 1e-8) + 1e-8)
        dh = tf.math.log(gt_h / (anchors[..., 3] + 1e-8) + 1e-8)

        return tf.stack([dx, dy, dw, dh], axis=-1)
