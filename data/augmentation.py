"""
Data augmentation for face detection training.
Handles rotation, scaling, blur, and color jitter to simulate head movement.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple


class FaceAugmentation:
    """TensorFlow-based augmentation for face detection data."""

    def __init__(
        self,
        rotation_range: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        blur_prob: float = 0.3,
        noise_prob: float = 0.2,
        horizontal_flip: bool = True,
    ):
        """
        Initialize augmentation parameters.

        Args:
            rotation_range: Maximum rotation angle in degrees
            scale_range: (min_scale, max_scale) for random scaling
            brightness_range: Random brightness adjustment range
            contrast_range: Random contrast adjustment range
            blur_prob: Probability of applying blur
            noise_prob: Probability of adding Gaussian noise
            horizontal_flip: Whether to apply random horizontal flip
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.horizontal_flip = horizontal_flip

    def __call__(
        self,
        image: tf.Tensor,
        bbox: tf.Tensor,
        landmarks: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply augmentation to image and labels.

        Args:
            image: Input image tensor [H, W, 3]
            bbox: Bounding box [x_min, y_min, x_max, y_max] normalized
            landmarks: Landmarks [5, 2] normalized (x, y)
            training: Whether in training mode

        Returns:
            Augmented (image, bbox, landmarks)
        """
        if not training:
            return image, bbox, landmarks

        # Random horizontal flip
        if self.horizontal_flip:
            image, bbox, landmarks = self._random_flip(image, bbox, landmarks)

        # Random rotation
        if self.rotation_range > 0:
            image, bbox, landmarks = self._random_rotation(image, bbox, landmarks)

        # Random scale
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            image, bbox, landmarks = self._random_scale(image, bbox, landmarks)

        # Color augmentation
        image = self._color_augmentation(image)

        # Random blur
        if self.blur_prob > 0:
            image = self._random_blur(image)

        # Random noise
        if self.noise_prob > 0:
            image = self._random_noise(image)

        return image, bbox, landmarks

    def _random_flip(
        self,
        image: tf.Tensor,
        bbox: tf.Tensor,
        landmarks: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Random horizontal flip with landmark reordering."""
        do_flip = tf.random.uniform([]) > 0.5

        def flip():
            flipped_image = tf.image.flip_left_right(image)

            # Flip bbox x coordinates
            flipped_bbox = tf.stack([
                1.0 - bbox[2],  # new x_min = 1 - old x_max
                bbox[1],       # y_min unchanged
                1.0 - bbox[0],  # new x_max = 1 - old x_min
                bbox[3],       # y_max unchanged
            ])

            # Flip landmarks x coordinates and swap left/right
            # Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
            flipped_landmarks = tf.stack([
                [1.0 - landmarks[1, 0], landmarks[1, 1]],  # right_eye -> left_eye
                [1.0 - landmarks[0, 0], landmarks[0, 1]],  # left_eye -> right_eye
                [1.0 - landmarks[2, 0], landmarks[2, 1]],  # nose
                [1.0 - landmarks[4, 0], landmarks[4, 1]],  # right_mouth -> left_mouth
                [1.0 - landmarks[3, 0], landmarks[3, 1]],  # left_mouth -> right_mouth
            ])

            return flipped_image, flipped_bbox, flipped_landmarks

        def no_flip():
            return image, bbox, landmarks

        return tf.cond(do_flip, flip, no_flip)

    def _random_rotation(
        self,
        image: tf.Tensor,
        bbox: tf.Tensor,
        landmarks: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Random rotation with coordinate transformation."""
        angle = tf.random.uniform(
            [],
            minval=-self.rotation_range,
            maxval=self.rotation_range,
        )
        angle_rad = angle * np.pi / 180.0

        # Get image dimensions
        shape = tf.cast(tf.shape(image), tf.float32)
        height, width = shape[0], shape[1]
        center_x, center_y = 0.5, 0.5

        # Rotate image
        rotated_image = self._rotate_image(image, angle_rad)

        # Rotate bbox corners and recompute axis-aligned bbox
        corners = tf.stack([
            [bbox[0], bbox[1]],  # top-left
            [bbox[2], bbox[1]],  # top-right
            [bbox[2], bbox[3]],  # bottom-right
            [bbox[0], bbox[3]],  # bottom-left
        ])

        rotated_corners = self._rotate_points(corners, -angle_rad, center_x, center_y)
        rotated_bbox = tf.stack([
            tf.reduce_min(rotated_corners[:, 0]),
            tf.reduce_min(rotated_corners[:, 1]),
            tf.reduce_max(rotated_corners[:, 0]),
            tf.reduce_max(rotated_corners[:, 1]),
        ])
        rotated_bbox = tf.clip_by_value(rotated_bbox, 0.0, 1.0)

        # Rotate landmarks
        rotated_landmarks = self._rotate_points(landmarks, -angle_rad, center_x, center_y)
        rotated_landmarks = tf.clip_by_value(rotated_landmarks, 0.0, 1.0)

        return rotated_image, rotated_bbox, rotated_landmarks

    def _rotate_image(self, image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
        """Rotate image using tf.raw_ops."""
        # Add batch dimension
        image = tf.expand_dims(image, 0)

        # Create rotation matrix
        cos_a = tf.cos(angle_rad)
        sin_a = tf.sin(angle_rad)

        # Transformation matrix for tf.raw_ops.ImageProjectiveTransformV3
        # [a, b, c, d, e, f, g, h] represents:
        # x' = (a*x + b*y + c) / (g*x + h*y + 1)
        # y' = (d*x + e*y + f) / (g*x + h*y + 1)
        transform = tf.stack([
            cos_a, -sin_a, 0.5 - 0.5 * cos_a + 0.5 * sin_a,
            sin_a, cos_a, 0.5 - 0.5 * sin_a - 0.5 * cos_a,
            0.0, 0.0
        ])
        transform = tf.reshape(transform, [1, 8])

        # Apply transformation
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=image,
            transforms=transform,
            output_shape=tf.shape(image)[1:3],
            fill_value=0.0,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
        )

        return rotated[0]

    def _rotate_points(
        self,
        points: tf.Tensor,
        angle_rad: tf.Tensor,
        center_x: float,
        center_y: float,
    ) -> tf.Tensor:
        """Rotate points around center."""
        cos_a = tf.cos(angle_rad)
        sin_a = tf.sin(angle_rad)

        # Translate to origin
        x = points[:, 0] - center_x
        y = points[:, 1] - center_y

        # Rotate
        new_x = x * cos_a - y * sin_a + center_x
        new_y = x * sin_a + y * cos_a + center_y

        return tf.stack([new_x, new_y], axis=1)

    def _random_scale(
        self,
        image: tf.Tensor,
        bbox: tf.Tensor,
        landmarks: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Random scaling with center crop/pad."""
        scale = tf.random.uniform(
            [],
            minval=self.scale_range[0],
            maxval=self.scale_range[1],
        )

        # Scale around center
        center_x, center_y = 0.5, 0.5

        # Scale bbox
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        new_bbox_w = bbox_w * scale
        new_bbox_h = bbox_h * scale
        new_bbox = tf.stack([
            bbox_center_x - new_bbox_w / 2,
            bbox_center_y - new_bbox_h / 2,
            bbox_center_x + new_bbox_w / 2,
            bbox_center_y + new_bbox_h / 2,
        ])

        # Scale landmarks relative to center
        scaled_landmarks = center_x + (landmarks - center_x) * scale
        scaled_landmarks = tf.clip_by_value(scaled_landmarks, 0.0, 1.0)

        # Scale image
        shape = tf.shape(image)
        new_h = tf.cast(tf.cast(shape[0], tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(shape[1], tf.float32) * scale, tf.int32)
        scaled_image = tf.image.resize(image, [new_h, new_w])

        # Crop or pad to original size
        scaled_image = tf.image.resize_with_crop_or_pad(
            scaled_image, shape[0], shape[1]
        )

        # Adjust bbox and landmarks for crop/pad offset
        offset = (1.0 - scale) / 2
        new_bbox = tf.clip_by_value(new_bbox + offset, 0.0, 1.0)
        scaled_landmarks = tf.clip_by_value(scaled_landmarks + offset, 0.0, 1.0)

        return scaled_image, new_bbox, scaled_landmarks

    def _color_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random color augmentation."""
        # Random brightness
        if self.brightness_range > 0:
            image = tf.image.random_brightness(image, self.brightness_range)

        # Random contrast
        if self.contrast_range > 0:
            image = tf.image.random_contrast(
                image,
                1.0 - self.contrast_range,
                1.0 + self.contrast_range,
            )

        # Random saturation
        image = tf.image.random_saturation(image, 0.8, 1.2)

        # Random hue
        image = tf.image.random_hue(image, 0.05)

        # Clip values
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def _random_blur(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random Gaussian blur."""
        do_blur = tf.random.uniform([]) < self.blur_prob

        def apply_blur():
            # Simple box blur as approximation
            kernel_size = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32) * 2 + 1
            kernel = tf.ones([kernel_size, kernel_size, 1, 1]) / tf.cast(
                kernel_size * kernel_size, tf.float32
            )

            # Apply per channel
            blurred = []
            for i in range(3):
                channel = image[:, :, i:i+1]
                channel = tf.expand_dims(channel, 0)
                channel = tf.nn.conv2d(channel, kernel, strides=1, padding='SAME')
                channel = channel[0]
                blurred.append(channel)

            return tf.concat(blurred, axis=-1)

        return tf.cond(do_blur, apply_blur, lambda: image)

    def _random_noise(self, image: tf.Tensor) -> tf.Tensor:
        """Add random Gaussian noise."""
        do_noise = tf.random.uniform([]) < self.noise_prob

        def add_noise():
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
            return tf.clip_by_value(image + noise, 0.0, 1.0)

        return tf.cond(do_noise, add_noise, lambda: image)


def create_augmentation_from_config(config: Dict) -> FaceAugmentation:
    """Create augmentation from config dictionary."""
    aug_config = config.get('augmentation', {})
    return FaceAugmentation(
        rotation_range=aug_config.get('rotation_range', 30),
        scale_range=tuple(aug_config.get('scale_range', [0.8, 1.2])),
        brightness_range=aug_config.get('brightness_range', 0.2),
        contrast_range=aug_config.get('contrast_range', 0.2),
        blur_prob=aug_config.get('blur_prob', 0.3),
        noise_prob=aug_config.get('noise_prob', 0.2),
        horizontal_flip=aug_config.get('horizontal_flip', True),
    )
