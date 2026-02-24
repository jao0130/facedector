"""
Data augmentation for face detection training using albumentations.
Handles rotation, scaling, perspective, blur, color jitter, and horizontal flip
with landmark coordination.
"""

import numpy as np
import albumentations as A
from typing import Tuple


# Landmark swap indices for horizontal flip: left_eye(0)<->right_eye(1), left_mouth(3)<->right_mouth(4)
FLIP_LANDMARK_INDICES = [1, 0, 2, 4, 3]


class FaceAugmentation:
    """Albumentations-based augmentation for face detection data."""

    def __init__(self, cfg=None, rotation_range=30.0, scale_min=0.7, scale_max=1.3,
                 brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1,
                 blur_prob=0.3, blur_limit=7, noise_prob=0.2,
                 horizontal_flip=True, perspective_prob=0.3):

        if cfg is not None:
            rotation_range = cfg.AUGMENTATION.ROTATION_RANGE
            scale_min = cfg.AUGMENTATION.SCALE_MIN
            scale_max = cfg.AUGMENTATION.SCALE_MAX
            brightness = cfg.AUGMENTATION.BRIGHTNESS
            contrast = cfg.AUGMENTATION.CONTRAST
            saturation = cfg.AUGMENTATION.SATURATION
            hue = cfg.AUGMENTATION.HUE
            blur_prob = cfg.AUGMENTATION.BLUR_PROB
            blur_limit = cfg.AUGMENTATION.BLUR_LIMIT
            noise_prob = cfg.AUGMENTATION.NOISE_PROB
            horizontal_flip = cfg.AUGMENTATION.HORIZONTAL_FLIP
            perspective_prob = cfg.AUGMENTATION.PERSPECTIVE_PROB

        self.horizontal_flip = horizontal_flip

        # Build albumentations pipeline (without flip — handled manually for landmarks)
        transforms_list = [
            A.Rotate(limit=rotation_range, p=0.5, border_mode=0),
            A.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=0.5),
            A.PadIfNeeded(min_height=None, min_width=None,
                          pad_height_divisor=1, pad_width_divisor=1),
            A.Perspective(scale=(0.02, 0.06), p=perspective_prob),
            A.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=blur_prob),
            A.GaussNoise(p=noise_prob),
            A.CoarseDropout(max_holes=1, max_height=0.15, max_width=0.15,
                            fill_value=0, p=0.15),
        ]

        self.transform = A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'],
                                     min_visibility=0.3),
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

    def __call__(self, image: np.ndarray, bbox: np.ndarray,
                 landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentation.

        Args:
            image: [H, W, 3] uint8 RGB
            bbox: [4] float32 normalized (x_min, y_min, x_max, y_max) in [0, 1]
            landmarks: [5, 2] float32 normalized

        Returns:
            Augmented (image, bbox, landmarks)
        """
        h, w = image.shape[:2]

        # Convert normalized coords to pixel coords for albumentations
        bbox_pixel = bbox.copy()
        bbox_pixel = np.clip(bbox_pixel, 0.0, 1.0)

        kps_pixel = []
        for lm in landmarks:
            kps_pixel.append((float(lm[0] * w), float(lm[1] * h)))

        # Manual horizontal flip (to handle landmark swapping)
        if self.horizontal_flip and np.random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            bbox_pixel = np.array([1.0 - bbox_pixel[2], bbox_pixel[1],
                                   1.0 - bbox_pixel[0], bbox_pixel[3]])
            landmarks_flipped = landmarks.copy()
            landmarks_flipped[:, 0] = 1.0 - landmarks[:, 0]
            landmarks = landmarks_flipped[FLIP_LANDMARK_INDICES]
            kps_pixel = [(float(lm[0] * w), float(lm[1] * h)) for lm in landmarks]
            bbox_pixel = np.clip(bbox_pixel, 0.0, 1.0)

        # albumentations bbox format: [x_min, y_min, x_max, y_max] normalized
        try:
            result = self.transform(
                image=image,
                bboxes=[bbox_pixel.tolist()],
                labels=[0],
                keypoints=kps_pixel,
            )

            aug_image = result['image']

            if len(result['bboxes']) > 0:
                aug_bbox = np.array(result['bboxes'][0], dtype=np.float32)
            else:
                aug_bbox = bbox

            if len(result['keypoints']) == 5:
                aug_h, aug_w = aug_image.shape[:2]
                aug_landmarks = np.array([
                    [kp[0] / aug_w, kp[1] / aug_h] for kp in result['keypoints']
                ], dtype=np.float32)
            else:
                aug_landmarks = landmarks

            # Clamp landmarks to [0, 1]
            aug_landmarks = np.clip(aug_landmarks, 0.0, 1.0)
            aug_bbox = np.clip(aug_bbox, 0.0, 1.0)

            # Resize back to original size if needed
            if aug_image.shape[0] != h or aug_image.shape[1] != w:
                aug_image = _resize_with_coords(aug_image, h, w)

            return aug_image, aug_bbox, aug_landmarks

        except Exception:
            return image, bbox, landmarks


def _resize_with_coords(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image. Bbox/landmarks are normalized so no coord adjustment needed."""
    import cv2
    return cv2.resize(image, (target_w, target_h))
