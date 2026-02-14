"""
TensorFlow data pipeline for face detection.
Supports PURE dataset, WIDER FACE dataset, and combined training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .augmentation import FaceAugmentation, create_augmentation_from_config


class PUREFaceDataset:
    """TensorFlow dataset loader for PURE face detection data."""

    LANDMARK_ORDER = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    def __init__(
        self,
        pure_dir: str,
        labels_dir: str,
        input_size: int = 224,
        augmentation: Optional[FaceAugmentation] = None,
    ):
        """
        Initialize dataset.

        Args:
            pure_dir: Path to PURE dataset directory
            labels_dir: Path to generated labels directory
            input_size: Target input size for the model
            augmentation: Augmentation instance (None for validation)
        """
        self.pure_dir = Path(pure_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size
        self.augmentation = augmentation

        # Load all labels
        self.samples = self._load_all_labels()

    def _load_all_labels(self) -> List[Dict]:
        """Load all labels from JSON files."""
        samples = []

        for label_file in sorted(self.labels_dir.glob("*.json")):
            if label_file.name == "stats.json":
                continue

            session_id = label_file.stem  # e.g., "01-01"

            with open(label_file, 'r') as f:
                session_labels = json.load(f)

            for image_name, label_data in session_labels.items():
                # PURE structure: XX-YY/XX-YY/*.png
                image_path = self.pure_dir / session_id / session_id / image_name
                if not image_path.exists():
                    # Fallback: try direct path
                    image_path = self.pure_dir / session_id / image_name
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'bbox': label_data['bbox'],
                        'landmarks': label_data['landmarks'],
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_train_val_split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split samples into train and validation sets."""
        samples = self.samples.copy()

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        return samples[:split_idx], samples[split_idx:]

    def create_tf_dataset(
        self,
        samples: List[Dict],
        batch_size: int = 32,
        training: bool = True,
        shuffle_buffer: int = 1000,
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from samples.

        Args:
            samples: List of sample dictionaries
            batch_size: Batch size
            training: Whether this is training dataset (enables augmentation)
            shuffle_buffer: Shuffle buffer size

        Returns:
            tf.data.Dataset yielding (image, bbox, landmarks)
        """
        # Extract paths and labels
        image_paths = [s['image_path'] for s in samples]
        bboxes = np.array([
            [s['bbox']['x_min'], s['bbox']['y_min'],
             s['bbox']['x_max'], s['bbox']['y_max']]
            for s in samples
        ], dtype=np.float32)
        landmarks = np.array([
            [[s['landmarks'][name]['x'], s['landmarks'][name]['y']]
             for name in self.LANDMARK_ORDER]
            for s in samples
        ], dtype=np.float32)

        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': image_paths,
            'bbox': bboxes,
            'landmarks': landmarks,
        })

        if training and shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)

        # Load and preprocess images
        dataset = dataset.map(
            lambda x: self._load_and_preprocess(x, training),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _load_and_preprocess(
        self,
        sample: Dict,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load image and preprocess with augmentation."""
        # Load image
        image_data = tf.io.read_file(sample['image_path'])
        image = tf.io.decode_png(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        # Resize to input size
        image = tf.image.resize(image, [self.input_size, self.input_size])

        bbox = sample['bbox']
        landmarks = sample['landmarks']

        # Apply augmentation if training
        if training and self.augmentation is not None:
            image, bbox, landmarks = self.augmentation(
                image, bbox, landmarks, training=True
            )

        return image, bbox, landmarks


class WiderFaceDataset:
    """Dataset loader for WIDER FACE data (converted to PURE-compatible JSON)."""

    LANDMARK_ORDER = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        input_size: int = 256,
        augmentation: Optional[FaceAugmentation] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size
        self.augmentation = augmentation
        self.samples = self._load_all_labels()

    def _load_all_labels(self) -> List[Dict]:
        """Load all labels from per-category JSON files."""
        samples = []

        for label_file in sorted(self.labels_dir.glob("*.json")):
            if label_file.name == "stats.json":
                continue

            category = label_file.stem  # e.g., "0--Parade"

            with open(label_file, 'r', encoding='utf-8') as f:
                category_labels = json.load(f)

            for image_name, label_data in category_labels.items():
                image_path = self.images_dir / category / image_name
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'bbox': label_data['bbox'],
                        'landmarks': label_data['landmarks'],
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)


class FaceDatasetBase:
    """Base class with shared tf.data pipeline logic."""

    LANDMARK_ORDER = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    def __init__(self, input_size: int, augmentation: Optional[FaceAugmentation] = None):
        self.input_size = input_size
        self.augmentation = augmentation

    def create_tf_dataset(
        self,
        samples: List[Dict],
        batch_size: int = 32,
        training: bool = True,
        shuffle_buffer: int = 1000,
    ) -> tf.data.Dataset:
        """Create tf.data.Dataset from sample list."""
        image_paths = [s['image_path'] for s in samples]
        bboxes = np.array([
            [s['bbox']['x_min'], s['bbox']['y_min'],
             s['bbox']['x_max'], s['bbox']['y_max']]
            for s in samples
        ], dtype=np.float32)
        landmarks = np.array([
            [[s['landmarks'][name]['x'], s['landmarks'][name]['y']]
             for name in self.LANDMARK_ORDER]
            for s in samples
        ], dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': image_paths,
            'bbox': bboxes,
            'landmarks': landmarks,
        })

        if training and shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.map(
            lambda x: self._load_and_preprocess(x, training),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def _load_and_preprocess(
        self, sample: Dict, training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load image and preprocess."""
        image_data = tf.io.read_file(sample['image_path'])
        # Decode any image format (png or jpg)
        image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [self.input_size, self.input_size])

        bbox = sample['bbox']
        landmarks = sample['landmarks']

        if training and self.augmentation is not None:
            image, bbox, landmarks = self.augmentation(
                image, bbox, landmarks, training=True
            )

        return image, bbox, landmarks


def create_datasets_from_config(
    config: Dict,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create train and validation datasets from config.
    Supports PURE-only, WIDER FACE-only, or combined datasets.

    Args:
        config: Configuration dictionary

    Returns:
        (train_dataset, val_dataset)
    """
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    input_size = model_config.get('input_size', 256)

    # Create augmentation for training
    augmentation = create_augmentation_from_config(config)

    # Collect samples from all configured datasets
    all_samples = []

    # PURE dataset
    pure_dir = data_config.get('pure_dir')
    labels_dir = data_config.get('labels_dir')
    if pure_dir and labels_dir and Path(labels_dir).exists():
        pure_dataset = PUREFaceDataset(
            pure_dir=pure_dir,
            labels_dir=labels_dir,
            input_size=input_size,
            augmentation=augmentation,
        )
        print(f"  PURE: {len(pure_dataset)} samples")
        all_samples.extend(pure_dataset.samples)

    # WIDER FACE dataset
    wider_dir = data_config.get('wider_face_dir')
    wider_labels = data_config.get('wider_face_labels_dir')
    if wider_dir and wider_labels and Path(wider_labels).exists():
        wider_dataset = WiderFaceDataset(
            images_dir=wider_dir,
            labels_dir=wider_labels,
            input_size=input_size,
            augmentation=augmentation,
        )
        print(f"  WIDER FACE: {len(wider_dataset)} samples")
        all_samples.extend(wider_dataset.samples)

    if not all_samples:
        raise ValueError("No dataset configured or found. Check data paths in config.")

    # Split into train and validation
    train_ratio = data_config.get('train_split', 0.8)
    samples = all_samples.copy()
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"  Total: {len(all_samples)} -> {len(train_samples)} train, {len(val_samples)} val")

    # Create tf.data pipelines
    pipeline_train = FaceDatasetBase(input_size, augmentation)
    pipeline_val = FaceDatasetBase(input_size, augmentation=None)

    batch_size = training_config.get('batch_size', 32)
    train_dataset = pipeline_train.create_tf_dataset(train_samples, batch_size, training=True)
    val_dataset = pipeline_val.create_tf_dataset(val_samples, batch_size, training=False)

    return train_dataset, val_dataset
