"""
Training script for face detection model.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from data.dataset import create_datasets_from_config
from data.augmentation import create_augmentation_from_config
from models.face_detector import FaceDetector, create_face_detector
from utils.losses import smooth_l1_loss, wing_loss, giou_loss
from utils.metrics import IoUMetric, NMEMetric, compute_iou, compute_nme
from utils.visualization import visualize_batch, plot_training_history


class FaceDetectorTrainer:
    """Training manager for face detection model."""

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.loss_config = config.get('loss', {})
        self.callback_config = config.get('callbacks', {})

        # Create model
        self.model = create_face_detector(config)

        # Build model
        input_size = self.model_config.get('input_size', 224)
        self.model.build((None, input_size, input_size, 3))

        # Print model summary
        self.model.summary()

        # Setup optimizer with warmup
        self.setup_optimizer()

        # Setup metrics
        self.train_loss_tracker = keras.metrics.Mean(name='loss')
        self.train_bbox_loss_tracker = keras.metrics.Mean(name='bbox_loss')
        self.train_landmark_loss_tracker = keras.metrics.Mean(name='landmark_loss')
        self.train_iou_metric = IoUMetric(name='iou')
        self.train_nme_metric = NMEMetric(name='nme')

        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')
        self.val_iou_metric = IoUMetric(name='val_iou')
        self.val_nme_metric = NMEMetric(name='val_nme')

        # Training history
        self.history = {
            'loss': [],
            'bbox_loss': [],
            'landmark_loss': [],
            'iou': [],
            'nme': [],
            'val_loss': [],
            'val_iou': [],
            'val_nme': [],
            'lr': [],
        }

    def setup_optimizer(self, steps_per_epoch: int = 100):
        """Setup optimizer with learning rate schedule.

        Args:
            steps_per_epoch: Number of training steps per epoch
        """
        initial_lr = self.training_config.get('initial_learning_rate', 0.001)
        min_lr = self.training_config.get('min_learning_rate', 0.00001)
        epochs = self.training_config.get('epochs', 100)
        warmup_epochs = self.training_config.get('warmup_epochs', 5)

        # Calculate total steps (not epochs)
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        # Cosine decay with warmup
        self.lr_schedule = WarmupCosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            alpha=min_lr / initial_lr,
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def compute_loss(
        self,
        predictions: Dict[str, tf.Tensor],
        gt_bbox: tf.Tensor,
        gt_landmarks: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute training loss.

        Args:
            predictions: Model predictions
            gt_bbox: Ground truth bounding boxes [B, 4]
            gt_landmarks: Ground truth landmarks [B, 5, 2]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        pred_bbox = predictions['bbox']
        pred_landmarks = predictions['landmarks']
        pred_confidence = predictions['confidence']

        # Bounding box loss (GIoU)
        bbox_loss = giou_loss(pred_bbox, gt_bbox)

        # Landmark loss (Wing loss)
        landmark_loss = wing_loss(pred_landmarks, gt_landmarks, w=10.0, epsilon=2.0)

        # Confidence loss
        confidence_target = tf.ones_like(pred_confidence)
        confidence_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(confidence_target, pred_confidence)
        )

        # Weighted combination
        bbox_weight = self.loss_config.get('bbox_weight', 1.0)
        landmark_weight = self.loss_config.get('landmark_weight', 0.5)
        confidence_weight = self.loss_config.get('confidence_weight', 1.0)

        total_loss = (
            bbox_weight * bbox_loss +
            landmark_weight * landmark_loss +
            confidence_weight * confidence_loss
        )

        return total_loss, {
            'bbox_loss': bbox_loss,
            'landmark_loss': landmark_loss,
            'confidence_loss': confidence_loss,
        }

    @tf.function
    def train_step(
        self,
        images: tf.Tensor,
        gt_bbox: tf.Tensor,
        gt_landmarks: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            total_loss, loss_dict = self.compute_loss(
                predictions, gt_bbox, gt_landmarks
            )

        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update metrics
        self.train_loss_tracker.update_state(total_loss)
        self.train_bbox_loss_tracker.update_state(loss_dict['bbox_loss'])
        self.train_landmark_loss_tracker.update_state(loss_dict['landmark_loss'])
        self.train_iou_metric.update_state(
            (gt_bbox, gt_landmarks), predictions
        )
        self.train_nme_metric.update_state(
            (gt_bbox, gt_landmarks), predictions
        )

        return {
            'loss': total_loss,
            **loss_dict,
        }

    @tf.function
    def val_step(
        self,
        images: tf.Tensor,
        gt_bbox: tf.Tensor,
        gt_landmarks: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Single validation step."""
        predictions = self.model(images, training=False)
        total_loss, loss_dict = self.compute_loss(
            predictions, gt_bbox, gt_landmarks
        )

        # Update metrics
        self.val_loss_tracker.update_state(total_loss)
        self.val_iou_metric.update_state(
            (gt_bbox, gt_landmarks), predictions
        )
        self.val_nme_metric.update_state(
            (gt_bbox, gt_landmarks), predictions
        )

        return {
            'loss': total_loss,
            **loss_dict,
            'predictions': predictions,
        }

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
    ):
        """
        Full training loop.

        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            epochs: Number of epochs (overrides config)
        """
        if epochs is None:
            epochs = self.training_config.get('epochs', 100)

        # Calculate steps per epoch and reinitialize optimizer with correct schedule
        steps_per_epoch = sum(1 for _ in train_dataset)
        self.setup_optimizer(steps_per_epoch=steps_per_epoch)
        print(f"Steps per epoch: {steps_per_epoch}")

        # Setup directories
        checkpoint_dir = Path(self.callback_config.get('checkpoint_dir', 'checkpoints'))
        log_dir = Path(self.callback_config.get('log_dir', 'logs'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.callback_config.get('early_stopping_patience', 15)

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

        for epoch in range(epochs):
            print(f"\n==== Epoch {epoch + 1}/{epochs} ====")

            # Reset metrics
            self.train_loss_tracker.reset_state()
            self.train_bbox_loss_tracker.reset_state()
            self.train_landmark_loss_tracker.reset_state()
            self.train_iou_metric.reset_state()
            self.train_nme_metric.reset_state()
            self.val_loss_tracker.reset_state()
            self.val_iou_metric.reset_state()
            self.val_nme_metric.reset_state()

            # Training with tqdm progress bar
            tbar = tqdm(train_dataset, ncols=150, desc=f"Train Epoch {epoch + 1}")
            for step, batch in enumerate(tbar):
                images, gt_bbox, gt_landmarks = batch
                result = self.train_step(images, gt_bbox, gt_landmarks)

                # Update progress bar with current metrics
                tbar.set_postfix(
                    loss=f"{float(self.train_loss_tracker.result()):.4f}",
                    bbox=f"{float(self.train_bbox_loss_tracker.result()):.4f}",
                    lmk=f"{float(self.train_landmark_loss_tracker.result()):.4f}",
                    iou=f"{float(self.train_iou_metric.result()):.4f}",
                )

            # Validation with tqdm progress bar
            vbar = tqdm(val_dataset, ncols=150, desc="Validation")
            for batch in vbar:
                images, gt_bbox, gt_landmarks = batch
                self.val_step(images, gt_bbox, gt_landmarks)

                # Update validation progress bar
                vbar.set_postfix(
                    val_loss=f"{float(self.val_loss_tracker.result()):.4f}",
                    val_iou=f"{float(self.val_iou_metric.result()):.4f}",
                    val_nme=f"{float(self.val_nme_metric.result()):.4f}",
                )

            # Get current learning rate
            lr = self.optimizer.learning_rate
            if callable(lr):
                current_lr = float(lr(self.optimizer.iterations))
            else:
                current_lr = float(lr)

            # Log metrics
            train_loss = float(self.train_loss_tracker.result())
            train_bbox_loss = float(self.train_bbox_loss_tracker.result())
            train_landmark_loss = float(self.train_landmark_loss_tracker.result())
            train_iou = float(self.train_iou_metric.result())
            train_nme = float(self.train_nme_metric.result())
            val_loss = float(self.val_loss_tracker.result())
            val_iou = float(self.val_iou_metric.result())
            val_nme = float(self.val_nme_metric.result())

            self.history['loss'].append(train_loss)
            self.history['bbox_loss'].append(train_bbox_loss)
            self.history['landmark_loss'].append(train_landmark_loss)
            self.history['iou'].append(train_iou)
            self.history['nme'].append(train_nme)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_nme'].append(val_nme)
            self.history['lr'].append(current_lr)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs} Summary:")
            print(f"  Train - loss: {train_loss:.4f}, bbox: {train_bbox_loss:.4f}, "
                  f"lmk: {train_landmark_loss:.4f}, iou: {train_iou:.4f}, nme: {train_nme:.4f}")
            print(f"  Val   - loss: {val_loss:.4f}, iou: {val_iou:.4f}, nme: {val_nme:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"{'='*60}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.model.save_weights(str(checkpoint_dir / 'best_model.weights.h5'))
                print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.model.save_weights(
                    str(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.weights.h5')
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Save final model
        self.model.save_weights(str(checkpoint_dir / 'final_model.weights.h5'))

        # Save training history
        import json
        with open(log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        # Plot training curves
        plot_training_history(self.history, save_path=str(log_dir / 'training_curves.png'))

        print("\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return self.history


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int = 5,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Handle edge case: no warmup
        if self.warmup_steps == 0:
            decay_step = step
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / (decay_steps + 1e-8)))
            decayed_lr = (1 - self.alpha) * cosine_decay + self.alpha
            return self.initial_learning_rate * decayed_lr

        # Warmup phase (add epsilon to prevent division by zero)
        warmup_lr = self.initial_learning_rate * (step / (warmup_steps + 1e-8))

        # Cosine decay phase
        decay_step = tf.maximum(step - warmup_steps, 0.0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / (decay_steps - warmup_steps + 1e-8)))
        decayed_lr = (1 - self.alpha) * cosine_decay + self.alpha
        decay_lr = self.initial_learning_rate * decayed_lr

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'warmup_steps': self.warmup_steps,
            'alpha': self.alpha,
        }


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_gpu(memory_limit_mb: int = 13000):
    """Configure GPU memory limit to avoid overloading.

    Args:
        memory_limit_mb: Maximum GPU memory to use in MB (default: 13000 = ~80% of 16GB)
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Limit GPU memory to specified amount
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            print(f"Found {len(gpus)} GPU(s), memory limited to {memory_limit_mb}MB (~80% of 16GB)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")


def main():
    # Setup GPU before any TensorFlow operations
    setup_gpu()

    parser = argparse.ArgumentParser(description='Train face detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create datasets
    print("Loading datasets...")
    train_dataset, val_dataset = create_datasets_from_config(config)

    # Create trainer
    trainer = FaceDetectorTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.model.load_weights(args.resume)

    # Train
    trainer.train(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
