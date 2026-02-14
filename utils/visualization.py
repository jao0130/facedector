"""
Visualization utilities for face detection results.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Landmark names and colors
LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
LANDMARK_COLORS = ['blue', 'blue', 'green', 'red', 'red']


def visualize_detection(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    confidence: Optional[float] = None,
    gt_bbox: Optional[np.ndarray] = None,
    gt_landmarks: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize face detection results.

    Args:
        image: Input image [H, W, 3] (normalized 0-1 or 0-255)
        bbox: Predicted bounding box [4] (x_min, y_min, x_max, y_max) normalized
        landmarks: Predicted landmarks [5, 2] normalized
        confidence: Detection confidence score
        gt_bbox: Ground truth bounding box (optional)
        gt_landmarks: Ground truth landmarks (optional)
        save_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure
    """
    # Convert image to displayable format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    h, w = image.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Draw ground truth (if provided)
    if gt_bbox is not None:
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox
        gt_rect = Rectangle(
            (gt_x_min * w, gt_y_min * h),
            (gt_x_max - gt_x_min) * w,
            (gt_y_max - gt_y_min) * h,
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            linestyle='--',
            label='Ground Truth',
        )
        ax.add_patch(gt_rect)

    if gt_landmarks is not None:
        for i, (lx, ly) in enumerate(gt_landmarks):
            ax.scatter(
                lx * w, ly * h,
                c='lime',
                s=50,
                marker='x',
                linewidths=2,
            )

    # Draw predictions
    x_min, y_min, x_max, y_max = bbox
    pred_rect = Rectangle(
        (x_min * w, y_min * h),
        (x_max - x_min) * w,
        (y_max - y_min) * h,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label='Prediction',
    )
    ax.add_patch(pred_rect)

    # Draw predicted landmarks
    for i, (lx, ly) in enumerate(landmarks):
        ax.scatter(
            lx * w, ly * h,
            c=LANDMARK_COLORS[i],
            s=80,
            marker='o',
            edgecolors='white',
            linewidths=1.5,
        )
        ax.annotate(
            LANDMARK_NAMES[i],
            (lx * w + 5, ly * h),
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    # Add confidence score
    if confidence is not None:
        title = f'Confidence: {confidence:.3f}'
        ax.set_title(title, fontsize=12)

    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_batch(
    images: np.ndarray,
    pred_bboxes: np.ndarray,
    pred_landmarks: np.ndarray,
    gt_bboxes: Optional[np.ndarray] = None,
    gt_landmarks: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    max_samples: int = 16,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a batch of detection results.

    Args:
        images: Batch of images [B, H, W, 3]
        pred_bboxes: Predicted bboxes [B, 4]
        pred_landmarks: Predicted landmarks [B, 5, 2]
        gt_bboxes: Ground truth bboxes [B, 4] (optional)
        gt_landmarks: Ground truth landmarks [B, 5, 2] (optional)
        confidences: Confidence scores [B] (optional)
        max_samples: Maximum samples to display
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_samples = min(len(images), max_samples)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        image = images[idx]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        ax.imshow(image)

        # Draw prediction
        x_min, y_min, x_max, y_max = pred_bboxes[idx]
        pred_rect = Rectangle(
            (x_min * w, y_min * h),
            (x_max - x_min) * w,
            (y_max - y_min) * h,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
        )
        ax.add_patch(pred_rect)

        # Draw landmarks
        for i, (lx, ly) in enumerate(pred_landmarks[idx]):
            ax.scatter(lx * w, ly * h, c=LANDMARK_COLORS[i], s=30, marker='o')

        # Draw ground truth
        if gt_bboxes is not None:
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bboxes[idx]
            gt_rect = Rectangle(
                (gt_x_min * w, gt_y_min * h),
                (gt_x_max - gt_x_min) * w,
                (gt_y_max - gt_y_min) * h,
                linewidth=2,
                edgecolor='green',
                facecolor='none',
                linestyle='--',
            )
            ax.add_patch(gt_rect)

        if gt_landmarks is not None:
            for i, (lx, ly) in enumerate(gt_landmarks[idx]):
                ax.scatter(lx * w, ly * h, c='lime', s=30, marker='x')

        # Title
        if confidences is not None:
            ax.set_title(f'Conf: {confidences[idx]:.3f}', fontsize=10)

        ax.axis('off')

    # Hide empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: Dictionary with training history
            Keys: 'loss', 'val_loss', 'bbox_loss', 'landmark_loss', etc.
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Separate losses and metrics
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    metric_keys = [k for k in history.keys() if 'loss' not in k.lower()]

    n_plots = int(bool(loss_keys)) + int(bool(metric_keys))
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot losses
    if loss_keys:
        ax = axes[plot_idx]
        for key in loss_keys:
            label = key.replace('_', ' ').title()
            ax.plot(history[key], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot metrics
    if metric_keys:
        ax = axes[plot_idx]
        for key in metric_keys:
            label = key.replace('_', ' ').title()
            ax.plot(history[key], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

    return fig


def create_detection_video(
    model,
    image_paths: List[str],
    output_path: str,
    input_size: int = 224,
    fps: int = 30,
):
    """
    Create a video showing face detection results.

    Args:
        model: Face detection model
        image_paths: List of image file paths
        output_path: Output video path
        input_size: Model input size
        fps: Output video FPS
    """
    import cv2

    # Get first image to determine size
    first_image = cv2.imread(image_paths[0])
    h, w = first_image.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_path in image_paths:
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_image = cv2.resize(rgb_image, (input_size, input_size))
        input_image = input_image.astype(np.float32) / 255.0
        input_batch = np.expand_dims(input_image, 0)

        # Predict
        predictions = model(input_batch, training=False)
        bbox = predictions['bbox'][0].numpy()
        landmarks = predictions['landmarks'][0].numpy()

        # Draw on original image
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(
            image,
            (int(x_min * w), int(y_min * h)),
            (int(x_max * w), int(y_max * h)),
            (0, 0, 255),
            2,
        )

        for lx, ly in landmarks:
            cv2.circle(
                image,
                (int(lx * w), int(ly * h)),
                3,
                (0, 255, 0),
                -1,
            )

        out.write(image)

    out.release()
    print(f"Video saved to {output_path}")
