"""
Face detection model: MobileNetV2 + FPN + detection heads.
Bbox/confidence use global pooling; landmarks use spatial soft-argmax for positional accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .backbone import create_mobilenetv2_backbone, FeaturePyramidNeck


class SpatialSoftArgmax(nn.Module):
    """
    Spatial soft-argmax: converts feature maps to heatmaps, then extracts
    (x, y) coordinates via differentiable weighted average.

    Preserves spatial information — critical for landmark localization.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmaps: [B, K, H, W] raw logits (one channel per landmark)
        Returns:
            coords: [B, K, 2] normalized (x, y) in [0, 1]
        """
        B, K, H, W = heatmaps.shape

        # Softmax over spatial dimensions
        flat = heatmaps.view(B, K, -1) / self.temperature
        weights = F.softmax(flat, dim=-1)  # [B, K, H*W]

        # Create coordinate grids normalized to [0, 1]
        # Use (0.5/W, 1.5/W, ...) so coords are pixel-centered
        grid_x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=heatmaps.device)
        grid_y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=heatmaps.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        grid_x = grid_x.reshape(-1)  # [H*W]
        grid_y = grid_y.reshape(-1)

        # Weighted sum of coordinates
        x = (weights * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, K]
        y = (weights * grid_y.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        return torch.stack([x, y], dim=-1)  # [B, K, 2]


class FaceDetector(nn.Module):
    """
    Lightweight face detector with bounding box and landmark prediction.

    Architecture:
        - Backbone: MobileNetV2 (multi-scale)
        - Neck: FPN (feature fusion)
        - Bbox/Confidence: Global pool -> FC (good for global regression)
        - Landmarks: Conv heatmaps -> Spatial Soft-Argmax (preserves spatial info)

    Input: [B, 3, H, W] (NCHW, float32, range [0, 1])
    Output: dict with:
        - 'bbox': [B, 4] (x_min, y_min, x_max, y_max), normalized [0, 1]
        - 'landmarks': [B, 5, 2] (x, y per landmark), normalized [0, 1]
        - 'confidence': [B, 1] sigmoid probability
    """

    def __init__(self, cfg=None, input_size=256, backbone_alpha=0.5,
                 num_landmarks=5, fpn_channels=64, pretrained=True):
        super().__init__()

        if cfg is not None:
            input_size = cfg.FACE_MODEL.INPUT_SIZE
            backbone_alpha = cfg.FACE_MODEL.BACKBONE_ALPHA
            num_landmarks = cfg.FACE_MODEL.NUM_LANDMARKS
            fpn_channels = cfg.FACE_MODEL.FPN_CHANNELS
            pretrained = cfg.FACE_MODEL.PRETRAINED

        self.input_size = input_size
        self.num_landmarks = num_landmarks

        # Backbone: MobileNetV2 multi-scale features
        self.backbone, out_channels_list = create_mobilenetv2_backbone(
            input_size=input_size,
            alpha=backbone_alpha,
            pretrained=pretrained,
        )

        # FPN neck
        self.fpn = FeaturePyramidNeck(
            in_channels_list=out_channels_list,
            out_channels=fpn_channels,
        )

        # ── Bbox + Confidence head (global pooling is fine for these) ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_shared = nn.Sequential(
            nn.Linear(fpn_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc_bbox = nn.Linear(128, 4)
        self.fc_confidence = nn.Linear(128, 1)

        # ── Landmark head (spatial — preserves positional information) ──
        self.landmark_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_landmarks, 1),  # one heatmap per landmark
        )
        self.spatial_softargmax = SpatialSoftArgmax(temperature=1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.backbone(x)

        # Apply FPN
        fpn_features = self.fpn(features)

        # Use second scale (1/8) — 32x32 for 256px input
        main_feature = fpn_features[1]  # [B, fpn_channels, H/8, W/8]

        # ── Bbox + Confidence (global) ──
        pooled = self.global_pool(main_feature).flatten(1)  # [B, fpn_channels]
        shared = self.fc_shared(pooled)  # [B, 128]

        bbox_raw = torch.sigmoid(self.fc_bbox(shared))  # [B, 4]
        cx, cy, w, h = bbox_raw[:, 0:1], bbox_raw[:, 1:2], bbox_raw[:, 2:3], bbox_raw[:, 3:4]
        bbox = torch.cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        bbox = bbox.clamp(0.0, 1.0)

        confidence = torch.sigmoid(self.fc_confidence(shared))  # [B, 1]

        # ── Landmarks (spatial) ──
        heatmaps = self.landmark_conv(main_feature)  # [B, 5, H/8, W/8]
        landmarks = self.spatial_softargmax(heatmaps)  # [B, 5, 2]

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }


class FaceDetectorLite(nn.Module):
    """Ultra-lightweight face detector for mobile inference."""

    def __init__(self, input_size: int = 128, num_landmarks: int = 5):
        super().__init__()
        self.input_size = input_size
        self.num_landmarks = num_landmarks

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self._dsconv_block(16, 32, stride=2),
            self._dsconv_block(32, 64, stride=2),
            self._dsconv_block(64, 64, stride=1),
            self._dsconv_block(64, 128, stride=2),
            self._dsconv_block(128, 128, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc_bbox = nn.Linear(128, 4)
        self.fc_landmarks = nn.Linear(128, num_landmarks * 2)
        self.fc_confidence = nn.Linear(128, 1)

    @staticmethod
    def _dsconv_block(in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)

        bbox_raw = torch.sigmoid(self.fc_bbox(feat))
        cx, cy, w, h = bbox_raw[:, 0:1], bbox_raw[:, 1:2], bbox_raw[:, 2:3], bbox_raw[:, 3:4]
        bbox = torch.cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        bbox = bbox.clamp(0.0, 1.0)

        landmarks = torch.sigmoid(self.fc_landmarks(feat))
        landmarks = landmarks.view(-1, self.num_landmarks, 2)
        confidence = torch.sigmoid(self.fc_confidence(feat))

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence,
        }


def create_face_detector(cfg, pretrained: bool = True) -> FaceDetector:
    """Create face detector model from config."""
    return FaceDetector(cfg=cfg)
