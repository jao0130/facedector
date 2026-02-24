"""
MobileNetV2 backbone for lightweight face detection.

Uses timm for pretrained MobileNetV2 with multi-scale feature extraction
and a Feature Pyramid Network (FPN) neck for feature fusion.
All tensors use NCHW format (PyTorch convention).
"""

from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: DepthwiseConv2d + BN + ReLU + Pointwise Conv2d + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class FeaturePyramidNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) neck for multi-scale feature fusion.
    Used in BlazeFace-like architectures.

    Takes a list of feature maps from the backbone at different scales
    and outputs a list of feature maps all with out_channels channels.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels
        num_scales = len(in_channels_list)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        self.fpn_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3)
            for _ in range(num_scales)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: list of [C2, C3, C4, C5] feature maps from backbone (NCHW).

        Returns:
            list of FPN feature maps, each with out_channels channels.
        """
        num_scales = len(features)

        # Apply lateral 1x1 convolutions
        laterals = [
            self.lateral_convs[i](features[i])
            for i in range(num_scales)
        ]

        # Top-down pathway with upsampling and element-wise addition
        for i in range(num_scales - 1, 0, -1):
            target_h = laterals[i - 1].shape[2]
            target_w = laterals[i - 1].shape[3]
            upsampled = F.interpolate(
                laterals[i],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply depthwise separable convolutions
        outputs = [
            self.fpn_convs[i](laterals[i])
            for i in range(num_scales)
        ]

        return outputs


def create_mobilenetv2_backbone(
    input_size: int = 224,
    alpha: float = 0.5,
    pretrained: bool = True,
) -> Tuple[nn.Module, List[int]]:
    """
    Create MobileNetV2 backbone with configurable width multiplier.

    Uses timm's feature extraction mode to get multi-scale feature maps
    at 4 scales (1/4, 1/8, 1/16, 1/32).

    Args:
        input_size: Input image spatial size (used only for documentation; not baked in).
        alpha: Width multiplier (0.5 maps to mobilenetv2_050 in timm).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Tuple of (backbone model, list of output channel counts per scale).
    """
    model_name = f"mobilenetv2_{int(alpha * 100):03d}"
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
    )

    out_channels_list = backbone.feature_info.channels()

    return backbone, out_channels_list
