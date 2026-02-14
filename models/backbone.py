"""
MobileNetV2 backbone for lightweight face detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_mobilenetv2_backbone(
    input_shape: tuple = (224, 224, 3),
    alpha: float = 0.5,
    include_top: bool = False,
    pretrained: bool = True,
) -> keras.Model:
    """
    Create MobileNetV2 backbone with configurable width multiplier.

    Args:
        input_shape: Input image shape (H, W, C)
        alpha: Width multiplier (0.25, 0.5, 0.75, 1.0)
        include_top: Whether to include classification head
        pretrained: Whether to load ImageNet weights (set False for inference with custom weights)

    Returns:
        MobileNetV2 backbone model outputting feature maps
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=include_top,
        weights='imagenet' if pretrained else None,
    )

    # Extract multi-scale feature maps
    # These layers provide features at different scales
    layer_names = [
        'block_3_expand_relu',   # 1/4 scale
        'block_6_expand_relu',   # 1/8 scale
        'block_13_expand_relu',  # 1/16 scale
        'out_relu',              # 1/32 scale
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]

    return keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name='mobilenetv2_backbone',
    )


class ConvBlock(layers.Layer):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        use_bn: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=not use_bn,
        )
        self.bn = layers.BatchNormalization() if use_bn else None
        self.relu = layers.ReLU()

    def call(self, x, training=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(layers.Layer):
    """Depthwise separable convolution for efficiency."""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pointwise = layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x, training=None):
        x = self.depthwise(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        return x


class FeaturePyramidNeck(layers.Layer):
    """
    Feature Pyramid Network (FPN) neck for multi-scale feature fusion.
    Used in BlazeFace-like architectures.
    """

    def __init__(self, out_channels: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to match channels)
        self.lateral_convs = []
        # Top-down feature maps
        self.fpn_convs = []

    def build(self, input_shapes):
        num_scales = len(input_shapes)

        for i in range(num_scales):
            self.lateral_convs.append(
                layers.Conv2D(
                    self.out_channels,
                    kernel_size=1,
                    padding='same',
                    name=f'lateral_{i}',
                )
            )
            self.fpn_convs.append(
                DepthwiseSeparableConv(
                    self.out_channels,
                    kernel_size=3,
                    name=f'fpn_{i}',
                )
            )

        super().build(input_shapes)

    def call(self, features, training=None):
        # features: list of [C2, C3, C4, C5] from backbone
        num_scales = len(features)

        # Apply lateral convolutions
        laterals = [
            self.lateral_convs[i](features[i])
            for i in range(num_scales)
        ]

        # Top-down pathway with upsampling
        for i in range(num_scales - 1, 0, -1):
            h, w = tf.shape(laterals[i-1])[1], tf.shape(laterals[i-1])[2]
            upsampled = tf.image.resize(laterals[i], [h, w])
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply FPN convolutions
        outputs = [
            self.fpn_convs[i](laterals[i], training=training)
            for i in range(num_scales)
        ]

        return outputs
