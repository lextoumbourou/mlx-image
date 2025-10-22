from collections import namedtuple
from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ._efficientnet_blocks import SqueezeExcite
from ..layers.misc import Conv2dNormActivation, StochasticDepth
from ..layers.pool import AdaptiveAvgPool2d
from ..layers.utils import _make_divisible

# Configuration for each stage in EfficientNet
StageConfig = namedtuple("StageConfig", ["expand_ratio", "out_channels", "num_blocks", "kernel_size", "stride"])


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        expand_ratio (int): Expansion ratio for the hidden dimension
        kernel_size (int): Kernel size for depthwise convolution
        stride (int): Stride for depthwise convolution
        se_ratio (float): Squeeze-and-excitation ratio
        drop_rate (float): Dropout rate for stochastic depth
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio

        if expand_ratio == 1:
            # MBConv1: no expansion phase
            # Structure: bn1 -> conv_dw -> SE -> conv_pw (projection) -> bn2
            self.bn1 = nn.BatchNorm(in_channels)

            self.conv_dw = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
                bias=False,
            )

            # SE operates on input channels
            self.se = SqueezeExcite(
                in_chs=in_channels,
                rd_ratio=se_ratio,
                act_layer=nn.SiLU,
                gate_layer=nn.Sigmoid,
            )

            # Projection
            self.conv_pw = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm(out_channels)

        else:
            # MBConv6: with expansion
            # Structure: conv_pw (expansion) -> bn1 -> conv_dw -> bn2 -> SE -> conv_pwl (projection) -> bn3

            # Expansion
            self.conv_pw = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm(hidden_dim)

            # Depthwise
            self.conv_dw = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            )
            self.bn2 = nn.BatchNorm(hidden_dim)

            # SE operates on expanded channels
            # Note: rd_channels is based on in_channels, not hidden_dim (EfficientNet design)
            rd_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcite(
                in_chs=hidden_dim,
                rd_channels=rd_channels,
                act_layer=nn.SiLU,
                gate_layer=nn.Sigmoid,
            )

            # Projection
            self.conv_pwl = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm(out_channels)

        # Stochastic depth
        self.drop_rate = drop_rate
        if self.use_residual and drop_rate > 0:
            self.stochastic_depth = StochasticDepth(drop_rate, mode="row")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        identity = x

        if self.expand_ratio == 1:
            # MBConv1 path
            x = self.bn1(x)
            x = nn.silu(x)
            x = self.conv_dw(x)
            x = self.se(x)
            x = self.conv_pw(x)
            x = self.bn2(x)
        else:
            # MBConv6 path
            x = self.conv_pw(x)
            x = self.bn1(x)
            x = nn.silu(x)
            x = self.conv_dw(x)
            x = self.bn2(x)
            x = nn.silu(x)
            x = self.se(x)
            x = self.conv_pwl(x)
            x = self.bn3(x)

        # Residual connection with stochastic depth
        if self.use_residual:
            if self.drop_rate > 0:
                x = self.stochastic_depth(x)
            x = x + identity

        return x


class EfficientNet(nn.Module):
    """EfficientNet model.

    Args:
        num_classes (int): number of output classes. Defaults to 1000.
        drop_rate (float): dropout rate before classifier. Defaults to 0.2.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.2.
        depth_multiplier (float): depth scaling multiplier. Defaults to 1.0.
        channel_multiplier (float): channel width scaling multiplier. Defaults to 1.0.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        depth_multiplier: float = 1.0,
        channel_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Base EfficientNet configuration (B0)
        # Each stage: (expand_ratio, out_channels, num_blocks, kernel_size, stride)
        base_settings = [
            (1, 16, 1, 3, 1),  # Stage 0: MBConv1, k3x3
            (6, 24, 2, 3, 2),  # Stage 1: MBConv6, k3x3
            (6, 40, 2, 5, 2),  # Stage 2: MBConv6, k5x5
            (6, 80, 3, 3, 2),  # Stage 3: MBConv6, k3x3
            (6, 112, 3, 5, 1),  # Stage 4: MBConv6, k5x5
            (6, 192, 4, 5, 2),  # Stage 5: MBConv6, k5x5
            (6, 320, 1, 3, 1),  # Stage 6: MBConv6, k3x3
        ]

        # Apply scaling to create the configuration
        settings = []
        for expand_ratio, base_channels, base_blocks, kernel_size, stride in base_settings:
            # Scale channels (round to nearest 8)
            out_channels = _make_divisible(base_channels * channel_multiplier, 8)
            # Scale depth (use ceiling to match timm)
            num_blocks = int(mx.ceil(base_blocks * depth_multiplier).item())
            settings.append(
                StageConfig(
                    expand_ratio=expand_ratio,
                    out_channels=out_channels,
                    num_blocks=num_blocks,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )

        # Stem: conv3x3, 3->32 channels (stem size is fixed across variants)
        stem_size = 32
        self.conv_stem = nn.Conv2d(
            in_channels=3, out_channels=stem_size, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(stem_size)

        # Build MBConv blocks
        self.blocks = []
        in_channels = stem_size

        # Calculate total number of blocks for stochastic depth
        total_blocks = sum([stage.num_blocks for stage in settings])
        block_idx = 0

        for stage in settings:
            stage_blocks = []
            for i in range(stage.num_blocks):
                # Stochastic depth linearly increases
                drop_rate_block = drop_path_rate * block_idx / total_blocks

                stage_blocks.append(
                    MBConvBlock(
                        in_channels=in_channels if i == 0 else stage.out_channels,
                        out_channels=stage.out_channels,
                        expand_ratio=stage.expand_ratio,
                        kernel_size=stage.kernel_size,
                        stride=stage.stride if i == 0 else 1,
                        se_ratio=0.25,
                        drop_rate=drop_rate_block,
                    )
                )
                block_idx += 1

            self.blocks.append(stage_blocks)
            in_channels = stage.out_channels

        # Head: 1x1 conv to expand to 1280 features (fixed across variants)
        head_size = 1280
        self.conv_head = nn.Conv2d(in_channels=in_channels, out_channels=head_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(head_size)

        # Pooling and classifier
        self.avgpool = AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(head_size, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x (mx.array): input tensor of shape (B, H, W, C)

        Returns:
            mx.array: output tensor of shape (B, num_classes)
        """
        # Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = nn.silu(x)

        # MBConv blocks
        for stage_blocks in self.blocks:
            for block in stage_blocks:
                x = block(x)

        # Head
        x = self.conv_head(x)
        x = self.bn2(x)
        x = nn.silu(x)

        # Classifier
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def efficientnet_b0(num_classes: int = 1000, drop_rate: float = 0.2, drop_path_rate: float = 0.2) -> EfficientNet:
    """Create EfficientNet B0 model.

    EfficientNet-B0 baseline:
    - Depth multiplier: 1.0
    - Width multiplier: 1.0
    - Input resolution: 224x224
    - Parameters: ~5.3M

    Args:
        num_classes (int): number of output classes. Defaults to 1000.
        drop_rate (float): dropout rate before classifier. Defaults to 0.2.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.2.

    Returns:
        EfficientNet: EfficientNet B0 model
    """
    return EfficientNet(
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        depth_multiplier=1.0,
        channel_multiplier=1.0,
    )


def efficientnet_b1(num_classes: int = 1000, drop_rate: float = 0.2, drop_path_rate: float = 0.2) -> EfficientNet:
    """Create EfficientNet B1 model.

    EfficientNet-B1:
    - Depth multiplier: 1.1
    - Width multiplier: 1.0
    - Input resolution: 240x240
    - Parameters: ~7.8M

    Args:
        num_classes (int): number of output classes. Defaults to 1000.
        drop_rate (float): dropout rate before classifier. Defaults to 0.2.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.2.

    Returns:
        EfficientNet: EfficientNet B1 model
    """
    return EfficientNet(
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        depth_multiplier=1.1,
        channel_multiplier=1.0,
    )


if __name__ == "__main__":
    # Test B0
    print("Testing EfficientNet-B0:")
    print("=" * 60)
    model_b0 = efficientnet_b0()
    x_b0 = mx.random.normal((1, 224, 224, 3))
    out_b0 = model_b0(x_b0)
    print(f"Input shape: {x_b0.shape}")
    print(f"Output shape: {out_b0.shape}")

    # Test B1
    print("\nTesting EfficientNet-B1:")
    print("=" * 60)
    model_b1 = efficientnet_b1()
    x_b1 = mx.random.normal((1, 240, 240, 3))
    out_b1 = model_b1(x_b1)
    print(f"Input shape: {x_b1.shape}")
    print(f"Output shape: {out_b1.shape}")
