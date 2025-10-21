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
    """

    def __init__(self, num_classes: int = 1000, drop_rate: float = 0.2, drop_path_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes

        # EfficientNet-B0 configuration
        settings = [
            StageConfig(expand_ratio=1, out_channels=16, num_blocks=1, kernel_size=3, stride=1),  # Stage 0
            StageConfig(expand_ratio=6, out_channels=24, num_blocks=2, kernel_size=3, stride=2),  # Stage 1
            StageConfig(expand_ratio=6, out_channels=40, num_blocks=2, kernel_size=5, stride=2),  # Stage 2
            StageConfig(expand_ratio=6, out_channels=80, num_blocks=3, kernel_size=3, stride=2),  # Stage 3
            StageConfig(expand_ratio=6, out_channels=112, num_blocks=3, kernel_size=5, stride=1),  # Stage 4
            StageConfig(expand_ratio=6, out_channels=192, num_blocks=4, kernel_size=5, stride=2),  # Stage 5
            StageConfig(expand_ratio=6, out_channels=320, num_blocks=1, kernel_size=3, stride=1),  # Stage 6
        ]

        # Stem: conv3x3 with 224x224 resolution, 3->32 channels
        self.conv_stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(32)

        # Build MBConv blocks
        self.blocks = []
        in_channels = 32

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

        # Head
        self.conv_head = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(1280)

        # Pooling and classifier
        self.avgpool = AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(1280, num_classes)

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


def efficientnet_b0(num_classes: int = 1000) -> EfficientNet:
    """Create EfficientNet B0 model.

    Args:
        num_classes (int): number of output classes. Defaults to 1000.

    Returns:
        EfficientNet: EfficientNet B0 model
    """
    return EfficientNet(num_classes=num_classes)


if __name__ == "__main__":
    model = efficientnet_b0()
    print(model)

    # Test with dummy input
    x = mx.random.normal((1, 224, 224, 3))
    out = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
