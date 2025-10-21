"""EfficientNet-specific blocks

This module contains block implementations specific to EfficientNet that match
the pytorch-image-models (timm) naming conventions, allowing direct weight loading
without renaming.

Based on code originally written by Ross Wightman.
"""

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family,
    based on original code written by Ross Wightman.

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction (default: 0.25)
        rd_channels (Optional[int]): explicit reduced channel count (overrides rd_ratio)
        act_layer: activation layer (default: nn.ReLU)
        gate_layer: attention gate function (default: nn.Sigmoid)
        force_act_layer (Optional): override block's activation fn if this is set
        rd_round_fn (Optional[Callable]): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        force_act_layer=None,
        rd_round_fn: Optional[Callable] = None,
    ):
        super().__init__()

        # Calculate reduced channels
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = int(rd_round_fn(in_chs * rd_ratio))

        # Use force_act_layer if provided, otherwise use act_layer
        act_layer = force_act_layer or act_layer

        # Note: Using conv_reduce/conv_expand naming to match timm,
        # not fc1/fc2, so weights can be loaded directly
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, kernel_size=1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x (mx.array): input tensor of shape (B, H, W, C)

        Returns:
            mx.array: output tensor of shape (B, H, W, C)
        """
        # Global average pooling - MLX uses (B, H, W, C) format
        # Mean over spatial dimensions (H, W), keeping dims
        x_se = mx.mean(x, axis=(1, 2), keepdims=True)

        # Squeeze
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)

        # Excite
        x_se = self.conv_expand(x_se)
        x_se = self.gate(x_se)

        # Scale input
        return x * x_se
