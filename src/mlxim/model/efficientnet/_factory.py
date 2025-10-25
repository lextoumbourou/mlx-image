"""Factory functions for EfficientNet models."""
from typing import List, Optional

from .._config import HFWeights, Metrics, ModelConfig, Transform
from ._blocks import MBConvConfig
from .efficientnet import EfficientNet


def _efficientnet_conf(
    arch: str,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
) -> tuple[List[MBConvConfig], Optional[int]]:
    """Get configuration for EfficientNet variants.

    Args:
        arch (str): Architecture name (e.g., 'efficientnet_b0')
        width_mult (float): Width multiplier for channels
        depth_mult (float): Depth multiplier for number of layers

    Returns:
        Tuple of (inverted_residual_setting, last_channel)
    """
    if arch.startswith("efficientnet_b"):
        # EfficientNet-B0 through B7 configuration
        # Format: (expand_ratio, kernel, stride, input_channels, out_channels, num_layers)
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
            MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
        ]
        last_channel = None
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(num_classes: int = 1000, dropout: float = 0.2) -> EfficientNet:
    """EfficientNet-B0 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B0 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


# Model configuration registry
efficientnet_configs = {
    "efficientnet_b0": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.77692, accuracy_at_5=0.93532),
        transform=Transform(img_size=224, crop_pct=224/256, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b0-mlxim", filename="model.safetensors"),
    ),
}
