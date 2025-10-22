#!/usr/bin/env python3
"""Test script for EfficientNet stem with Conv2dNormActivation."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlxim.model.efficientnet import EfficientNet
from mlxim.model.layers import Conv2dNormActivation
import mlx.nn as nn


def test_conv2d_norm_activation():
    """Test the Conv2dNormActivation block."""
    print("Testing Conv2dNormActivation block...")

    # Create a simple Conv2dNormActivation block
    block = Conv2dNormActivation(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=2,
        norm_layer=nn.BatchNorm,
        activation_layer=nn.SiLU,
    )

    # Create dummy input (batch=2, height=224, width=224, channels=3)
    x = mx.random.normal((2, 224, 224, 3))

    print(f"Input shape: {x.shape}")

    # Forward pass
    output = block(x)

    print(f"Output shape: {output.shape}")

    # Check output shape (should be batch=2, h=112, w=112, c=32 due to stride=2)
    expected_h = (224 - 3 + 2 * 1) // 2 + 1  # (H - K + 2P) / S + 1
    expected_w = (224 - 3 + 2 * 1) // 2 + 1
    expected_shape = (2, expected_h, expected_w, 32)

    print(f"Expected shape: {expected_shape}")

    assert output.shape == expected_shape, f"Shape mismatch! Got {output.shape}, expected {expected_shape}"
    print("✓ Conv2dNormActivation test passed!\n")


def test_efficientnet_stem():
    """Test EfficientNet with stem layer."""
    print("Testing EfficientNet with stem...")

    # Create model
    model = EfficientNet(num_classes=1000, dropout_rate=0.2)

    # Create dummy input (batch=2, height=224, width=224, channels=3)
    x = mx.random.normal((2, 224, 224, 3))

    print(f"Input shape: {x.shape}")

    # Forward pass through stem only
    stem_output = model.stem(x)
    print(f"Stem output shape: {stem_output.shape}")

    # Forward pass through entire model (will go through stem, avgpool, classifier)
    output = model(x)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (2, 1000), f"Output shape mismatch! Got {output.shape}"
    print("✓ EfficientNet stem test passed!\n")


def test_stem_components():
    """Test the components of the stem."""
    print("Testing stem components...")

    model = EfficientNet()

    # Check that stem is a Conv2dNormActivation
    assert isinstance(model.stem, Conv2dNormActivation), "Stem should be Conv2dNormActivation"

    # Check stem layers
    print(f"Stem has {len(model.stem.layers)} layers")
    print("Stem layers:")
    for i, layer in enumerate(model.stem.layers):
        print(f"  {i}: {type(layer).__name__}")

    # Expected: Conv2d, BatchNorm, SiLU
    assert len(model.stem.layers) == 3, f"Expected 3 layers, got {len(model.stem.layers)}"
    assert isinstance(model.stem.layers[0], nn.Conv2d), "First layer should be Conv2d"
    assert isinstance(model.stem.layers[1], nn.BatchNorm), "Second layer should be BatchNorm"
    assert isinstance(model.stem.layers[2], nn.SiLU), "Third layer should be SiLU"

    print("✓ Stem components test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("EfficientNet Stem Tests")
    print("=" * 60 + "\n")

    test_conv2d_norm_activation()
    test_efficientnet_stem()
    test_stem_components()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
