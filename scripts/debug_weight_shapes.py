#!/usr/bin/env python3
"""Debug weight shapes in the MLX model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlxim.model.efficientnet import efficientnet_b0, load_efficientnet_weights


def main():
    # Create model
    print("Creating model...")
    model = efficientnet_b0(num_classes=1000, dropout=0.2)

    # Load weights
    weights_path = Path("weights/efficientnet_b0_torch.safetensors")
    print(f"Loading weights from {weights_path}...")
    load_efficientnet_weights(model, weights_path)

    # Check stem weights
    print("\n" + "=" * 80)
    print("Stem Layer Shapes")
    print("=" * 80)
    print(f"stem.layers[0] (Conv2d):")
    print(f"  weight shape: {model.stem.layers[0].weight.shape}")
    print(f"  stride: {model.stem.layers[0].stride}")
    print(f"  padding: {model.stem.layers[0].padding}")

    print(f"\nstem.layers[1] (BatchNorm):")
    print(f"  weight shape: {model.stem.layers[1].weight.shape}")
    print(f"  bias shape: {model.stem.layers[1].bias.shape}")

    print(f"\nstem.layers[2] (SiLU): {type(model.stem.layers[2])}")

    # Test just the conv layer
    print("\n" + "=" * 80)
    print("Testing Stem Conv Layer")
    print("=" * 80)

    x = mx.random.normal((1, 224, 224, 3))
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    conv = model.stem.layers[0]
    print(f"\nConv weight shape: {conv.weight.shape}")
    print(f"Conv weight dtype: {conv.weight.dtype}")
    print(f"Conv stride: {conv.stride}")
    print(f"Conv padding: {conv.padding}")

    try:
        out = conv(x)
        print(f"✓ Conv output shape: {out.shape}")
    except Exception as e:
        print(f"✗ Conv failed: {e}")


if __name__ == "__main__":
    main()
