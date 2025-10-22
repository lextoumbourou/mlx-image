#!/usr/bin/env python3
"""Test loading PyTorch weights into MLX EfficientNet model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlxim.model.efficientnet import efficientnet_b0, load_efficientnet_weights


def main():
    print("=" * 80)
    print("Testing Weight Loading: PyTorch → MLX EfficientNet-B0")
    print("=" * 80)

    # Create model
    print("\n1. Creating MLX EfficientNet-B0 model...")
    model = efficientnet_b0(num_classes=1000, dropout=0.2)
    print("   ✓ Model created")

    # Load weights
    weights_path = Path("weights/efficientnet_b0_torch.safetensors")
    if not weights_path.exists():
        print(f"\n✗ Error: Weights file not found at {weights_path}")
        print("  Please run scripts/efficientnet_torch_inference.py first to download weights")
        return

    print(f"\n2. Loading weights from {weights_path}...")
    try:
        stats = load_efficientnet_weights(model, weights_path)
        print(f"\n   ✓ Weight loading complete!")
        print(f"   Statistics: {stats}")
    except Exception as e:
        print(f"\n✗ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test forward pass
    print("\n3. Testing forward pass with loaded weights...")
    x = mx.random.normal((1, 224, 224, 3))
    print(f"   Input shape: {x.shape}")

    try:
        output = model(x)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{float(mx.min(output)):.4f}, {float(mx.max(output)):.4f}]")

        # Check if outputs look reasonable (not all zeros or NaN)
        if mx.isnan(output).any():
            print("   ✗ Warning: Output contains NaN values!")
        elif mx.abs(output).max() < 1e-6:
            print("   ✗ Warning: Output values are too small!")
        else:
            print("   ✓ Forward pass successful!")
    except Exception as e:
        print(f"   ✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("Weight Loading Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
