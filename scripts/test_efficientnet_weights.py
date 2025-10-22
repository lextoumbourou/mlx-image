"""Test EfficientNet with MLX-format weights.

Before running this test, download and convert the PyTorch weights using:
  python scripts/convert_efficientnet_weights.py --model efficientnet_b0
  python scripts/convert_efficientnet_weights.py --model efficientnet_b1
"""

import mlx.core as mx

from mlxim.model.efficientnet.efficientnet import efficientnet_b0, efficientnet_b1


def test_model(model_name: str, model_fn, weights_path: str, input_size: int):
    """Test a single EfficientNet model."""
    print("\n" + "=" * 60)
    print(f"Testing {model_name}")
    print("=" * 60)

    # Create model
    model = model_fn()

    # Load MLX-format weights
    weights = mx.load(weights_path)
    print(f"Loaded {len(weights)} weight tensors from {weights_path}")

    # Load weights into model
    try:
        model.load_weights(list(weights.items()))
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        import sys

        sys.exit(1)

    # Test forward pass
    print(f"\nTesting forward pass with {input_size}x{input_size} input...")
    x = mx.random.normal((2, input_size, input_size, 3))
    output = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Output mean:  {output.mean():.3f}")
    print(f"  Output std:   {output.std():.3f}")

    # Test with different input sizes
    print(f"\nTesting with different input sizes...")
    for size in [224, 256, 384]:
        x_test = mx.random.normal((1, size, size, 3))
        out_test = model(x_test)
        print(f"  Input {size}x{size} -> Output shape: {out_test.shape}")

    print(f"✓ {model_name} tests passed!")


# Test EfficientNet-B0
test_model(
    model_name="EfficientNet-B0",
    model_fn=efficientnet_b0,
    weights_path="weights/efficientnet_b0/model_mlx.safetensors",
    input_size=224,
)

# Test EfficientNet-B1
test_model(
    model_name="EfficientNet-B1",
    model_fn=efficientnet_b1,
    weights_path="weights/efficientnet_b1/model_mlx.safetensors",
    input_size=240,
)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
