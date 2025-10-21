"""Test EfficientNet with MLX-format weights.

Before running this test, download and convert the PyTorch weights using:
  python scripts/convert_efficientnet_weights.py
"""

import mlx.core as mx

from mlxim.model.efficientnet.efficientnet import efficientnet_b0

# Create model
model = efficientnet_b0()

# Load MLX-format weights
weights = mx.load("weights/model_mlx.safetensors")
print(f"Loaded {len(weights)} weight tensors from MLX-format file")

# Load weights into model
try:
    model.load_weights(list(weights.items()))
    print("✓ Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    import sys

    sys.exit(1)

# Test forward pass
print("\n" + "=" * 60)
print("Testing forward pass with loaded weights...")
print("=" * 60)

x = mx.random.normal((2, 224, 224, 3))
output = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Output mean:  {output.mean():.3f}")
print(f"Output std:   {output.std():.3f}")

# Test with smaller input
print("\nTesting with different input sizes...")
for size in [224, 256, 384]:
    x_test = mx.random.normal((1, size, size, 3))
    out_test = model(x_test)
    print(f"  Input {size}x{size} -> Output shape: {out_test.shape}")

print("\n✓ All tests passed!")
