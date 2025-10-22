#!/usr/bin/env python3
"""Test MLX Linear weight format."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Create a Linear layer
linear = nn.Linear(1280, 1000, bias=True)  # MLX format: (input_dims, output_dims)

print("=" * 80)
print("MLX Linear Weight Format Analysis")
print("=" * 80)

print(f"\nLinear(input_dims=1280, output_dims=1000)")
print(f"  Initial weight shape: {linear.weight.shape}")
print(f"  Initial bias shape: {linear.bias.shape}")

# Test forward pass
x = mx.random.normal((1, 1280))
print(f"\nInput shape: {x.shape}")

try:
    out = linear(x)
    print(f"✓ Output shape: {out.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Now try PyTorch format weight [out_features, in_features]
print("\n" + "-" * 80)
print("Testing PyTorch format weight [out_features, in_features]")
print("-" * 80)

pytorch_weight = np.random.randn(1000, 1280).astype(np.float32)
print(f"PyTorch weight shape: {pytorch_weight.shape} [out=1000, in=1280]")

# Try setting it directly (PyTorch format)
linear.weight = mx.array(pytorch_weight)
print(f"After setting: weight shape = {linear.weight.shape}")

try:
    out = linear(x)
    print(f"✓ Output shape: {out.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Try transposed format [in_features, out_features]
print("\n" + "-" * 80)
print("Testing transposed format [in_features, out_features]")
print("-" * 80)

transposed_weight = np.transpose(pytorch_weight, (1, 0))
print(f"Transposed weight shape: {transposed_weight.shape} [in=1280, out=1000]")

linear.weight = mx.array(transposed_weight)
print(f"After setting: weight shape = {linear.weight.shape}")

try:
    out = linear(x)
    print(f"✓ Output shape: {out.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("Need to test which format MLX Linear expects...")
