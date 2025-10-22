#!/usr/bin/env python3
"""Test MLX Conv2d weight shape expectations."""
import mlx.core as mx
import mlx.nn as nn

# Create a fresh Conv2d layer
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

print("Fresh Conv2d layer:")
print(f"  Initial weight shape: {conv.weight.shape}")

# Test with input
x = mx.random.normal((1, 224, 224, 3))
print(f"\nInput shape: {x.shape}")

try:
    out = conv(x)
    print(f"✓ Output shape: {out.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Now try to set the weight manually
import numpy as np
new_weight = np.random.randn(3, 3, 3, 32).astype(np.float32)
print(f"\nSetting weight manually with shape: {new_weight.shape}")
conv.weight = mx.array(new_weight)
print(f"Weight shape after setting: {conv.weight.shape}")

try:
    out = conv(x)
    print(f"✓ Output shape after setting weight: {out.shape}")
except Exception as e:
    print(f"✗ Error after setting weight: {e}")
