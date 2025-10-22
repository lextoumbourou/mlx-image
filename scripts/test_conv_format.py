#!/usr/bin/env python3
"""Test MLX Conv2d weight format."""
import mlx.core as mx
import mlx.nn as nn

# Create Conv2d layers with different shapes
configs = [
    (3, 32, 3),  # in_channels=3, out_channels=32, kernel_size=3
    (64, 128, 5),  # in_channels=64, out_channels=128, kernel_size=5
    (1, 10, 1),  # in_channels=1, out_channels=10, kernel_size=1
]

print("=" * 80)
print("MLX Conv2d Weight Format Analysis")
print("=" * 80)

for in_c, out_c, k in configs:
    conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, bias=False)
    weight_shape = conv.weight.shape

    print(f"\nin_channels={in_c}, out_channels={out_c}, kernel_size={k}")
    print(f"  Weight shape: {weight_shape}")
    print(f"  Interpretation: ", end="")

    if weight_shape == (out_c, k, k, in_c):
        print(f"[out_channels={out_c}, kernel_h={k}, kernel_w={k}, in_channels={in_c}]")
        print(f"  Format: [C_out, H, W, C_in]")
    elif weight_shape == (out_c, in_c, k, k):
        print(f"[out_channels={out_c}, in_channels={in_c}, kernel_h={k}, kernel_w={k}]")
        print(f"  Format: [C_out, C_in, H, W] (PyTorch format)")
    else:
        print(f"Unknown format!")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("MLX Conv2d weight format is: [C_out, kernel_h, kernel_w, C_in]")
print("PyTorch Conv2d weight format is: [C_out, C_in, kernel_h, kernel_w]")
print("\nSo we need to transpose from [C_out, C_in, H, W] → [C_out, H, W, C_in]")
