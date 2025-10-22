#!/usr/bin/env python3
"""Test eval mode in MLX model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlxim.model.efficientnet import efficientnet_b0, load_efficientnet_weights


def main():
    # Create and load model
    model = efficientnet_b0(num_classes=1000, dropout=0.2)
    weights_path = Path("weights/efficientnet_b0_torch.safetensors")
    load_efficientnet_weights(model, weights_path)

    print("=" * 80)
    print("Testing Training vs Eval Mode")
    print("=" * 80)

    # Test input
    x = mx.random.normal((1, 224, 224, 3))

    # Training mode
    print("\n1. Training mode (default):")
    print(f"   model.training = {model.training}")
    out_train = model(x)
    probs_train = mx.softmax(out_train[0], axis=-1)
    top1_train = mx.argmax(probs_train)
    print(f"   Top prediction: class {int(top1_train)}, prob {float(probs_train[top1_train])*100:.2f}%")
    print(f"   Output std: {float(mx.std(out_train)):.4f}")

    # Eval mode
    print("\n2. Eval mode:")
    model.eval()
    print(f"   model.training = {model.training}")
    out_eval = model(x)
    probs_eval = mx.softmax(out_eval[0], axis=-1)
    top1_eval = mx.argmax(probs_eval)
    print(f"   Top prediction: class {int(top1_eval)}, prob {float(probs_eval[top1_eval])*100:.2f}%")
    print(f"   Output std: {float(mx.std(out_eval)):.4f}")

    # Check if outputs differ
    print("\n" + "-" * 80)
    diff = mx.abs(out_train - out_eval).max()
    print(f"Max difference between train and eval: {float(diff):.6f}")

    if float(diff) > 0.001:
        print("✓ Outputs differ significantly - eval mode is working!")
    else:
        print("✗ Outputs are the same - eval mode may not be working")

    print("=" * 80)


if __name__ == "__main__":
    main()
