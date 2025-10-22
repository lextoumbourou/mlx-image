#!/usr/bin/env python3
"""Inspect the safetensors weights structure."""
from pathlib import Path
from safetensors import safe_open

weights_path = Path("weights/efficientnet_b0_torch.safetensors")

print("=" * 80)
print("PyTorch EfficientNet-B0 Weights Structure")
print("=" * 80)

with safe_open(weights_path, framework="pt") as f:
    keys = f.keys()
    print(f"\nTotal number of tensors: {len(keys)}")

    # Group by layer type
    features_keys = [k for k in keys if k.startswith("features.")]
    classifier_keys = [k for k in keys if k.startswith("classifier.")]

    print(f"Features layers: {len(features_keys)}")
    print(f"Classifier layers: {len(classifier_keys)}")

    print("\n" + "-" * 80)
    print("First 20 layer names and shapes:")
    print("-" * 80)

    for i, key in enumerate(list(keys)[:20]):
        tensor = f.get_tensor(key)
        print(f"{i+1:3d}. {key:60s} {list(tensor.shape)}")

    print("\n" + "-" * 80)
    print("Last 10 layer names and shapes:")
    print("-" * 80)

    for i, key in enumerate(list(keys)[-10:], start=len(keys)-9):
        tensor = f.get_tensor(key)
        print(f"{i:3d}. {key:60s} {list(tensor.shape)}")

    # Analyze structure
    print("\n" + "=" * 80)
    print("Structure Analysis")
    print("=" * 80)

    # Find unique prefixes
    prefixes = set()
    for key in keys:
        parts = key.split(".")
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")

    print(f"\nUnique layer prefixes ({len(prefixes)}):")
    for prefix in sorted(prefixes):
        matching = [k for k in keys if k.startswith(prefix + ".")]
        print(f"  {prefix:40s} ({len(matching)} tensors)")
