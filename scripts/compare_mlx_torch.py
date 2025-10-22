#!/usr/bin/env python3
"""
Compare MLX and Torch EfficientNet outputs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from torchvision.models import efficientnet_b0 as torch_efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

from mlxim.model.efficientnet import efficientnet_b0 as mlx_efficientnet_b0


def preprocess_image_torch(image_path: Path) -> torch.Tensor:
    """Preprocess image for torch model."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)


def preprocess_image_mlx(image_path: Path, image_size: int = 224) -> mx.array:
    """Preprocess image for MLX model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)

    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)

    return mx.array(img_array)


def load_imagenet_labels() -> list:
    """Load ImageNet class labels."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    return weights.meta["categories"]


def compare_outputs(image_path: str):
    """Compare outputs from MLX and Torch models."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    print("=" * 80)
    print("EfficientNet-B0: MLX vs PyTorch Comparison")
    print("=" * 80)
    print(f"\nImage: {image_path}")
    print(f"Image class folder: {image_path.parent.name}")

    # Load labels
    categories = load_imagenet_labels()

    # ========== PyTorch Model ==========
    print("\n" + "-" * 80)
    print("PyTorch Model (with pretrained weights)")
    print("-" * 80)

    torch_model = torch_efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    torch_model.eval()

    torch_image = preprocess_image_torch(image_path)
    print(f"Input shape (PyTorch): {torch_image.shape} [N, C, H, W]")

    with torch.no_grad():
        torch_output = torch_model(torch_image)
        torch_probs = torch.nn.functional.softmax(torch_output[0], dim=0)

    torch_top5_prob, torch_top5_idx = torch.topk(torch_probs, 5)

    print("\nTop 5 predictions (PyTorch):")
    for i in range(5):
        idx = int(torch_top5_idx[i])
        prob = float(torch_top5_prob[i])
        print(f"  {i+1}. {categories[idx]:30s} {prob*100:6.2f}%")

    # ========== MLX Model ==========
    print("\n" + "-" * 80)
    print("MLX Model (with loaded PyTorch weights)")
    print("-" * 80)

    mlx_model = mlx_efficientnet_b0(num_classes=1000, dropout=0.2)

    # Load weights
    from mlxim.model.efficientnet import load_efficientnet_weights

    weights_path = Path("weights/efficientnet_b0_torch.safetensors")
    if weights_path.exists():
        print("Loading weights...")
        load_efficientnet_weights(mlx_model, weights_path)
        print("Weights loaded!")
    else:
        print(f"Warning: Weights not found at {weights_path}")

    # Set to eval mode
    mlx_model.eval()
    print("Model set to eval mode")

    mlx_image = preprocess_image_mlx(image_path)
    print(f"Input shape (MLX):     {mlx_image.shape} [N, H, W, C]")

    mlx_output = mlx_model(mlx_image)
    mlx_probs = mx.softmax(mlx_output[0], axis=-1)

    mlx_top5_indices = mx.argpartition(-mlx_probs, kth=5)[:5]
    mlx_top5_indices = mlx_top5_indices[mx.argsort(-mlx_probs[mlx_top5_indices])]
    mlx_top5_prob = mlx_probs[mlx_top5_indices]

    print("\nTop 5 predictions (MLX):")
    for i in range(5):
        idx = int(mlx_top5_indices[i])
        prob = float(mlx_top5_prob[i])
        print(f"  {i+1}. {categories[idx]:30s} {prob*100:6.2f}%")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nPyTorch prediction: {categories[int(torch_top5_idx[0])]} ({float(torch_top5_prob[0])*100:.2f}%)")
    print(f"MLX prediction:     {categories[int(mlx_top5_indices[0])]} ({float(mlx_top5_prob[0])*100:.2f}%)")

    print("\n" + "-" * 80)
    print("Notes:")
    print("-" * 80)
    print("• Both models use pretrained ImageNet weights")
    print("• Input shapes differ: PyTorch [N,C,H,W] vs MLX [N,H,W,C]")
    if abs(float(torch_top5_prob[0]) - float(mlx_top5_prob[0])) < 0.01:
        print("• ✓ Predictions MATCH! Models are equivalent.")
    else:
        print("• ✗ Predictions differ. Possible issues:")
        print("  - BatchNorm running statistics may need eval mode")
        print("  - Dropout should be disabled during inference")
        print("  - Preprocessing differences")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare MLX and Torch EfficientNet outputs")
    parser.add_argument(
        "--image",
        type=str,
        default="tests/data/n01667778/ILSVRC2012_val_00002832.JPEG",
        help="Path to the image to classify",
    )
    args = parser.parse_args()

    compare_outputs(args.image)


if __name__ == "__main__":
    main()
