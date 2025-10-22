#!/usr/bin/env python3
"""
Script to classify an image using EfficientNetB0 in MLX.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
import numpy as np
from PIL import Image
from mlxim.model.efficientnet import efficientnet_b0


def preprocess_image(image_path: Path, image_size: int = 224) -> mx.array:
    """Preprocess image for EfficientNet.

    Args:
        image_path: Path to image file
        image_size: Target image size

    Returns:
        Preprocessed image array
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize with bilinear interpolation
    # Use BICUBIC for better quality (similar to torchvision default)
    image = image.resize((image_size, image_size), Image.BICUBIC)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Convert to MLX array
    return mx.array(img_array)


def load_imagenet_labels() -> list:
    """Load ImageNet class labels.

    Returns:
        List of ImageNet class names
    """
    # Using the same categories as torchvision
    # This is a simplified version - in production you'd load from a file
    from torchvision.models import EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    return weights.meta["categories"]


def main():
    parser = argparse.ArgumentParser(description="Classify an image using EfficientNetB0 (MLX)")
    parser.add_argument(
        "--image",
        type=str,
        default="tests/data/n01667778/ILSVRC2012_val_00002832.JPEG",
        help="Path to the image to classify",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (not implemented yet)",
    )
    args = parser.parse_args()

    # Set paths
    weights_dir = Path("./weights")
    weights_dir.mkdir(exist_ok=True)

    image_path = Path(args.image)

    print("Creating EfficientNetB0 model (MLX)...")
    model = efficientnet_b0(num_classes=1000, dropout=0.2)
    print("Model created successfully!")

    # Load pretrained weights
    if args.weights:
        weights_file = Path(args.weights)
    else:
        weights_file = weights_dir / "efficientnet_b0_torch.safetensors"

    if weights_file.exists():
        print(f"\nLoading pretrained weights from {weights_file}...")
        from mlxim.model.efficientnet import load_efficientnet_weights

        load_efficientnet_weights(model, weights_file)
        print("Weights loaded successfully!")
    else:
        print(f"\nWarning: No weights found at {weights_file}")
        print("Using random initialization (predictions will be meaningless)")

    # Set model to eval mode
    model.eval()
    print("Model set to eval mode")

    # Load and preprocess the image
    print(f"\nLoading image from {image_path}...")
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    image_tensor = preprocess_image(image_path)
    print(f"Image preprocessed: {image_tensor.shape}")

    # Run inference
    print("Running inference...")
    output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = mx.softmax(output[0], axis=-1)

    # Get top 5 predictions
    top5_indices = mx.argpartition(-probabilities, kth=5)[:5]
    top5_indices = top5_indices[mx.argsort(-probabilities[top5_indices])]
    top5_prob = probabilities[top5_indices]

    # Get ImageNet class labels
    categories = load_imagenet_labels()

    print("\nTop 5 predictions (MLX):")
    print("-" * 60)
    for i in range(5):
        category = categories[int(top5_indices[i])]
        probability = float(top5_prob[i])
        print(f"{i+1}. {category:30s} {probability*100:6.2f}%")

    # Save predictions
    predictions_path = weights_dir / "efficientnet_b0_mlx_predictions.json"
    predictions = {
        "model": "efficientnet_b0_mlx",
        "image_path": str(image_path),
        "top5": [
            {
                "rank": i + 1,
                "category": categories[int(top5_indices[i])],
                "probability": float(top5_prob[i]),
            }
            for i in range(5)
        ],
    }

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
