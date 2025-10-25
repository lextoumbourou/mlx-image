"""Test EfficientNet MLX implementation with PyTorch weights."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlxim.model.efficientnet._factory import efficientnet_b0
from mlxim.model._utils import load_weights


def preprocess_image(image_path: str, size: int = 224):
    """Preprocess image for EfficientNet.

    Args:
        image_path: Path to image file
        size: Target size for resizing

    Returns:
        Preprocessed image as MLX array (1, H, W, C)
    """
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)

    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)

    # Convert to MLX array (already in NHWC format)
    return mx.array(img_array)


def load_imagenet_classes():
    """Load ImageNet class labels."""
    # Simple version - just return index
    # You can load full labels from a file if needed
    return None


def main():
    # Paths
    weights_path = "/Users/lex/code/mlx-image/weights/efficientnet_b0_mlx.safetensors"
    image_path = "/Users/lex/code/hf_datasets/mlx-vision/imagenet-1k/val/n02123045/ILSVRC2012_val_00033490.JPEG"

    print("Creating EfficientNet-B0 model...")
    model = efficientnet_b0()

    print(f"Loading MLX weights from {weights_path}...")
    model = load_weights(model, weights_path, strict=True, verbose=True)

    print(f"Loading and preprocessing image from {image_path}...")
    img = preprocess_image(image_path)

    print(f"Image shape: {img.shape}")
    print("Running inference...")

    # Set model to eval mode (disable dropout, use running stats for batchnorm)
    model.eval()

    # Run inference
    mx.eval(model.parameters())
    logits = model(img)
    mx.eval(logits)

    print(f"Logits shape: {logits.shape}")

    # Get top-5 predictions
    probs = mx.softmax(logits, axis=-1)
    top5_indices = mx.argsort(probs[0])[-5:][::-1]
    top5_probs = probs[0][top5_indices]

    print("\nTop-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist())):
        print(f"{i+1}. Class {idx}: {prob:.4f}")

    print(f"\nTop prediction: Class {top5_indices[0].item()}")
    print("(Expected: Should be a cat class - typically around class 281-285)")


if __name__ == "__main__":
    main()
