"""Test EfficientNet PyTorch implementation for comparison."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Use installed torchvision
import torchvision.models as models


def preprocess_image(image_path: str, size: int = 224):
    """Preprocess image for EfficientNet.

    Args:
        image_path: Path to image file
        size: Target size for resizing

    Returns:
        Preprocessed image as torch tensor (1, C, H, W)
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

    # Convert to torch tensor and change to NCHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

    return img_tensor


def main():
    # Paths
    weights_path = "/Users/lex/code/mlx-image/weights/efficientnet_b0_torch.safetensors"
    image_path = "/Users/lex/code/hf_datasets/mlx-vision/imagenet-1k/val/n02123045/ILSVRC2012_val_00033490.JPEG"

    print("Creating EfficientNet-B0 model...")
    model = models.efficientnet_b0()

    print(f"Loading weights from {weights_path}...")
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    model.eval()

    print(f"Loading and preprocessing image from {image_path}...")
    img = preprocess_image(image_path)

    print(f"Image shape: {img.shape}")
    print("Running inference...")

    # Run inference
    with torch.no_grad():
        logits = model(img)

    print(f"Logits shape: {logits.shape}")

    # Get top-5 predictions
    probs = F.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs[0], 5)

    print("\nTop-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist())):
        print(f"{i+1}. Class {idx}: {prob:.4f}")

    print(f"\nTop prediction: Class {top5_indices[0].item()}")
    print("(Expected: Should be a cat class - typically around class 281-285)")


if __name__ == "__main__":
    main()
