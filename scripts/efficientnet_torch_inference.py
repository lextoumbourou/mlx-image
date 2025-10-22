#!/usr/bin/env python3
"""
Script to download EfficientNetB0 weights using torchvision and classify an image.
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def main():
    parser = argparse.ArgumentParser(description="Classify an image using EfficientNetB0")
    parser.add_argument(
        "--image",
        type=str,
        default="../hf_datasets/mlx-vision/imagenet-1k/val/n02123045/ILSVRC2012_val_00040745.JPEG",
        help="Path to the image to classify"
    )
    args = parser.parse_args()

    # Set paths
    weights_dir = Path("./weights")
    weights_dir.mkdir(exist_ok=True)

    image_path = Path(args.image)

    print("Loading EfficientNetB0 model with pretrained weights...")
    # Load pretrained model
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.eval()

    # Save weights to the weights directory
    weights_path = weights_dir / "efficientnet_b0_torch.pth"
    print(f"Saving weights to {weights_path}...")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved successfully!")

    # Get preprocessing transforms
    preprocess = weights.transforms()

    # Load and preprocess the image
    print(f"\nLoading image from {image_path}...")
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Get ImageNet class labels
    categories = weights.meta["categories"]

    print("\nTop 5 predictions:")
    print("-" * 60)
    for i in range(top5_prob.size(0)):
        category = categories[top5_catid[i]]
        probability = top5_prob[i].item()
        print(f"{i+1}. {category:30s} {probability*100:6.2f}%")

    # Save predictions
    predictions_path = weights_dir / "efficientnet_b0_predictions.json"
    predictions = {
        "model": "efficientnet_b0",
        "image_path": str(image_path),
        "top5": [
            {
                "rank": i+1,
                "category": categories[top5_catid[i]],
                "probability": float(top5_prob[i].item())
            }
            for i in range(5)
        ]
    }

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
