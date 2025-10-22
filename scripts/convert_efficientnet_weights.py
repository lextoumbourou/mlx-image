"""Convert PyTorch EfficientNet weights to MLX format.

This script:
1. Downloads PyTorch EfficientNet weights from HuggingFace Hub (timm models)
2. Transposes conv weights from PyTorch format (O,I,H,W) to MLX format (O,H,W,I)
3. Filters out PyTorch-specific parameters (num_batches_tracked)
4. Saves the converted weights to weights/ directory

Note: SE layer weights (conv_reduce/conv_expand) are kept as-is, matching timm naming.
"""

import argparse
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download


def convert_weights(input_path: str, output_path: str) -> None:
    """Convert PyTorch EfficientNet weights to MLX format.

    Args:
        input_path: Path to input safetensors file (PyTorch format)
        output_path: Path to output safetensors file (MLX format)
    """
    print(f"Loading weights from: {input_path}")
    weights = mx.load(input_path)
    print(f"Loaded {len(weights)} weight tensors")

    converted_weights = {}
    skipped_count = 0
    transposed_count = 0

    for key, value in weights.items():
        # Skip PyTorch-specific parameters
        if key.endswith("num_batches_tracked"):
            skipped_count += 1
            continue

        # Transpose 4D conv weights from PyTorch (O,I,H,W) to MLX (O,H,W,I)
        if "weight" in key and len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
            transposed_count += 1

        converted_weights[key] = value

    print("\nConversion summary:")
    print(f"  - Skipped {skipped_count} num_batches_tracked parameters")
    print(f"  - Transposed {transposed_count} conv weight tensors")
    print(f"  - Total output parameters: {len(converted_weights)}")

    print(f"\nSaving converted weights to: {output_path}")
    mx.save_safetensors(output_path, converted_weights)
    print("✓ Conversion complete!")


def download_timm_weights(model_name: str, repo_id: str, output_dir: Path) -> str:
    """Download timm model weights from HuggingFace Hub.

    Args:
        model_name: Name of the model (e.g., "efficientnet_b0")
        repo_id: HuggingFace Hub repository ID (e.g., "timm/efficientnet_b0.ra_in1k")
        output_dir: Directory to save the downloaded weights

    Returns:
        Path to the downloaded weights file
    """
    filename = "model.safetensors"

    print(f"Downloading {model_name} from {repo_id}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
    )

    print(f"✓ Downloaded {model_name} weights to: {model_path}")
    return model_path


# Mapping of model names to their HuggingFace Hub repository IDs
MODEL_REPO_MAP = {
    "efficientnet_b0": "timm/efficientnet_b0.ra_in1k",
    "efficientnet_b1": "timm/efficientnet_b1.ft_in1k",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert PyTorch EfficientNet weights to MLX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=list(MODEL_REPO_MAP.keys()),
        help=f"Model name (choices: {', '.join(MODEL_REPO_MAP.keys())})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Output directory for weights (default: weights)",
    )

    args = parser.parse_args()

    # Get repository ID from model name
    repo_id = MODEL_REPO_MAP[args.model]

    # Create model-specific output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting {args.model}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}\n")

    # Download PyTorch weights from HuggingFace Hub
    pytorch_weights_path = download_timm_weights(args.model, repo_id, output_dir)

    # Convert to MLX format
    mlx_weights_path = output_dir / "model_mlx.safetensors"
    convert_weights(pytorch_weights_path, str(mlx_weights_path))

    print(f"\n✓ All done! MLX weights saved to: {mlx_weights_path}")
    return 0


if __name__ == "__main__":
    exit(main())
