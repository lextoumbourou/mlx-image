"""Convert PyTorch EfficientNet weights to MLX format.

This script:
1. Loads PyTorch EfficientNet weights from a safetensors file
2. Renames conv_reduce/conv_expand to fc1/fc2 (MLX naming convention)
3. Transposes conv weights from PyTorch format (O,I,H,W) to MLX format (O,H,W,I)
4. Filters out PyTorch-specific parameters (num_batches_tracked)
5. Saves the converted weights to a new safetensors file
"""

import argparse
from pathlib import Path

import mlx.core as mx


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
    renamed_count = 0
    transposed_count = 0

    for key, value in weights.items():
        # Skip PyTorch-specific parameters
        if key.endswith("num_batches_tracked"):
            skipped_count += 1
            continue

        # Rename SE layer parameters from PyTorch to MLX convention
        if ".se.conv_reduce." in key:
            key = key.replace(".se.conv_reduce.", ".se.fc1.")
            renamed_count += 1
        elif ".se.conv_expand." in key:
            key = key.replace(".se.conv_expand.", ".se.fc2.")
            renamed_count += 1

        # Transpose 4D conv weights from PyTorch (O,I,H,W) to MLX (O,H,W,I)
        if "weight" in key and len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
            transposed_count += 1

        converted_weights[key] = value

    print("\nConversion summary:")
    print(f"  - Skipped {skipped_count} num_batches_tracked parameters")
    print(f"  - Renamed {renamed_count} SE layer parameters")
    print(f"  - Transposed {transposed_count} conv weight tensors")
    print(f"  - Total output parameters: {len(converted_weights)}")

    print(f"\nSaving converted weights to: {output_path}")
    mx.save_safetensors(output_path, converted_weights)
    print("✓ Conversion complete!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PyTorch EfficientNet weights to MLX format")
    parser.add_argument("input_path", type=str, help="Path to input safetensors file (PyTorch format)")
    parser.add_argument("output_path", type=str, help="Path to output safetensors file (MLX format)")

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input_path).exists():
        print(f"Error: Input file not found: {args.input_path}")
        return 1

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    if output_dir != Path(".") and not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    convert_weights(args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    exit(main())
