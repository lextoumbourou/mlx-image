#!/usr/bin/env python3
"""
Script to convert PyTorch weights (.pth) to safetensors format.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to safetensors format")
    parser.add_argument(
        "--input",
        type=str,
        default="weights/efficientnet_b0_torch.pth",
        help="Path to the input PyTorch weights file (.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output safetensors file (default: same name with .safetensors extension)"
    )
    args = parser.parse_args()

    # Set paths
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    if args.output is None:
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = Path(args.output)

    print(f"Loading PyTorch weights from {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu")

    print(f"State dict contains {len(state_dict)} tensors")

    # Print some statistics
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")

    # Calculate size
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"Total size: {total_size / (1024**2):.2f} MB")

    print(f"\nSaving to safetensors format at {output_path}...")
    save_file(state_dict, output_path)

    # Verify the saved file
    output_size = output_path.stat().st_size
    print(f"Saved successfully! Output file size: {output_size / (1024**2):.2f} MB")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
