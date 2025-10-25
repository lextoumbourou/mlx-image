#!/usr/bin/env python3
"""Convert PyTorch weights to safetensors format."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_pytorch_to_safetensors(input_path: str, output_path: str | None = None) -> None:
    """Convert PyTorch weights file to safetensors format.

    Args:
        input_path: Path to the input PyTorch .pth file
        output_path: Path to the output .safetensors file (optional)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load PyTorch weights
    print(f"Loading PyTorch weights from: {input_path}")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    # Handle case where the .pth file contains a nested state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Determine output path
    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)

    # Save as safetensors
    print(f"Saving safetensors to: {output_path}")
    save_file(state_dict, str(output_path))

    print(f"✓ Conversion complete!")
    print(f"  Input:  {input_path} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Output: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with automatic output naming
  python scripts/convert_to_safetensors.py weights/efficientnet_b0_torch.pth

  # Convert with custom output path
  python scripts/convert_to_safetensors.py weights/efficientnet_b0_torch.pth -o weights/model.safetensors
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input PyTorch .pth file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output .safetensors file (default: replaces .pth with .safetensors)",
    )

    args = parser.parse_args()
    convert_pytorch_to_safetensors(args.input, args.output)


if __name__ == "__main__":
    main()
