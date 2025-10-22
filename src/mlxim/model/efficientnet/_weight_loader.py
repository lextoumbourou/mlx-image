"""Weight loader for EfficientNet from PyTorch safetensors."""
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def convert_conv2d_weight(weight: np.ndarray) -> mx.array:
    """Convert PyTorch Conv2d weight [C_out, C_in, H, W] to MLX [C_out, H, W, C_in].

    Args:
        weight: PyTorch weight array

    Returns:
        MLX weight array with transposed dimensions
    """
    # PyTorch: [C_out, C_in, H, W] -> MLX: [C_out, H, W, C_in]
    return mx.array(np.transpose(weight, (0, 2, 3, 1)))


def convert_linear_weight(weight: np.ndarray) -> mx.array:
    """Convert PyTorch Linear weight to MLX format.

    Args:
        weight: PyTorch weight array [C_out, C_in]

    Returns:
        MLX weight array (same format as PyTorch)
    """
    # PyTorch: [C_out, C_in] -> MLX: [C_out, C_in] (same format!)
    return mx.array(weight)


def load_efficientnet_weights(model, weights_path: Path) -> Dict[str, int]:
    """Load PyTorch EfficientNet weights into MLX model.

    Args:
        model: MLX EfficientNet model
        weights_path: Path to safetensors file

    Returns:
        Dictionary with loading statistics
    """
    stats = {"loaded": 0, "skipped": 0, "missing": 0, "shape_mismatches": 0}

    print(f"Loading weights from {weights_path}...")

    with safe_open(weights_path, framework="np") as f:
        torch_keys = list(f.keys())

        # Load stem (features.0)
        print("\nLoading stem...")
        stem_conv_weight = f.get_tensor("features.0.0.weight")
        model.stem.layers[0].weight = convert_conv2d_weight(stem_conv_weight)
        stats["loaded"] += 1

        # BatchNorm for stem
        model.stem.layers[1].weight = mx.array(f.get_tensor("features.0.1.weight"))
        model.stem.layers[1].bias = mx.array(f.get_tensor("features.0.1.bias"))
        model.stem.layers[1].running_mean = mx.array(f.get_tensor("features.0.1.running_mean"))
        model.stem.layers[1].running_var = mx.array(f.get_tensor("features.0.1.running_var"))
        stats["loaded"] += 4

        # Load MBConv blocks (features.1 through features.7)
        print("Loading MBConv blocks...")
        for stage_idx, stage in enumerate(model.blocks, start=1):
            print(f"  Stage {stage_idx} ({len(stage)} blocks)")
            for block_idx, block in enumerate(stage):
                prefix = f"features.{stage_idx}.{block_idx}.block"

                # Check what layers exist in this block
                block_keys = [k for k in torch_keys if k.startswith(prefix)]

                # Determine block structure based on keys
                layer_idx = 0

                # Check if there's an expansion layer (block.0 with 1x1 conv before depthwise)
                has_expansion = any(
                    "block.0.0.weight" in k and f.get_tensor(k).shape[2:] == (1, 1)
                    for k in block_keys
                    if k.startswith(prefix + ".0")
                )

                # Expansion layer (optional)
                if has_expansion:
                    exp_conv = f.get_tensor(f"{prefix}.0.0.weight")
                    block.block[layer_idx].layers[0].weight = convert_conv2d_weight(exp_conv)
                    block.block[layer_idx].layers[1].weight = mx.array(
                        f.get_tensor(f"{prefix}.0.1.weight")
                    )
                    block.block[layer_idx].layers[1].bias = mx.array(f.get_tensor(f"{prefix}.0.1.bias"))
                    block.block[layer_idx].layers[1].running_mean = mx.array(
                        f.get_tensor(f"{prefix}.0.1.running_mean")
                    )
                    block.block[layer_idx].layers[1].running_var = mx.array(
                        f.get_tensor(f"{prefix}.0.1.running_var")
                    )
                    stats["loaded"] += 5
                    layer_idx += 1

                # Depthwise conv
                dw_prefix = f"{prefix}.{0 if not has_expansion else 1}"
                dw_conv = f.get_tensor(f"{dw_prefix}.0.weight")
                block.block[layer_idx].layers[0].weight = convert_conv2d_weight(dw_conv)
                block.block[layer_idx].layers[1].weight = mx.array(f.get_tensor(f"{dw_prefix}.1.weight"))
                block.block[layer_idx].layers[1].bias = mx.array(f.get_tensor(f"{dw_prefix}.1.bias"))
                block.block[layer_idx].layers[1].running_mean = mx.array(
                    f.get_tensor(f"{dw_prefix}.1.running_mean")
                )
                block.block[layer_idx].layers[1].running_var = mx.array(
                    f.get_tensor(f"{dw_prefix}.1.running_var")
                )
                stats["loaded"] += 5
                layer_idx += 1

                # Squeeze-and-Excitation
                se_prefix = f"{prefix}.{1 if not has_expansion else 2}"
                se_fc1_weight = f.get_tensor(f"{se_prefix}.fc1.weight")
                se_fc2_weight = f.get_tensor(f"{se_prefix}.fc2.weight")

                # SE uses Conv2d in PyTorch [C_out, C_in, 1, 1] but we need [1, 1, C_in, C_out]
                block.block[layer_idx].fc1.weight = convert_conv2d_weight(se_fc1_weight)
                block.block[layer_idx].fc1.bias = mx.array(f.get_tensor(f"{se_prefix}.fc1.bias"))
                block.block[layer_idx].fc2.weight = convert_conv2d_weight(se_fc2_weight)
                block.block[layer_idx].fc2.bias = mx.array(f.get_tensor(f"{se_prefix}.fc2.bias"))
                stats["loaded"] += 4
                layer_idx += 1

                # Projection layer
                proj_prefix = f"{prefix}.{2 if not has_expansion else 3}"
                proj_conv = f.get_tensor(f"{proj_prefix}.0.weight")
                block.block[layer_idx].layers[0].weight = convert_conv2d_weight(proj_conv)
                block.block[layer_idx].layers[1].weight = mx.array(f.get_tensor(f"{proj_prefix}.1.weight"))
                block.block[layer_idx].layers[1].bias = mx.array(f.get_tensor(f"{proj_prefix}.1.bias"))
                block.block[layer_idx].layers[1].running_mean = mx.array(
                    f.get_tensor(f"{proj_prefix}.1.running_mean")
                )
                block.block[layer_idx].layers[1].running_var = mx.array(
                    f.get_tensor(f"{proj_prefix}.1.running_var")
                )
                stats["loaded"] += 5

        # Load head (features.8)
        print("Loading head...")
        head_conv_weight = f.get_tensor("features.8.0.weight")
        model.head.layers[0].weight = convert_conv2d_weight(head_conv_weight)
        model.head.layers[1].weight = mx.array(f.get_tensor("features.8.1.weight"))
        model.head.layers[1].bias = mx.array(f.get_tensor("features.8.1.bias"))
        model.head.layers[1].running_mean = mx.array(f.get_tensor("features.8.1.running_mean"))
        model.head.layers[1].running_var = mx.array(f.get_tensor("features.8.1.running_var"))
        stats["loaded"] += 5

        # Load classifier
        print("Loading classifier...")
        classifier_weight = f.get_tensor("classifier.1.weight")
        classifier_bias = f.get_tensor("classifier.1.bias")
        model.classifier.layers[1].weight = convert_linear_weight(classifier_weight)
        model.classifier.layers[1].bias = mx.array(classifier_bias)
        stats["loaded"] += 2

    print(f"\n✓ Loaded {stats['loaded']} weight tensors")
    return stats
