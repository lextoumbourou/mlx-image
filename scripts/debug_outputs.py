"""Debug script to compare layer outputs between PyTorch and MLX."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlxim.model.efficientnet._factory import efficientnet_b0 as mlx_efficientnet_b0
import torchvision.models as models


def preprocess_image_mlx(image_path: str, size: int = 224):
    """Preprocess image for MLX EfficientNet."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, 0)
    return mx.array(img_array)


def preprocess_image_torch(image_path: str, size: int = 224):
    """Preprocess image for PyTorch EfficientNet."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor


def main():
    weights_path = "/Users/lex/code/mlx-image/weights/efficientnet_b0_torch.safetensors"
    image_path = "/Users/lex/code/hf_datasets/mlx-vision/imagenet-1k/val/n02123045/ILSVRC2012_val_00033490.JPEG"

    print("=== Creating models ===")
    mlx_model = mlx_efficientnet_b0()
    torch_model = models.efficientnet_b0()

    print("\n=== Loading weights ===")
    from safetensors.torch import load_file
    mlx_model.load_pytorch_weights(weights_path)
    torch_model.load_state_dict(load_file(weights_path))

    mlx_model.eval()
    torch_model.eval()

    print("\n=== Loading image ===")
    mlx_img = preprocess_image_mlx(image_path)
    torch_img = preprocess_image_torch(image_path)

    print(f"MLX image shape: {mlx_img.shape}, dtype: {mlx_img.dtype}")
    print(f"PyTorch image shape: {torch_img.shape}, dtype: {torch_img.dtype}")

    # Check if preprocessing is the same
    mlx_img_np = np.array(mlx_img[0])  # H, W, C
    torch_img_np = torch_img[0].permute(1, 2, 0).numpy()  # H, W, C
    print(f"\nPreprocessing difference (max abs): {np.max(np.abs(mlx_img_np - torch_img_np))}")

    print("\n=== Running inference ===")
    mx.eval(mlx_model.parameters())

    # MLX
    mlx_logits = mlx_model(mlx_img)
    mx.eval(mlx_logits)
    mlx_probs = mx.softmax(mlx_logits, axis=-1)
    mlx_top5 = mx.argsort(mlx_probs[0])[-5:][::-1]

    # PyTorch
    with torch.no_grad():
        torch_logits = torch_model(torch_img)
    torch_probs = torch.nn.functional.softmax(torch_logits, dim=-1)
    torch_top5 = torch.topk(torch_probs[0], 5).indices

    print("\n=== Results ===")
    print(f"MLX top-5: {mlx_top5.tolist()}")
    print(f"PyTorch top-5: {torch_top5.tolist()}")

    print(f"\nMLX logits stats: min={float(mx.min(mlx_logits)):.4f}, max={float(mx.max(mlx_logits)):.4f}, mean={float(mx.mean(mlx_logits)):.4f}")
    print(f"PyTorch logits stats: min={torch_logits.min():.4f}, max={torch_logits.max():.4f}, mean={torch_logits.mean():.4f}")

    # Compare stem output
    print("\n=== Checking stem output ===")
    mlx_stem_out = mlx_model.features[0](mlx_img)
    mx.eval(mlx_stem_out)

    with torch.no_grad():
        torch_stem_out = torch_model.features[0](torch_img)

    print(f"MLX stem output shape: {mlx_stem_out.shape}")
    print(f"PyTorch stem output shape: {torch_stem_out.shape}")
    print(f"MLX stem stats: min={float(mx.min(mlx_stem_out)):.4f}, max={float(mx.max(mlx_stem_out)):.4f}, mean={float(mx.mean(mlx_stem_out)):.4f}")
    print(f"PyTorch stem stats: min={torch_stem_out.min():.4f}, max={torch_stem_out.max():.4f}, mean={torch_stem_out.mean():.4f}")

    # Convert to comparable format
    mlx_stem_np = np.array(mlx_stem_out[0])  # H, W, C
    torch_stem_np = torch_stem_out[0].permute(1, 2, 0).cpu().numpy()  # H, W, C
    print(f"Stem output difference (max abs): {np.max(np.abs(mlx_stem_np - torch_stem_np)):.6f}")


if __name__ == "__main__":
    main()
