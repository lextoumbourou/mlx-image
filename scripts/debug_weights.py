"""Debug script to inspect PyTorch weights structure."""
from safetensors import safe_open

weights_path = "/Users/lex/code/mlx-image/weights/efficientnet_b0_torch.safetensors"

print("Inspecting weight keys...")
with safe_open(weights_path, framework="numpy") as f:
    keys = list(f.keys())

# Group by stage
stage_keys = {}
for key in keys:
    if key.startswith("features."):
        parts = key.split(".")
        if len(parts) >= 2:
            stage_num = parts[1]
            if stage_num not in stage_keys:
                stage_keys[stage_num] = []
            stage_keys[stage_num].append(key)

print("\nKeys by stage:")
for stage, keys_list in sorted(stage_keys.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
    print(f"\n=== Stage {stage} ===")
    for key in sorted(keys_list):
        print(f"  {key}")

print("\nClassifier keys:")
for key in sorted([k for k in keys if k.startswith("classifier.")]):
    print(f"  {key}")
