from .._config import HFWeights, Metrics, ModelConfig, Transform
from .efficientnet import efficientnet_b0, efficientnet_b1

efficientnet_configs = {
    "efficientnet_b0": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=None, accuracy_at_5=None),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="lexandstuff/efficientnet_b0-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b1": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=None, accuracy_at_5=None),
        transform=Transform(img_size=240),
        weights=HFWeights(repo_id="lexandstuff/efficientnet_b1-mlxim", filename="model.safetensors"),
    ),
}
