from ._blocks import MBConv, MBConvConfig
from ._factory import efficientnet_b0, efficientnet_configs
from .efficientnet import EfficientNet

__all__ = [
    "EfficientNet",
    "MBConv",
    "MBConvConfig",
    "efficientnet_b0",
    "EFFICIENTNET_ENTRYPOINT",
    "EFFICIENTNET_CONFIG",
]

EFFICIENTNET_ENTRYPOINT = {
    "efficientnet_b0": efficientnet_b0,
}

EFFICIENTNET_CONFIG = {
    "efficientnet_b0": efficientnet_configs["efficientnet_b0"],
}
