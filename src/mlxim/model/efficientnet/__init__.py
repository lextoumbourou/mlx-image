from ._factory import efficientnet_b0, efficientnet_b1, efficientnet_configs

__all__ = [
    "EFFICIENTNET_ENTRYPOINT",
    "EFFICIENTNET_CONFIG",
]

EFFICIENTNET_ENTRYPOINT = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
}

EFFICIENTNET_CONFIG = {
    "efficientnet_b0": efficientnet_configs["efficientnet_b0"],
    "efficientnet_b1": efficientnet_configs["efficientnet_b1"],
}
