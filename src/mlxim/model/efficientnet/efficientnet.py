"""EfficientNet model implementation in MLX."""
import copy
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..layers import AdaptiveAvgPool2d, Conv2dNormActivation
from ._blocks import MBConv, MBConvConfig


class EfficientNet(nn.Module):
    """EfficientNet model.

    Args:
        inverted_residual_setting (List[MBConvConfig]): Network structure configuration
        dropout (float): Dropout probability
        stochastic_depth_prob (float): Stochastic depth probability
        num_classes (int): Number of output classes
        norm_layer (Optional[type]): Normalization layer to use
        last_channel (Optional[int]): Number of channels in the final conv layer
    """

    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[type] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")

        if norm_layer is None:
            norm_layer = nn.BatchNorm

        # Building stem (first layer)
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.stem = Conv2dNormActivation(
            3,
            firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        # Building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0

        self.blocks = []
        for cnf in inverted_residual_setting:
            stage = []
            for _ in range(cnf.num_layers):
                # Copy to avoid modifications
                block_cnf = copy.copy(cnf)

                # Overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(MBConv(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            self.blocks.append(stage)

        # Building head (last several layers)
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        self.head = Conv2dNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        self.avgpool = AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm):
                nn.init.constant(1, m.weight)

    def get_features(self, x: mx.array) -> mx.array:
        """Extract features before the classifier.

        Args:
            x (mx.array): Input array of shape (batch, height, width, channels)

        Returns:
            mx.array: Feature array
        """
        # Pass through stem
        x = self.stem(x)

        # Pass through all MBConv blocks
        for stage in self.blocks:
            for block in stage:
                x = block(x)

        # Pass through head
        x = self.head(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))  # flatten operation

        return x

    def __call__(self, x: mx.array) -> mx.array:
        """EfficientNet forward pass.

        Args:
            x (mx.array): Input array of shape (batch, height, width, channels)

        Returns:
            mx.array: Output logits of shape (batch, num_classes)
        """
        x = self.get_features(x)
        x = self.classifier(x)
        return x
