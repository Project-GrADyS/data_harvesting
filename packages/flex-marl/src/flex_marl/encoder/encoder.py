from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import MLP
from validation_core import validate_positive_integer

from .configs import (
    FlatHeadConfig,
    HeadConfig,
    SequentialHeadConfig,
    _validate_module_class,
    validate_head_config,
)
from .heads import FlatHead, SequentialHead


class MultiHeadEncoderModule(nn.Module):
    def __init__(
        self,
        head_configs: Sequence[HeadConfig],
        mix_layer_depth: int,
        mix_layer_num_cells: int,
        mix_activation_class: type[nn.Module] | None,
        output_dim: int,
        device: DEVICE_TYPING | None = None,
    ) -> None:
        super().__init__()
        self.head_configs = tuple(head_configs)
        if not self.head_configs:
            raise ValueError("head_configs must contain at least one head configuration.")
        validate_positive_integer("output_dim", output_dim)
        validate_positive_integer("mix_layer_depth", mix_layer_depth)
        validate_positive_integer("mix_layer_num_cells", mix_layer_num_cells)
        if mix_activation_class is not None:
            _validate_module_class("mix_activation_class", mix_activation_class)

        self.output_dim = output_dim
        self.mix_layer_depth = mix_layer_depth
        self.mix_layer_num_cells = mix_layer_num_cells
        self.mix_activation_class = mix_activation_class if mix_activation_class is not None else nn.Tanh
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.heads = nn.ModuleDict()
        for config in self.head_configs:
            validate_head_config(config)
            if config.key in self.heads:
                raise ValueError(f"Duplicate head key found: {config.key}")
            if isinstance(config, SequentialHeadConfig):
                self.heads[config.key] = SequentialHead(config, device=self.device)
            else:
                self.heads[config.key] = FlatHead(config, device=self.device)

        mix_input_dim = sum(config.output_size for config in self.head_configs)
        self.mix_layer = MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device,
        )

    def _pre_forward_checks(self, input_dict: dict[str, torch.Tensor]) -> None:
        for config in self.head_configs:
            if config.key not in input_dict:
                raise KeyError(f"Input dictionary is missing required key: {config.key}")
            if isinstance(config, SequentialHeadConfig):
                if config.mask_key not in input_dict:
                    raise KeyError(
                        f"Input dictionary is missing required mask key: {config.mask_key}"
                    )
                positional = config.positional_encoding_config
                if positional is not None and positional.idx_key not in input_dict:
                    raise KeyError(
                        f"Input dictionary is missing required positional index key: {positional.idx_key}"
                    )

    def forward(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self._pre_forward_checks(input_dict)

        head_outputs: list[torch.Tensor] = []
        batch_shape: torch.Size | None = None
        for config in self.head_configs:
            head = self.heads[config.key]
            head_input = input_dict[config.key]
            if isinstance(config, SequentialHeadConfig):
                positional = config.positional_encoding_config
                idx = input_dict[positional.idx_key] if positional is not None else None
                head_output = head(head_input, input_dict[config.mask_key], idx)
            else:
                head_output = head(head_input)

            if head_output.shape[-1] != config.output_size:
                raise ValueError(
                    f"Head {config.key!r} returned size {head_output.shape[-1]}, "
                    f"expected {config.output_size}."
                )
            if batch_shape is None:
                batch_shape = head_output.shape[:-1]
            elif head_output.shape[:-1] != batch_shape:
                raise ValueError(
                    f"Head {config.key!r} has batch shape {tuple(head_output.shape[:-1])}, "
                    f"expected {tuple(batch_shape)}."
                )
            head_outputs.append(head_output)

        return self.mix_layer(torch.cat(head_outputs, dim=-1))
