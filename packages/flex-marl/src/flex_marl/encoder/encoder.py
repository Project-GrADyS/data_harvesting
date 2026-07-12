import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING

from .configs import (
    FlatHeadConfig,
    SequentialHeadConfig,
    validate_head_config,
)
from .heads import SequentialHead, FlatHead

class MultiHeadEncoderModule(nn.Module):
    def __init__(
        self,
        head_configs: list[SequentialHeadConfig | FlatHeadConfig],
        mix_layer_depth: int,
        mix_layer_num_cells: int,
        mix_activation_class: type[nn.Module] | None,
        output_dim: int,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        if not head_configs:
            raise ValueError("head_configs must contain at least one head configuration.")
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}.")
        if not isinstance(mix_layer_depth, int) or isinstance(mix_layer_depth, bool) or mix_layer_depth <= 0:
            raise ValueError(f"mix_layer_depth must be a positive integer, got {mix_layer_depth}.")
        if not isinstance(mix_layer_num_cells, int) or isinstance(mix_layer_num_cells, bool) or mix_layer_num_cells <= 0:
            raise ValueError(f"mix_layer_num_cells must be a positive integer, got {mix_layer_num_cells}.")
        if mix_activation_class is not None and (
            not isinstance(mix_activation_class, type) or not issubclass(mix_activation_class, nn.Module)
        ):
            raise ValueError("mix_activation_class must be an nn.Module class or None.")

        self.head_configs = head_configs
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
            
            is_sequential = isinstance(config, SequentialHeadConfig)
            if is_sequential:
                self.heads[config.key] = self._build_sequential_head(config)
                continue

            is_flat = isinstance(config, FlatHeadConfig)
            if is_flat:
                self.heads[config.key] = self._build_flat_head(config)
                continue
            
            raise ValueError(f"Head config must be either SequentialHeadConfig or FlatHeadConfig, got {type(config)}.")

        mix_input_dim = (
            sum(head_config.output_size for head_config in self.head_configs)
        )
        self.mix_layer = self._build_mix_layer(mix_input_dim)

    def _build_sequential_head(self, head_config: SequentialHeadConfig) -> SequentialHead:
        return SequentialHead(head_config, device=self.device)

    def _build_flat_head(self, head_config: FlatHeadConfig) -> FlatHead:
        return FlatHead(head_config, device=self.device)

    def _build_mix_layer(self, mix_input_dim: int) -> nn.Module:
        return MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device,
        )
    
    def _pre_forward_checks(self, input_dict: dict[str, torch.Tensor]) -> None:
        """Perform pre-forward checks on the input dictionary."""
        for head_config in self.head_configs:
            key = head_config.key
            mask_key = head_config.mask_key if isinstance(head_config, SequentialHeadConfig) else None

            if key not in input_dict:
                raise KeyError(f"Input dictionary is missing required key: {key}")

            if mask_key is not None and mask_key not in input_dict:
                raise KeyError(f"Input dictionary is missing required mask key: {mask_key}")
    
    def forward(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a multi-head observation."""
        self._pre_forward_checks(input_dict)

        head_outputs: list[torch.Tensor] = []
        batch_shape: torch.Size | None = None
        for head_config in self.head_configs:
            key = head_config.key
            mask_key = head_config.mask_key if isinstance(head_config, SequentialHeadConfig) else None
            idx_key = head_config.positional_encoding_config.idx_key \
                if isinstance(head_config, SequentialHeadConfig) and head_config.positional_encoding_config is not None else None
            head = self.heads[key]
            
            mask = input_dict[mask_key] if mask_key is not None else None
            head_input = input_dict[key]
            idx = input_dict[idx_key] if idx_key is not None else None


            head_output = head(head_input, mask, idx)
            if head_output.shape[-1] != head_config.output_size:
                raise ValueError(
                    f"Head {key!r} returned size {head_output.shape[-1]}, expected {head_config.output_size}."
                )
            if batch_shape is None:
                batch_shape = head_output.shape[:-1]
            elif head_output.shape[:-1] != batch_shape:
                raise ValueError(
                    f"Head {key!r} has batch shape {tuple(head_output.shape[:-1])}, "
                    f"expected {tuple(batch_shape)}."
                )
            head_outputs.append(head_output)

        return self.mix_layer(torch.cat(head_outputs, dim=-1))
