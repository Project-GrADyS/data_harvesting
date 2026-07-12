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
    head_output_buffer: torch.Tensor  # type hint for the buffer registered in __init__

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
        self.head_configs = head_configs
        self.output_dim = output_dim
        self.mix_layer_depth = mix_layer_depth
        self.mix_layer_num_cells = mix_layer_num_cells
        self.mix_activation_class = mix_activation_class if mix_activation_class is not None else nn.Tanh
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.heads = nn.ModuleDict()
        for config in self.head_configs:
            if config.key in self.heads:
                raise ValueError(f"Duplicate head key found: {config.key}")
            
            validate_head_config(config)
            
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

        # Fixed size buffer to store the outputs of all of the heads before the mix layer.
        # This is used to avoid re-allocating memory for the head outputs on every forward pass.
        # Register as a buffer so it's moved with the module. This is scratch memory
        # (avoids reallocations) so mark it non-persistent (won't be saved in state_dict).
        self.register_buffer(
            "head_output_buffer",
            torch.empty(mix_input_dim, device=self.device),
            persistent=False,
        )


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

        curr_index = 0
        for head_config in self.head_configs:
            key = head_config.key
            mask_key = head_config.mask_key if isinstance(head_config, SequentialHeadConfig) else None
            idx_key = head_config.positional_encoding_config.idx_key \
                if isinstance(head_config, SequentialHeadConfig) and head_config.positional_encoding_config is not None else None
            head = self.heads[key]
            
            mask = input_dict[mask_key] if mask_key is not None else None
            head_input = input_dict[key]
            idx = input_dict[idx_key] if idx_key is not None else None


            self.head_output_buffer[curr_index:curr_index + head_config.output_size] = head(head_input, mask, idx)
            curr_index += head_config.output_size

        return self.mix_layer(self.head_output_buffer)