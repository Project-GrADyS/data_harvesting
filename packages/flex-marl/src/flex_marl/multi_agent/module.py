from __future__ import annotations

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING

from flex_marl.encoder import (
    FlatHeadConfig,
    MultiHeadEncoderModule,
    PositionalEncodingConfig,
    SequentialHeadConfig,
)

from .configs import (
    CentralizedOutput,
    FieldConfig,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentMode,
    SequentialFieldConfig,
)


class MultiAgentEncoderModule(nn.Module):
    """Apply a structured encoder according to a fixed-slot multi-agent execution mode."""

    def __init__(
        self,
        config: MultiAgentEncoderConfig,
        device: DEVICE_TYPING | None = None,
    ) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._head_configs = [self._compile_head_config(field) for field in config.fields]

        if config.mode is MultiAgentMode.INDEPENDENT:
            self.encoders = nn.ModuleList(self._build_encoder() for _ in range(config.num_agents))
        else:
            self.encoder = self._build_encoder()

    def _build_encoder(self) -> MultiHeadEncoderModule:
        return MultiHeadEncoderModule(
            head_configs=self._head_configs,
            mix_layer_depth=self.config.mix_layer_depth,
            mix_layer_num_cells=self.config.mix_layer_num_cells,
            mix_activation_class=self.config.mix_activation_class,
            output_dim=self.config.output_dim,
            device=self.device,
        )

    def _compile_head_config(self, field: FieldConfig) -> FlatHeadConfig | SequentialHeadConfig:
        if isinstance(field, FlatFieldConfig) and self.config.mode is not MultiAgentMode.CENTRALIZED:
            return FlatHeadConfig(
                key=field.key,
                input_size=field.input_size,
                output_size=field.output_size,
                depth=field.depth,
                hidden_layer_size=field.hidden_layer_size,
                activation_class=field.activation_class,
            )

        mask_key = self._internal_key(field.key, "mask")
        idx_key = self._internal_key(field.key, "agent_idx")
        positional_config: PositionalEncodingConfig | None = None
        if field.encode_agent_identity:
            positional_config = PositionalEncodingConfig(idx_key=idx_key, num_positions=self.config.num_agents)

        if isinstance(field, FlatFieldConfig):
            return SequentialHeadConfig(
                key=field.key,
                mask_key=mask_key,
                input_size=field.input_size,
                output_size=field.output_size,
                positional_encoding_config=positional_config,
                num_heads=field.centralized_num_heads,
                ff_dim=field.centralized_ff_dim,
                depth=field.centralized_depth,
                dropout=field.centralized_dropout,
            )

        return SequentialHeadConfig(
            key=field.key,
            mask_key=mask_key,
            input_size=field.input_size,
            output_size=field.output_size,
            positional_encoding_config=positional_config,
            num_heads=field.num_heads,
            ff_dim=field.ff_dim,
            depth=field.depth,
            dropout=field.dropout,
        )

    def forward(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        agent_mask = self._get_agent_mask(input_dict)

        if self.config.mode is MultiAgentMode.INDEPENDENT:
            outputs = [
                encoder(self._prepare_independent(input_dict, agent_mask, agent_index))
                for agent_index, encoder in enumerate(self.encoders)
            ]
            output = torch.stack(outputs, dim=-2)
            return self._mask_per_agent_output(output, agent_mask)

        if self.config.mode is MultiAgentMode.SHARED:
            output = self.encoder(self._prepare_shared(input_dict, agent_mask))
            return self._mask_per_agent_output(output, agent_mask)

        output = self.encoder(self._prepare_centralized(input_dict, agent_mask))
        if self.config.centralized_output is CentralizedOutput.BROADCAST:
            return output.unsqueeze(-2).expand(*output.shape[:-1], self.config.num_agents, output.shape[-1])
        return output

    def _prepare_shared(
        self,
        input_dict: dict[str, torch.Tensor],
        agent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = self._get_field(input_dict, field)
            self._validate_agent_axis(value, field)
            prepared[field.key] = value
            if isinstance(field, SequentialFieldConfig):
                element_mask = self._get_sequence_mask(input_dict, field, value)
                prepared[self._internal_key(field.key, "mask")] = element_mask & agent_mask.unsqueeze(-1)
                if field.encode_agent_identity:
                    prepared[self._internal_key(field.key, "agent_idx")] = self._agent_indices(value)
        return prepared

    def _prepare_independent(
        self,
        input_dict: dict[str, torch.Tensor],
        agent_mask: torch.Tensor,
        agent_index: int,
    ) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = self._get_field(input_dict, field)
            self._validate_agent_axis(value, field)
            prepared[field.key] = value.select(-3 if isinstance(field, SequentialFieldConfig) else -2, agent_index)
            if isinstance(field, SequentialFieldConfig):
                element_mask = self._get_sequence_mask(input_dict, field, value)
                active = agent_mask.select(-1, agent_index).unsqueeze(-1)
                prepared[self._internal_key(field.key, "mask")] = element_mask.select(-2, agent_index) & active
                if field.encode_agent_identity:
                    idx_shape = (*prepared[field.key].shape[:-1], 1)
                    prepared[self._internal_key(field.key, "agent_idx")] = torch.full(
                        idx_shape,
                        agent_index,
                        dtype=torch.long,
                        device=value.device,
                    )
        return prepared

    def _prepare_centralized(
        self,
        input_dict: dict[str, torch.Tensor],
        agent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = self._get_field(input_dict, field)
            self._validate_agent_axis(value, field)
            mask_key = self._internal_key(field.key, "mask")
            idx_key = self._internal_key(field.key, "agent_idx")

            if isinstance(field, FlatFieldConfig):
                prepared[field.key] = value
                prepared[mask_key] = agent_mask
                if field.encode_agent_identity:
                    prepared[idx_key] = self._agent_indices(value.unsqueeze(-2)).squeeze(-2)
                continue

            element_mask = self._get_sequence_mask(input_dict, field, value)
            effective_mask = element_mask & agent_mask.unsqueeze(-1)
            prepared[field.key] = value.flatten(-3, -2)
            prepared[mask_key] = effective_mask.flatten(-2)
            if field.encode_agent_identity:
                prepared[idx_key] = self._agent_indices(value).flatten(-3, -2)
        return prepared

    def _get_agent_mask(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.agent_mask_key not in input_dict:
            raise KeyError(f"Input dictionary is missing agent mask key: {self.config.agent_mask_key}")
        mask = input_dict[self.config.agent_mask_key]
        if mask.dtype != torch.bool:
            raise TypeError(f"Agent mask must have boolean dtype, got {mask.dtype}.")
        if mask.ndim < 1 or mask.shape[-1] != self.config.num_agents:
            raise ValueError(
                f"Agent mask must have shape (*B, {self.config.num_agents}), got {tuple(mask.shape)}."
            )
        return mask

    @staticmethod
    def _get_field(input_dict: dict[str, torch.Tensor], field: FieldConfig) -> torch.Tensor:
        if field.key not in input_dict:
            raise KeyError(f"Input dictionary is missing field key: {field.key}")
        return input_dict[field.key]

    @staticmethod
    def _get_sequence_mask(
        input_dict: dict[str, torch.Tensor],
        field: SequentialFieldConfig,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if field.mask_key not in input_dict:
            raise KeyError(f"Input dictionary is missing sequence mask key: {field.mask_key}")
        mask = input_dict[field.mask_key]
        if mask.dtype != torch.bool:
            raise TypeError(f"Sequence mask for {field.key!r} must have boolean dtype, got {mask.dtype}.")
        if mask.shape != value.shape[:-1]:
            raise ValueError(
                f"Sequence mask for {field.key!r} must have shape {tuple(value.shape[:-1])}, got {tuple(mask.shape)}."
            )
        return mask

    def _validate_agent_axis(self, value: torch.Tensor, field: FieldConfig) -> None:
        agent_axis = -3 if isinstance(field, SequentialFieldConfig) else -2
        minimum_dims = 3 if isinstance(field, SequentialFieldConfig) else 2
        if value.ndim < minimum_dims or value.shape[agent_axis] != self.config.num_agents:
            kind = "sequential" if isinstance(field, SequentialFieldConfig) else "flat"
            raise ValueError(
                f"{kind.capitalize()} field {field.key!r} must have {self.config.num_agents} agents at axis "
                f"{agent_axis}, got shape {tuple(value.shape)}."
            )
        if value.shape[-1] != field.input_size:
            raise ValueError(
                f"Field {field.key!r} last dimension must be {field.input_size}, got {value.shape[-1]}."
            )

    def _agent_indices(self, sequential_value: torch.Tensor) -> torch.Tensor:
        batch_shape = sequential_value.shape[:-3]
        sequence_length = sequential_value.shape[-2]
        view_shape = (*((1,) * len(batch_shape)), self.config.num_agents, 1, 1)
        return torch.arange(self.config.num_agents, device=sequential_value.device).view(view_shape).expand(
            *batch_shape, self.config.num_agents, sequence_length, 1
        )

    @staticmethod
    def _mask_per_agent_output(output: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        return output * agent_mask.unsqueeze(-1).to(output.dtype)

    @staticmethod
    def _internal_key(field_key: str, suffix: str) -> str:
        return f"__flex_marl_{field_key}_{suffix}"

    @staticmethod
    def _validate_config(config: MultiAgentEncoderConfig) -> None:
        if not config.fields:
            raise ValueError("fields must contain at least one field configuration.")
        if not isinstance(config.num_agents, int) or isinstance(config.num_agents, bool) or config.num_agents <= 0:
            raise ValueError(f"num_agents must be a positive integer, got {config.num_agents}.")
        if not isinstance(config.mode, MultiAgentMode):
            raise TypeError(f"mode must be a MultiAgentMode, got {type(config.mode)}.")
        if not isinstance(config.centralized_output, CentralizedOutput):
            raise TypeError(
                f"centralized_output must be a CentralizedOutput, got {type(config.centralized_output)}."
            )
        if not isinstance(config.agent_mask_key, str) or not config.agent_mask_key:
            raise ValueError("agent_mask_key must be a non-empty string.")
        keys = [field.key for field in config.fields]
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Every field key must be a non-empty string.")
        if len(keys) != len(set(keys)):
            raise ValueError("Field keys must be unique.")
