from __future__ import annotations

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING

from flex_marl.encoder import MultiHeadEncoderModule

from .configs import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentMode,
    SequentialFieldConfig,
    _internal_key,
    compile_head_config,
    validate_multi_agent_encoder_config,
)


def _build_encoder(
    config: MultiAgentEncoderConfig,
    device: torch.device,
    run_pre_forward_checks: bool,
) -> MultiHeadEncoderModule:
    """Compile all fields and construct one mode-agnostic structured encoder."""

    head_configs = tuple(compile_head_config(field, config.mode, config.num_agents) for field in config.fields)
    return MultiHeadEncoderModule(
        head_configs=head_configs,
        mix_layer_depth=config.mix_layer_depth,
        mix_layer_num_cells=config.mix_layer_num_cells,
        mix_activation_class=config.mix_activation_class,
        output_dim=config.output_dim,
        device=device,
        run_pre_forward_checks=run_pre_forward_checks,
    )


def _agent_dimension_indices(sequential_value: torch.Tensor, num_agents: int) -> torch.Tensor:
    """
    Return one agent ID per sequence element with shape ``(*B, agents, sequence, 1)``. Agent IDs are  derived
    from the agent dimension of the input tensor, which is assumed to be at index -3. The sequence dimension
    is assumed to be at index -2. Each agent id is repeated for every element in its sequence and broadcast 
    across all batch dimensions.
    """

    batch_shape = sequential_value.shape[:-3]
    sequence_length = sequential_value.shape[-2]
    # Start with one scalar ID per fixed slot, then broadcast it over every
    # batch dimension and every element owned by that agent.
    view_shape = (*((1,) * len(batch_shape)), num_agents, 1, 1)
    return torch.arange(num_agents, device=sequential_value.device).view(view_shape).expand(
        *batch_shape, num_agents, sequence_length, 1
    )


def _mask_inactive_agent_outputs(output: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """Force representations belonging to inactive fixed slots to zero."""

    return output * agent_mask.unsqueeze(-1).to(output.dtype)


class MultiAgentEncoderModule(nn.Module):
    """Encode fixed-slot multi-agent observations in one of three execution modes.

    Shared and independent modes return ``(*B, agents, output_dim)``. Centralized
    mode returns ``(*B, output_dim)`` by default, or the same global vector
    broadcast to ``(*B, agents, output_dim)`` when configured accordingly.

    Args:
        config: Field descriptions, execution mode, mask key, and encoder dimensions.
        device: Device on which the underlying encoders are constructed. Defaults to CPU.
        run_pre_forward_checks: Whether this module and its child encoders validate
            tensor contracts before each forward pass.
    """

    def __init__(
        self,
        config: MultiAgentEncoderConfig,
        device: DEVICE_TYPING | None = None,
        run_pre_forward_checks: bool = True,
    ) -> None:
        super().__init__()
        validate_multi_agent_encoder_config(config)
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.run_pre_forward_checks = run_pre_forward_checks

        if config.mode is MultiAgentMode.INDEPENDENT:
            # Each fixed slot owns a completely separate encoder and parameter set.
            self.encoders = nn.ModuleList(
                _build_encoder(config, self.device, run_pre_forward_checks)
                for _ in range(config.num_agents)
            )
        else:
            # Shared and centralized execution each require only one parameter set.
            self.encoder = _build_encoder(config, self.device, run_pre_forward_checks)

    def _pre_forward_checks(self, input_dict: dict[str, torch.Tensor]) -> None:
        """Validate all required keys, dtypes, and shapes before transforming tensors."""

        required_keys = {self.config.agent_mask_key}
        required_keys.update(field.key for field in self.config.fields)
        required_keys.update(
            field.mask_key for field in self.config.fields if isinstance(field, SequentialFieldConfig)
        )
        missing_keys = required_keys.difference(input_dict)
        if missing_keys:
            raise KeyError(f"Input dictionary is missing required keys: {sorted(missing_keys)}")

        agent_mask = input_dict[self.config.agent_mask_key]
        if not isinstance(agent_mask, torch.Tensor):
            raise TypeError("Agent mask must be a torch.Tensor.")
        if agent_mask.dtype != torch.bool:
            raise TypeError(f"Agent mask must have boolean dtype, got {agent_mask.dtype}.")
        if agent_mask.ndim < 1 or agent_mask.shape[-1] != self.config.num_agents:
            raise ValueError(
                f"Agent mask must have shape (*B, {self.config.num_agents}), got {tuple(agent_mask.shape)}."
            )
        batch_shape = agent_mask.shape[:-1]

        for field in self.config.fields:
            value = input_dict[field.key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Field {field.key!r} must be a torch.Tensor.")

            if isinstance(field, FlatFieldConfig):
                expected_shape = (*batch_shape, self.config.num_agents, field.input_size)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"Flat field {field.key!r} must have shape {expected_shape}, got {tuple(value.shape)}."
                    )
                continue

            expected_prefix = (*batch_shape, self.config.num_agents)
            if value.ndim != len(batch_shape) + 3 or value.shape[:-2] != expected_prefix:
                raise ValueError(
                    f"Sequential field {field.key!r} must have shape "
                    f"(*B, {self.config.num_agents}, sequence_length, {field.input_size}), "
                    f"got {tuple(value.shape)}."
                )
            if value.shape[-2] == 0 or value.shape[-1] != field.input_size:
                raise ValueError(
                    f"Sequential field {field.key!r} must have a non-empty sequence and last dimension "
                    f"{field.input_size}, got {tuple(value.shape)}."
                )

            element_mask = input_dict[field.mask_key]
            if not isinstance(element_mask, torch.Tensor):
                raise TypeError(f"Sequence mask for {field.key!r} must be a torch.Tensor.")
            if element_mask.dtype != torch.bool:
                raise TypeError(
                    f"Sequence mask for {field.key!r} must have boolean dtype, got {element_mask.dtype}."
                )
            if element_mask.shape != value.shape[:-1]:
                raise ValueError(
                    f"Sequence mask for {field.key!r} must have shape {tuple(value.shape[:-1])}, "
                    f"got {tuple(element_mask.shape)}."
                )

    def forward(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Validate and encode a dictionary of fixed-slot multi-agent observations."""

        if self.run_pre_forward_checks:
            self._pre_forward_checks(input_dict)
        agent_mask = input_dict[self.config.agent_mask_key]

        if self.config.mode is MultiAgentMode.INDEPENDENT:
            outputs = [
                encoder(self._prepare_independent(input_dict, agent_mask, agent_index))
                for agent_index, encoder in enumerate(self.encoders)
            ]
            output = torch.stack(outputs, dim=-2)
            return _mask_inactive_agent_outputs(output, agent_mask)

        if self.config.mode is MultiAgentMode.SHARED:
            output = self.encoder(self._prepare_shared(input_dict, agent_mask))
            return _mask_inactive_agent_outputs(output, agent_mask)

        output = self.encoder(self._prepare_centralized(input_dict, agent_mask))
        if self.config.centralized_output is CentralizedOutput.BROADCAST:
            # Expose the single global representation at every agent position.
            # `expand` broadcasts it without evaluating the encoder again.
            return output.unsqueeze(-2).expand(*output.shape[:-1], self.config.num_agents, output.shape[-1])
        return output

    def _prepare_shared(
        self,
        input_dict: dict[str, torch.Tensor],
        agent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Keep the agent axis as a batch dimension for one shared encoder."""
        
        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = input_dict[field.key]
            prepared[field.key] = value
            if isinstance(field, SequentialFieldConfig):
                # An element is usable only when both it and its owning agent are active.
                prepared[_internal_key(field.key, "mask")] = (
                    input_dict[field.mask_key] & agent_mask.unsqueeze(-1)
                )
                if field.sequential_options.encode_agent_identity:
                    prepared[_internal_key(field.key, "agent_idx")] = _agent_dimension_indices(
                        value, self.config.num_agents
                    )
        return prepared

    def _prepare_independent(
        self,
        input_dict: dict[str, torch.Tensor],
        agent_mask: torch.Tensor,
        agent_index: int,
    ) -> dict[str, torch.Tensor]:
        """Select one fixed slot before invoking that slot's private encoder."""

        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = input_dict[field.key]
            # Sequential tensors place agents at -3; flat tensors place them at -2.
            agent_axis = -3 if isinstance(field, SequentialFieldConfig) else -2
            prepared[field.key] = value.select(agent_axis, agent_index)
            if isinstance(field, SequentialFieldConfig):
                # Selecting one agent removes the agent axis from both the values and mask.
                active = agent_mask.select(-1, agent_index).unsqueeze(-1)
                prepared[_internal_key(field.key, "mask")] = (
                    input_dict[field.mask_key].select(-2, agent_index) & active
                )
                if field.sequential_options.encode_agent_identity:
                    idx_shape = (*prepared[field.key].shape[:-1], 1)
                    prepared[_internal_key(field.key, "agent_idx")] = torch.full(
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
        """Convert all agent observations into global masked sequences."""

        prepared: dict[str, torch.Tensor] = {}
        for field in self.config.fields:
            value = input_dict[field.key]
            mask_key = _internal_key(field.key, "mask")
            idx_key = _internal_key(field.key, "agent_idx")

            if isinstance(field, FlatFieldConfig):
                # A flat value from each slot becomes one sequence element:
                # (*B, agents, features) remains (*B, sequence=agents, features).
                prepared[field.key] = value
                prepared[mask_key] = agent_mask
                assert field.sequential_options is not None
                if field.sequential_options.encode_agent_identity:
                    # Reuse the sequential index builder with a synthetic length-one
                    # axis, then remove that axis to obtain (*B, agents, 1).
                    prepared[idx_key] = _agent_dimension_indices(value.unsqueeze(-2), self.config.num_agents).squeeze(-2)
                continue

            # Concatenate each agent's sequence into one global sequence:
            # (*B, agents, sequence, features) -> (*B, agents * sequence, features).
            effective_mask = input_dict[field.mask_key] & agent_mask.unsqueeze(-1)
            prepared[field.key] = value.flatten(-3, -2)
            prepared[mask_key] = effective_mask.flatten(-2)
            if field.sequential_options.encode_agent_identity:
                # Flatten IDs in exactly the same order as their corresponding elements.
                prepared[idx_key] = _agent_dimension_indices(value, self.config.num_agents).flatten(-3, -2)
        return prepared
