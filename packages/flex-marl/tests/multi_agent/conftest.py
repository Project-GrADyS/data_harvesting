from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flex_marl.multi_agent import (
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
)


@pytest.fixture
def sequential_options() -> SequentialFieldOptions:
    return SequentialFieldOptions(num_heads=2, ff_dim=16, depth=1, dropout=0.0)


@pytest.fixture
def flat_field(sequential_options: SequentialFieldOptions) -> FlatFieldConfig:
    return FlatFieldConfig(
        key="flat",
        input_size=2,
        output_size=8,
        depth=1,
        hidden_layer_size=16,
        sequential_options=sequential_options,
    )


@pytest.fixture
def sequential_field(sequential_options: SequentialFieldOptions) -> SequentialFieldConfig:
    return SequentialFieldConfig(
        key="sequence",
        mask_key="sequence_mask",
        input_size=3,
        output_size=8,
        sequential_options=sequential_options,
    )


@pytest.fixture
def make_config(flat_field, sequential_field):
    def factory(mode: MultiAgentMode, **changes) -> MultiAgentEncoderConfig:
        config = MultiAgentEncoderConfig(
            fields=(sequential_field, flat_field),
            num_agents=3,
            mode=mode,
            agent_mask_key="agent_mask",
            output_dim=5,
            mix_layer_depth=1,
            mix_layer_num_cells=16,
        )
        return replace(config, **changes)

    return factory


@pytest.fixture
def make_inputs():
    def factory(batch_shape: tuple[int, ...] = (2,), sequence_length: int = 4) -> dict[str, torch.Tensor]:
        agent_mask = torch.ones(*batch_shape, 3, dtype=torch.bool)
        sequence_mask = torch.ones(*batch_shape, 3, sequence_length, dtype=torch.bool)
        if batch_shape == (2,):
            agent_mask = torch.tensor([[True, False, True], [True, True, False]])
            sequence_mask = torch.tensor(
                [
                    [[True, True, False, False], [True, True, True, True], [True, False, False, False]],
                    [[True, True, True, True], [True, False, True, False], [True, True, False, False]],
                ]
            )
        return {
            "sequence": torch.randn(*batch_shape, 3, sequence_length, 3),
            "sequence_mask": sequence_mask,
            "flat": torch.randn(*batch_shape, 3, 2),
            "agent_mask": agent_mask,
        }

    return factory
