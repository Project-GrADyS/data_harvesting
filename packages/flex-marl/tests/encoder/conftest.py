from __future__ import annotations

import pytest
import torch

from flex_marl.encoder import FlatHeadConfig, PositionalEncodingConfig, SequentialHeadConfig


@pytest.fixture
def flat_config() -> FlatHeadConfig:
    return FlatHeadConfig(key="flat", input_size=5, output_size=7, depth=2, hidden_layer_size=11)


@pytest.fixture
def sequential_config() -> SequentialHeadConfig:
    return SequentialHeadConfig(
        key="sequence",
        mask_key="sequence_mask",
        input_size=3,
        output_size=8,
        positional_encoding_config=None,
        num_heads=2,
        ff_dim=16,
        depth=1,
        dropout=0.0,
    )


@pytest.fixture
def positional_config() -> SequentialHeadConfig:
    return SequentialHeadConfig(
        key="sequence",
        mask_key="sequence_mask",
        input_size=3,
        output_size=8,
        positional_encoding_config=PositionalEncodingConfig(idx_key="position", num_positions=4),
        num_heads=2,
        ff_dim=16,
        depth=1,
        dropout=0.0,
    )


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    torch.manual_seed(0)
