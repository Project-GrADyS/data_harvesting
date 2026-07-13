from dataclasses import replace

import pytest
from torch import nn

from flex_marl.encoder import FlatHeadConfig, SequentialHeadConfig
from flex_marl.multi_agent import MultiAgentMode
from flex_marl.multi_agent.configs import compile_head_config


@pytest.mark.parametrize("mode", [MultiAgentMode.SHARED, MultiAgentMode.INDEPENDENT])
def test_flat_field_compiles_to_flat_head_in_per_agent_modes(mode, flat_field) -> None:
    assert isinstance(compile_head_config(flat_field, mode, 3), FlatHeadConfig)


def test_flat_head_compilation_preserves_all_mlp_settings(flat_field) -> None:
    field = replace(flat_field, input_size=4, output_size=12, depth=2, hidden_layer_size=24, activation_class=nn.SiLU)
    compiled = compile_head_config(field, MultiAgentMode.SHARED, 3)
    assert isinstance(compiled, FlatHeadConfig)
    assert (compiled.key, compiled.input_size, compiled.output_size) == ("flat", 4, 12)
    assert (compiled.depth, compiled.hidden_layer_size, compiled.activation_class) == (2, 24, nn.SiLU)


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_sequential_field_compiles_to_sequential_head_in_every_mode(mode, sequential_field) -> None:
    assert isinstance(compile_head_config(sequential_field, mode, 3), SequentialHeadConfig)


def test_centralized_flat_field_compiles_to_sequential_head(flat_field) -> None:
    assert isinstance(compile_head_config(flat_field, MultiAgentMode.CENTRALIZED, 3), SequentialHeadConfig)


def test_sequential_head_compilation_preserves_transformer_settings(sequential_field, sequential_options) -> None:
    options = replace(sequential_options, num_heads=4, ff_dim=31, depth=2, dropout=0.25)
    field = replace(sequential_field, output_size=12, sequential_options=options)
    compiled = compile_head_config(field, MultiAgentMode.SHARED, 5)
    assert isinstance(compiled, SequentialHeadConfig)
    assert (compiled.input_size, compiled.output_size) == (field.input_size, 12)
    assert (compiled.num_heads, compiled.ff_dim, compiled.depth, compiled.dropout) == (4, 31, 2, 0.25)


def test_compilation_uses_private_mask_key(sequential_field) -> None:
    compiled = compile_head_config(replace(sequential_field, mask_key="source_mask"), MultiAgentMode.SHARED, 3)
    assert isinstance(compiled, SequentialHeadConfig)
    assert compiled.mask_key == "__flex_marl_sequence_mask"
    assert compiled.mask_key != "source_mask"


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_compilation_adds_agent_identity_encoding_when_enabled(mode, sequential_field) -> None:
    compiled = compile_head_config(sequential_field, mode, 5)
    assert isinstance(compiled, SequentialHeadConfig)
    assert compiled.positional_encoding_config is not None
    assert compiled.positional_encoding_config.idx_key == "__flex_marl_sequence_agent_idx"
    assert compiled.positional_encoding_config.num_positions == 5


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_compilation_omits_agent_identity_encoding_when_disabled(mode, sequential_field) -> None:
    options = replace(sequential_field.sequential_options, encode_agent_identity=False)
    compiled = compile_head_config(replace(sequential_field, sequential_options=options), mode, 3)
    assert isinstance(compiled, SequentialHeadConfig)
    assert compiled.positional_encoding_config is None


def test_internal_keys_are_distinct_between_fields_and_purposes(sequential_field) -> None:
    other = replace(sequential_field, key="other", mask_key="other_source_mask")
    first = compile_head_config(sequential_field, MultiAgentMode.SHARED, 3)
    second = compile_head_config(other, MultiAgentMode.SHARED, 3)
    keys = {
        first.mask_key,
        first.positional_encoding_config.idx_key,
        second.mask_key,
        second.positional_encoding_config.idx_key,
    }
    assert len(keys) == 4


def test_compile_head_config_validates_mode_specific_requirements(flat_field) -> None:
    with pytest.raises(ValueError, match="requires sequential_options"):
        compile_head_config(replace(flat_field, sequential_options=None), MultiAgentMode.CENTRALIZED, 3)


@pytest.mark.parametrize("num_agents", [0, -1, 1.5, True])
def test_compile_head_config_rejects_invalid_num_agents(num_agents, flat_field) -> None:
    with pytest.raises(ValueError, match="num_agents"):
        compile_head_config(flat_field, MultiAgentMode.SHARED, num_agents)
