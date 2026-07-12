from dataclasses import FrozenInstanceError, replace

import pytest
from torch import nn

from flex_marl.encoder import (
    FlatHeadConfig,
    PositionalEncodingConfig,
    SequentialHeadConfig,
    validate_head_config,
)


def test_flat_config_accepts_valid_values(flat_config):
    assert validate_head_config(flat_config) is None


def test_sequential_config_accepts_valid_values(sequential_config):
    assert validate_head_config(sequential_config) is None


def test_sequential_positional_config_accepts_valid_values(positional_config):
    assert validate_head_config(positional_config) is None


@pytest.mark.parametrize("kind", ["flat", "sequential"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive_input_size(kind, value, flat_config, sequential_config):
    config = replace(flat_config if kind == "flat" else sequential_config, input_size=value)
    with pytest.raises(ValueError, match="input_size"):
        validate_head_config(config)


@pytest.mark.parametrize("kind", ["flat", "sequential"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive_output_size(kind, value, flat_config, sequential_config):
    config = replace(flat_config if kind == "flat" else sequential_config, output_size=value)
    with pytest.raises(ValueError, match="output_size"):
        validate_head_config(config)


@pytest.mark.parametrize("value", [0, -1])
def test_flat_config_rejects_non_positive_depth(flat_config, value):
    with pytest.raises(ValueError, match="depth"):
        validate_head_config(replace(flat_config, depth=value))


@pytest.mark.parametrize("value", [0, -1])
def test_flat_config_rejects_non_positive_hidden_layer_size(flat_config, value):
    with pytest.raises(ValueError, match="hidden_layer_size"):
        validate_head_config(replace(flat_config, hidden_layer_size=value))


@pytest.mark.parametrize("field", ["num_heads", "ff_dim", "depth"])
@pytest.mark.parametrize("value", [0, -1])
def test_sequential_config_rejects_non_positive_integer_fields(sequential_config, field, value):
    with pytest.raises(ValueError, match=field):
        validate_head_config(replace(sequential_config, **{field: value}))


@pytest.mark.parametrize("value", [0.0, 0.999])
def test_sequential_config_accepts_dropout_boundaries(sequential_config, value):
    validate_head_config(replace(sequential_config, dropout=value))


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1, float("inf"), float("nan")])
def test_sequential_config_rejects_invalid_dropout(sequential_config, value):
    with pytest.raises(ValueError, match="dropout"):
        validate_head_config(replace(sequential_config, dropout=value))


@pytest.mark.parametrize("value", [0, -1])
def test_sequential_config_rejects_non_positive_num_positions(positional_config, value):
    positional = replace(positional_config.positional_encoding_config, num_positions=value)
    with pytest.raises(ValueError, match="num_positions"):
        validate_head_config(replace(positional_config, positional_encoding_config=positional))


def test_sequential_config_requires_output_size_divisible_by_num_heads(sequential_config):
    with pytest.raises(ValueError, match="divisible"):
        validate_head_config(replace(sequential_config, output_size=10, num_heads=3))


@pytest.mark.parametrize("output_size,num_heads", [(8, 1), (8, 8)])
def test_sequential_config_accepts_attention_boundary_shapes(sequential_config, output_size, num_heads):
    validate_head_config(replace(sequential_config, output_size=output_size, num_heads=num_heads))


def test_configs_are_keyword_only():
    with pytest.raises(TypeError):
        FlatHeadConfig("flat", 4)
    with pytest.raises(TypeError):
        PositionalEncodingConfig("idx", 4)


@pytest.mark.parametrize("config_name", ["flat", "sequential", "positional"])
def test_configs_are_frozen(config_name, flat_config, sequential_config, positional_config):
    config = {"flat": flat_config, "sequential": sequential_config, "positional": positional_config.positional_encoding_config}[config_name]
    with pytest.raises(FrozenInstanceError):
        config.num_positions = 5 if config_name == "positional" else 5


def test_configs_apply_documented_defaults():
    flat = FlatHeadConfig(key="x", input_size=2)
    seq = SequentialHeadConfig(key="x", mask_key="m", input_size=2, positional_encoding_config=None)
    assert (flat.output_size, flat.depth, flat.hidden_layer_size, flat.activation_class) == (64, 3, 128, nn.ReLU)
    assert (seq.output_size, seq.num_heads, seq.ff_dim, seq.depth, seq.dropout) == (64, 8, 128, 3, 0.1)


@pytest.mark.parametrize("value", [None, ""])
def test_positional_config_requires_nonempty_idx_key(positional_config, value):
    positional = replace(positional_config.positional_encoding_config, idx_key=value)
    with pytest.raises(ValueError, match="idx_key"):
        validate_head_config(replace(positional_config, positional_encoding_config=positional))


@pytest.mark.parametrize("kind", ["flat", "sequential"])
def test_head_config_requires_nonempty_input_key(kind, flat_config, sequential_config):
    with pytest.raises(ValueError, match="key"):
        validate_head_config(replace(flat_config if kind == "flat" else sequential_config, key=""))


def test_sequential_config_requires_nonempty_mask_key(sequential_config):
    with pytest.raises(ValueError, match="mask_key"):
        validate_head_config(replace(sequential_config, mask_key=""))


def test_flat_config_rejects_invalid_activation_class(flat_config):
    with pytest.raises(ValueError, match="activation_class"):
        validate_head_config(replace(flat_config, activation_class="relu"))


def test_validate_head_config_rejects_unsupported_type():
    with pytest.raises(TypeError, match="head_config"):
        validate_head_config(object())
