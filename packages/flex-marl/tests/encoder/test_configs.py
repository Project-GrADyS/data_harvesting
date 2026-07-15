import pickle
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
    config, field = {
        "flat": (flat_config, "depth"),
        "sequential": (sequential_config, "depth"),
        "positional": (positional_config.positional_encoding_config, "num_positions"),
    }[config_name]
    with pytest.raises(FrozenInstanceError):
        setattr(config, field, 5)


def test_configs_apply_documented_defaults():
    flat = FlatHeadConfig(key="x", input_size=2)
    seq = SequentialHeadConfig(key="x", mask_key="m", input_size=2, positional_encoding_config=None)
    assert (flat.output_size, flat.depth, flat.hidden_layer_size, flat.activation_class) == (64, 3, 128, nn.ReLU)
    assert (seq.output_size, seq.num_heads, seq.ff_dim, seq.depth, seq.dropout) == (64, 8, 128, 3, 0.1)


@pytest.mark.parametrize("value", [""])
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


@pytest.mark.parametrize("value", ["relu", nn.ReLU(), None])
def test_flat_config_rejects_wrong_activation_class_types(flat_config, value):
    with pytest.raises(TypeError, match="activation_class"):
        validate_head_config(replace(flat_config, activation_class=value))


def test_validate_head_config_rejects_unsupported_type():
    with pytest.raises(TypeError, match="head_config"):
        validate_head_config(object())


@pytest.mark.parametrize("config_name", ["flat", "sequential", "positional"])
def test_configs_use_slots(config_name, flat_config, sequential_config, positional_config):
    config = {
        "flat": flat_config,
        "sequential": sequential_config,
        "positional": positional_config.positional_encoding_config,
    }[config_name]

    assert not hasattr(config, "__dict__")


@pytest.mark.parametrize("kind", ["flat", "sequential"])
@pytest.mark.parametrize("field", ["input_size", "output_size"])
@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_base_integer_fields_reject_wrong_types(kind, field, value, flat_config, sequential_config):
    config = replace(flat_config if kind == "flat" else sequential_config, **{field: value})

    with pytest.raises(TypeError, match=field):
        validate_head_config(config)


@pytest.mark.parametrize("field", ["depth", "hidden_layer_size"])
@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_flat_integer_fields_reject_wrong_types(flat_config, field, value):
    with pytest.raises(TypeError, match=field):
        validate_head_config(replace(flat_config, **{field: value}))


@pytest.mark.parametrize("field", ["num_heads", "ff_dim", "depth"])
@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_sequential_integer_fields_reject_wrong_types(sequential_config, field, value):
    with pytest.raises(TypeError, match=field):
        validate_head_config(replace(sequential_config, **{field: value}))


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_dropout_rejects_wrong_types(sequential_config, value):
    with pytest.raises(TypeError, match="dropout"):
        validate_head_config(replace(sequential_config, dropout=value))


@pytest.mark.parametrize("field", ["key", "mask_key", "idx_key"])
@pytest.mark.parametrize("value", [None, 1, b"key"])
def test_key_fields_reject_wrong_types(field, value, flat_config, sequential_config, positional_config):
    if field == "key":
        config = replace(flat_config, key=value)
    elif field == "mask_key":
        config = replace(sequential_config, mask_key=value)
    else:
        positional = replace(positional_config.positional_encoding_config, idx_key=value)
        config = replace(positional_config, positional_encoding_config=positional)

    with pytest.raises(TypeError, match=field):
        validate_head_config(config)


@pytest.mark.parametrize("value", [True, 1.0, "4", None])
def test_num_positions_rejects_wrong_types(positional_config, value):
    positional = replace(positional_config.positional_encoding_config, num_positions=value)

    with pytest.raises(TypeError, match="num_positions"):
        validate_head_config(replace(positional_config, positional_encoding_config=positional))


@pytest.mark.parametrize("config_name", ["flat", "sequential", "positional"])
def test_configs_support_pickle_round_trip(config_name, flat_config, sequential_config, positional_config):
    config = {
        "flat": flat_config,
        "sequential": sequential_config,
        "positional": positional_config.positional_encoding_config,
    }[config_name]

    assert pickle.loads(pickle.dumps(config)) == config
