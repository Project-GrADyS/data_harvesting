from dataclasses import FrozenInstanceError, replace

import pytest
from torch import nn

from flex_marl.multi_agent import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
    validate_field_config,
    validate_multi_agent_encoder_config,
    validate_sequential_field_options,
)


def test_sequential_options_accept_documented_defaults() -> None:
    options = SequentialFieldOptions()
    assert (options.num_heads, options.ff_dim, options.depth, options.dropout, options.encode_agent_identity) == (
        8, 128, 3, 0.1, True
    )
    assert validate_sequential_field_options(options) is None


@pytest.mark.parametrize("dropout", [0.0, 0.999999])
def test_sequential_options_accept_dropout_boundaries(dropout: float, sequential_options) -> None:
    assert validate_sequential_field_options(replace(sequential_options, dropout=dropout)) is None


@pytest.mark.parametrize("field_name", ["num_heads", "ff_dim", "depth"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_sequential_options_reject_invalid_positive_integer_fields(field_name: str, value, sequential_options) -> None:
    with pytest.raises(ValueError, match=field_name):
        validate_sequential_field_options(replace(sequential_options, **{field_name: value}))


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1, True, "0.1", float("inf"), float("nan")])
def test_sequential_options_reject_invalid_dropout(value, sequential_options) -> None:
    with pytest.raises(ValueError, match="dropout"):
        validate_sequential_field_options(replace(sequential_options, dropout=value))


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_sequential_options_require_boolean_agent_identity(value, sequential_options) -> None:
    with pytest.raises(TypeError, match="encode_agent_identity"):
        validate_sequential_field_options(replace(sequential_options, encode_agent_identity=value))


def test_sequential_options_reject_wrong_object_type() -> None:
    with pytest.raises(TypeError, match="SequentialFieldOptions"):
        validate_sequential_field_options(object())


@pytest.mark.parametrize("field_type", [FlatFieldConfig, SequentialFieldConfig])
def test_field_configs_are_keyword_only(field_type) -> None:
    with pytest.raises(TypeError):
        field_type("field", 3)


def test_field_configs_are_frozen(flat_field, sequential_field) -> None:
    for field in (flat_field, sequential_field):
        with pytest.raises(FrozenInstanceError):
            field.input_size = 99


def test_field_config_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="FlatFieldConfig or SequentialFieldConfig"):
        validate_field_config(object(), MultiAgentMode.SHARED)


@pytest.mark.parametrize("key", ["", None, 3])
def test_field_configs_require_nonempty_keys(key, flat_field, sequential_field) -> None:
    for field in (replace(flat_field, key=key), replace(sequential_field, key=key)):
        with pytest.raises(ValueError, match="field key"):
            validate_field_config(field, MultiAgentMode.SHARED)


@pytest.mark.parametrize("mask_key", ["", None, 3])
def test_sequential_fields_require_nonempty_mask_keys(mask_key, sequential_field) -> None:
    with pytest.raises(ValueError, match="mask_key"):
        validate_field_config(replace(sequential_field, mask_key=mask_key), MultiAgentMode.SHARED)


@pytest.mark.parametrize("field_name", ["input_size", "output_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_field_configs_reject_invalid_sizes(field_name, value, flat_field, sequential_field) -> None:
    for field in (replace(flat_field, **{field_name: value}), replace(sequential_field, **{field_name: value})):
        with pytest.raises(ValueError, match=field_name):
            validate_field_config(field, MultiAgentMode.SHARED)


@pytest.mark.parametrize("field_name", ["depth", "hidden_layer_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_flat_field_rejects_invalid_mlp_dimensions(field_name, value, flat_field) -> None:
    with pytest.raises(ValueError, match=field_name):
        validate_field_config(replace(flat_field, **{field_name: value}), MultiAgentMode.SHARED)


@pytest.mark.parametrize("activation", [nn.ReLU(), str, lambda: nn.ReLU()])
def test_flat_field_rejects_invalid_activation_class(activation, flat_field) -> None:
    with pytest.raises(ValueError, match="activation_class"):
        validate_field_config(replace(flat_field, activation_class=activation), MultiAgentMode.SHARED)


@pytest.mark.parametrize("mode", [MultiAgentMode.SHARED, MultiAgentMode.INDEPENDENT])
def test_flat_field_accepts_sequential_options_outside_centralized_mode(mode, flat_field) -> None:
    assert validate_field_config(flat_field, mode) is None


def test_centralized_flat_field_requires_sequential_options(flat_field) -> None:
    with pytest.raises(ValueError, match="requires sequential_options"):
        validate_field_config(replace(flat_field, sequential_options=None), MultiAgentMode.CENTRALIZED)


def test_sequential_field_requires_options_at_construction() -> None:
    with pytest.raises(TypeError, match="sequential_options"):
        SequentialFieldConfig(key="sequence", mask_key="mask", input_size=3)


def test_sequential_field_rejects_wrong_options_type(sequential_field) -> None:
    with pytest.raises(TypeError, match="SequentialFieldOptions"):
        validate_field_config(replace(sequential_field, sequential_options=object()), MultiAgentMode.SHARED)


def test_sequential_field_output_size_must_divide_attention_heads(sequential_field) -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_field_config(replace(sequential_field, output_size=7), MultiAgentMode.SHARED)


def test_complete_config_requires_at_least_one_field(make_config) -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), fields=()))


@pytest.mark.parametrize("field_name", ["num_agents", "output_dim", "mix_layer_depth", "mix_layer_num_cells"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_complete_config_rejects_invalid_scalar_settings(field_name, value, make_config) -> None:
    with pytest.raises(ValueError, match=field_name):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), **{field_name: value}))


@pytest.mark.parametrize("field_name,value", [("mode", "shared"), ("centralized_output", "global")])
def test_complete_config_requires_enum_instances(field_name, value, make_config) -> None:
    with pytest.raises(TypeError, match=field_name):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), **{field_name: value}))


@pytest.mark.parametrize("key", ["", None, 3])
def test_complete_config_rejects_invalid_agent_mask_key(key, make_config) -> None:
    with pytest.raises(ValueError, match="agent_mask_key"):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), agent_mask_key=key))


def test_complete_config_rejects_duplicate_field_keys_across_field_types(make_config, flat_field, sequential_field) -> None:
    duplicate = replace(sequential_field, key=flat_field.key)
    with pytest.raises(ValueError, match="unique"):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), fields=(flat_field, duplicate)))


@pytest.mark.parametrize("activation", [None, nn.ReLU])
def test_complete_config_accepts_default_and_custom_mix_activation(activation, make_config) -> None:
    config = replace(make_config(MultiAgentMode.SHARED), mix_activation_class=activation)
    assert validate_multi_agent_encoder_config(config) is None


@pytest.mark.parametrize("activation", [nn.ReLU(), str, lambda: nn.ReLU()])
def test_complete_config_rejects_invalid_mix_activation(activation, make_config) -> None:
    with pytest.raises(ValueError, match="mix_activation_class"):
        validate_multi_agent_encoder_config(replace(make_config(MultiAgentMode.SHARED), mix_activation_class=activation))


def test_complete_config_rejects_wrong_object_type() -> None:
    with pytest.raises(TypeError, match="MultiAgentEncoderConfig"):
        validate_multi_agent_encoder_config(object())


def test_module_constructor_runs_complete_config_validation(make_config) -> None:
    invalid = replace(make_config(MultiAgentMode.SHARED), centralized_output="global")
    with pytest.raises(TypeError, match="centralized_output"):
        MultiAgentEncoderModule(invalid)
