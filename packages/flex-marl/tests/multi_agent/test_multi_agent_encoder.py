from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flex_marl.encoder import FlatHeadConfig, SequentialHeadConfig
from flex_marl.multi_agent import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
)


def make_config(mode: MultiAgentMode, **changes) -> MultiAgentEncoderConfig:
    config = MultiAgentEncoderConfig(
        fields=(
            SequentialFieldConfig(
                key="sequence",
                mask_key="sequence_mask",
                input_size=3,
                output_size=8,
                num_heads=2,
                ff_dim=16,
                depth=1,
                dropout=0.0,
            ),
            FlatFieldConfig(
                key="flat",
                input_size=2,
                output_size=8,
                depth=1,
                hidden_layer_size=16,
                centralized_num_heads=2,
                centralized_ff_dim=16,
                centralized_depth=1,
                centralized_dropout=0.0,
            ),
        ),
        num_agents=3,
        mode=mode,
        agent_mask_key="agent_mask",
        output_dim=5,
        mix_layer_depth=1,
        mix_layer_num_cells=16,
    )
    return replace(config, **changes)


def make_inputs() -> dict[str, torch.Tensor]:
    return {
        "sequence": torch.randn(2, 3, 4, 3),
        "sequence_mask": torch.tensor(
            [
                [[True, True, False, False], [True, True, True, True], [True, False, False, False]],
                [[True, True, True, True], [True, False, True, False], [True, True, False, False]],
            ]
        ),
        "flat": torch.randn(2, 3, 2),
        "agent_mask": torch.tensor([[True, False, True], [True, True, False]]),
    }


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_multi_agent_encoder_output_shapes(mode: MultiAgentMode) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).eval()
    output = module(make_inputs())
    expected = (2, 5) if mode is MultiAgentMode.CENTRALIZED else (2, 3, 5)
    assert output.shape == expected
    assert torch.isfinite(output).all()


def test_shared_mode_compiles_field_types_without_exposing_head_configs() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    assert isinstance(module._head_configs[0], SequentialHeadConfig)
    assert isinstance(module._head_configs[1], FlatHeadConfig)


def test_centralized_mode_compiles_flat_field_as_sequence() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    assert all(isinstance(config, SequentialHeadConfig) for config in module._head_configs)


def test_shared_mode_combines_agent_and_element_masks() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs()
    prepared = module._prepare_shared(inputs, inputs["agent_mask"])
    expected = inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)
    torch.testing.assert_close(prepared["__flex_marl_sequence_mask"], expected)
    assert prepared["__flex_marl_sequence_agent_idx"].shape == (2, 3, 4, 1)


def test_centralized_mode_flattens_agents_and_sequence_elements() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])

    assert prepared["sequence"].shape == (2, 12, 3)
    expected_mask = (inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)).flatten(-2)
    torch.testing.assert_close(prepared["__flex_marl_sequence_mask"], expected_mask)
    assert prepared["__flex_marl_sequence_agent_idx"].shape == (2, 12, 1)
    torch.testing.assert_close(
        prepared["__flex_marl_sequence_agent_idx"][0, :, 0],
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    )


def test_centralized_mode_treats_flat_field_as_agent_sequence() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])

    assert prepared["flat"] is inputs["flat"]
    torch.testing.assert_close(prepared["__flex_marl_flat_mask"], inputs["agent_mask"])
    assert prepared["__flex_marl_flat_agent_idx"].shape == (2, 3, 1)
    torch.testing.assert_close(prepared["__flex_marl_flat_agent_idx"][0, :, 0], torch.arange(3))


def test_shared_and_independent_modes_zero_inactive_agent_outputs() -> None:
    inputs = make_inputs()
    for mode in (MultiAgentMode.SHARED, MultiAgentMode.INDEPENDENT):
        output = MultiAgentEncoderModule(make_config(mode)).eval()(inputs)
        torch.testing.assert_close(output[~inputs["agent_mask"]], torch.zeros_like(output[~inputs["agent_mask"]]))


def test_independent_mode_owns_one_distinct_encoder_per_agent() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    assert len(module.encoders) == 3
    first_parameters = {id(parameter) for parameter in module.encoders[0].parameters()}
    assert all(
        first_parameters.isdisjoint({id(parameter) for parameter in encoder.parameters()})
        for encoder in module.encoders[1:]
    )


def test_centralized_output_can_be_broadcast_per_agent() -> None:
    config = make_config(
        MultiAgentMode.CENTRALIZED,
        centralized_output=CentralizedOutput.BROADCAST,
    )
    output = MultiAgentEncoderModule(config).eval()(make_inputs())
    assert output.shape == (2, 3, 5)
    torch.testing.assert_close(output[:, 0], output[:, 1])
    torch.testing.assert_close(output[:, 1], output[:, 2])


def test_missing_or_invalid_agent_mask_is_rejected() -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs()
    del inputs["agent_mask"]
    with pytest.raises(KeyError, match="agent mask"):
        module(inputs)

    inputs = make_inputs()
    inputs["agent_mask"] = torch.ones(2, 3)
    with pytest.raises(TypeError, match="boolean"):
        module(inputs)


def test_duplicate_field_keys_are_rejected() -> None:
    field = FlatFieldConfig(key="flat", input_size=2, output_size=8)
    config = replace(make_config(MultiAgentMode.SHARED), fields=(field, field))
    with pytest.raises(ValueError, match="unique"):
        MultiAgentEncoderModule(config)
