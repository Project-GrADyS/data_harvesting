from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.multi_agent import CentralizedOutput, MultiAgentEncoderModule, MultiAgentMode


def expected_shape(mode, batch_shape, centralized_output=CentralizedOutput.GLOBAL):
    if mode is MultiAgentMode.CENTRALIZED and centralized_output is CentralizedOutput.GLOBAL:
        return (*batch_shape, 5)
    return (*batch_shape, 3, 5)


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_module_honors_constructor_device(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode), device="cpu")
    output = module(make_inputs())
    assert output.device.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_module_to_moves_all_mode_specific_parameters(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).to("cpu")
    output = module(make_inputs())
    assert output.device.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_module_supports_float64_on_cpu(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).double()
    inputs = make_inputs()
    inputs["flat"] = inputs["flat"].double().requires_grad_()
    inputs["sequence"] = inputs["sequence"].double().requires_grad_()
    output = module(inputs)
    output.sum().backward()
    assert output.dtype == torch.float64
    assert inputs["flat"].grad is not None and inputs["sequence"].grad is not None
    assert torch.isfinite(inputs["flat"].grad).all() and torch.isfinite(inputs["sequence"].grad).all()


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_forward_supports_noncontiguous_inputs(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).eval()
    contiguous = make_inputs(batch_shape=(2, 4))
    noncontiguous = {
        key: value.transpose(0, 1).contiguous().transpose(0, 1)
        for key, value in contiguous.items()
    }
    assert all(not value.is_contiguous() for value in noncontiguous.values())
    torch.testing.assert_close(module(noncontiguous), module(contiguous))


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_forward_does_not_mutate_input_dictionary_or_tensors(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).eval()
    inputs = make_inputs()
    original_keys = tuple(inputs)
    copies = {key: value.clone() for key, value in inputs.items()}
    module(inputs)
    assert tuple(inputs) == original_keys
    for key, expected in copies.items():
        torch.testing.assert_close(inputs[key], expected)


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_repeated_forward_does_not_leak_prepared_state(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).eval()
    original = make_inputs()
    expected = module(original)
    module(make_inputs())
    torch.testing.assert_close(module(original), expected)


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_eval_output_is_repeatable(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode)).eval()
    inputs = make_inputs()
    torch.testing.assert_close(module(inputs), module(inputs))


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_end_to_end_backward_is_finite(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode))
    inputs = make_inputs()
    inputs["flat"].requires_grad_()
    inputs["sequence"].requires_grad_()
    module(inputs).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    assert torch.isfinite(inputs["flat"].grad).all() and torch.isfinite(inputs["sequence"].grad).all()


@pytest.mark.parametrize(
    "mode,centralized_output",
    [
        (MultiAgentMode.SHARED, CentralizedOutput.GLOBAL),
        (MultiAgentMode.INDEPENDENT, CentralizedOutput.GLOBAL),
        (MultiAgentMode.CENTRALIZED, CentralizedOutput.GLOBAL),
        (MultiAgentMode.CENTRALIZED, CentralizedOutput.BROADCAST),
    ],
)
def test_state_dict_round_trip_preserves_output(mode, centralized_output, make_config, make_inputs) -> None:
    config = replace(make_config(mode), centralized_output=centralized_output)
    first = MultiAgentEncoderModule(config).eval()
    second = MultiAgentEncoderModule(config).eval()
    second.load_state_dict(first.state_dict())
    inputs = make_inputs()
    torch.testing.assert_close(second(inputs), first(inputs))


def test_state_dict_layout_reflects_execution_mode(make_config) -> None:
    shared_keys = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).state_dict()
    centralized_keys = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).state_dict()
    independent_keys = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT)).state_dict()
    assert shared_keys and all(key.startswith("encoder.") for key in shared_keys)
    assert centralized_keys and all(key.startswith("encoder.") for key in centralized_keys)
    assert independent_keys and all(key.startswith("encoders.") for key in independent_keys)
    assert all(any(key.startswith(f"encoders.{agent}.") for key in independent_keys) for agent in range(3))


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_output_contains_no_nan_or_inf_for_valid_mixed_masks(mode, make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(mode))
    inputs = make_inputs()
    inputs["sequence_mask"][0, 2] = False
    inputs["flat"].requires_grad_()
    inputs["sequence"].requires_grad_()
    output = module(inputs)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs["flat"].grad).all() and torch.isfinite(inputs["sequence"].grad).all()


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_module_uses_custom_mix_activation(mode, make_config) -> None:
    module = MultiAgentEncoderModule(replace(make_config(mode), mix_activation_class=nn.SiLU))
    encoders = module.encoders if mode is MultiAgentMode.INDEPENDENT else [module.encoder]
    assert all(encoder.mix_activation_class is nn.SiLU for encoder in encoders)
    assert all(any(isinstance(layer, nn.SiLU) for layer in encoder.mix_layer.modules()) for encoder in encoders)


@pytest.mark.parametrize("mode", list(MultiAgentMode))
def test_module_uses_default_mix_activation_when_none(mode, make_config) -> None:
    module = MultiAgentEncoderModule(replace(make_config(mode), mix_activation_class=None))
    encoders = module.encoders if mode is MultiAgentMode.INDEPENDENT else [module.encoder]
    assert all(encoder.mix_activation_class is nn.Tanh for encoder in encoders)
    assert all(any(isinstance(layer, nn.Tanh) for layer in encoder.mix_layer.modules()) for encoder in encoders)
