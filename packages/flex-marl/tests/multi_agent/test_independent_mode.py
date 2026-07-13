from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.multi_agent import MultiAgentEncoderModule, MultiAgentMode


class RecordingEncoder(nn.Module):
    def __init__(self, value: float, output_dim: int = 5):
        super().__init__()
        self.value = value
        self.output_dim = output_dim
        self.received = None

    def forward(self, data):
        self.received = data
        batch_shape = data["flat"].shape[:-1]
        return torch.full((*batch_shape, self.output_dim), self.value, device=data["flat"].device)


def test_independent_mode_constructs_one_encoder_per_slot(make_config) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    assert len(module.encoders) == 3
    assert not hasattr(module, "encoder")


def test_independent_encoders_do_not_share_parameters(make_config) -> None:
    encoders = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT)).encoders
    parameter_sets = [{id(parameter) for parameter in encoder.parameters()} for encoder in encoders]
    assert all(parameter_sets[i].isdisjoint(parameter_sets[j]) for i in range(3) for j in range(i + 1, 3))


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_independent_mode_preserves_arbitrary_batch_dimensions(batch_shape, make_config, make_inputs) -> None:
    output = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT)).eval()(make_inputs(batch_shape))
    assert output.shape == (*batch_shape, 3, 5)


def test_independent_preparation_selects_correct_flat_slot(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    inputs = make_inputs()
    for agent in range(3):
        inputs["flat"][:, agent] = agent + 1
        prepared = module._prepare_independent(inputs, inputs["agent_mask"], agent)
        torch.testing.assert_close(prepared["flat"], torch.full((2, 2), agent + 1.0))


def test_independent_preparation_selects_correct_sequence_slot(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    inputs = make_inputs()
    for agent in range(3):
        inputs["sequence"][:, agent] = agent + 1
        prepared = module._prepare_independent(inputs, inputs["agent_mask"], agent)
        torch.testing.assert_close(prepared["sequence"], torch.full((2, 4, 3), agent + 1.0))


def test_independent_preparation_combines_selected_masks(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    inputs = make_inputs()
    for agent in range(3):
        actual = module._prepare_independent(inputs, inputs["agent_mask"], agent)["__flex_marl_sequence_mask"]
        expected = inputs["sequence_mask"][:, agent] & inputs["agent_mask"][:, agent].unsqueeze(-1)
        torch.testing.assert_close(actual, expected)


def test_independent_preparation_fills_selected_agent_identity(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    inputs = make_inputs()
    for agent in range(3):
        idx = module._prepare_independent(inputs, inputs["agent_mask"], agent)["__flex_marl_sequence_agent_idx"]
        assert idx.shape == (2, 4, 1)
        torch.testing.assert_close(idx, torch.full_like(idx, agent))


def test_independent_preparation_omits_identity_when_disabled(make_config, sequential_field, make_inputs) -> None:
    options = replace(sequential_field.sequential_options, encode_agent_identity=False)
    config = replace(
        make_config(MultiAgentMode.INDEPENDENT), fields=(replace(sequential_field, sequential_options=options),)
    )
    module = MultiAgentEncoderModule(config)
    inputs = make_inputs()
    prepared = module._prepare_independent(inputs, inputs["agent_mask"], 0)
    assert "__flex_marl_sequence_agent_idx" not in prepared


def test_independent_mode_routes_each_slot_to_matching_encoder(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    recorders = [RecordingEncoder(float(agent)) for agent in range(3)]
    module.encoders = nn.ModuleList(recorders)
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    output = module(inputs)
    for agent, recorder in enumerate(recorders):
        torch.testing.assert_close(recorder.received["flat"], inputs["flat"][:, agent])
        torch.testing.assert_close(output[:, agent], torch.full((2, 5), float(agent)))


def test_independent_mode_zeroes_inactive_outputs(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    module.encoders = nn.ModuleList([RecordingEncoder(7.0) for _ in range(3)])
    inputs = make_inputs()
    output = module(inputs)
    torch.testing.assert_close(output[~inputs["agent_mask"]], torch.zeros_like(output[~inputs["agent_mask"]]))
    torch.testing.assert_close(output[inputs["agent_mask"]], torch.full_like(output[inputs["agent_mask"]], 7.0))


def test_independent_encoder_parameter_changes_affect_only_own_slot(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT)).eval()
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    before = module(inputs)
    with torch.no_grad():
        for parameter in module.encoders[1].parameters():
            parameter.add_(1.0)
    after = module(inputs)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 2], after[:, 2])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_independent_mode_allows_different_outputs_for_identical_agent_inputs(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT)).eval()
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    inputs["sequence_mask"][:] = True
    inputs["flat"][:] = inputs["flat"][:, :1]
    inputs["sequence"][:] = inputs["sequence"][:, :1]
    output = module(inputs)
    assert not (torch.allclose(output[:, 0], output[:, 1]) and torch.allclose(output[:, 1], output[:, 2]))


def test_independent_mode_backpropagates_only_through_active_slot_outputs(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.INDEPENDENT))
    inputs = make_inputs()
    inputs["flat"].requires_grad_()
    inputs["sequence"].requires_grad_()
    module(inputs).sum().backward()
    torch.testing.assert_close(
        inputs["flat"].grad[~inputs["agent_mask"]], torch.zeros_like(inputs["flat"].grad[~inputs["agent_mask"]])
    )
    effective = inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)
    torch.testing.assert_close(
        inputs["sequence"].grad[~effective], torch.zeros_like(inputs["sequence"].grad[~effective])
    )
    for agent, encoder in enumerate(module.encoders):
        grads = [parameter.grad for parameter in encoder.parameters()]
        active_in_batch = inputs["agent_mask"][:, agent].any()
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        if not active_in_batch:
            assert all(torch.count_nonzero(grad) == 0 for grad in grads)
