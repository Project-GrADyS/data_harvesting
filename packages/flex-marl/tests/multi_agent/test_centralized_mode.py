from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.multi_agent import CentralizedOutput, MultiAgentEncoderModule, MultiAgentMode


class CountingEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        batch_shape = data["flat"].shape[:-2]
        return torch.arange(self.output_dim, dtype=data["flat"].dtype).expand(*batch_shape, self.output_dim)


def test_centralized_mode_constructs_one_encoder(make_config) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    assert hasattr(module, "encoder")
    assert not hasattr(module, "encoders")


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_centralized_global_output_preserves_batch_dimensions(batch_shape, make_config, make_inputs) -> None:
    output = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).eval()(make_inputs(batch_shape))
    assert output.shape == (*batch_shape, 5)
    assert torch.isfinite(output).all()


def test_centralized_flat_field_becomes_agent_sequence(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])
    assert prepared["flat"] is inputs["flat"]
    assert prepared["flat"].shape == (2, 3, 2)


def test_centralized_flat_mask_is_agent_mask(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])
    assert prepared["__flex_marl_flat_mask"] is inputs["agent_mask"]


def test_centralized_flat_agent_indices_follow_slot_order(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs(batch_shape=(2, 2))
    idx = module._prepare_centralized(inputs, inputs["agent_mask"])["__flex_marl_flat_agent_idx"]
    assert idx.shape == (2, 2, 3, 1)
    torch.testing.assert_close(idx[0, 0, :, 0], torch.arange(3))


def test_centralized_flat_field_omits_indices_when_identity_disabled(make_config, flat_field, make_inputs) -> None:
    options = replace(flat_field.sequential_options, encode_agent_identity=False)
    config = replace(make_config(MultiAgentMode.CENTRALIZED), fields=(replace(flat_field, sequential_options=options),))
    module = MultiAgentEncoderModule(config)
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])
    assert "__flex_marl_flat_agent_idx" not in prepared


def test_centralized_sequence_flattens_agents_in_slot_major_order(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs(sequence_length=4)
    for agent in range(3):
        for timestep in range(4):
            inputs["sequence"][:, agent, timestep] = agent * 10 + timestep
    flattened = module._prepare_centralized(inputs, inputs["agent_mask"])["sequence"]
    assert flattened.shape == (2, 12, 3)
    torch.testing.assert_close(flattened[0, :, 0], torch.tensor([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23.0]))


def test_centralized_sequence_flattens_effective_mask_in_same_order(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    prepared = module._prepare_centralized(inputs, inputs["agent_mask"])
    expected = (inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)).flatten(-2)
    torch.testing.assert_close(prepared["__flex_marl_sequence_mask"], expected)


def test_centralized_sequence_indices_retain_owning_agent(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    idx = module._prepare_centralized(inputs, inputs["agent_mask"])["__flex_marl_sequence_agent_idx"]
    expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    torch.testing.assert_close(idx[0, :, 0], expected)


def test_centralized_fields_may_have_different_sequence_lengths(make_config, sequential_field, make_inputs) -> None:
    other = replace(sequential_field, key="other", mask_key="other_mask")
    config = replace(make_config(MultiAgentMode.CENTRALIZED), fields=(sequential_field, other))
    inputs = make_inputs()
    inputs["other"] = torch.randn(2, 3, 7, 3)
    inputs["other_mask"] = torch.ones(2, 3, 7, dtype=torch.bool)
    assert MultiAgentEncoderModule(config).eval()(inputs).shape == (2, 5)


def test_centralized_masked_agent_values_do_not_affect_global_output(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).eval()
    inputs = make_inputs()
    before = module(inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["flat"][~inputs["agent_mask"]] = 100_000
    inactive_sequence = (~inputs["agent_mask"]).unsqueeze(-1).expand_as(inputs["sequence_mask"])
    changed["sequence"][inactive_sequence] = -100_000
    torch.testing.assert_close(module(changed), before, atol=1e-6, rtol=1e-6)


def test_centralized_masked_elements_do_not_affect_global_output(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).eval()
    inputs = make_inputs()
    before = module(inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["sequence"][~inputs["sequence_mask"]] = 100_000
    torch.testing.assert_close(module(changed), before, atol=1e-6, rtol=1e-6)


def test_centralized_active_agent_values_can_affect_global_output(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).eval()
    inputs = make_inputs()
    before = module(inputs)
    inputs["flat"][:, 0] += 100
    assert not torch.allclose(module(inputs), before)


def test_centralized_all_agents_inactive_produces_finite_output(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED))
    inputs = make_inputs()
    inputs["agent_mask"][:] = False
    inputs["flat"].requires_grad_()
    inputs["sequence"].requires_grad_()
    output = module(inputs)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs["flat"].grad).all() and torch.isfinite(inputs["sequence"].grad).all()


def test_centralized_broadcast_shape_matches_fixed_slots(make_config, make_inputs) -> None:
    config = replace(make_config(MultiAgentMode.CENTRALIZED), centralized_output=CentralizedOutput.BROADCAST)
    assert MultiAgentEncoderModule(config).eval()(make_inputs()).shape == (2, 3, 5)


def test_centralized_broadcast_repeats_exact_global_vector(make_config, make_inputs) -> None:
    global_module = MultiAgentEncoderModule(make_config(MultiAgentMode.CENTRALIZED)).eval()
    broadcast_config = replace(
        make_config(MultiAgentMode.CENTRALIZED), centralized_output=CentralizedOutput.BROADCAST
    )
    broadcast_module = MultiAgentEncoderModule(broadcast_config).eval()
    broadcast_module.load_state_dict(global_module.state_dict())
    inputs = make_inputs()
    global_output = global_module(inputs)
    broadcast = broadcast_module(inputs)
    for agent in range(3):
        torch.testing.assert_close(broadcast[:, agent], global_output)


def test_centralized_broadcast_invokes_encoder_once(make_config, make_inputs) -> None:
    config = replace(make_config(MultiAgentMode.CENTRALIZED), centralized_output=CentralizedOutput.BROADCAST)
    module = MultiAgentEncoderModule(config)
    counter = CountingEncoder(config.output_dim)
    module.encoder = counter
    module(make_inputs())
    assert counter.calls == 1


def test_centralized_broadcast_is_not_masked_per_agent(make_config, make_inputs) -> None:
    config = replace(make_config(MultiAgentMode.CENTRALIZED), centralized_output=CentralizedOutput.BROADCAST)
    module = MultiAgentEncoderModule(config).eval()
    output = module(make_inputs())
    torch.testing.assert_close(output[:, 0], output[:, 1])
    torch.testing.assert_close(output[:, 1], output[:, 2])
