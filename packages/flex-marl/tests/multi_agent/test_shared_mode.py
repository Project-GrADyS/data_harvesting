from dataclasses import replace

import pytest
import torch

from flex_marl.multi_agent import MultiAgentEncoderModule, MultiAgentMode


def test_shared_mode_constructs_one_encoder(make_config) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    assert hasattr(module, "encoder")
    assert not hasattr(module, "encoders")


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_shared_mode_preserves_arbitrary_batch_dimensions(batch_shape, make_config, make_inputs) -> None:
    output = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).eval()(make_inputs(batch_shape))
    assert output.shape == (*batch_shape, 3, 5)
    assert torch.isfinite(output).all()


def test_shared_mode_passes_flat_values_without_reshaping(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs()
    prepared = module._prepare_shared(inputs, inputs["agent_mask"])
    assert prepared["flat"] is inputs["flat"]


def test_shared_mode_combines_agent_and_element_masks(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs()
    prepared = module._prepare_shared(inputs, inputs["agent_mask"])
    expected = inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)
    torch.testing.assert_close(prepared["__flex_marl_sequence_mask"], expected)


def test_shared_mode_builds_agent_indices_for_every_sequence_element(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs(batch_shape=(2, 2), sequence_length=3)
    idx = module._prepare_shared(inputs, inputs["agent_mask"])["__flex_marl_sequence_agent_idx"]
    assert idx.shape == (2, 2, 3, 3, 1)
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    torch.testing.assert_close(idx[0, 0, ..., 0].flatten(), expected)


def test_shared_mode_omits_agent_indices_when_identity_is_disabled(make_config, sequential_field) -> None:
    options = replace(sequential_field.sequential_options, encode_agent_identity=False)
    config = replace(
        make_config(MultiAgentMode.SHARED),
        fields=(replace(sequential_field, sequential_options=options),),
    )
    module = MultiAgentEncoderModule(config)
    inputs = {
        "sequence": torch.randn(2, 3, 4, 3),
        "sequence_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "agent_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    assert "__flex_marl_sequence_agent_idx" not in module._prepare_shared(inputs, inputs["agent_mask"])


def test_shared_mode_zeroes_inactive_outputs(make_config, make_inputs) -> None:
    inputs = make_inputs()
    output = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).eval()(inputs)
    torch.testing.assert_close(output[~inputs["agent_mask"]], torch.zeros_like(output[~inputs["agent_mask"]]))


def test_shared_mode_active_outputs_depend_only_on_their_own_agent_values(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).eval()
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    before = module(inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["flat"][:, 1] += 10_000
    changed["sequence"][:, 1] -= 10_000
    after = module(changed)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 2], after[:, 2])


def test_shared_mode_uses_same_parameters_for_identical_agents(make_config, make_inputs, sequential_field) -> None:
    options = replace(sequential_field.sequential_options, encode_agent_identity=False)
    config = replace(
        make_config(MultiAgentMode.SHARED),
        fields=(replace(sequential_field, sequential_options=options),),
    )
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    inputs["sequence_mask"][:] = True
    inputs["sequence"][:, 1] = inputs["sequence"][:, 0]
    output = MultiAgentEncoderModule(config).eval()(inputs)
    torch.testing.assert_close(output[:, 0], output[:, 1])


def test_shared_mode_agent_identity_can_distinguish_identical_sequences(make_config, make_inputs, sequential_field) -> None:
    config = replace(make_config(MultiAgentMode.SHARED), fields=(sequential_field,))
    module = MultiAgentEncoderModule(config).eval()
    inputs = make_inputs()
    inputs["agent_mask"][:] = True
    inputs["sequence_mask"][:] = True
    inputs["sequence"][:, 1] = inputs["sequence"][:, 0]
    with torch.no_grad():
        embedding = module.encoder.heads["sequence"].positional_encoder
        embedding.weight.copy_(torch.arange(embedding.weight.numel()).reshape_as(embedding.weight))
    output = module(inputs)
    assert not torch.allclose(output[:, 0], output[:, 1])


def test_shared_mode_masked_sequence_values_do_not_affect_output(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).eval()
    inputs = make_inputs()
    before = module(inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    effective = inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)
    changed["sequence"][~effective] = 100_000
    torch.testing.assert_close(module(changed), before, atol=1e-6, rtol=1e-6)


def test_shared_mode_inactive_inputs_receive_zero_gradient(make_config, make_inputs) -> None:
    module = MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED))
    inputs = make_inputs()
    inputs["flat"].requires_grad_()
    inputs["sequence"].requires_grad_()
    module(inputs).sum().backward()
    assert inputs["flat"].grad is not None and inputs["sequence"].grad is not None
    torch.testing.assert_close(
        inputs["flat"].grad[~inputs["agent_mask"]], torch.zeros_like(inputs["flat"].grad[~inputs["agent_mask"]])
    )
    effective = inputs["sequence_mask"] & inputs["agent_mask"].unsqueeze(-1)
    torch.testing.assert_close(
        inputs["sequence"].grad[~effective], torch.zeros_like(inputs["sequence"].grad[~effective])
    )
