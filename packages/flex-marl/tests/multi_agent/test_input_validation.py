from __future__ import annotations

import pytest
import torch

from flex_marl.multi_agent import MultiAgentEncoderModule, MultiAgentMode


@pytest.fixture
def module(make_config):
    return MultiAgentEncoderModule(make_config(MultiAgentMode.SHARED)).eval()


def test_forward_reports_all_missing_required_keys(module) -> None:
    with pytest.raises(KeyError) as error:
        module({})
    message = str(error.value)
    assert all(key in message for key in ("agent_mask", "flat", "sequence", "sequence_mask"))
    assert message.index("agent_mask") < message.index("flat") < message.index("sequence")


def test_plural_pre_forward_checks_method_enforces_input_contract(module) -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        module._pre_forward_checks({})


def test_forward_ignores_unrelated_input_keys(module, make_inputs) -> None:
    inputs = make_inputs()
    expected = module(inputs)
    actual = module({**inputs, "unused": torch.randn(9)})
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("value", [[True, True, True], 1, None])
def test_agent_mask_must_be_tensor(value, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["agent_mask"] = value
    with pytest.raises(TypeError, match="Agent mask.*torch.Tensor"):
        module(inputs)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_agent_mask_must_be_boolean(dtype, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["agent_mask"] = torch.ones(2, 3, dtype=dtype)
    with pytest.raises(TypeError, match="boolean"):
        module(inputs)


def test_agent_mask_requires_agent_axis(module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["agent_mask"] = torch.tensor(True)
    with pytest.raises(ValueError, match=r"\(\*B, 3\)"):
        module(inputs)


@pytest.mark.parametrize("slots", [2, 4])
def test_agent_mask_requires_configured_slot_count(slots, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["agent_mask"] = torch.ones(2, slots, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"\(\*B, 3\)"):
        module(inputs)


@pytest.mark.parametrize("value", [[[1.0, 2.0]], 1, None])
def test_flat_field_must_be_tensor(value, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["flat"] = value
    with pytest.raises(TypeError, match="'flat'.*torch.Tensor"):
        module(inputs)


@pytest.mark.parametrize("shape", [(3, 2), (2, 2, 2), (2, 3, 3), (2, 3, 2, 1), (1, 2, 3, 2)])
def test_flat_field_requires_exact_batch_agent_and_feature_shape(shape, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["flat"] = torch.randn(shape)
    with pytest.raises(ValueError, match="Flat field 'flat'.*shape"):
        module(inputs)


@pytest.mark.parametrize("value", [[], 1, None])
def test_sequential_field_must_be_tensor(value, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence"] = value
    with pytest.raises(TypeError, match="'sequence'.*torch.Tensor"):
        module(inputs)


@pytest.mark.parametrize("shape", [(2, 4, 3), (2, 3, 4, 3, 1), (1, 2, 3, 4, 3)])
def test_sequential_field_rejects_wrong_rank_or_batch_shape(shape, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence"] = torch.randn(shape)
    with pytest.raises(ValueError, match="Sequential field 'sequence'.*shape"):
        module(inputs)


@pytest.mark.parametrize("slots", [2, 4])
def test_sequential_field_rejects_wrong_agent_count(slots, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence"] = torch.randn(2, slots, 4, 3)
    with pytest.raises(ValueError, match="Sequential field 'sequence'.*shape"):
        module(inputs)


def test_sequential_field_rejects_empty_sequence(module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence"] = torch.randn(2, 3, 0, 3)
    inputs["sequence_mask"] = torch.ones(2, 3, 0, dtype=torch.bool)
    with pytest.raises(ValueError, match="non-empty sequence"):
        module(inputs)


def test_sequential_field_rejects_wrong_feature_width(module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence"] = torch.randn(2, 3, 4, 2)
    with pytest.raises(ValueError, match="last dimension 3"):
        module(inputs)


@pytest.mark.parametrize("value", [[], 1, None])
def test_sequence_mask_must_be_tensor(value, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence_mask"] = value
    with pytest.raises(TypeError, match="Sequence mask.*'sequence'.*torch.Tensor"):
        module(inputs)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_sequence_mask_must_be_boolean(dtype, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence_mask"] = torch.ones(2, 3, 4, dtype=dtype)
    with pytest.raises(TypeError, match="boolean"):
        module(inputs)


@pytest.mark.parametrize("shape", [(2, 3, 3), (2, 2, 4), (1, 2, 3, 4), (2, 3, 4, 1)])
def test_sequence_mask_must_match_value_prefix_exactly(shape, module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["sequence_mask"] = torch.ones(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="Sequence mask.*shape"):
        module(inputs)


def test_multiple_fields_must_share_agent_mask_batch_shape(module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["flat"] = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="Flat field 'flat'.*shape"):
        module(inputs)


def test_validation_does_not_mutate_inputs(module, make_inputs) -> None:
    inputs = make_inputs()
    keys = tuple(inputs)
    copies = {key: value.clone() for key, value in inputs.items()}
    module(inputs)
    assert tuple(inputs) == keys
    for key, expected in copies.items():
        torch.testing.assert_close(inputs[key], expected)


def test_failing_validation_does_not_mutate_inputs(module, make_inputs) -> None:
    inputs = make_inputs()
    inputs["flat"] = torch.randn(2, 2, 2)
    keys = tuple(inputs)
    copies = {key: value.clone() for key, value in inputs.items()}
    with pytest.raises(ValueError):
        module(inputs)
    assert tuple(inputs) == keys
    for key, expected in copies.items():
        torch.testing.assert_close(inputs[key], expected)
