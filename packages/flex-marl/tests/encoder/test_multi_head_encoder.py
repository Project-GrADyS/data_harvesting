from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.encoder import FlatHeadConfig, MultiHeadEncoderModule, PositionalEncodingConfig, SequentialHeadConfig
from flex_marl.encoder.heads import FlatHead, SequentialHead


def make_encoder(configs, **kwargs):
    return MultiHeadEncoderModule(
        configs,
        mix_layer_depth=kwargs.pop("mix_layer_depth", 1),
        mix_layer_num_cells=kwargs.pop("mix_layer_num_cells", 12),
        mix_activation_class=kwargs.pop("mix_activation_class", None),
        output_dim=kwargs.pop("output_dim", 6),
        **kwargs,
    )


class RecordingHead(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.args = None

    def forward(self, *args):
        self.args = args
        return self.output


class CaptureMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x.clone()
        return x


def test_multi_head_builds_flat_head_for_flat_config(flat_config):
    assert isinstance(make_encoder([flat_config]).heads[flat_config.key], FlatHead)


def test_multi_head_builds_sequential_head_for_sequential_config(sequential_config):
    assert isinstance(make_encoder([sequential_config]).heads[sequential_config.key], SequentialHead)


def test_multi_head_builds_mixed_head_types(flat_config, sequential_config):
    heads = make_encoder([flat_config, sequential_config]).heads
    assert isinstance(heads["flat"], FlatHead) and isinstance(heads["sequence"], SequentialHead)


def test_multi_head_rejects_duplicate_keys(flat_config):
    with pytest.raises(ValueError, match="Duplicate"):
        make_encoder([flat_config, replace(flat_config, input_size=3)])


def test_multi_head_rejects_unsupported_config_type():
    with pytest.raises(TypeError, match="head_config"):
        make_encoder([object()])


def test_multi_head_rejects_empty_head_list():
    with pytest.raises(ValueError, match="at least one"):
        make_encoder([])


def test_multi_head_mix_input_size_is_sum_of_head_outputs():
    configs = [
        FlatHeadConfig(key="a", input_size=2, output_size=5),
        FlatHeadConfig(key="b", input_size=3, output_size=7),
        FlatHeadConfig(key="c", input_size=4, output_size=11),
    ]
    encoder = make_encoder(configs)
    assert encoder.mix_layer(torch.randn(23)).shape == (6,)


def test_multi_head_uses_default_tanh_mix_activation(flat_config):
    assert any(isinstance(module, nn.Tanh) for module in make_encoder([flat_config]).mix_layer.modules())


def test_multi_head_uses_custom_mix_activation(flat_config):
    assert any(
        isinstance(module, nn.ReLU)
        for module in make_encoder([flat_config], mix_activation_class=nn.ReLU).mix_layer.modules()
    )


@pytest.mark.parametrize("field", ["output_dim", "mix_layer_depth", "mix_layer_num_cells"])
@pytest.mark.parametrize("value", [0, -1])
def test_multi_head_rejects_non_positive_dimensions(flat_config, field, value):
    with pytest.raises(ValueError, match=field):
        make_encoder([flat_config], **{field: value})


def test_multi_head_rejects_invalid_mix_activation(flat_config):
    with pytest.raises(ValueError, match="mix_activation_class"):
        make_encoder([flat_config], mix_activation_class="relu")


def test_multi_head_has_no_persistent_forward_scratch_buffer(flat_config):
    encoder = make_encoder([flat_config])
    assert "head_output_buffer" not in dict(encoder.named_buffers())
    assert "head_output_buffer" not in encoder.state_dict()


def test_multi_head_constructor_runs_config_validation(flat_config):
    with pytest.raises(ValueError, match="input_size"):
        make_encoder([replace(flat_config, input_size=0)])


def test_multi_head_routes_observation_and_mask_to_matching_heads(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config])
    flat_out, seq_out = torch.randn(7), torch.randn(8)
    flat_rec, seq_rec = RecordingHead(flat_out), RecordingHead(seq_out)
    encoder.heads["flat"], encoder.heads["sequence"] = flat_rec, seq_rec
    encoder.mix_layer = CaptureMix()
    data = {"flat": torch.randn(5), "sequence": torch.randn(4, 3), "sequence_mask": torch.ones(4, dtype=torch.bool)}
    encoder(data)
    assert flat_rec.args[0] is data["flat"] and flat_rec.args[1:] == (None, None)
    assert seq_rec.args[0] is data["sequence"] and seq_rec.args[1] is data["sequence_mask"]


def test_multi_head_routes_idx_to_configured_sequential_head(positional_config):
    encoder, rec = make_encoder([positional_config]), RecordingHead(torch.randn(8))
    encoder.heads["sequence"], encoder.mix_layer = rec, CaptureMix()
    data = {
        "sequence": torch.randn(4, 3),
        "sequence_mask": torch.ones(4, dtype=torch.bool),
        "position": torch.tensor([1]),
    }
    encoder(data)
    assert rec.args[2] is data["position"]


def test_multi_head_without_positional_encoding_requires_no_idx_key(sequential_config):
    output = make_encoder([sequential_config]).eval()(
        {"sequence": torch.randn(4, 3), "sequence_mask": torch.ones(4, dtype=torch.bool)}
    )
    assert output.shape == (6,)


def test_multi_head_concatenates_outputs_in_config_order(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config])
    encoder.heads["flat"] = RecordingHead(torch.arange(7.0))
    encoder.heads["sequence"] = RecordingHead(torch.arange(8.0) + 10)
    capture = CaptureMix()
    encoder.mix_layer = capture
    encoder({"flat": torch.randn(5), "sequence": torch.randn(4, 3), "sequence_mask": torch.ones(4, dtype=torch.bool)})
    torch.testing.assert_close(capture.input, torch.cat((torch.arange(7.0), torch.arange(8.0) + 10)))


def test_multi_head_order_does_not_depend_on_input_dict_order(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config]).eval()
    flat, seq, mask = torch.randn(5), torch.randn(4, 3), torch.ones(4, dtype=torch.bool)
    first = {"flat": flat, "sequence": seq, "sequence_mask": mask}
    second = {"sequence_mask": mask, "sequence": seq, "flat": flat}
    torch.testing.assert_close(encoder(first), encoder(second))


@pytest.mark.parametrize(
    "batch_shape",
    [(), (4,), (2, 3)],
)
def test_multi_head_preserves_batch_dimensions(flat_config, sequential_config, batch_shape):
    encoder = make_encoder([flat_config, sequential_config]).eval()
    data = {
        "flat": torch.randn(*batch_shape, 5),
        "sequence": torch.randn(*batch_shape, 4, 3),
        "sequence_mask": torch.ones(*batch_shape, 4, dtype=torch.bool),
    }
    assert encoder(data).shape == (*batch_shape, 6)


def test_multi_head_ignores_unrelated_input_keys(flat_config):
    encoder, value = make_encoder([flat_config]).eval(), torch.randn(5)
    torch.testing.assert_close(encoder({"flat": value}), encoder({"flat": value, "unused": torch.randn(9)}))


def test_multi_head_rejects_missing_observation_key(flat_config):
    with pytest.raises(KeyError, match="flat"):
        make_encoder([flat_config])({})


def test_multi_head_rejects_missing_sequential_mask_key(sequential_config):
    with pytest.raises(KeyError, match="sequence_mask"):
        make_encoder([sequential_config])({"sequence": torch.randn(4, 3)})


def test_multi_head_rejects_missing_positional_idx_key(positional_config):
    with pytest.raises(KeyError, match="position"):
        make_encoder([positional_config])(
            {"sequence": torch.randn(4, 3), "sequence_mask": torch.ones(4, dtype=torch.bool)}
        )


def test_multi_head_rejects_inconsistent_batch_shapes(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config])
    with pytest.raises(ValueError, match="batch shape"):
        encoder(
            {
                "flat": torch.randn(2, 5),
                "sequence": torch.randn(3, 4, 3),
                "sequence_mask": torch.ones(3, 4, dtype=torch.bool),
            }
        )


def test_multi_head_supports_different_sequence_lengths(sequential_config):
    other = replace(sequential_config, key="other", mask_key="other_mask")
    encoder = make_encoder([sequential_config, other]).eval()
    output = encoder(
        {
            "sequence": torch.randn(2, 4, 3),
            "sequence_mask": torch.ones(2, 4, dtype=torch.bool),
            "other": torch.randn(2, 7, 3),
            "other_mask": torch.ones(2, 7, dtype=torch.bool),
        }
    )
    assert output.shape == (2, 6)


def test_multi_head_repeated_forward_does_not_leak_previous_values(flat_config):
    encoder = make_encoder([flat_config]).eval()
    first, second = torch.randn(3, 5), torch.randn(3, 5)
    encoder({"flat": first})
    expected = encoder({"flat": second})
    torch.testing.assert_close(expected, encoder({"flat": second}))


def test_multi_head_forward_does_not_mutate_inputs(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config]).eval()
    data = {
        "flat": torch.randn(2, 5),
        "sequence": torch.randn(2, 4, 3),
        "sequence_mask": torch.tensor([[True, False, True, True], [True] * 4]),
    }
    copies = {key: value.clone() for key, value in data.items()}
    encoder(data)
    for key in data:
        torch.testing.assert_close(data[key], copies[key])


def test_multi_head_supports_end_to_end_backward(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config])
    flat, seq = torch.randn(2, 5, requires_grad=True), torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [True] * 4])
    encoder({"flat": flat, "sequence": seq, "sequence_mask": mask}).sum().backward()
    assert flat.grad is not None and seq.grad is not None
    torch.testing.assert_close(seq.grad[~mask], torch.zeros_like(seq.grad[~mask]))
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_state_dict_round_trip_preserves_output(flat_config, sequential_config):
    first, second = make_encoder([flat_config, sequential_config]).eval(), make_encoder([flat_config, sequential_config]).eval()
    second.load_state_dict(first.state_dict())
    data = {
        "flat": torch.randn(2, 5),
        "sequence": torch.randn(2, 4, 3),
        "sequence_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    torch.testing.assert_close(first(data), second(data))


def test_module_to_device_moves_all_parameters_and_output(flat_config):
    encoder = make_encoder([flat_config], device="cpu")
    output = encoder({"flat": torch.randn(2, 5)})
    assert output.device.type == "cpu" and all(p.device.type == "cpu" for p in encoder.parameters())


def test_module_supports_float64_on_cpu(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config]).double()
    flat = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    seq = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    output = encoder({"flat": flat, "sequence": seq, "sequence_mask": torch.ones(2, 4, dtype=torch.bool)})
    output.sum().backward()
    assert output.dtype == torch.float64 and flat.grad is not None and seq.grad is not None


def test_eval_output_is_repeatable_after_multiple_forwards(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config]).eval()
    data = {
        "flat": torch.randn(2, 5),
        "sequence": torch.randn(2, 4, 3),
        "sequence_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    expected = encoder(data)
    encoder({"flat": torch.randn(2, 5), "sequence": torch.randn(2, 4, 3), "sequence_mask": data["sequence_mask"]})
    torch.testing.assert_close(expected, encoder(data))


def test_encoder_output_contains_no_nan_or_inf_for_valid_inputs(flat_config, sequential_config):
    encoder = make_encoder([flat_config, sequential_config])
    flat, seq = torch.randn(2, 5, requires_grad=True), torch.randn(2, 4, 3, requires_grad=True)
    output = encoder(
        {"flat": flat, "sequence": seq, "sequence_mask": torch.tensor([[True, False, True, True], [False] * 4])}
    )
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(flat.grad).all() and torch.isfinite(seq.grad).all()
