from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.encoder.heads import SequentialHead


class CaptureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.mask = None

    def forward(self, x, src_key_padding_mask=None):
        self.input = x.detach().clone()
        self.mask = src_key_padding_mask.detach().clone()
        return x


class MaskedValueTransformer(nn.Module):
    def forward(self, x, src_key_padding_mask=None):
        result = torch.full_like(x, 3.0)
        result[src_key_padding_mask] = 1000.0
        return result


class ZeroProjection(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.zeros((*x.shape[:-1], self.size), device=x.device, dtype=x.dtype)


@pytest.mark.parametrize(
    "x_shape,mask_shape,expected",
    [((5, 3), (5,), (8,)), ((4, 5, 3), (4, 5), (4, 8)), ((2, 4, 5, 3), (2, 4, 5), (2, 4, 8))],
)
def test_sequential_head_preserves_batch_dimensions(sequential_config, x_shape, mask_shape, expected):
    head = SequentialHead(sequential_config).eval()
    output = head(torch.randn(x_shape), torch.ones(mask_shape, dtype=torch.bool))
    assert output.shape == expected and torch.isfinite(output).all()


def test_sequential_head_builds_transformer_with_output_size_as_d_model(sequential_config):
    head = SequentialHead(sequential_config)
    layer = head.transformer.layers[0]
    assert head.encoder.out_features == sequential_config.output_size
    assert layer.self_attn.embed_dim == sequential_config.output_size


def test_sequential_head_is_deterministic_in_eval_mode(sequential_config):
    config = replace(sequential_config, dropout=0.5)
    head, x, mask = SequentialHead(config).eval(), torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool)
    torch.testing.assert_close(head(x, mask), head(x, mask))


def test_sequential_head_supports_backward(sequential_config):
    head, x = SequentialHead(sequential_config), torch.randn(2, 5, 3, requires_grad=True)
    head(x, torch.ones(2, 5, dtype=torch.bool)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_sequential_head_rejects_input_with_fewer_than_two_dimensions(sequential_config):
    with pytest.raises(ValueError, match="at least 2"):
        SequentialHead(sequential_config)(torch.randn(3), torch.ones(1, dtype=torch.bool))


def test_sequential_head_rejects_wrong_input_feature_size(sequential_config):
    with pytest.raises(ValueError, match="input_size"):
        SequentialHead(sequential_config)(torch.randn(5, 4), torch.ones(5, dtype=torch.bool))


def test_sequential_head_rejects_empty_sequence(sequential_config):
    with pytest.raises(ValueError, match="at least one timestep"):
        SequentialHead(sequential_config)(torch.randn(2, 0, 3), torch.ones(2, 0, dtype=torch.bool))


def test_sequential_head_supports_noncontiguous_input(sequential_config):
    head = SequentialHead(sequential_config).eval()
    x = torch.randn(4, 2, 5, 3).transpose(0, 1)
    mask = torch.ones(x.shape[:-1], dtype=torch.bool)
    torch.testing.assert_close(head(x, mask), head(x.contiguous(), mask.contiguous()))


def test_sequential_head_requires_mask_argument(sequential_config):
    with pytest.raises(TypeError):
        SequentialHead(sequential_config)(torch.randn(2, 5, 3))


def test_sequential_head_rejects_none_mask(sequential_config):
    with pytest.raises(ValueError, match="mask is required"):
        SequentialHead(sequential_config)(torch.randn(2, 5, 3), None)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_sequential_head_rejects_non_boolean_mask(sequential_config, dtype):
    with pytest.raises(TypeError, match="boolean"):
        SequentialHead(sequential_config)(torch.randn(2, 5, 3), torch.ones(2, 5, dtype=dtype))


@pytest.mark.parametrize("shape", [(2, 4), (2, 5, 1), (5,)])
def test_sequential_head_rejects_wrong_mask_shape(sequential_config, shape):
    with pytest.raises(ValueError, match="Mask tensor"):
        SequentialHead(sequential_config)(torch.randn(2, 5, 3), torch.ones(shape, dtype=torch.bool))


def test_sequential_head_all_true_mask_keeps_every_timestep(sequential_config):
    head, capture = SequentialHead(sequential_config), CaptureTransformer()
    head.transformer = capture
    head(torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool))
    assert not capture.mask.any()


def test_sequential_head_inverts_validity_mask_for_transformer(sequential_config):
    head, capture = SequentialHead(sequential_config), CaptureTransformer()
    head.transformer = capture
    valid = torch.tensor([[True, False, True]])
    head(torch.randn(1, 3, 3), valid)
    torch.testing.assert_close(capture.mask, ~valid)


def test_sequential_head_false_values_do_not_affect_output(sequential_config):
    head = SequentialHead(sequential_config).eval()
    valid = torch.tensor([[True, False, True, False, True], [False, True, True, False, True]])
    x1 = torch.randn(2, 5, 3)
    x2 = x1.clone()
    x2[~valid] = 10_000
    torch.testing.assert_close(head(x1, valid), head(x2, valid), atol=1e-6, rtol=1e-6)


def test_sequential_head_true_values_affect_output(sequential_config):
    head = SequentialHead(sequential_config).eval()
    valid = torch.ones(1, 5, dtype=torch.bool)
    x1, x2 = torch.zeros(1, 5, 3), torch.zeros(1, 5, 3)
    x2[0, 0] = 10
    assert not torch.allclose(head(x1, valid), head(x2, valid))


def test_sequential_head_pooling_averages_only_true_positions(sequential_config):
    head = SequentialHead(sequential_config)
    head.encoder = ZeroProjection(sequential_config.output_size)
    head.transformer = MaskedValueTransformer()
    valid = torch.tensor([[True, False, True], [False, True, True]])
    output = head(torch.randn(2, 3, 3), valid)
    torch.testing.assert_close(output, torch.full_like(output, 3.0))


def test_sequential_head_false_positions_receive_zero_gradient(sequential_config):
    head = SequentialHead(sequential_config)
    x = torch.randn(2, 5, 3, requires_grad=True)
    valid = torch.tensor([[True, False, True, False, True], [False, True, True, False, True]])
    head(x, valid).sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad[~valid], torch.zeros_like(x.grad[~valid]))
    assert torch.count_nonzero(x.grad[valid]) > 0


def test_sequential_head_all_false_mask_returns_zero(sequential_config):
    output = SequentialHead(sequential_config).eval()(torch.randn(3, 5, 3), torch.zeros(3, 5, dtype=torch.bool))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_sequential_head_mixed_fully_masked_batch_is_isolated(sequential_config):
    head, x = SequentialHead(sequential_config).eval(), torch.randn(3, 5, 3)
    valid = torch.tensor([[False] * 5, [True, False, True, False, True], [True] * 5])
    output = head(x, valid)
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    assert torch.isfinite(output).all() and torch.count_nonzero(output[1:]) > 0


def test_sequential_head_masked_values_do_not_change_other_batch_items(sequential_config):
    head, x = SequentialHead(sequential_config).eval(), torch.randn(2, 5, 3)
    valid = torch.tensor([[True, False, True, False, True], [True] * 5])
    before = head(x, valid)
    x[0, ~valid[0]] = 10000
    after = head(x, valid)
    torch.testing.assert_close(before, after)


def test_sequential_head_without_positional_config_builds_no_embedding(sequential_config):
    assert SequentialHead(sequential_config).positional_encoder is None


def test_positional_embedding_has_configured_shape(positional_config):
    embedding = SequentialHead(positional_config).positional_encoder
    assert embedding is not None
    assert (embedding.num_embeddings, embedding.embedding_dim) == (4, positional_config.output_size)


def test_positional_encoding_requires_idx(positional_config):
    with pytest.raises(ValueError, match="no positional index"):
        SequentialHead(positional_config)(torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool))


@pytest.mark.parametrize("value", [0, 3])
def test_positional_encoding_accepts_boundary_indices(positional_config, value):
    output = SequentialHead(positional_config).eval()(
        torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool), torch.full((2, 5, 1), value)
    )
    assert output.shape == (2, 8) and torch.isfinite(output).all()


@pytest.mark.parametrize("value", [-1, 4])
def test_positional_encoding_rejects_out_of_range_index(positional_config, value):
    with pytest.raises(ValueError, match="range"):
        SequentialHead(positional_config)(
            torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool), torch.full((2, 5, 1), value)
        )


def test_positional_encoding_rejects_non_integer_idx(positional_config):
    with pytest.raises(TypeError, match="integer dtype"):
        SequentialHead(positional_config)(
            torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool), torch.full((2, 5, 1), 1.5)
        )


@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 5), (2, 4, 1), (2, 5, 1, 1)])
def test_positional_encoding_rejects_wrong_idx_shape(positional_config, shape):
    with pytest.raises(ValueError, match="must have shape"):
        SequentialHead(positional_config)(
            torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool), torch.zeros(shape, dtype=torch.long)
        )


@pytest.mark.parametrize(
    "x_shape,mask_shape,idx_shape,expected",
    [
        ((5, 3), (5,), (5, 1), (8,)),
        ((2, 5, 3), (2, 5), (2, 5, 1), (2, 8)),
        ((2, 4, 5, 3), (2, 4, 5), (2, 4, 5, 1), (2, 4, 8)),
    ],
)
def test_positional_encoding_preserves_batch_dimensions(positional_config, x_shape, mask_shape, idx_shape, expected):
    output = SequentialHead(positional_config).eval()(
        torch.randn(x_shape), torch.ones(mask_shape, dtype=torch.bool), torch.zeros(idx_shape, dtype=torch.long)
    )
    assert output.shape == expected


def test_positional_embedding_is_added_to_corresponding_sequence_element(positional_config):
    head, capture = SequentialHead(positional_config), CaptureTransformer()
    head.encoder = ZeroProjection(positional_config.output_size)
    head.transformer = capture
    with torch.no_grad():
        head.positional_encoder.weight.copy_(torch.arange(32).reshape(4, 8))
    idx = torch.tensor([[[0], [1], [2], [3]], [[3], [2], [1], [0]]])
    head(torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool), idx)
    expected = head.positional_encoder(idx.squeeze(-1))
    torch.testing.assert_close(capture.input, expected)


def test_different_indices_can_change_output(positional_config):
    head, x, valid = SequentialHead(positional_config).eval(), torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool)
    x[1] = x[0]
    idx = torch.stack((torch.zeros(5, 1, dtype=torch.long), torch.ones(5, 1, dtype=torch.long)))
    output = head(x, valid, idx)
    assert not torch.allclose(output[0], output[1])


def test_idx_is_ignored_when_positional_encoding_is_disabled(sequential_config):
    head, x, valid = SequentialHead(sequential_config).eval(), torch.randn(2, 5, 3), torch.ones(2, 5, dtype=torch.bool)
    idx = torch.stack((torch.zeros(5, 1, dtype=torch.long), torch.ones(5, 1, dtype=torch.long)))
    torch.testing.assert_close(head(x, valid), head(x, valid, idx))
