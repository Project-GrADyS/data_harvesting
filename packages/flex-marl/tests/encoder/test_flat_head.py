from dataclasses import replace

import pytest
import torch
from torch import nn

from flex_marl.encoder.heads import FlatHead


@pytest.mark.parametrize("shape,expected", [((5,), (7,)), ((4, 5), (4, 7)), ((2, 3, 5), (2, 3, 7))])
def test_flat_head_preserves_batch_dimensions(flat_config, shape, expected):
    assert FlatHead(flat_config)(torch.randn(shape)).shape == expected


def test_flat_head_uses_configured_architecture(flat_config):
    head = FlatHead(flat_config)
    assert head.mlp(torch.randn(2, flat_config.input_size)).shape[-1] == flat_config.output_size
    assert any(isinstance(module, nn.ReLU) for module in head.modules())


def test_flat_head_is_deterministic_for_fixed_input(flat_config):
    head, x = FlatHead(flat_config).eval(), torch.randn(3, 5)
    torch.testing.assert_close(head(x), head(x))


def test_flat_head_supports_backward(flat_config):
    head, x = FlatHead(flat_config), torch.randn(3, 5, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_flat_head_rejects_wrong_feature_size(flat_config):
    with pytest.raises(RuntimeError):
        FlatHead(flat_config)(torch.randn(3, 4))


def test_flat_head_supports_noncontiguous_input(flat_config):
    head = FlatHead(flat_config).eval()
    x = torch.randn(4, 3, 5).transpose(0, 1)
    assert not x.is_contiguous()
    torch.testing.assert_close(head(x), head(x.contiguous()))


def test_flat_head_honors_requested_device(flat_config):
    head = FlatHead(flat_config, device="cpu")
    output = head(torch.randn(2, 5))
    assert output.device.type == "cpu" and all(p.device.type == "cpu" for p in head.parameters())


def test_flat_head_honors_supported_dtype(flat_config):
    head = FlatHead(flat_config).double()
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float64


def test_flat_head_constructor_validates_config(flat_config):
    with pytest.raises(ValueError, match="input_size"):
        FlatHead(replace(flat_config, input_size=0))


def test_flat_head_rejects_sequential_only_arguments(flat_config):
    head = FlatHead(flat_config)

    with pytest.raises(TypeError):
        head(torch.randn(2, 5), torch.ones(2, dtype=torch.bool))
