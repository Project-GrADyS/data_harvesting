import torch

from data_harvesting.encoder import (
    SequentialEncoder,
    SequentialEncoderConfig,
    SequentialEncoderInput,
)


def _make_sequential_head(*, agentic_encoding: bool = False) -> SequentialEncoder:
    return SequentialEncoder(
        input=SequentialEncoderInput(key="drones", input_size=2),
        config=SequentialEncoderConfig(
            embed_dim=16,
            head_dim=8,
            num_heads=2,
            ff_dim=32,
            depth=1,
            dropout=0.0,
            max_num_agents=4,
            agentic_encoding=agentic_encoding,
        ),
        device=torch.device("cpu"),
    )


def test_sequential_head_mask_argument_changes_forward_when_elements_masked() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 5, 2)
    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out_none = head(x, agent_idx, None)
    out_masked = head(x, agent_idx, mask)
    out_all_true = head(x, agent_idx, torch.ones_like(mask))

    assert not torch.allclose(out_none, out_masked, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_none, out_all_true, atol=1e-6, rtol=1e-6)


def test_sequential_head_mask_argument_changes_backward_when_elements_masked() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x_nomask = torch.randn(2, 5, 2, requires_grad=True)
    x_masked = x_nomask.detach().clone().requires_grad_(True)

    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out_nomask = head(x_nomask, agent_idx, None)
    out_masked = head(x_masked, agent_idx, mask)

    out_nomask.sum().backward()
    out_masked.sum().backward()

    assert x_nomask.grad is not None
    assert x_masked.grad is not None
    assert torch.allclose(x_masked.grad[~mask], torch.zeros_like(x_masked.grad[~mask]), atol=1e-8, rtol=0)
    assert torch.max(torch.abs(x_nomask.grad[~mask])) > 0


def test_sequential_head_masked_positions_receive_zero_gradients() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 5, 2, requires_grad=True)
    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out = head(x, agent_idx, mask)
    out.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad[~mask], torch.zeros_like(x.grad[~mask]), atol=1e-8, rtol=0)
    assert torch.count_nonzero(x.grad[mask]) > 0


def test_sequential_head_output_shape_preserves_single_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(4, 5, 2)
    agent_idx = torch.zeros((4, 1), dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (4, 16)
    assert out.device == x.device


def test_sequential_head_output_shape_preserves_multi_batch_dims() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 3, 5, 2)
    agent_idx = torch.zeros((2, 3, 1), dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (2, 3, 16)
    assert out.device == x.device


def test_sequential_head_output_shape_preserves_no_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(5, 2)
    agent_idx = torch.tensor([0], dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (16,)
    assert out.device == x.device


def test_sequential_head_agentic_encoding_changes_output_by_agent() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head(agentic_encoding=True)

    x = torch.randn(2, 5, 2)
    x[1] = x[0]
    mask = torch.ones((2, 5), dtype=torch.bool)
    agent_idx = torch.tensor([[0], [1]], dtype=torch.long)

    out = head(x, agent_idx, mask)

    assert not torch.allclose(out[0], out[1], atol=1e-6, rtol=1e-6)