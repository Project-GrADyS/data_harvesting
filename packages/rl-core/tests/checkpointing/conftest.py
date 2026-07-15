from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from rl_core.checkpointing import Checkpoint


@pytest.fixture
def checkpoint_factory():
    def make_checkpoint(
        step: int = 0,
        value: float | None = None,
        *,
        state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Checkpoint:
        value = float(step) if value is None else value
        return Checkpoint(
            step=step,
            state=state if state is not None else {"policy": {"weight": torch.tensor(value)}},
            metadata=metadata if metadata is not None else {"algorithm": "test"},
        )

    return make_checkpoint
