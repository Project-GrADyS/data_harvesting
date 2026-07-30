from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def kaz_config(monkeypatch) -> dict:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    params = Path(__file__).resolve().parents[1] / "params.yaml"
    config = yaml.safe_load(params.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["environment"].update(
        {
            "num_archers": 1,
            "num_knights": 1,
            "max_zombies": 2,
            "max_arrows": 2,
            "max_cycles": 8,
            "spawn_rate": 2,
        }
    )
    config["actor"].update({"entity_embed_dim": 8, "role_embed_dim": 4})
    config["critic"].update(
        {"entity_embed_dim": 8, "role_embed_dim": 4, "action_embed_dim": 4}
    )
    config["encoder"].update(
        {
            "num_heads": 2,
            "ff_dim": 16,
            "depth": 1,
            "mix_layer_depth": 1,
            "mix_layer_num_cells": 16,
        }
    )
    config["training"].update(
        {
            "total_timesteps": 16,
            "warmup_steps": 4,
            "batch_size": 4,
            "device": "cpu",
            "exploration_annealing_steps": 8,
        }
    )
    config["collector"].update(
        {"num_collectors": 1, "frames_per_batch": 8, "device": "cpu"}
    )
    config["replay_buffer"].update(
        {"buffer_size": 64, "prefetch": 0, "device": "cpu"}
    )
    config["optimization"].update(
        {"updates_per_batch": 1, "grad_clip": 1.0}
    )
    config["evaluation"].update({"num_episodes": 1, "eval_every_n_steps": 1000})
    config["checkpoint"].update(
        {"enabled": False, "save_final_model": False}
    )
    return config
