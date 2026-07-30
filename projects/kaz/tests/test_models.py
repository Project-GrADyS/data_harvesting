from __future__ import annotations

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from kaz_training.environment import make_env
from kaz_training.models import create_actor, create_critic, create_exploratory_actor


def test_actor_outputs_hard_one_hot_actions_with_gradients(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        actor = create_actor(env, kaz_config, torch.device("cpu"))
        reset = env.reset()
        output = actor(reset.clone())
        action = output["agents", "action"]

        assert torch.all(action.sum(dim=-1) == 1)
        assert torch.all((action == 0) | (action == 1))
        action[..., 0].sum().backward()
        assert any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in actor.parameters()
        )
    finally:
        env.close()


def test_actor_masks_inactive_agents_and_critic_returns_team_value(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        actor = create_actor(env, kaz_config, torch.device("cpu"))
        exploratory_actor, _ = create_exploratory_actor(
            actor, env, kaz_config, torch.device("cpu")
        )
        critic = create_critic(env, kaz_config, torch.device("cpu"))
        reset = env.reset()
        reset["agents", "mask"][1] = False
        with set_exploration_type(ExplorationType.RANDOM):
            acted = exploratory_actor(reset)
        assert torch.all(acted["agents", "action"][1] == 0)
        valued = critic(acted)
        assert valued["state_action_value"].shape == torch.Size([1])
        assert torch.isfinite(valued["state_action_value"]).all()
    finally:
        env.close()
