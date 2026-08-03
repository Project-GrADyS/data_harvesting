from copy import deepcopy
from pathlib import Path

import torch
from torchrl.envs import check_env_specs

from sushigo.config import load_config, with_preset
from sushigo.environment import ACTION_KEY, MASK_KEY, make_env
from sushigo.policy import (
    build_q_policy,
    checkpoint_payload,
    load_checkpoint_policy,
)

PARAMS = Path(__file__).parents[1] / "params.yaml"


def test_mlp_and_flex_policies_select_legal_actions():
    base = load_config(PARAMS)
    for preset in ("fixed_2p", "variable_encoder_2_4"):
        config = with_preset(base, preset)
        environment = make_env(config)
        check_env_specs(environment)
        policy = build_q_policy(environment, config, device="cpu")
        tensordict = environment.reset()
        policy(tensordict)
        actions = tensordict.get(ACTION_KEY)
        masks = tensordict.get(MASK_KEY)
        assert masks.gather(-1, actions.unsqueeze(-1)).all()
        environment.close()


def test_checkpoint_round_trip(tmp_path):
    config = with_preset(load_config(PARAMS), "fixed_2p")
    environment = make_env(config)
    policy = build_q_policy(environment, config, device="cpu")
    policy(environment.reset())
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload(policy, config), path)

    loaded, loaded_config = load_checkpoint_policy(path)
    initial = environment.reset()
    tensordict = initial.clone()
    replay = initial.clone()
    policy(tensordict)
    loaded(replay)
    assert torch.equal(tensordict.get(ACTION_KEY), replay.get(ACTION_KEY))
    assert loaded_config["environment"] == config["environment"]
    environment.close()
