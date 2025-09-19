import torch
from typing import Any, Dict, TypedDict
from torchrl.objectives import DDPGLoss, ValueEstimators, SoftUpdate

def create_loss(policy: torch.nn.Module, critic: torch.nn.Module, config: Dict[str, Any]) -> DDPGLoss:
    """
    Creates the DDPG loss module using parameters from config.
    Args:
        policy: The actor network.
        critic: The critic network.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Configured DDPGLoss instance.
    """
    gamma = config["optimization"]["gamma"]
    loss_module = DDPGLoss(
        actor_network=policy,
        value_network=critic,
        delay_value=True,
        loss_function="l2",
    )
    loss_module.set_keys(
        state_action_value=("agents", "state_action_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module

OptimizerDict = TypedDict("OptimizerDict", {"loss_actor": torch.optim.Optimizer, "loss_value": torch.optim.Optimizer})

def create_optimizers(loss_module: DDPGLoss, config: Dict[str, Any]) -> OptimizerDict:
    """
    Creates optimizers for the actor and critic using parameters from config.
    Args:
        loss_module: The DDPGLoss module.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Dictionary with 'loss_actor' and 'loss_value' optimizers.
    """
    lr = config["optimization"]["lr"]
    optimizers = {
        "loss_actor": torch.optim.Adam(
            loss_module.actor_network_params.flatten_keys().values(), lr=lr
        ),
        "loss_value": torch.optim.Adam(
            loss_module.value_network_params.flatten_keys().values(), lr=lr
        ),
    }
    return optimizers

def create_updater(loss_module: DDPGLoss, config: Dict[str, Any]) -> SoftUpdate:
    """
    Creates a SoftUpdate target network updater using parameters from config.
    Args:
        loss_module: The DDPGLoss module.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Configured SoftUpdate instance.
    """
    tau = config["optimization"]["tau"]
    return SoftUpdate(loss_module, tau=tau)

