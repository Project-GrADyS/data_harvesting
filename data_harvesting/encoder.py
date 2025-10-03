from dataclasses import dataclass
import torch
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING
from torch import nn
from tensordict.nn import TensorDictModule

from data_harvesting.transformer import Transformer
from data_harvesting.utils import get_activation_class

@dataclass
class SequentialConfig:
    """
    Configuration for a sequential head in the multi-agent network. A sequential head processes a
    sequential key of the observation using Transformer blocks. The expected input shape is
    (*B, n_agents, seq_len, input_dim) where *B denotes zero or more leading batch dimensions
    shared across all observation keys.
    """
    key: str

    obs_size: int
    embed_dim: int
    head_dim: int
    num_heads: int
    ff_dim: int
    depth: int
    dropout: float

@dataclass
class FlatConfig:
    """
    Configuration for a flat head in the multi-agent network. A flat head processes a flat key of the
    observation using an MLP. The expected input shape is (*B, n_agents, input_dim) where *B denotes
    zero or more leading batch dimensions shared across all observation keys.
    """
    key: str

    obs_size: int
    embed_dim: int
    depth: int
    num_cells: int
    activation_class: type[nn.Module]


class AgentBlock(nn.Module):
    """Container module for per-agent processing stack.

    Holds:
        seq_nets: ModuleDict of Transformer-based sequential heads
        flat_nets: ModuleDict of MLP-based flat heads
        mix_layer: Final MLP that fuses head outputs
    """
    def __init__(self, seq_nets: dict[str, nn.Module], flat_nets: dict[str, nn.Module], mix_layer: nn.Module):
        super().__init__()
        self.seq_nets = nn.ModuleDict(seq_nets)
        self.flat_nets = nn.ModuleDict(flat_nets)
        self.mix_layer = mix_layer


class MultiAgentFlexEncoder(nn.Module):
    """
    A flexible multi-agent encoder that can process both sequential and flat observation keys of a dict-like observation space.
    Each agent's observation can contain multiple keys, some of which are sequential (e.g., time series data) and some are flat
    (e.g., scalar features). The encoder processes sequential keys using Transformer blocks and flat keys using MLPs. The outputs
    from all heads are concatenated and passed through a final mixing MLP to produce the final output.

    Inputs should be provided as keyword arguments, where each key corresponds to an observation key and the value is a tensor
    of shape (*B, n_agents, ...). Sequential keys should have shape (*B, n_agents, seq_len, input_dim) and flat keys
    should have shape (*B, n_agents, input_dim). The network validates that: (1) all required keys are present, (2) the
    agent dimension matches `n_agents`, (3) the last feature dimension matches the configured obs size, and (4) all leading
    batch dimensions *B are identical across keys. The number of leading batch dimensions may be zero.

    Args:
        sequential_configs (list[SequentialConfig]): List of configurations for sequential heads.
        flat_configs (list[FlatConfig]): List of configurations for flat heads.
        mix_layer_depth (int): Depth of the final mixing MLP.
        mix_layer_num_cells (int): Number of cells per layer in the final mixing MLP
        output_dim (int): Dimension of the final output.
        n_agents (int): Number of agents.
        centralized (bool | None): If True, the network is centralized (processes all agents' observations together).
                                   If False, the network is decentralized (processes each agent's observation independently).
                                   If None, defaults to False.
        share_params (bool | None): If True, all agents share the same network parameters.
                                    If False, each agent has its own set of parameters.
                                    If None, defaults to True if centralized else False.
        device (DEVICE_TYPING | None): Device to place the networks on. Defaults to CPU when ``None``.
    """

    def __init__(
        self,
        sequential_configs: list[SequentialConfig],
        flat_configs: list[FlatConfig],
        mix_layer_depth: int,
        mix_layer_num_cells: int,
        mix_activation_class: type[nn.Module] | None,
        output_dim: int,
        n_agents: int,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device: DEVICE_TYPING | None = None
    ):
        """Initialize the flexible multi-agent encoder.

        Args:
            sequential_configs: Transformer head configurations for observation keys with a
                temporal dimension.
            flat_configs: MLP head configurations for flat observation keys.
            mix_layer_depth: Number of hidden layers in the final mixing MLP.
            mix_layer_num_cells: Hidden size used by each layer of the mixing MLP.
            mix_activation_class: Activation function class for the mixing MLP. Defaults to
                :class:`torch.nn.Tanh` when ``None``.
            output_dim: Size of the final per-agent embedding produced by the network.
            n_agents: Total number of agents in the environment.
            centralized: When ``True`` all agent observations are pooled before processing.
            share_params: When ``True`` one set of parameters is shared across all agents.
            device: Device handle used to place all learnable modules and validate inputs. When
                ``None``, the network defaults to CPU placement.

        The constructor stores the requested configuration, resolves defaults for the operating
        mode (centralized vs. decentralized and shared vs. per-agent parameters) and builds the
        per-agent processing stacks ahead of time so they can be reused on every forward pass.
        """
        super().__init__()
        self.sequential_configs = sequential_configs
        self.flat_configs = flat_configs
        self.output_dim = output_dim
        self.mix_layer_depth = mix_layer_depth
        self.mix_layer_num_cells = mix_layer_num_cells
        self.mix_activation_class = mix_activation_class if mix_activation_class is not None else nn.Tanh
        self.n_agents = n_agents
        self.centralized = centralized if centralized is not None else False
        self.share_params = share_params if share_params is not None else (True if centralized else False)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.agent_networks = nn.ModuleList(
            [self._build_agent_network() for _ in range(1 if self.share_params else self.n_agents)]
        )

    def _pre_forward_check(self, inputs):
        """Validate the structure, shape and device of the provided observation tensors.

        Args:
            inputs: A mapping from observation key to tensor provided to :meth:`forward`.

        Raises:
            KeyError: If an expected key is missing or there are unexpected extras.
            ValueError: When a tensor has the wrong dimensionality, agent count, last dimension,
                batch size, or is placed on the wrong device.
        """
        # Validate every sequential key independently to surface precise, actionable errors.
        for config in self.sequential_configs:
            if config.key not in inputs:
                raise KeyError(f"Sequential key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.obs_size:
                raise ValueError(f"Sequential input '{config.key}' last dimension must be {config.obs_size}, got {inputs[config.key].shape[-1]}.")

        # Repeat the checks for flat keys to ensure downstream modules receive correctly shaped data.
        for config in self.flat_configs:
            if config.key not in inputs:
                raise KeyError(f"Flat key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.obs_size:
                raise ValueError(f"Flat input '{config.key}' last dimension must be {config.obs_size}, got {inputs[config.key].shape[-1]}.")
        
        # Check for missing keys
        input_keys = set(inputs.keys())
        expected_keys = {config.key for config in self.sequential_configs + self.flat_configs}
        if input_keys != expected_keys:
            raise KeyError(f"Input keys do not match expected keys. Expected: {expected_keys}, got: {input_keys}.")


        batch_dim = next(iter(inputs.values())).shape[0]
        for key, tensor in inputs.items():
            if tensor.shape[0] != batch_dim:
                raise ValueError(f"All input tensors must have the same batch size. Tensor '{key}' has batch size {tensor.shape[0]}, expected {batch_dim}.")
    
    def _build_sequence_net(self, config: SequentialConfig, device):
        """Instantiate the Transformer block stack that processes a sequential observation key."""

        embedder = nn.Linear(config.obs_size, config.embed_dim, device=device)

        transformer = Transformer(
            input_dim=config.embed_dim,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            depth=config.depth,
            dropout=config.dropout,
            device=device
        )

        return nn.Sequential(embedder, transformer)

    def _build_flat_net(self, config: FlatConfig, device):
        """Instantiate the MLP head that processes a flat observation key."""
        input_dim = config.obs_size
        if self.centralized:
            input_dim *= self.n_agents
        return MLP(
            in_features=input_dim,
            out_features=config.embed_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device
        )
    
    def _build_agent_network(self) -> nn.Module:
        """Assemble and register the per-agent processing stacks.

        Returns:
            nn.Module: container with attributes ``seq_nets``, ``flat_nets`` and ``mix_layer``.
        """
        seq_nets: dict[str, nn.Module] = {}
        for cfg in self.sequential_configs:
            seq_nets[cfg.key] = self._build_sequence_net(cfg, self.device)

        flat_nets: dict[str, nn.Module] = {}
        for cfg in self.flat_configs:
            flat_nets[cfg.key] = self._build_flat_net(cfg, self.device)

        mix_input_dim = (
            sum(cfg.embed_dim for cfg in self.sequential_configs)
            + sum(cfg.embed_dim for cfg in self.flat_configs)
        )
        mix_layer = MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device,
        )
        return AgentBlock(seq_nets, flat_nets, mix_layer)

    def forward(self, **observation: torch.Tensor) -> torch.Tensor:
        """Encode multi-agent observations into per-agent embeddings.

        Keyword Args:
            observation: Tensors keyed by observation name. Each tensor must obey the shape
                requirements validated in :meth:`_pre_forward_check`.

        Returns:
            torch.Tensor: A tensor of shape ``(*B, n_agents, output_dim)`` containing the
            encoded representation for every agent. *B denotes the (possibly empty) leading
            batch dimensions shared across all input keys.
        """
        self._pre_forward_check(observation)

        num_agents = self.n_agents if not self.centralized else 1

        all_agent_outputs = []
        for agent_idx in range(num_agents):
            block = self.agent_networks[0] if self.share_params else self.agent_networks[agent_idx]
            seq_nets = block.seq_nets
            flat_nets = block.flat_nets
            mix_layer = block.mix_layer

            head_outputs = []
            for config in self.sequential_configs:
                seq_input = observation[config.key]
                if self.centralized:
                    # Collapse agent and temporal dimensions generically. Assumes trailing dims
                    # are (..., n_agents, seq_len, feature). This supports any (or no) leading
                    # batch dimensions without explicit unpacking.
                    seq_input = seq_input.flatten(start_dim=-3, end_dim=-2)
                else:
                    # Select the agent along the -3 (agent) axis without assuming any specific
                    # number of leading batch dimensions. Result keeps shape (..., seq_len, feature).
                    seq_input = seq_input.select(dim=-3, index=agent_idx)

                seq_output = seq_nets[config.key](seq_input)
                #  Aggregate the temporal dimension so every head contributes a fixed-size vector.
                seq_output = seq_output.mean(dim=-2)
                head_outputs.append(seq_output)

            for config in self.flat_configs:
                flat_input = observation[config.key]
                if self.centralized:
                    # Flattening along the agent dimension
                    flat_input = flat_input.flatten(start_dim=-2, end_dim=-1)
                else:
                    # Select specific agent slice
                    flat_input = flat_input.select(dim=-2, index=agent_idx)
                flat_output = flat_nets[config.key](flat_input)
                head_outputs.append(flat_output)

            # Fuse the contributions from every head and record the per-agent output.
            agent_input = torch.cat(head_outputs, dim=-1)
            agent_output = mix_layer(agent_input)
            all_agent_outputs.append(agent_output)

        if self.centralized:
            # Replicate the single output across all agents
            all_agent_outputs = all_agent_outputs * self.n_agents

        # Assemble final tensor with agent dimension immediately before the feature dimension.
        return torch.stack(all_agent_outputs, dim=-2)
