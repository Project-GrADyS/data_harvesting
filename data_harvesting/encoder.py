from dataclasses import dataclass
import torch
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING
from torch import nn
import tensordict

from data_harvesting.transformer import Transformer

@dataclass
class SequentialConfig:
    """
    Configuration for a sequential head in the multi-agent network. A sequential head processes a
    sequential key of the observation using Transformer blocks. The input should be of shape
    (batch_size, n_agents, seq_len, input_dim).
    """
    key: str

    input_dim: int
    head_dim: int
    num_heads: int
    ff_dim: int
    depth: int
    dropout: float = 0.1

@dataclass
class FlatConfig:
    """
    Configuration for a flat head in the multi-agent network. A flat head processes a flat key of the
    observation using an MLP. The input should be of shape (batch_size, n_agents, input_dim).
    """
    key: str

    input_dim: int
    output_dim: int
    depth: int
    num_cells: int
    activation_class: type[nn.Module]


@torch.compile
class MultiAgentFlexEncoder(nn.Module):
    """
    A flexible multi-agent encoder that can process both sequential and flat observation keys of a dict-like observation space.
    Each agent's observation can contain multiple keys, some of which are sequential (e.g., time series data) and some are flat
    (e.g., scalar features). The encoder processes sequential keys using Transformer blocks and flat keys using MLPs. The outputs
    from all heads are concatenated and passed through a final mixing MLP to produce the final output.

    Inputs should be provided as keyword arguments, where each key corresponds to an observation key and the value is a tensor
    of shape (batch_size, n_agents, ...). Sequential keys should have shape (batch_size, n_agents, seq_len, input_dim) and flat keys
    should have shape (batch_size, n_agents, input_dim). The network will validate the input shapes based on the provided configurations.

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
        self.agent_networks = [
            self._build_agent_network() for _ in range(1 if self.share_params else self.n_agents)
        ]

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
            if inputs[config.key].ndim != 4:
                raise ValueError(f"Sequential input '{config.key}' must be 4D (batch_size, n_agents, seq_len, input_dim).")
            if inputs[config.key].shape[-1] != config.input_dim:
                raise ValueError(f"Sequential input '{config.key}' last dimension must be {config.input_dim}, got {inputs[config.key].shape[-1]}.")
            if inputs[config.key].shape[1] != self.n_agents:
                raise ValueError(f"Sequential input '{config.key}' second dimension must be {self.n_agents}, got {inputs[config.key].shape[1]}.")

        # Repeat the checks for flat keys to ensure downstream modules receive correctly shaped data.
        for config in self.flat_configs:
            if config.key not in inputs:
                raise KeyError(f"Flat key '{config.key}' not found in inputs.")
            if inputs[config.key].ndim != 3:
                raise ValueError(f"Flat input '{config.key}' must be 3D (batch_size, n_agents, input_dim).")
            if inputs[config.key].shape[-1] != config.input_dim:
                raise ValueError(f"Flat input '{config.key}' last dimension must be {config.input_dim}, got {inputs[config.key].shape[-1]}.")
            if inputs[config.key].shape[1] != self.n_agents:
                raise ValueError(f"Flat input '{config.key}' second dimension must be {self.n_agents}, got {inputs[config.key].shape[1]}.")
            
        input_keys = set(inputs.keys())
        expected_keys = {config.key for config in self.sequential_configs + self.flat_configs}
        if input_keys != expected_keys:
            raise KeyError(f"Input keys do not match expected keys. Expected: {expected_keys}, got: {input_keys}.")
        
        batch_dim = next(iter(inputs.values())).shape[0]
        for key, tensor in inputs.items():
            if tensor.device != self.device:
                raise ValueError(f"Input tensor '{key}' is on device {tensor.device}, expected {self.device}.")
            if tensor.shape[0] != batch_dim:
                raise ValueError(f"All input tensors must have the same batch size. Tensor '{key}' has batch size {tensor.shape[0]}, expected {batch_dim}.")
    
    def _build_sequence_net(self, config: SequentialConfig, device):
        """Instantiate the Transformer block stack that processes a sequential observation key."""
        return Transformer(
            input_dim=config.input_dim,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            depth=config.depth,
            dropout=config.dropout,
            device=device
        )

    def _build_flat_net(self, config: FlatConfig, device):
        """Instantiate the MLP head that processes a flat observation key."""
        input_dim = config.input_dim
        if self.centralized:
            input_dim *= self.n_agents
        return MLP(
            in_features=input_dim,
            out_features=config.output_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device
        )
    
    def _build_agent_network(self):
        """Assemble the per-agent processing stacks for sequential, flat, and mixed features.

        Returns:
            tuple[dict[str, nn.Module], dict[str, nn.Module], MLP]:
                * A mapping from sequential keys to Transformer heads.
                * A mapping from flat keys to MLP heads.
                * The final mixing MLP that fuses all head outputs into a single embedding.
        """
        if self.centralized:
            n_agents = 1
        else:
            n_agents = self.n_agents

        # Build a dedicated Transformer for each sequential key.
        seq_nets = {}
        for config in self.sequential_configs:
            seq_nets[config.key] = self._build_sequence_net(config, self.device)

        # Flat heads operate on flattened per-agent features; adjust the configuration accordingly.
        flat_nets = {}
        for config in self.flat_configs:
            flat_nets[config.key] = self._build_flat_net(config, self.device)

        # The mixing network consumes the concatenated outputs from every head.
        mix_input_dim = sum(
            config.head_dim * config.num_heads for config in self.sequential_configs
        ) + sum(
            config.output_dim for config in self.flat_configs
        )
        mix_layer = MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device
        )
        return seq_nets, flat_nets, mix_layer

    def forward(self, **observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-agent observations into per-agent embeddings.

        Keyword Args:
            observation: Tensors keyed by observation name. Each tensor must obey the shape
                requirements validated in :meth:`_pre_forward_check`.

        Returns:
            torch.Tensor: A tensor of shape ``(batch_size, n_agents, output_dim)`` containing the
            encoded representation for every agent in the batch.
        """
        self._pre_forward_check(observation)
        batch_size = next(iter(observation.values())).shape[0]

        num_agents = self.n_agents if not self.centralized else 1

        all_agent_outputs = []
        for agent_idx in range(num_agents):
            if self.share_params:
                seq_nets, flat_nets, mix_layer = self.agent_networks[0]
            else:
                seq_nets, flat_nets, mix_layer = self.agent_networks[agent_idx]

            head_outputs = []
            for config in self.sequential_configs:
                seq_input = observation[config.key]
                if self.centralized:
                    b, n, t, i = seq_input.shape
                    # Collapse the agent dimension into the temporal axis for centralized processing.
                    seq_input = seq_input.view(b, n * t, i)
                else:
                    # Select the agent-specific slice; Transformers operate on (batch, seq_len, feature).
                    seq_input = seq_input[:, agent_idx, :, :]

                seq_output = seq_nets[config.key](seq_input)
                # Aggregate the temporal dimension so every head contributes a fixed-size vector.
                seq_output = seq_output.mean(dim=1)
                head_outputs.append(seq_output)

            for config in self.flat_configs:
                flat_input = observation[config.key]
                if self.centralized:
                    # All agent features have already been merged; flatten directly.
                    flat_input = flat_input.view(batch_size, -1)
                else:
                    flat_input = flat_input[:, agent_idx, :].view(batch_size, -1)
                flat_output = flat_nets[config.key](flat_input)
                head_outputs.append(flat_output)

            # Fuse the contributions from every head and record the per-agent output.
            agent_input = torch.cat(head_outputs, dim=-1)
            agent_output = mix_layer(agent_input)
            all_agent_outputs.append(agent_output)
        return torch.stack(all_agent_outputs, dim=1)  # (batch_size, n_agents, output_dim)