from dataclasses import dataclass
import torch
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING
from torch import nn

@dataclass
class SequentialEncoderConfig:
    """
    Configuration for a SequentialEncoder
    """
    key: str
    """The key this encoder processes."""
    input_size: int
    """Dimensionality of a single input feature in the sequence"""
    embed_dim: int
    """Dimensionality of the output embedding produced by this head."""
    head_dim: int
    """Dimensionality of the output of each attention head."""
    num_heads: int
    """Number of attention heads in the Transformer blocks."""
    ff_dim: int
    """Dimensionality of the feedforward layers in the Transformer blocks."""
    depth: int
    """Number of Transformer blocks in the head."""
    dropout: float
    """Dropout rate used in the Transformer blocks."""
    max_num_agents: int
    """Maximum number of agents expected in the environment. Necessary for generating agentic embeddings."""
    agentic_encoding: bool
    """When ``True``, adds an agent embedding to the input based on the agent index."""

@dataclass
class FlatEncoderConfig:
    """
    Configuration for a FlatEncoder
    """
    key: str
    """The key this encoder processes."""
    input_size: int
    """Dimensionality of the flat input feature."""
    embed_dim: int
    """Dimensionality of the output embedding produced by this head."""
    depth: int
    """Number of hidden layers in the MLP."""
    num_cells: int
    """Number of cells per hidden layer in the MLP."""
    activation_class: type[nn.Module]
    """Activation function class used between MLP layers."""

class SequentialEncoder(nn.Module):
    """
    A Transformer-based encoder that processes sequential observations. It receives data of shape
    (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input sequence.
    """
    def __init__(self, 
                 config: SequentialEncoderConfig,
                 device: DEVICE_TYPING | None = None):
        """Initialize the sequential head.

        Args:
            config: Configuration for the sequential head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        # Linear layer to project input features to the embedding dimension
        self.obs_encoder = nn.Linear(config.input_size, config.embed_dim, device=device)
        # Embedding layer for agent indices
        self.agent_embedder = nn.Embedding(
            num_embeddings=config.max_num_agents,
            embedding_dim=config.embed_dim,
            device=device
        ) if config.agentic_encoding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            device=device
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
            enable_nested_tensor=True
        )

    def forward(self, x: torch.Tensor, agent_idx: torch.Tensor) -> torch.Tensor:
        """Process a sequential observation for one agent.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).
            agent_idx: Input tensor of shape (*B, 1) containing the agent index for each batch entry. Will be used for 
                agentic encoding if it's enabled.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """

        # Observations where all features are -1 are padding and should be ignored by the
        # attention mechanism. Create a mask that marks these timesteps so the attention
        # layers do not attend to them. 
        padded_input_mask = torch.all(x == -1, dim=-1)

        embed_output = self.obs_encoder(x)

        # If agentic encoding is enabled, add the agent embedding to every step in the sequence.
        if self.config.agentic_encoding:
            agent_embeddings = self.agent_embedder(agent_idx).squeeze(dim=-2)
            embed_output += agent_embeddings

        seq_output = self.transformer(embed_output, src_key_padding_mask=padded_input_mask)

        # Aggregate the sequence dimension so every head contributes a fixed-size vector.
        seq_output = seq_output.mean(dim=-2)

        return seq_output
    
class FlatEncoder(nn.Module):
    """
    An MLP-based head for processing a single flat observation key for one agent. The input is a tensor of shape
    (*B, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input.
    """
    def __init__(self, 
                 config: FlatEncoderConfig,
                 centralized: bool,
                 n_agents: int,
                 device: DEVICE_TYPING | None = None):
        """Initialize the flat head.

        Args:
            config: Configuration for the flat head.
            centralized: When ``True``, the head processes concatenated observations from all agents.
            n_agents: Total number of agents in the environment.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config
        self.centralized = centralized

        input_dim = config.input_size
        # If centralized, the input contains observations from all agents concatenated together.
        if self.centralized:
            input_dim *= n_agents
        self.mlp = MLP(
            in_features=input_dim,
            out_features=config.embed_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a flat observation for one agent.

        Args:
            x: Input tensor of shape (*B, input_dim) if decentralized or (*B, n_agents * input_dim)
                if centralized.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        return self.mlp(x)

class AgentBlock(nn.Module):
    """
    A container module for the per-agent processing stack. Each agent's inputs are processed by a set of
    Transformer-based sequential heads and MLP-based flat heads. The outputs from all heads are concatenated
    and passed through a final mixing MLP to produce the final output.

    Holds:
        seq_heads: ModuleDict of Transformer-based sequential heads
        flat_heads: ModuleDict of MLP-based flat heads
        mix_layer: Final MLP that fuses head outputs
    """
    def __init__(self, 
                 seq_heads: dict[str, SequentialEncoder], 
                 flat_heads: dict[str, FlatEncoder], 
                 mix_layer: nn.Module,
                 centralized: bool = False,
                 n_agents: int = 1,):
        super().__init__()
        self.seq_heads = nn.ModuleDict(seq_heads)
        self.flat_heads = nn.ModuleDict(flat_heads)
        self.mix_layer = mix_layer
        self.centralized = centralized
        self.n_agents = n_agents

    def forward(self, observation: dict[str, torch.Tensor], agent_idx: torch.Tensor):
        """
        Processes observations and produces a per-agent output. 
        Args:
            observation: A mapping from observation key to tensor. Each tensor must have shape
                (*B, n_agents, ...) where *B denotes zero or more leading batch dimensions.
                Sequential keys must have shape (*B, n_agents, seq_len, input_dim) and flat keys
                must have shape (*B, n_agents, input_dim).
            agent_idx: Tensor of shape (*B, 1) containing the index of the agent being processed.
        """
        head_outputs = []
        for key in self.seq_heads.keys():
            seq_input = observation[key]
            if self.centralized:
                # The agent dimension of sequential observations will be collapsed into a single sequence of size
                # (*B, n_agents * seq_len). The model has no way of knowing which items in the sequence belong to which agent,
                # so we provide an agent index tensor that, for each item in the sequence, indicates which agent it came from.
                agent_idx_tensor = torch.arange(self.n_agents, device=seq_input.device).repeat_interleave(seq_input.shape[-2])
                agent_idx_tensor = agent_idx_tensor.unsqueeze(0).expand(seq_input.shape[0], -1).unsqueeze(-1)

                # Collapse agent and temporal dimensions generically. Assumes trailing dims
                # are (..., n_agents, seq_len, feature). This supports any (or no) leading
                # batch dimensions without explicit unpacking.
                seq_input = seq_input.flatten(start_dim=-3, end_dim=-2)
                
            else:
                # Select the agent along the -3 (agent) axis without assuming any specific
                # number of leading batch dimensions. Result keeps shape (..., seq_len, feature).
                seq_input = seq_input.select(dim=-3, index=agent_idx)
                agent_idx_tensor = torch.full_like(seq_input[..., :1, 0:1], agent_idx)

            seq_output = self.seq_heads[key](seq_input, agent_idx_tensor)

            head_outputs.append(seq_output)

        for key in self.flat_heads.keys():
            flat_input = observation[key]
            if self.centralized:
                # Flattening along the agent dimension
                flat_input = flat_input.flatten(start_dim=-2, end_dim=-1)
            else:
                # Select specific agent slice
                flat_input = flat_input.select(dim=-2, index=agent_idx)
            flat_output = self.flat_heads[key](flat_input)
            head_outputs.append(flat_output)

        # Fuse the contributions from every head and record the per-agent output.
        agent_input = torch.cat(head_outputs, dim=-1)
        return self.mix_layer(agent_input)

@torch.compile(dynamic=True)
class MultiAgentFlexModule(nn.Module):
    """
    A flexible multi-agent module that can process both sequential and flat observation keys of a dict-like observation space.
    Each agent's observation can contain multiple keys, some of which are sequential (e.g., time series data) and some are flat
    (e.g., scalar features). The module processes sequential keys using Transformer blocks and flat keys using MLPs. The outputs
    from all heads are concatenated and passed through a final mixing MLP to produce the final output.

    Inputs should be provided as keyword arguments, where each key corresponds to an input key and the value is a tensor
    of shape (*B, n_agents, ...). Sequential keys should have shape (*B, n_agents, seq_len, input_dim) and flat keys
    should have shape (*B, n_agents, input_dim). The network validates that: (1) all required keys are present, (2) the
    agent dimension matches `n_agents`, (3) the last feature dimension matches the configured obs size, and (4) all leading
    batch dimensions *B are identical across keys. The number of leading batch dimensions may be zero.

    Args:
        sequential_configs (list[SequentialConfig]): List of configurations for sequential heads.
        flat_configs (list[FlatConfig]): List of configurations for flat heads.
        mix_layer_depth (int): Depth of the final mixing MLP.
        mix_layer_num_cells (int): Number of cells per layer in the final mixing MLP
        mix_activation_class (type[nn.Module] | None): Activation function class for the mixing MLP. Defaults to
            :class:`torch.nn.Tanh` when ``None``.
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
        sequential_configs: list[SequentialEncoderConfig],
        flat_configs: list[FlatEncoderConfig],
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
        self.share_params = share_params
        # Centralized networks always share parameters.
        if self.centralized:
            self.share_params = True
        else:
            self.share_params = share_params if share_params is not None else False

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
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(f"Sequential input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}.")

        # Repeat the checks for flat keys to ensure downstream modules receive correctly shaped data.
        for config in self.flat_configs:
            if config.key not in inputs:
                raise KeyError(f"Flat key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(f"Flat input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}.")
        
        # Check for missing keys
        input_keys = set(inputs.keys())
        expected_keys = {config.key for config in self.sequential_configs + self.flat_configs}
        if input_keys != expected_keys:
            raise KeyError(f"Input keys do not match expected keys. Expected: {expected_keys}, got: {input_keys}.")


        batch_dim = next(iter(inputs.values())).shape[0]
        for key, tensor in inputs.items():
            if tensor.shape[0] != batch_dim:
                raise ValueError(f"All input tensors must have the same batch size. Tensor '{key}' has batch size {tensor.shape[0]}, expected {batch_dim}.")
    
    def _build_sequence_head(self, config: SequentialEncoderConfig, device) -> SequentialEncoder:
        """Instantiate the Transformer block stack that processes a sequential observation key."""
        return SequentialEncoder(config, device=device)

    def _build_flat_head(self, config: FlatEncoderConfig, device) -> FlatEncoder:
        """Instantiate the MLP head that processes a flat observation key."""
        return FlatEncoder(config, self.centralized, self.n_agents, device=device)
    
    def _build_agent_network(self) -> nn.Module:
        """Assemble and register the per-agent processing stacks.

        Returns:
            nn.Module: container with attributes ``seq_heads``, ``flat_heads`` and ``mix_layer``.
        """
        seq_heads: dict[str, nn.Module] = {}
        for cfg in self.sequential_configs:
            seq_heads[cfg.key] = self._build_sequence_head(cfg, self.device)

        flat_heads: dict[str, nn.Module] = {}
        for cfg in self.flat_configs:
            flat_heads[cfg.key] = self._build_flat_head(cfg, self.device)

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
        return AgentBlock(seq_heads, flat_heads, mix_layer, self.centralized, self.n_agents)

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
            block: AgentBlock = self.agent_networks[0] if self.share_params else self.agent_networks[agent_idx]
            agent_output = block(observation, agent_idx)
            all_agent_outputs.append(agent_output)

        if self.centralized:
            # Replicate the single output across all agents
            all_agent_outputs = all_agent_outputs * self.n_agents

        # Assemble final tensor with agent dimension immediately before the feature dimension.
        return torch.stack(all_agent_outputs, dim=-2)
