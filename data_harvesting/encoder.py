from dataclasses import dataclass
import torch
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING
from torch import nn

from data_harvesting.utils import get_activation_class

@dataclass
class SequentialConfig:
    """
    Configuration for a sequential head in the observation encoder. A sequential head processes a
    sequential key of the observation dictionary using Transformer blocks. The expected input shape is
    (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions
    shared across all observation keys.
    """
    key: str

    obs_size: int
    embed_dim: int
    num_heads: int
    ff_dim: int
    depth: int
    dropout: float

@dataclass
class FlatConfig:
    """
    Configuration for a flat head in the observation encoder. A flat head processes a flat key of the
    observation dictionary using an MLP. The expected input shape is (*B, input_dim) where *B denotes
    zero or more leading batch dimensions shared across all observation keys.
    """
    key: str

    obs_size: int
    embed_dim: int
    depth: int
    num_cells: int
    activation_class: type[nn.Module]

class SequentialHead(nn.Module):
    """
    A Transformer-based head for processing a single sequential observation key for one agent. It's job is 
    to encode a sequence of observations into a fixed-size embedding.

    Inputs should have shape (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions
    shared across all observation keys. The output has shape (*B, embed_dim).
    """
    def __init__(self, 
                 config: SequentialConfig,
                 device: DEVICE_TYPING | None = None):
        """Initialize the sequential head.

        Args:
            config: Configuration for the sequential head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        self.obs_embedder = nn.Linear(config.obs_size, config.embed_dim, device=device)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a sequential observation.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """

        # Observations where all features are -1 are padding and should be ignored by the
        # attention mechanism. Create a mask that marks these timesteps so the attention
        # layers do not attend to them. 
        padded_inputs = torch.all(x == -1, dim=-1)

        embed_output = self.obs_embedder(x)
        seq_output = self.transformer(embed_output, src_key_padding_mask=padded_inputs)
        
        # The output of the transformer has shape (*B, seq_len, embed_dim). We need to aggregate
        # across the sequence dimension to produce a single fixed-size embedding. 
        # We can't do a simple mean because some positions are padding and should be ignored. We
        # fill the mask with zeros and them divide by the number of non-padded positions, performing
        # masked mean pooling.
        non_padded_inputs = ~padded_inputs  # True for valid positions
        seq_output_masked = seq_output.masked_fill(padded_inputs.unsqueeze(-1), 0.0) # Filling padded positions with 0
        seq_counts = (non_padded_inputs
            .sum(dim=-1, keepdim=True) # Number of valid (non-padded) positions
            .clamp(min=1))  # Avoid division by zero
        seq_output = seq_output_masked.sum(dim=-2) / seq_counts # Average over valid positions only
        return seq_output

class FlatHead(nn.Module):
    """
    An MLP-based head for processing a single flat observation key for one agent. It's job is to
    encode a flat observation into a fixed-size embedding. 

    Inputs should have shape (*B, input_dim) where *B denotes zero or more leading batch dimensions
    shared across all observation keys. The output has shape (*B, embed_dim).
    """
    def __init__(self, 
                 config: FlatConfig,
                 device: DEVICE_TYPING | None = None):
        """
        Initialize the flat head.

        Args:
            config: Configuration for the flat head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config
        input_dim = config.obs_size

        self.mlp = MLP(
            in_features=input_dim,
            out_features=config.embed_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a flat observation.

        Args:
            x: Input tensor of shape (*B, input_dim) if decentralized or (*B, n_agents * input_dim)
                if centralized.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        return self.mlp(x)

@torch.compile(dynamic=True)
class FlexObservationEncoder(nn.Module):
    """
    A flexible observation encoder that can process both sequential and flat observation keys of a dict-like observation space.
    Each agent's observation can contain multiple keys, some of which are sequential and of variable size (e.g., a buffer of
    messages) and some are flat (e.g., scalar features). The encoder processes sequential keys using Transformer blocks and
    flat keys using MLPs. Sequential keys are transformed into fixed-size embeddings and flat keys are projected into embeddings.
    The embeddings from each head will be concatenated and passed through a final mixing MLP to produce the final output.

    Inputs should be provided as keyword arguments, where each key corresponds to an observation key and the value is a tensor
    of shape (*B, ...). Sequential keys should have shape (*B, seq_len, input_dim) and flat keys
    should have shape (*B, input_dim). The network validates that:

    1. all required keys are present
    2. each tensor has the correct last dimension
    3. all tensors share the same leading batch dimensions

    Args:
        sequential_configs (list[SequentialConfig]): List of configurations for each sequential head.
        flat_configs (list[FlatConfig]): List of configurations for each flat head.
        mix_layer_depth (int): Depth of the final mixing MLP.
        mix_layer_num_cells (int): Number of cells per layer in the final mixing MLP
        output_dim (int): Dimension of the final output.
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
        device: DEVICE_TYPING | None = None
    ):
        """
        Initialize the flexible multi-agent encoder.

        Args:
            sequential_configs: Transformer head configurations for observation keys with a
                temporal dimension.
            flat_configs: MLP head configurations for flat observation keys.
            mix_layer_depth: Number of hidden layers in the final mixing MLP.
            mix_layer_num_cells: Hidden size used by each layer of the mixing MLP.
            mix_activation_class: Activation function class for the mixing MLP. Defaults to
                :class:`torch.nn.Tanh` when ``None``.
            output_dim: Size of the final per-agent embedding produced by the network.
            device: Device handle used to place all learnable modules and validate inputs. When
                ``None``, the network defaults to CPU placement.

        The constructor stores the requested configuration, resolves defaults for the operating
        mode (centralized vs. decentralized and shared vs. per-agent parameters) and builds the
        per-agent processing stacks ahead of time so they can be reused on every forward pass.
        """
        super().__init__()
        
        if len(sequential_configs) == 0 and len(flat_configs) == 0:
            raise ValueError("At least one of `sequential_configs` or `flat_configs` must be non-empty.")
        
        self.sequential_configs = sequential_configs
        self.flat_configs = flat_configs
        self.output_dim = output_dim
        self.mix_layer_depth = mix_layer_depth
        self.mix_layer_num_cells = mix_layer_num_cells
        self.mix_activation_class = mix_activation_class if mix_activation_class is not None else nn.Tanh

        self.device = torch.device(device) if device is not None else torch.device("cpu")

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

        self.seq_heads = nn.ModuleDict(seq_heads)
        self.flat_heads = nn.ModuleDict(flat_heads)
        self.mix_layer = MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device,
        )

    def _pre_forward_check(self, inputs):
        """
        Validate the structure, shape and device of the provided observation tensors.

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
    
    def _build_sequence_head(self, config: SequentialConfig, device) -> SequentialHead:
        """Instantiate the Transformer block stack that processes a sequential observation key."""
        return SequentialHead(config, device=device)

    def _build_flat_head(self, config: FlatConfig, device) -> FlatHead:
        """Instantiate the MLP head that processes a flat observation key."""
        return FlatHead(config, device=device)
    
    def forward(self, **observation: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-agent observations into per-agent embeddings.

        Keyword Args:
            observation: Tensors keyed by observation name. Each tensor must obey the shape
                requirements validated in :meth:`_pre_forward_check`.

        Returns:
            torch.Tensor: A tensor of shape ``(*B, output_dim)`` containing the
            encoded representation for every agent. *B denotes the (possibly empty) leading
            batch dimensions shared across all input keys.
        """
        self._pre_forward_check(observation)

        head_outputs = []
        for config in self.sequential_configs:
            seq_input = observation[config.key]

            seq_output = self.seq_heads[config.key](seq_input)

            head_outputs.append(seq_output)

        for config in self.flat_configs:
            flat_input = observation[config.key]

            flat_output = self.flat_heads[config.key](flat_input)
            head_outputs.append(flat_output)

        # Fuse the contributions from every head and record the per-agent output.
        agent_input = torch.cat(head_outputs, dim=-1)
        agent_output = self.mix_layer(agent_input)

        return agent_output
