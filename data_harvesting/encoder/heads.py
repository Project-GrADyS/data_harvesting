import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING

from .configs import SequentialEncoderConfig, SequentialEncoderInput, FlatEncoderConfig, FlatEncoderInput


class SequentialEncoder(nn.Module):
    """
    A Transformer-based encoder that processes sequential observations. It receives data of shape
    (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input sequence.
    """
    def __init__(
        self,
        input: SequentialEncoderInput,
        config: SequentialEncoderConfig,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the sequential head.

        Args:
            config: Configuration for the sequential head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config
        self.input = input

        # Linear layer to project input features to the embedding dimension
        self.obs_encoder = nn.Linear(input.input_size, config.embed_dim, device=device)
        # Embedding layer for agent indices
        self.agent_embedder = nn.Embedding(
            num_embeddings=config.max_num_agents,
            embedding_dim=config.embed_dim,
            device=device,
        ) if config.agentic_encoding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            device=device,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
            enable_nested_tensor=True,
        )

    def forward(self, x: torch.Tensor, agent_idx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequential observation for one agent.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).
            agent_idx: Input tensor of shape (*B, 1) containing the agent index for each batch entry. Will be used for
                agentic encoding if it's enabled.
            mask: Optional tensor of shape (*B, seq_len) indicating valid timesteps. Not currently used.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """

        # Observations where all features are -1 are padding and should be ignored by the
        # attention mechanism. Create a mask that marks these timesteps so the attention
        # layers do not attend to them.
        padded_input_mask = torch.all(x == -1, dim=-1)

        # Combine with the externally provided agent mask
        if mask is not None:
            padded_input_mask |= ~mask

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
    def __init__(
        self,
        input: FlatEncoderInput,
        config: FlatEncoderConfig,
        input_dim: int | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the flat head.

        Args:
            config: Configuration for the flat head.
            input_dim: Optional override for input size (useful when inputs are concatenated externally).
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        resolved_input_dim = input_dim if input_dim is not None else input.input_size
        self.mlp = MLP(
            in_features=resolved_input_dim,
            out_features=config.embed_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a flat observation for one agent.

        Args:
            x: Input tensor of shape (*B, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        return self.mlp(x)