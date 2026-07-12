import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING

from .configs import SequentialHeadConfig, FlatHeadConfig


class SequentialHead(nn.Module):
    """
    A Transformer-based encoder that processes sequential observations. It receives data of shape
    (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input sequence.
    """

    positional_encoder: nn.Embedding | None  # type hint for the positional encoder registered in __init__

    def __init__(
        self,
        config: SequentialHeadConfig,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the sequential head.

        Args:
            config: Configuration for the sequential head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        # Linear layer to project input features to the embedding dimension
        self.encoder = nn.Linear(config.input_size, config.output_size, device=device)
        # Embedding layer for agent indices
        if config.positional_encoding_config is not None:
            self.positional_encoder = nn.Embedding(
                num_embeddings=config.positional_encoding_config.num_positions,
                embedding_dim=config.output_size,
                device=device,
            )
        else:
            self.positional_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.output_size,
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

    def _pre_forward_checks(self, x: torch.Tensor, idx: torch.Tensor | None, mask: torch.Tensor | None):
        """Perform checks on the input tensors before the forward pass.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).
            idx: Input tensor of shape (*B, 1) containing the positional index for each batch entry. Will be used for
                positional encoding if it's enabled.
            mask: Optional tensor of shape (*B, seq_len) indicating valid timesteps. Not currently used.

        Raises:
            ValueError: If the input tensor shapes do not match the expected dimensions.
        """
        if x.ndim < 2:
            raise ValueError(f"Input tensor x must have at least 2 dimensions, got {x.ndim}.")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Last dimension of input tensor x must match input_size ({self.config.input_size}), got {x.shape[-1]}."
            )
        if idx is not None and idx.ndim != x.ndim - 1:
            raise ValueError(
                f"Positional index tensor idx must have one less dimension than input tensor x, got {idx.ndim} vs {x.ndim}."
            )
        if self.positional_encoder is not None and idx is None:
            raise ValueError("Positional encoding is enabled, but no positional index tensor idx was provided.")
        if mask is not None and mask.shape != x.shape[:-1]:
            raise ValueError(
                f"Mask tensor must have the same leading dimensions as input tensor x, got {mask.shape} vs {x.shape[:-1]}."
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequential observation for one agent.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).
            mask: Tensor of shape (*B, seq_len) indicating valid timesteps.
            idx: Input tensor of shape (*B, 1) containing the positional index for each batch entry. Will be used for
                positional encoding if it's enabled.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        self._pre_forward_checks(x, idx, mask)

        leading_batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        x_flat = x.reshape(-1, seq_len, x.shape[-1])

        embed_output = self.encoder(x_flat)

        # If positional encoding is enabled, add the positional embedding to every step in the sequence.
        if self.positional_encoder is not None and idx is not None:
            # If the idx shape matches the leading batch dimensions of x, we can directly embed it. 
            # Otherwise, we need to reshape it to match the flattened batch dimensions of x_flat.
            if idx.shape[:-1] == x_flat.shape[:-1]:
                positional_embeddings: torch.Tensor = self.positional_encoder(idx.squeeze(-1).to(torch.long))
            else:
                positional_embeddings: torch.Tensor = self.positional_encoder(idx.reshape(-1, 1).to(torch.long)).squeeze(dim=-2)
                positional_embeddings = positional_embeddings.unsqueeze(-2)
            embed_output += positional_embeddings


        fully_masked_out = mask.all(dim=-1)
        transformer_padding_mask = mask.reshape(-1, seq_len).to(torch.bool)
        
        seq_output = self.transformer(embed_output, src_key_padding_mask=transformer_padding_mask)

        # Aggregate only valid timesteps so masked entries do not influence the output.
        valid_timestep_mask = ~fully_masked_out
        valid_timestep_mask = valid_timestep_mask.unsqueeze(-1).to(seq_output.dtype)
        seq_output = (seq_output * valid_timestep_mask).sum(dim=-2) / valid_timestep_mask.sum(dim=-2).clamp_min(1.0)

        return seq_output.reshape(*leading_batch_shape, seq_output.shape[-1])


class FlatHead(nn.Module):
    """
    An MLP-based head for processing a single flat observation key for one agent. The input is a tensor of shape
    (*B, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input.
    """
    def __init__(
        self,
        config: FlatHeadConfig,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the flat head.

        Args:
            config: Configuration for the flat head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        self.mlp = MLP(
            in_features=config.input_size,
            out_features=config.output_size,
            depth=config.depth,
            num_cells=config.hidden_layer_size,
            activation_class=config.activation_class,
            device=device,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Process a flat observation for one agent.

        Args:
            x: Input tensor of shape (*B, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        return self.mlp(x)
