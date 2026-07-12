from dataclasses import dataclass
from torch import nn

@dataclass(frozen=True, kw_only=True)
class _BaseHeadConfig:
    """
    Configuration for a single head in a multi-head encoder.
    """
    key: str
    """The TensorDict key this head processes."""
    input_size: int
    """Dimensionality of the input feature for this head."""
    output_size: int = 64
    """Dimensionality of the output embedding produced by this head."""

@dataclass(frozen=True, kw_only=True)
class PositionalEncodingConfig:
    """
    Configuration for positional encoding in a sequential head.
    """
    idx_key: str | None
    """The TensorDict key for the positional index corresponding to this head, if any."""
    num_positions: int
    """Number of unique positions to encode. This determines the size of the embedding layer for positional encoding."""

@dataclass(frozen=True, kw_only=True)
class SequentialHeadConfig(_BaseHeadConfig):
    """
    Configuration for a single sequential head in a multi-head encoder.
    """
    mask_key: str
    """The TensorDict key for the mask corresponding to this head"""
    positional_encoding_config: PositionalEncodingConfig | None
    """Configuration for positional encoding in this head, if any."""
    num_heads: int = 8
    """Number of attention heads in the Transformer blocks."""
    ff_dim: int = 128
    """Dimensionality of the feedforward layers in the Transformer blocks."""
    depth: int = 3
    """Number of Transformer blocks in the head."""
    dropout: float = 0.1
    """Dropout rate used in the Transformer blocks."""

@dataclass(frozen=True, kw_only=True)
class FlatHeadConfig(_BaseHeadConfig):
    """
    Configuration for a single flat head in a multi-head encoder.
    """
    depth: int = 3
    """Number of hidden layers in the MLP."""
    hidden_layer_size: int = 128
    """Number of cells per hidden layer in the MLP."""
    activation_class: type[nn.Module] = nn.ReLU
    """Activation function class used between MLP layers."""

type HeadConfig = SequentialHeadConfig | FlatHeadConfig

def _validate_base_head_config(head_config: _BaseHeadConfig) -> None:
    """
    Validate the base configuration of a head.

    Args:
        head_config: The base head configuration to validate.

    Raises:
        ValueError: If any of the base configuration parameters are invalid.
    """
    if head_config.input_size <= 0:
        raise ValueError(f"input_size must be a positive integer, got {head_config.input_size}.")
    if head_config.output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {head_config.output_size}.")
    
def _validate_sequential_head_config(head_config: SequentialHeadConfig) -> None:
    """
    Validate the configuration of a sequential head.

    Args:
        head_config: The sequential head configuration to validate.

    Raises:
        ValueError: If any of the configuration parameters are invalid.
    """
    _validate_base_head_config(head_config)
    
    if head_config.num_heads <= 0:
        raise ValueError(f"num_heads must be a positive integer, got {head_config.num_heads}.")
    if head_config.ff_dim <= 0:
        raise ValueError(f"ff_dim must be a positive integer, got {head_config.ff_dim}.")
    if head_config.depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {head_config.depth}.")
    if not (0.0 <= head_config.dropout < 1.0):
        raise ValueError(f"dropout must be in the range [0.0, 1.0), got {head_config.dropout}.")
    if head_config.positional_encoding_config is not None:
        if head_config.positional_encoding_config.num_positions <= 0:
            raise ValueError(
                f"num_positions must be a positive integer, got {head_config.positional_encoding_config.num_positions}."
            )
    if head_config.output_size % head_config.num_heads != 0:
        raise ValueError(
            f"output_size ({head_config.output_size}) must be divisible by num_heads ({head_config.num_heads})."
        )
    
def _validate_flat_head_config(head_config: FlatHeadConfig) -> None:
    """
    Validate the configuration of a flat head.

    Args:
        head_config: The flat head configuration to validate.

    Raises:
        ValueError: If any of the configuration parameters are invalid.
    """
    _validate_base_head_config(head_config)
    
    if head_config.depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {head_config.depth}.")
    if head_config.hidden_layer_size <= 0:
        raise ValueError(f"hidden_layer_size must be a positive integer, got {head_config.hidden_layer_size}.")
    

def validate_head_config(
    head_config: HeadConfig,
) -> None:
    """
    Validate the configuration of a head.

    Args:
        head_config: The head configuration to validate.

    Raises:
        ValueError: If any of the configuration parameters are invalid.
    """
    _validate_base_head_config(head_config)

    if isinstance(head_config, SequentialHeadConfig):
        _validate_sequential_head_config(head_config)
    elif isinstance(head_config, FlatHeadConfig):
        _validate_flat_head_config(head_config)