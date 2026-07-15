from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import nn
from validation_core import (
    validate_non_empty_string,
    validate_positive_integer,
    validate_probability,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _BaseHeadConfig:
    """Configuration shared by every encoder head."""

    key: str
    """The input-dictionary key this head processes."""

    input_size: int
    """Dimensionality of the input feature processed by this head."""

    output_size: int = 64
    """Dimensionality of the embedding produced by this head."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionalEncodingConfig:
    """Configure learned integer-index embeddings for a sequential head."""

    idx_key: str
    """The input-dictionary key containing each sequence element's positional index."""

    num_positions: int
    """Number of unique indices represented by the positional embedding table."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SequentialHeadConfig(_BaseHeadConfig):
    """Configure a Transformer-based sequential encoder head."""

    mask_key: str
    """The input-dictionary key containing this head's Boolean validity mask."""

    positional_encoding_config: PositionalEncodingConfig | None
    """Positional-encoding settings, or ``None`` to disable positional encoding."""

    num_heads: int = 8
    """Number of attention heads in each Transformer block."""

    ff_dim: int = 128
    """Dimensionality of the feedforward layers in each Transformer block."""

    depth: int = 3
    """Number of Transformer blocks in the head."""

    dropout: float = 0.1
    """Dropout probability used in the Transformer blocks."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FlatHeadConfig(_BaseHeadConfig):
    """Configure an MLP-based flat encoder head."""

    depth: int = 3
    """Number of hidden layers in the MLP."""

    hidden_layer_size: int = 128
    """Number of cells in each hidden MLP layer."""

    activation_class: type[nn.Module] = nn.ReLU
    """Activation module class used between MLP layers."""


HeadConfig: TypeAlias = SequentialHeadConfig | FlatHeadConfig


def _validate_module_class(name: str, value: object) -> None:
    if not isinstance(value, type) or not issubclass(value, nn.Module):
        raise TypeError(f"{name} must be an nn.Module subclass.")


def _validate_base_head_config(head_config: _BaseHeadConfig) -> None:
    """Validate fields shared by every head configuration.

    Args:
        head_config: The base head configuration to validate.

    Raises:
        TypeError: If a field has the wrong runtime type.
        ValueError: If a field has the correct type but an invalid value.
    """

    validate_non_empty_string("key", head_config.key)
    validate_positive_integer("input_size", head_config.input_size)
    validate_positive_integer("output_size", head_config.output_size)


def _validate_sequential_head_config(head_config: SequentialHeadConfig) -> None:
    """Validate a sequential-head configuration.

    Args:
        head_config: The sequential head configuration to validate.

    Raises:
        TypeError: If a field has the wrong runtime type.
        ValueError: If a field has the correct type but an invalid value.
    """

    _validate_base_head_config(head_config)
    validate_non_empty_string("mask_key", head_config.mask_key)
    validate_positive_integer("num_heads", head_config.num_heads)
    validate_positive_integer("ff_dim", head_config.ff_dim)
    validate_positive_integer("depth", head_config.depth)
    validate_probability("dropout", head_config.dropout)

    positional_config = head_config.positional_encoding_config
    if positional_config is not None:
        if not isinstance(positional_config, PositionalEncodingConfig):
            raise TypeError(
                "positional_encoding_config must be a PositionalEncodingConfig or None."
            )
        validate_non_empty_string("idx_key", positional_config.idx_key)
        validate_positive_integer("num_positions", positional_config.num_positions)

    if head_config.output_size % head_config.num_heads != 0:
        raise ValueError(
            f"output_size ({head_config.output_size}) must be divisible by "
            f"num_heads ({head_config.num_heads})."
        )


def _validate_flat_head_config(head_config: FlatHeadConfig) -> None:
    """Validate a flat-head configuration.

    Args:
        head_config: The flat head configuration to validate.

    Raises:
        TypeError: If a field has the wrong runtime type.
        ValueError: If a field has the correct type but an invalid value.
    """

    _validate_base_head_config(head_config)
    validate_positive_integer("depth", head_config.depth)
    validate_positive_integer("hidden_layer_size", head_config.hidden_layer_size)
    _validate_module_class("activation_class", head_config.activation_class)


def validate_head_config(head_config: HeadConfig) -> None:
    """Validate one encoder-head configuration.

    Args:
        head_config: The head configuration to validate.

    Raises:
        TypeError: If the object or one of its fields has the wrong runtime type.
        ValueError: If a field has the correct type but an invalid value.
    """

    if isinstance(head_config, SequentialHeadConfig):
        _validate_sequential_head_config(head_config)
    elif isinstance(head_config, FlatHeadConfig):
        _validate_flat_head_config(head_config)
    else:
        raise TypeError(
            "head_config must be a SequentialHeadConfig or FlatHeadConfig, "
            f"got {type(head_config)}."
        )
