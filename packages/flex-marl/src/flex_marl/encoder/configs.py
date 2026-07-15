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
    input_size: int
    output_size: int = 64


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionalEncodingConfig:
    """Configure learned integer-index embeddings for a sequential head."""

    idx_key: str
    num_positions: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SequentialHeadConfig(_BaseHeadConfig):
    """Configure a Transformer-based sequential encoder head."""

    mask_key: str
    positional_encoding_config: PositionalEncodingConfig | None
    num_heads: int = 8
    ff_dim: int = 128
    depth: int = 3
    dropout: float = 0.1


@dataclass(frozen=True, slots=True, kw_only=True)
class FlatHeadConfig(_BaseHeadConfig):
    """Configure an MLP-based flat encoder head."""

    depth: int = 3
    hidden_layer_size: int = 128
    activation_class: type[nn.Module] = nn.ReLU


HeadConfig: TypeAlias = SequentialHeadConfig | FlatHeadConfig


def _validate_module_class(name: str, value: object) -> None:
    if not isinstance(value, type) or not issubclass(value, nn.Module):
        raise TypeError(f"{name} must be an nn.Module subclass.")


def _validate_base_head_config(head_config: _BaseHeadConfig) -> None:
    validate_non_empty_string("key", head_config.key)
    validate_positive_integer("input_size", head_config.input_size)
    validate_positive_integer("output_size", head_config.output_size)


def _validate_sequential_head_config(head_config: SequentialHeadConfig) -> None:
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
    _validate_base_head_config(head_config)
    validate_positive_integer("depth", head_config.depth)
    validate_positive_integer("hidden_layer_size", head_config.hidden_layer_size)
    _validate_module_class("activation_class", head_config.activation_class)


def validate_head_config(head_config: HeadConfig) -> None:
    """Validate one encoder-head configuration."""

    if isinstance(head_config, SequentialHeadConfig):
        _validate_sequential_head_config(head_config)
    elif isinstance(head_config, FlatHeadConfig):
        _validate_flat_head_config(head_config)
    else:
        raise TypeError(
            "head_config must be a SequentialHeadConfig or FlatHeadConfig, "
            f"got {type(head_config)}."
        )
