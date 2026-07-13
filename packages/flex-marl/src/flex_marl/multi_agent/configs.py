from dataclasses import dataclass
from enum import StrEnum

from torch import nn

from flex_marl.encoder import (
    FlatHeadConfig,
    PositionalEncodingConfig,
    SequentialHeadConfig,
    validate_head_config,
)


class MultiAgentMode(StrEnum):
    """How encoder parameters and observations are distributed across agents."""

    SHARED = "shared"
    """All agents share the same encoder parameters and receive the same representation for a given field."""
    INDEPENDENT = "independent"
    """Each agent has its own encoder parameters and receives its own representation for a given field."""
    CENTRALIZED = "centralized"
    """All agents are encoded jointly into a single global representation."""


class CentralizedOutput(StrEnum):
    """Shape of the representation returned by a centralized encoder."""

    GLOBAL = "global"
    """Return one global vector for all agents."""
    BROADCAST = "broadcast"
    """Return one global vector for each agent slot, broadcasting the same vector to all slots."""


@dataclass(frozen=True, kw_only=True)
class SequentialFieldOptions:
    """Transformer settings used whenever a field is encoded as a sequence."""

    num_heads: int = 8
    """Number of self-attention heads in each Transformer layer."""

    ff_dim: int = 128
    """Width of the feedforward network inside each Transformer layer."""

    depth: int = 3
    """Number of Transformer layers."""

    dropout: float = 0.1
    """Dropout probability used by the Transformer layers."""

    encode_agent_identity: bool = True
    """Whether sequence elements receive a learned embedding of their owning agent slot."""


@dataclass(frozen=True, kw_only=True)
class FlatFieldConfig:
    """Describe a fixed-size field and how it should be encoded."""

    key: str
    """Input-dictionary key containing a tensor shaped ``(*B, agents, input_size)``."""

    input_size: int
    """Width of the raw field for each agent."""

    output_size: int = 64
    """Width of the intermediate representation produced for this field."""

    depth: int = 3
    """Number of hidden layers used when the field is encoded independently per agent."""

    hidden_layer_size: int = 128
    """Width of each hidden layer in the per-agent MLP."""

    activation_class: type[nn.Module] = nn.ReLU
    """Activation module class used by the per-agent MLP."""

    sequential_options: SequentialFieldOptions | None = None
    """Transformer settings for centralized mode, where agents form a sequence.

    This must be provided when ``MultiAgentMode.CENTRALIZED`` is selected. It is
    unused in shared and independent modes.
    """

@dataclass(frozen=True, kw_only=True)
class SequentialFieldConfig:
    """Describe a variable-size per-agent sequence and its Transformer encoding."""

    key: str
    """Input-dictionary key containing ``(*B, agents, sequence_length, input_size)``."""

    mask_key: str
    """Key containing the Boolean element mask shaped ``(*B, agents, sequence_length)``."""

    input_size: int
    """Width of each raw sequence element."""

    sequential_options: SequentialFieldOptions
    """Transformer architecture used to encode the sequence."""

    output_size: int = 64
    """Width of the intermediate representation produced for this field."""

type FieldConfig = FlatFieldConfig | SequentialFieldConfig


@dataclass(frozen=True, kw_only=True)
class MultiAgentEncoderConfig:
    """Configure multi-agent orchestration around the structured encoder."""

    fields: tuple[FieldConfig, ...]
    """Ordered field descriptions; their encoded representations are mixed in this order."""

    num_agents: int
    """Number of fixed agent slots expected on every input tensor."""

    mode: MultiAgentMode
    """Whether agents share parameters, own independent encoders, or are encoded jointly."""

    agent_mask_key: str
    """Key containing the Boolean active-slot mask shaped ``(*B, num_agents)``."""

    output_dim: int
    """Width of the final representation produced by each underlying encoder."""

    mix_layer_depth: int
    """Number of hidden layers in the MLP that mixes the field representations."""

    mix_layer_num_cells: int
    """Width of each hidden layer in the mix MLP."""

    mix_activation_class: type[nn.Module] | None = None
    """Mix-MLP activation class; ``None`` selects the encoder default of ``nn.Tanh``."""

    centralized_output: CentralizedOutput = CentralizedOutput.GLOBAL
    """Return one global vector or broadcast it across all fixed agent slots."""


def _validate_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def _validate_key(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}.")


def validate_sequential_field_options(options: SequentialFieldOptions) -> None:
    """Validate Transformer settings independently of a field configuration."""

    if not isinstance(options, SequentialFieldOptions):
        raise TypeError(f"options must be a SequentialFieldOptions, got {type(options)}.")
    _validate_positive_int("num_heads", options.num_heads)
    _validate_positive_int("ff_dim", options.ff_dim)
    _validate_positive_int("depth", options.depth)
    if isinstance(options.dropout, bool) or not isinstance(options.dropout, (int, float)) or not (
        0.0 <= options.dropout < 1.0
    ):
        raise ValueError(f"dropout must be in the range [0.0, 1.0), got {options.dropout}.")
    if not isinstance(options.encode_agent_identity, bool):
        raise TypeError("encode_agent_identity must be a bool.")


def validate_field_config(field: FieldConfig, mode: MultiAgentMode) -> None:
    """Validate one field, including requirements imposed by the execution mode."""

    if not isinstance(field, (FlatFieldConfig, SequentialFieldConfig)):
        raise TypeError(f"field must be a FlatFieldConfig or SequentialFieldConfig, got {type(field)}.")
    _validate_key("field key", field.key)
    _validate_positive_int("input_size", field.input_size)
    _validate_positive_int("output_size", field.output_size)
    if isinstance(field, FlatFieldConfig):
        _validate_positive_int("depth", field.depth)
        _validate_positive_int("hidden_layer_size", field.hidden_layer_size)
        if not isinstance(field.activation_class, type) or not issubclass(field.activation_class, nn.Module):
            raise ValueError("activation_class must be an nn.Module class.")
        if field.sequential_options is not None:
            validate_sequential_field_options(field.sequential_options)
        if mode is MultiAgentMode.CENTRALIZED and field.sequential_options is None:
            raise ValueError(
                f"Flat field {field.key!r} requires sequential_options in centralized mode because agents are "
                "encoded as a sequence."
            )
        return

    _validate_key("mask_key", field.mask_key)
    validate_sequential_field_options(field.sequential_options)
    if field.output_size % field.sequential_options.num_heads != 0:
        raise ValueError(
            f"output_size ({field.output_size}) must be divisible by num_heads "
            f"({field.sequential_options.num_heads}) for field {field.key!r}."
        )


def validate_multi_agent_encoder_config(config: MultiAgentEncoderConfig) -> None:
    """Validate a complete multi-agent encoder configuration."""

    if not isinstance(config, MultiAgentEncoderConfig):
        raise TypeError(f"config must be a MultiAgentEncoderConfig, got {type(config)}.")
    if not config.fields:
        raise ValueError("fields must contain at least one field configuration.")
    _validate_positive_int("num_agents", config.num_agents)
    if not isinstance(config.mode, MultiAgentMode):
        raise TypeError(f"mode must be a MultiAgentMode, got {type(config.mode)}.")
    if not isinstance(config.centralized_output, CentralizedOutput):
        raise TypeError(f"centralized_output must be a CentralizedOutput, got {type(config.centralized_output)}.")
    _validate_key("agent_mask_key", config.agent_mask_key)
    _validate_positive_int("output_dim", config.output_dim)
    _validate_positive_int("mix_layer_depth", config.mix_layer_depth)
    _validate_positive_int("mix_layer_num_cells", config.mix_layer_num_cells)
    if config.mix_activation_class is not None and (
        not isinstance(config.mix_activation_class, type) or not issubclass(config.mix_activation_class, nn.Module)
    ):
        raise ValueError("mix_activation_class must be an nn.Module class or None.")

    keys: list[str] = []
    for field in config.fields:
        validate_field_config(field, config.mode)
        keys.append(field.key)
    if len(keys) != len(set(keys)):
        raise ValueError("Field keys must be unique.")


def _internal_key(field_key: str, suffix: str) -> str:
    """Build a private dictionary key shared by compilation and input preparation."""

    return f"__flex_marl_{field_key}_{suffix}"


def compile_head_config(
    field: FieldConfig,
    mode: MultiAgentMode,
    num_agents: int,
) -> FlatHeadConfig | SequentialHeadConfig:
    """Compile a user-facing field into the low-level head required by one mode."""

    validate_field_config(field, mode)
    _validate_positive_int("num_agents", num_agents)

    if isinstance(field, FlatFieldConfig) and mode is not MultiAgentMode.CENTRALIZED:
        head_config: FlatHeadConfig | SequentialHeadConfig = FlatHeadConfig(
            key=field.key,
            input_size=field.input_size,
            output_size=field.output_size,
            depth=field.depth,
            hidden_layer_size=field.hidden_layer_size,
            activation_class=field.activation_class,
        )
    else:
        options = field.sequential_options
        assert options is not None  # Guaranteed by field validation for every sequentially encoded field.
        positional_config = None
        if options.encode_agent_identity:
            positional_config = PositionalEncodingConfig(
                idx_key=_internal_key(field.key, "agent_idx"),
                num_positions=num_agents,
            )
        head_config = SequentialHeadConfig(
            key=field.key,
            mask_key=_internal_key(field.key, "mask"),
            input_size=field.input_size,
            output_size=field.output_size,
            positional_encoding_config=positional_config,
            num_heads=options.num_heads,
            ff_dim=options.ff_dim,
            depth=options.depth,
            dropout=options.dropout,
        )

    validate_head_config(head_config)
    return head_config
