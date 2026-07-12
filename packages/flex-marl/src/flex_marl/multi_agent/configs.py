from dataclasses import dataclass
from enum import StrEnum

from torch import nn


class MultiAgentMode(StrEnum):
    SHARED = "shared"
    INDEPENDENT = "independent"
    CENTRALIZED = "centralized"


class CentralizedOutput(StrEnum):
    GLOBAL = "global"
    BROADCAST = "broadcast"


@dataclass(frozen=True, kw_only=True)
class FlatFieldConfig:
    key: str
    input_size: int

    output_size: int = 64
    depth: int = 3
    hidden_layer_size: int = 128
    activation_class: type[nn.Module] = nn.ReLU

    encode_agent_identity: bool = True
    centralized_num_heads: int = 8
    centralized_ff_dim: int = 128
    centralized_depth: int = 3
    centralized_dropout: float = 0.1


@dataclass(frozen=True, kw_only=True)
class SequentialFieldConfig:
    key: str
    mask_key: str
    input_size: int

    output_size: int = 64
    num_heads: int = 8
    ff_dim: int = 128
    depth: int = 3
    dropout: float = 0.1

    encode_agent_identity: bool = True


type FieldConfig = FlatFieldConfig | SequentialFieldConfig


@dataclass(frozen=True, kw_only=True)
class MultiAgentEncoderConfig:
    fields: tuple[FieldConfig, ...]
    num_agents: int
    mode: MultiAgentMode
    agent_mask_key: str
    output_dim: int
    mix_layer_depth: int
    mix_layer_num_cells: int
    mix_activation_class: type[nn.Module] | None = None
    centralized_output: CentralizedOutput = CentralizedOutput.GLOBAL
