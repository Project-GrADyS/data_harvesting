# Flex MARL

`flex-marl` provides reusable neural-network building blocks for multi-agent
reinforcement learning.

## Encoder

Its encoder turns heterogeneous observations into one
fixed-size representation that can be consumed by a policy, value network, or
other downstream model.

The encoder is useful when an observation is composed of several independent
parts. For example, an agent might receive:

- a flat vector describing its own state;
- a variable-length sequence describing nearby entities; and
- another sequence whose representation depends on an agent or position ID.

Each observation part is processed by a dedicated **head**. The head outputs
are concatenated in configuration order and passed through a final MLP, called
the **mix layer**.

```text
flat observation ───────────────→ FlatHead ───────────┐
                                                      │
sequential observation + mask ─→ SequentialHead ──────┼─→ concatenate → mix MLP → encoded output
                                                      │
another observation ───────────→ another head ────────┘
```

All heads preserve arbitrary leading batch dimensions. If every input uses a
batch shape `*B`, the complete encoder returns a tensor shaped
`(*B, output_dim)`.

### Encoder heads

#### Flat head

A flat head uses an MLP to encode a tensor shaped:

```text
(*B, input_size) → (*B, output_size)
```

Its depth, hidden-layer width, and activation class are configurable through
`FlatHeadConfig`.

#### Sequential head

A sequential head uses a Transformer encoder followed by masked mean pooling:

```text
(*B, sequence_length, input_size)
                 ↓ linear projection
(*B, sequence_length, output_size)
                 ↓ Transformer
(*B, sequence_length, output_size)
                 ↓ masked mean pooling
(*B, output_size)
```

`output_size` is the Transformer's `d_model`: the width of each timestep's
internal representation. PyTorch divides that representation evenly among the
configured attention heads, so `output_size` must be divisible by `num_heads`.

Every sequential input requires a Boolean validity mask shaped
`(*B, sequence_length)`:

- `True` means the timestep is valid and participates in attention and pooling.
- `False` means the timestep is padding and must not affect the result.

A sequence whose mask is entirely `False` produces a zero representation.

#### Positional encoding

A sequential head can optionally add a learned positional embedding to every
timestep. The caller supplies one zero-based integer index per batch entry,
shaped `(*B, 1)`.

`num_positions` is the number of available embeddings, so valid indices satisfy:

```text
0 <= index < num_positions
```

Despite the name, the index does not have to represent a timestep. It can also
represent an agent ID, role, source, or any other finite category whose learned
embedding should condition the complete sequence.

### Public API

The following symbols are exported from both `flex_marl` and
`flex_marl.encoder`:

| Symbol                     | Purpose                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `FlatHeadConfig`           | Configures an MLP-based flat-observation head.                                                     |
| `SequentialHeadConfig`     | Configures a Transformer-based sequential head.                                                    |
| `PositionalEncodingConfig` | Configures the optional learned index embedding for a sequential head.                             |
| `MultiHeadEncoderModule`   | Builds the heads, routes dictionary inputs, concatenates their outputs, and applies the mix layer. |
| `validate_head_config`     | Validates a head configuration independently of module construction.                               |

The configuration dataclasses are frozen and keyword-only. Invalid
configurations are also rejected automatically when a
`MultiHeadEncoderModule` is constructed.

### Example

```python
import torch
from torch import nn

from flex_marl import (
    FlatHeadConfig,
    MultiHeadEncoderModule,
    PositionalEncodingConfig,
    SequentialHeadConfig,
)


encoder = MultiHeadEncoderModule(
    head_configs=[
        FlatHeadConfig(
            key="agent_state",
            input_size=12,
            output_size=32,
            depth=2,
            hidden_layer_size=64,
        ),
        SequentialHeadConfig(
            key="neighbors",
            mask_key="neighbors_mask",
            input_size=6,
            output_size=64,
            positional_encoding_config=PositionalEncodingConfig(
                idx_key="agent_index",
                num_positions=8,
            ),
            num_heads=8,
            ff_dim=128,
            depth=2,
            dropout=0.1,
        ),
    ],
    mix_layer_depth=2,
    mix_layer_num_cells=128,
    mix_activation_class=nn.ReLU,
    output_dim=96,
)

batch_size = 4
sequence_length = 10

observations = {
    "agent_state": torch.randn(batch_size, 12),
    "neighbors": torch.randn(batch_size, sequence_length, 6),
    "neighbors_mask": torch.tensor(
        [
            [True, True, True, False, False, False, False, False, False, False],
            [True, True, True, True, True, False, False, False, False, False],
            [True, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
        ]
    ),
    "agent_index": torch.tensor([[0], [1], [2], [3]]),
}

encoded = encoder(observations)
assert encoded.shape == (batch_size, 96)
```

Input dictionary order does not matter. Head outputs are always concatenated in
the order of `head_configs`. All heads must receive the same leading batch
shape, although sequential heads may use different sequence lengths.

### Configuration reference

#### Common head fields

| Field         | Meaning                                          |
| ------------- | ------------------------------------------------ |
| `key`         | Input-dictionary key containing the observation. |
| `input_size`  | Width of each raw observation item.              |
| `output_size` | Width produced by the head.                      |

#### `FlatHeadConfig`

| Field               |   Default | Meaning                                  |
| ------------------- | --------: | ---------------------------------------- |
| `depth`             |       `3` | Number of hidden MLP layers.             |
| `hidden_layer_size` |     `128` | Width of each hidden layer.              |
| `activation_class`  | `nn.ReLU` | Activation module class used by the MLP. |

#### `SequentialHeadConfig`

| Field                        |  Default | Meaning                                                                 |
| ---------------------------- | -------: | ----------------------------------------------------------------------- |
| `mask_key`                   | required | Input-dictionary key containing the Boolean validity mask.              |
| `positional_encoding_config` | required | A `PositionalEncodingConfig`, or `None` to disable positional encoding. |
| `num_heads`                  |      `8` | Number of parallel self-attention heads.                                |
| `ff_dim`                     |    `128` | Hidden width of each Transformer block's feed-forward network.          |
| `depth`                      |      `3` | Number of Transformer encoder blocks.                                   |
| `dropout`                    |    `0.1` | Dropout probability; must be in `[0, 1)`.                               |

#### `PositionalEncodingConfig`

| Field           | Meaning                                                             |
| --------------- | ------------------------------------------------------------------- |
| `idx_key`       | Input-dictionary key containing indices shaped `(*B, 1)`.           |
| `num_positions` | Number of learned embeddings and exclusive upper bound for indices. |

### Project organization

```text
packages/flex-marl/
├── README.md
├── pyproject.toml
├── src/flex_marl/
│   ├── __init__.py             # Root public exports
│   ├── py.typed                # Marks the package as typed
│   └── encoder/
│       ├── __init__.py         # Encoder public exports
│       ├── configs.py          # Frozen configs and validation
│       ├── heads.py            # FlatHead and SequentialHead
│       └── flex.py             # MultiHeadEncoderModule and mix layer
└── tests/encoder/
    ├── conftest.py             # Shared deterministic fixtures
    ├── test_configs.py         # Configuration validation
    ├── test_flat_head.py       # Flat-head behavior
    ├── test_sequential_head.py # Transformer, masks, and positions
    ├── test_multi_head_encoder.py
    └── test_public_api.py
```

`FlatHead` and `SequentialHead` are implementation building blocks. Most users
should construct them indirectly through `MultiHeadEncoderModule` and the public
configuration types.

### Development

This package is a member of the repository's uv workspace. From the repository
root, run its tests with:

```bash
uv run pytest packages/flex-marl/tests/encoder
```

The suite covers configuration boundaries, tensor shapes, mask semantics,
positional encoding, input routing, gradients, serialization, devices, and
dtypes.
