# Multi-agent encoder

`MultiAgentEncoderModule` separates multi-agent orchestration from structured
representation learning. Users describe each observation once with a field
configuration. The module compiles those descriptions into the existing
`MultiHeadEncoderModule`, reshapes the tensors for the selected execution mode,
and restores the expected output layout.

The module assumes fixed agent slots. Every input contains exactly
`num_agents` slots, and `agent_mask_key` identifies which slots are active.

## Input shapes

With arbitrary leading batch dimensions `*B`:

```text
agent mask:       (*B, agents)
flat field:       (*B, agents, input_size)
sequential field: (*B, agents, sequence_length, input_size)
sequence mask:    (*B, agents, sequence_length)
```

Masks are Boolean and use `True` for valid agents or elements. For sequential
fields, an element is used only when both its element mask and its owning
agent's mask are `True`.

## Execution modes

### Shared

One encoder is reused for all agents. The agent axis remains a leading batch
dimension, so each agent is encoded independently with shared parameters:

```text
(*B, agents, ..., features) -> (*B, agents, output_dim)
```

Inactive slots are zeroed in the final output.

### Independent

Each fixed slot owns a separate encoder and parameter set. The module selects
one slot from every field, invokes that slot's encoder, and stacks all results:

```text
encoder[0](agent 0), ..., encoder[A-1](agent A-1)
                         -> (*B, agents, output_dim)
```

Inactive slots are zeroed in the final output.

### Centralized

One encoder observes all active agents jointly. A flat field becomes a sequence
whose elements are the fixed agent slots:

```text
(*B, agents, features) -> (*B, sequence=agents, features)
```

A sequential field concatenates the per-agent sequences in slot order:

```text
(*B, agents, sequence, features)
    -> (*B, agents * sequence, features)
```

The element mask is combined with the active-agent mask and flattened in the
same order. Agent-ID indices are reshaped identically, so every flattened
element retains the identity of its owning slot.

Centralized mode returns one global vector by default. Set
`centralized_output=CentralizedOutput.BROADCAST` to expose the same vector at
every fixed slot without recomputing it.

## Field configuration

`FlatFieldConfig` configures the MLP used in shared and independent modes. In
centralized mode the agent values must instead be processed as a sequence, so
the field must explicitly provide `SequentialFieldOptions`. Construction fails
when centralized mode contains a flat field without those options.

`SequentialFieldConfig` always requires `SequentialFieldOptions`, which contains
only Transformer architecture settings:

- `num_heads`
- `ff_dim`
- `depth`
- `dropout`
- `encode_agent_identity`

`output_size` remains on the field because it is the representation width of
that field regardless of execution mode. `encode_agent_identity` belongs to the
sequential options because it controls whether the module adds learned agent-ID
embeddings while a field is being encoded as a sequence.

Configurations can be checked without constructing a module through
`validate_sequential_field_options`, `validate_field_config`, and
`validate_multi_agent_encoder_config`. Module construction performs the complete
validation automatically.

## Example

```python
import torch

from flex_marl import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
)


transformer = SequentialFieldOptions(
    num_heads=4,
    ff_dim=128,
    depth=2,
    dropout=0.1,
)

config = MultiAgentEncoderConfig(
    fields=(
        SequentialFieldConfig(
            key="neighbors",
            mask_key="neighbors_mask",
            input_size=6,
            output_size=32,
            sequential_options=transformer,
        ),
        FlatFieldConfig(
            key="action",
            input_size=2,
            output_size=32,
            sequential_options=transformer,
        ),
    ),
    num_agents=3,
    mode=MultiAgentMode.CENTRALIZED,
    agent_mask_key="agent_mask",
    output_dim=64,
    mix_layer_depth=2,
    mix_layer_num_cells=128,
    centralized_output=CentralizedOutput.GLOBAL,
)

encoder = MultiAgentEncoderModule(config)
inputs = {
    "neighbors": torch.randn(5, 3, 8, 6),
    "neighbors_mask": torch.ones(5, 3, 8, dtype=torch.bool),
    "action": torch.randn(5, 3, 2),
    "agent_mask": torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, True],
            [True, False, True],
        ]
    ),
}

encoded = encoder(inputs)
assert encoded.shape == (5, 64)
```

The core module currently consumes a string-keyed tensor dictionary. Adapting
nested TensorDict keys belongs to a separate integration layer.
