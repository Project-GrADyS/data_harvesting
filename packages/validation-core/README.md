# Validation Core

`validation-core` provides small, dependency-free runtime validation helpers
shared by the reusable packages in this workspace.

The helpers distinguish invalid types from invalid values:

- a value of the wrong runtime type raises `TypeError`;
- a value of the expected type but outside its accepted domain raises
  `ValueError`.

Package-specific invariants remain in the package that owns them.

## Public helpers

- `validate_positive_integer`
- `validate_non_negative_integer`
- `validate_non_empty_string`
- `validate_probability`
- `validate_bool`
- `validate_mapping`
- `validate_callable`

Each helper accepts a field name and a value, raises on failure, and returns
`None` on success. In particular, Boolean values are not accepted as integers,
and probabilities must be finite values in the half-open interval `[0, 1)`.

The package intentionally owns only broadly reusable checks. Structural and
cross-field validation stays beside the dataclass or runtime boundary whose
contract defines it.

## Development

From the repository root, run:

```bash
uv run pytest packages/validation-core/tests
```
