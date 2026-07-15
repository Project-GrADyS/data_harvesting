# Validation Core

`validation-core` provides small, dependency-free runtime validation helpers
shared by the reusable packages in this workspace.

The helpers distinguish invalid types from invalid values:

- a value of the wrong runtime type raises `TypeError`;
- a value of the expected type but outside its accepted domain raises
  `ValueError`.

Package-specific invariants remain in the package that owns them.
