# Reinforcement-Learning Workspace

This repository is a uv workspace containing reusable reinforcement-learning
packages and the data-harvesting application.

```text
packages/
  flex-marl/
  rl-core/
  validation-core/
projects/
  data-harvesting/
```

Run the application test suite from the workspace root with:

```bash
uv run --package data-harvesting pytest projects/data-harvesting/tests
```

Application commands and MLflow setup are documented in
`projects/data-harvesting/README.md`.
