# D-ATC

Reinforcement-learning experiments for distributed air traffic control.

## BlueSky benchmark

The benchmark runs one headless BlueSky simulator per process, because BlueSky
uses process-global simulation and traffic state. It measures rollout throughput
separately from process startup and environment resets.

Run a small smoke benchmark:

```powershell
uv run --package d-atc python projects/d-atc/scripts/benchmark_bsky.py `
  --aircraft 10 --workers 1 --steps 10 --episodes 1
```

Sweep traffic scale and parallelism, recording machine-readable results:

```powershell
uv run --package d-atc python projects/d-atc/scripts/benchmark_bsky.py `
  --aircraft 100 1000 5000 --workers 1 2 4 8 --steps 200 --episodes 3 `
  --json-output projects/d-atc/benchmark-results.json
```

Observation copying is enabled by default to resemble an RL environment. Use
`--no-observe` to isolate simulator throughput. BlueSky navigation data is cached
under `projects/d-atc/.bluesky/`; the first run will therefore take longer.
