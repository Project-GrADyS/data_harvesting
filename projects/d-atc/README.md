# D-ATC

Reinforcement-learning experiments for distributed air traffic control.

## GrADyS/BlueSky circle simulation

Run a finite integration scenario with aircraft randomly placed near a geographic
center and controlled by `AircraftCircleProtocol`:

```powershell
uv run --package d-atc circle-simulation `
  --latitude -23.5505 --longitude -46.6333 --altitude-m 1500 `
  --aircraft 5 --radius-m 5000 --altitude-spread-m 250 `
  --speed-mps 120 --duration-s 120 --seed 42
```

Placement is uniform over the configured disk and deterministic for a given
seed. Altitudes and speeds use meters and meters per second; headings generated
for BlueSky use degrees. Use `--real-time` to pace the simulation against wall
clock time and `--verbose` to enable GrADyS event logging. Run
`uv run --package d-atc circle-simulation --help` for every option.

Add `--visualization` to open BlueSky's interactive QtGL traffic view. The
view supports navigation, zoom, labels, and map layers, but does not forward
commands that would change the GrADyS-owned simulation. Pair it with
`--real-time` for comfortable interactive viewing; without real-time pacing,
the simulation is allowed to run as fast as possible. Closing the view does
not stop the simulation.
