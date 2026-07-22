import math
from types import SimpleNamespace

import pytest

from d_atc import circle_simulation
from d_atc.circle_simulation import (
    CircleSimulationConfiguration,
    build_circle_simulation,
    generate_aircraft_initial_states,
    validate_circle_simulation_config,
)
from d_atc.environment.protocol import AircraftCircleProtocol


def test_generated_aircraft_are_deterministic_and_within_configured_volume() -> None:
    config = CircleSimulationConfiguration(
        center=(10.0, 179.9, 2_000.0),
        aircraft_count=20,
        placement_radius_m=3_000.0,
        altitude_spread_m=200.0,
        seed=7,
    )

    first = generate_aircraft_initial_states(config)
    second = generate_aircraft_initial_states(config)

    assert first == second
    assert len(first) == 20
    for state in first:
        east, north, up = state.local_position
        assert math.hypot(east, north) <= config.placement_radius_m
        assert abs(up) <= config.altitude_spread_m
        assert state.altitude_m == pytest.approx(config.center[2] + up)
        assert -90.0 <= state.latitude <= 90.0
        assert -180.0 <= state.longitude <= 180.0
        assert 0.0 <= state.heading_deg < 360.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"aircraft_count": 0}, "aircraft_count"),
        ({"placement_radius_m": 0.0}, "placement_radius_m"),
        ({"altitude_spread_m": -1.0}, "altitude_spread_m"),
        ({"initial_speed_mps": 0.0}, "initial_speed_mps"),
        ({"aircraft_type": " "}, "aircraft_type"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"update_rate_s": float("nan")}, "update_rate_s"),
        ({"seed": -1}, "seed"),
        ({"center": (91.0, 0.0, 0.0)}, "latitude"),
        ({"visualization": "yes"}, "visualization"),
    ],
)
def test_circle_configuration_rejects_invalid_values(overrides, message: str) -> None:
    values = {
        "center": (0.0, 0.0, 1_000.0),
        "aircraft_count": 2,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_circle_simulation_config(CircleSimulationConfiguration(**values))


def test_builder_connects_gradysim_nodes_to_matching_bluesky_aircraft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    built_simulator = SimpleNamespace(start_simulation=lambda: None)

    class FakeMobilityHandler:
        def __init__(self, config) -> None:
            self.config = config
            self.initialized: list[dict[str, object]] = []

        def initialize_aircraft(self, **kwargs) -> None:
            self.initialized.append(kwargs)

    class FakeBuilder:
        instance = None

        def __init__(self, config) -> None:
            self.config = config
            self.handler = None
            self.nodes: list[tuple[type, tuple[float, float, float]]] = []
            FakeBuilder.instance = self

        def add_handler(self, handler):
            self.handler = handler
            return self

        def add_node(self, protocol, position):
            self.nodes.append((protocol, position))
            return len(self.nodes) - 1

        def build(self):
            return built_simulator

    monkeypatch.setattr(circle_simulation, "BlueSkyMobilityHandler", FakeMobilityHandler)
    monkeypatch.setattr(circle_simulation, "SimulationBuilder", FakeBuilder)
    config = CircleSimulationConfiguration(
        center=(1.0, 2.0, 1_000.0),
        aircraft_count=3,
        seed=5,
        bluesky_workdir=tmp_path,
    )

    simulation = build_circle_simulation(config)

    builder = FakeBuilder.instance
    assert simulation.simulator is built_simulator
    assert len(simulation.aircraft) == 3
    assert [protocol for protocol, _ in builder.nodes] == [AircraftCircleProtocol] * 3
    assert [call["node_id"] for call in simulation.mobility_handler.initialized] == [0, 1, 2]
    for call, state in zip(simulation.mobility_handler.initialized, simulation.aircraft, strict=True):
        assert call["initial_latitude"] == state.latitude
        assert call["initial_longitude"] == state.longitude
        assert call["initial_altitude"] == state.altitude_m
        assert call["initial_heading"] == state.heading_deg
        assert call["initial_speed"] == config.initial_speed_mps


def test_circle_protocol_advances_heading_as_aircraft_reaches_targets() -> None:
    commands = []
    protocol = AircraftCircleProtocol()
    protocol.provider = SimpleNamespace(send_mobility_command=commands.append)

    def telemetry(heading: float):
        return SimpleNamespace(heading=heading)

    protocol.handle_telemetry(telemetry(90.0))
    protocol.handle_telemetry(telemetry(92.0))
    protocol.handle_telemetry(telemetry(96.0))

    assert [command.param_1 for command in commands] == [100.0, 110.0]


def test_cli_builds_and_runs_configured_simulation(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured = {}

    class FakeSimulation:
        def run(self) -> None:
            captured["ran"] = True

    def fake_build(config):
        captured["config"] = config
        return FakeSimulation()

    monkeypatch.setattr(circle_simulation, "build_circle_simulation", fake_build)

    result = circle_simulation.main(
        [
            "--latitude", "51.0",
            "--longitude", "-1.0",
            "--altitude-m", "2000",
            "--aircraft", "3",
            "--duration-s", "10",
            "--seed", "9",
        ]
    )

    assert result == 0
    assert captured["ran"] is True
    assert captured["config"].center == (51.0, -1.0, 2_000.0)
    assert captured["config"].aircraft_count == 3
    assert captured["config"].duration_s == 10.0
    assert captured["config"].seed == 9
    assert "Building 3 aircraft" in capsys.readouterr().out
