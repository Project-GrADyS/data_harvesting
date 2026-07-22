from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Sequence

from gradysim.simulator.simulation import (
    SimulationBuilder,
    SimulationConfiguration,
    Simulator,
)
from validation_core import (
    validate_bool,
    validate_non_empty_string,
    validate_non_negative_integer,
    validate_non_negative_real,
    validate_positive_integer,
    validate_positive_real,
)

from d_atc.environment.coordinates import HomeCoordinateFrame
from d_atc.environment.mobility import (
    BlueSkyMobilityConfiguration,
    BlueSkyMobilityHandler,
)
from d_atc.environment.protocol import AircraftCircleProtocol


DEFAULT_CENTER = (-23.5505, -46.6333, 1_500.0)
DEFAULT_BLUESKY_WORKDIR = Path(__file__).resolve().parents[3] / ".bluesky"


@dataclass(frozen=True, kw_only=True, slots=True)
class CircleSimulationConfiguration:
    center: tuple[float, float, float] = DEFAULT_CENTER
    aircraft_count: int = 5
    placement_radius_m: float = 5_000.0
    altitude_spread_m: float = 250.0
    initial_speed_mps: float = 120.0
    aircraft_type: str = "A320"
    duration_s: float = 120.0
    update_rate_s: float = 0.05
    seed: int = 42
    bluesky_workdir: Path = DEFAULT_BLUESKY_WORKDIR
    real_time: bool = False
    execution_logging: bool = False
    visualization: bool = False


def validate_circle_simulation_config(config: CircleSimulationConfiguration) -> None:
    if not isinstance(config, CircleSimulationConfiguration):
        raise TypeError(
            "config must be a CircleSimulationConfiguration, "
            f"got {type(config)}."
        )

    HomeCoordinateFrame(config.center)
    validate_positive_integer("aircraft_count", config.aircraft_count)
    validate_positive_real("placement_radius_m", config.placement_radius_m)
    validate_non_negative_real("altitude_spread_m", config.altitude_spread_m)
    validate_positive_real("initial_speed_mps", config.initial_speed_mps)
    validate_non_empty_string("aircraft_type", config.aircraft_type)
    if not config.aircraft_type.strip():
        raise ValueError("aircraft_type must not be blank")
    validate_positive_real("duration_s", config.duration_s)
    validate_positive_real("update_rate_s", config.update_rate_s)
    validate_non_negative_integer("seed", config.seed)
    if not isinstance(config.bluesky_workdir, Path):
        raise TypeError("bluesky_workdir must be a pathlib.Path")
    validate_bool("real_time", config.real_time)
    validate_bool("execution_logging", config.execution_logging)
    validate_bool("visualization", config.visualization)


@dataclass(frozen=True, slots=True)
class AircraftInitialState:
    local_position: tuple[float, float, float]
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float


@dataclass(frozen=True, slots=True)
class CircleSimulation:
    simulator: Simulator
    mobility_handler: BlueSkyMobilityHandler
    aircraft: tuple[AircraftInitialState, ...]

    def run(self) -> None:
        self.simulator.start_simulation()


def generate_aircraft_initial_states(
    config: CircleSimulationConfiguration,
) -> tuple[AircraftInitialState, ...]:
    validate_circle_simulation_config(config)
    coordinate_frame = HomeCoordinateFrame(config.center)
    rng = random.Random(config.seed)
    aircraft: list[AircraftInitialState] = []

    for _ in range(config.aircraft_count):
        radius = config.placement_radius_m * math.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * math.pi)
        east = radius * math.cos(angle)
        north = radius * math.sin(angle)
        up = rng.uniform(-config.altitude_spread_m, config.altitude_spread_m)
        latitude, longitude, altitude = coordinate_frame.local_to_geographic(east, north, up)
        heading = rng.uniform(0.0, 360.0)

        aircraft.append(
            AircraftInitialState(
                local_position=(east, north, up),
                latitude=float(latitude),
                longitude=float(longitude),
                altitude_m=float(altitude),
                heading_deg=heading,
            )
        )

    return tuple(aircraft)


def build_circle_simulation(config: CircleSimulationConfiguration) -> CircleSimulation:
    validate_circle_simulation_config(config)
    aircraft = generate_aircraft_initial_states(config)
    mobility_handler = BlueSkyMobilityHandler(
        BlueSkyMobilityConfiguration(
            base_workdir=config.bluesky_workdir,
            home=config.center,
            update_rate=config.update_rate_s,
            visualization=config.visualization,
        )
    )
    builder = SimulationBuilder(
        SimulationConfiguration(
            duration=config.duration_s,
            real_time=config.real_time,
            execution_logging=config.execution_logging,
        )
    )
    builder.add_handler(mobility_handler)

    node_ids = [
        builder.add_node(AircraftCircleProtocol, state.local_position)
        for state in aircraft
    ]
    simulator = builder.build()

    for node_id, state in zip(node_ids, aircraft, strict=True):
        mobility_handler.initialize_aircraft(
            node_id=node_id,
            aircraft_type=config.aircraft_type,
            initial_latitude=state.latitude,
            initial_longitude=state.longitude,
            initial_altitude=state.altitude_m,
            initial_heading=state.heading_deg,
            initial_speed=config.initial_speed_mps,
        )

    return CircleSimulation(
        simulator=simulator,
        mobility_handler=mobility_handler,
        aircraft=aircraft,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run circling aircraft through the GrADyS/BlueSky mobility integration.",
    )
    parser.add_argument("--latitude", type=float, default=DEFAULT_CENTER[0])
    parser.add_argument("--longitude", type=float, default=DEFAULT_CENTER[1])
    parser.add_argument("--altitude-m", type=float, default=DEFAULT_CENTER[2])
    parser.add_argument("--aircraft", type=int, default=5, dest="aircraft_count")
    parser.add_argument("--radius-m", type=float, default=5_000.0, dest="placement_radius_m")
    parser.add_argument("--altitude-spread-m", type=float, default=250.0)
    parser.add_argument("--speed-mps", type=float, default=120.0, dest="initial_speed_mps")
    parser.add_argument("--aircraft-type", default="A320")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--update-rate-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_BLUESKY_WORKDIR)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable GrADyS execution logging.")
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Open BlueSky's read-only interactive QtGL view.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = CircleSimulationConfiguration(
        center=(args.latitude, args.longitude, args.altitude_m),
        aircraft_count=args.aircraft_count,
        placement_radius_m=args.placement_radius_m,
        altitude_spread_m=args.altitude_spread_m,
        initial_speed_mps=args.initial_speed_mps,
        aircraft_type=args.aircraft_type,
        duration_s=args.duration_s,
        update_rate_s=args.update_rate_s,
        seed=args.seed,
        bluesky_workdir=args.workdir,
        real_time=args.real_time,
        execution_logging=args.verbose,
        visualization=args.visualization,
    )
    try:
        validate_circle_simulation_config(config)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Building {config.aircraft_count} aircraft within {config.placement_radius_m:.0f} m "
        f"of ({config.center[0]:.5f}, {config.center[1]:.5f})..."
    )
    simulation = build_circle_simulation(config)
    simulation.run()
    print(f"Completed {config.duration_s:.1f} simulated seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
