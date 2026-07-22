import uuid
from dataclasses import dataclass
from pathlib import Path

import bluesky as bs
from bluesky.core.walltime import Timer
from bluesky.simulation import Simulation
from bluesky.tools.aero import ft
from bluesky.traffic import Traffic
from gradysim.protocol.messages.mobility import MobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.node import Node
from validation_core import (
    validate_bool,
    validate_finite_real,
    validate_non_empty_string,
    validate_non_negative_real,
    validate_positive_real,
)

from d_atc.environment.coordinates import HomeCoordinateFrame
from d_atc.environment.visualization import BlueSkyVisualization


@dataclass(frozen=True, kw_only=True, slots=True)
class BlueSkyMobilityConfiguration:
    base_workdir: Path = Path("./.bluesky")
    """Base directory for the BlueSky simulation working directory. 
    A unique subdirectory will be created for each simulation instance."""
    home: tuple[float, float, float]
    """Geographic coordinates of the home location (latitude, longitude, altitude) in degrees and meters."""
    update_rate: float = 0.05
    """How often to update the mobility."""
    visualization: bool = False
    """Enable BlueSky's interactive, read-only QtGL visualization."""


def validate_bluesky_mobility_config(config: BlueSkyMobilityConfiguration) -> None:
    """Validate a BlueSky mobility configuration before constructing its handler."""

    if not isinstance(config, BlueSkyMobilityConfiguration):
        raise TypeError(
            "config must be a BlueSkyMobilityConfiguration, "
            f"got {type(config)}."
        )
    if not isinstance(config.base_workdir, Path):
        raise TypeError("base_workdir must be a pathlib.Path")
    if config.base_workdir.exists() and not config.base_workdir.is_dir():
        raise ValueError("base_workdir must refer to a directory")

    validate_positive_real("update_rate", config.update_rate)
    validate_bool("visualization", config.visualization)

    HomeCoordinateFrame(config.home)


@dataclass(frozen=True, kw_only=True, slots=True)
class _BlueSkyAircraft:
    node: Node
    callsign: str


class BlueSkyMobilityCommand(MobilityCommand):
    def __init__(self, heading: float | None = None,
                 speed: float | None = None,
                 vertical_speed: float | None = None):
        """
        :param heading: Target heading of the aircraft in degrees.
        :param speed: Target speed of the aircraft in knots
        :param vertical_speed: Target vertical speed of the aircraft in feet per minutes.
        """
        heading = _finite_float("heading", heading) if heading is not None else None
        if speed is not None:
            validate_non_negative_real("speed", speed)

        vertical_speed = _finite_float("vertical_speed", vertical_speed) if vertical_speed is not None else None

        if heading is not None and not 0.0 <= heading < 360.0:
            raise ValueError("heading must be in the range [0, 360)")

        super().__init__(4, heading, speed, vertical_speed)


@dataclass
class BlueSkyTelemetry(Telemetry):
    geo_coords: tuple[float, float, float]
    heading: float
    speed: float
    vertical_speed: float


class BlueSkyMobilityHandler(INodeHandler):
    _event_loop: EventLoop

    def __init__(self, config: BlueSkyMobilityConfiguration) -> None:
        super().__init__()
        validate_bluesky_mobility_config(config)
        # Create a unique working directory for this simulation instance
        self._workdir = config.base_workdir.resolve() / uuid.uuid4().hex
        self._workdir.mkdir(parents=True)

        self._config = config
        self._uninitialized_nodes: list[Node] = []
        self._aircraft: dict[int, _BlueSkyAircraft] = {}
        self._coord_frame = HomeCoordinateFrame(self._config.home)
        self._visualization: BlueSkyVisualization | None = None

        # Initialize bluesky simulation
        try:
            if self._config.visualization:
                self._visualization = BlueSkyVisualization.start(
                    workdir=self._workdir,
                )
            bs.init(
                "sim",
                workdir=self._workdir,
                detached=self._visualization is None,
                group_id=(
                    self._visualization.group_id
                    if self._visualization is not None
                    else None
                ),
            )
            if self._visualization is not None:
                bs.net.connect(
                    hostname="127.0.0.1",
                    recv_port=self._visualization.recv_port,
                    send_port=self._visualization.send_port,
                )
        except BaseException:
            if self._visualization is not None:
                self._visualization.close()
            raise

        # Set update rate
        bs.stack.stack(f"DT {self._config.update_rate}")
        if self._visualization is not None:
            bs.stack.stack(
                f"PAN {self._config.home[0]},{self._config.home[1]};ZOOM 1"
            )

    @staticmethod
    def get_label() -> str:
        return "mobility"

    def inject(self, event_loop: EventLoop) -> None:
        self._event_loop = event_loop

        self._event_loop.schedule_event(self._config.update_rate, self._step)

    def register_node(self, node: Node) -> None:
        if node.id in self._aircraft or any(registered.id == node.id for registered in self._uninitialized_nodes):
            raise ValueError(f"Node with ID {node.id} is already registered.")
        self._uninitialized_nodes.append(node)

    @property
    def traffic(self) -> Traffic:
        return bs.traf

    @property
    def simulation(self) -> Simulation:
        return bs.sim

    def initialize_aircraft(self, node_id: int, aircraft_type: str, initial_latitude: float,
                            initial_longitude: float, initial_altitude: float, initial_heading: float,
                            initial_speed: float) -> None:
        node = next((n for n in self._uninitialized_nodes if n.id == node_id), None)
        if node is None:
            raise ValueError(f"Node with ID {node_id} is not registered or already initialized.")

        validate_non_empty_string("aircraft_type", aircraft_type)
        if not aircraft_type.strip():
            raise ValueError("aircraft_type must not be blank")

        initial_latitude = _finite_float("initial_latitude", initial_latitude)
        initial_longitude = _finite_float("initial_longitude", initial_longitude)
        initial_altitude = _finite_float("initial_altitude", initial_altitude)
        initial_heading = _finite_float("initial_heading", initial_heading)
        validate_non_negative_real("initial_speed", initial_speed)
        initial_speed = float(initial_speed)

        if not -90.0 <= initial_latitude <= 90.0:
            raise ValueError("initial_latitude must be in the range [-90, 90]")
        if not -180.0 <= initial_longitude <= 180.0:
            raise ValueError("initial_longitude must be in the range [-180, 180]")
        if not 0.0 <= initial_heading < 360.0:
            raise ValueError("initial_heading must be in the range [0, 360)")
        callsign = f"AC{node_id}"
        self.traffic.cre(callsign, aircraft_type, initial_latitude, initial_longitude, initial_heading,
                         initial_altitude, initial_speed)
        self._aircraft[node_id] = _BlueSkyAircraft(node=node, callsign=callsign)
        self._uninitialized_nodes.remove(node)

    def _step(self):
        if self._visualization is not None:
            Timer.update_timers()
            bs.net.update()

        self.simulation.step()

        if abs(self._event_loop.current_time - self.simulation.simt) > 0.01:
            raise RuntimeError("Event loop time is not synchronized with BlueSky update rate.")

        self._update_nodes()

        if self._visualization is not None:
            bs.scr.update()

        self._event_loop.schedule_event(self._event_loop.current_time + self._config.update_rate, self._step)

    def _update_nodes(self) -> None:
        # Resolving node position
        latitudes = self.traffic.lat.tolist()
        longitudes = self.traffic.lon.tolist()
        altitudes = self.traffic.alt.tolist()
        headings = self.traffic.hdg.tolist()
        speeds = self.traffic.cas.tolist()
        vertical_speeds = self.traffic.vs.tolist()
        x_arr, y_arr, z_arr = self._coord_frame.geographic_to_local(latitudes, longitudes, altitudes)
        x_coords = x_arr.tolist()
        y_coords = y_arr.tolist()
        z_coords = z_arr.tolist()

        for node_id, aircraft in self._aircraft.items():
            node = aircraft.node
            aircraft_index = self.traffic.id2idx(aircraft.callsign)
            position = (
                float(x_coords[aircraft_index]),
                float(y_coords[aircraft_index]),
                float(z_coords[aircraft_index])
            )
            node.position = position

            self._send_telemetry_update(
                node,
                position,
                (float(latitudes[aircraft_index]), float(longitudes[aircraft_index]), float(altitudes[aircraft_index])),
                float(headings[aircraft_index]),
                float(speeds[aircraft_index]),
                float(vertical_speeds[aircraft_index]))

    def _send_telemetry_update(self, node: Node,
                               position: tuple[float, float, float],
                               geo_coords: tuple[float, float, float],
                               heading: float,
                               speed: float,
                               vertical_speed: float):
        telemetry = BlueSkyTelemetry(
            current_position=position,
            geo_coords=geo_coords,
            heading=heading,
            speed=speed,
            vertical_speed=vertical_speed
        )

        def handle_telemetry():
            node.protocol_encapsulator.handle_telemetry(telemetry)

        self._event_loop.schedule_event(
            self._event_loop.current_time,
            handle_telemetry,
            f"Node {node.id} handle telemetry"
        )

    def handle_command(self, command: MobilityCommand, node: Node):
        if not isinstance(command, BlueSkyMobilityCommand):
            raise ValueError("Command must be an instance of BlueSkyMobilityCommand.")

        if node.id not in self._aircraft:
            raise ValueError(f"Node with ID {node.id} is not initialized as an aircraft.")

        heading: float | None = command.param_1
        speed: float | None = command.param_2
        vertical_speed: float | None = command.param_3

        callsign = self._aircraft[node.id].callsign

        instruction = ""
        if heading is not None:
            instruction += f"HDG {callsign},{heading};"

        if speed is not None:
            instruction += f"SPD {callsign},{speed};"

        if vertical_speed is not None:
            # BluSky autopilot requires both vertical speed and altitude to be set. Since altitude is not part of the
            # instruction, we set the target altitude based on the current altitude and the vertical speed, with a wide
            # margin of 1 minute.
            alt = self.traffic.alt[self.traffic.id2idx(callsign)]
            alt_ft = alt / ft
            target_alt_ft = max(alt_ft + vertical_speed * 1.0, 0)

            instruction += f"ALT {callsign},{target_alt_ft},{abs(vertical_speed)};"

        # Update the aircraft's heading, speed, vertical speed and altitude in BlueSky
        if len(instruction) > 0:
            bs.stack.stack(instruction)

    def finalize(self) -> None:
        try:
            bs.sim.quit()
            if self._visualization is not None:
                bs.net.close()
        finally:
            if self._visualization is not None:
                self._visualization.close()


def _finite_float(name: str, value: float) -> float:
    validate_finite_real(name, value)
    return float(value)
