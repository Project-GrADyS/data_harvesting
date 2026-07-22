from types import SimpleNamespace

import numpy as np
import pytest

from bluesky.tools.aero import ft
from gradysim.protocol.messages.mobility import SetSpeedMobilityCommand
from gradysim.simulator.event import EventLoop

from d_atc.environment import mobility
from d_atc.environment.mobility import (
    BlueSkyMobilityCommand,
    BlueSkyMobilityConfiguration,
    BlueSkyMobilityHandler,
    BlueSkyTelemetry,
    validate_bluesky_mobility_config,
)


HOME = (-23.5505, -46.6333, 760.0)


class _FakeStack:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def stack(self, command: str) -> None:
        self.commands.append(command)


class _FakeSimulation:
    def __init__(self, update_rate: float) -> None:
        self.simt = 0.0
        self._update_rate = update_rate
        self.quit_calls = 0

    def step(self) -> None:
        self.simt += self._update_rate

    def quit(self) -> None:
        self.quit_calls += 1


class _FakeTraffic:
    def __init__(self) -> None:
        self.id: list[str] = []
        self.lat = np.array([], dtype=np.float64)
        self.lon = np.array([], dtype=np.float64)
        self.alt = np.array([], dtype=np.float64)
        self.hdg = np.array([], dtype=np.float64)
        self.cas = np.array([], dtype=np.float64)
        self.vs = np.array([], dtype=np.float64)
        self.creation_result: bool | tuple[bool, str] = True

    def cre(
        self,
        callsign: str,
        aircraft_type: str,
        latitude: float,
        longitude: float,
        heading: float,
        altitude: float,
        speed: float,
    ) -> bool | tuple[bool, str]:
        del aircraft_type
        if self.creation_result is not True:
            return self.creation_result

        self.id.append(callsign)
        self.lat = np.append(self.lat, latitude)
        self.lon = np.append(self.lon, longitude)
        self.alt = np.append(self.alt, altitude)
        self.hdg = np.append(self.hdg, heading)
        self.cas = np.append(self.cas, speed)
        self.vs = np.append(self.vs, 0.0)
        return True

    def id2idx(self, callsign: str) -> int:
        try:
            return self.id.index(callsign)
        except ValueError:
            return -1


class _TelemetryRecorder:
    def __init__(self) -> None:
        self.updates: list[BlueSkyTelemetry] = []

    def handle_telemetry(self, telemetry: BlueSkyTelemetry) -> None:
        self.updates.append(telemetry)


def _node(node_id: int):
    return SimpleNamespace(
        id=node_id,
        position=(0.0, 0.0, 0.0),
        protocol_encapsulator=_TelemetryRecorder(),
    )


@pytest.fixture
def fake_handler(monkeypatch: pytest.MonkeyPatch, tmp_path):
    update_rate = 0.05
    stack = _FakeStack()
    simulation = _FakeSimulation(update_rate)
    traffic = _FakeTraffic()
    init_calls: list[tuple[str, object, bool, object]] = []

    monkeypatch.setattr(
        mobility.bs,
        "init",
        lambda mode, *, workdir, detached, group_id: init_calls.append(
            (mode, workdir, detached, group_id)
        ),
    )
    monkeypatch.setattr(mobility.bs, "stack", stack)
    monkeypatch.setattr(mobility.bs, "sim", simulation, raising=False)
    monkeypatch.setattr(mobility.bs, "traf", traffic, raising=False)

    handler = BlueSkyMobilityHandler(
        BlueSkyMobilityConfiguration(
            base_workdir=tmp_path,
            home=HOME,
            update_rate=update_rate,
        )
    )
    return handler, traffic, simulation, stack, init_calls


@pytest.mark.parametrize(
    ("update_rate", "exception"),
    [
        (0.0, ValueError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("fast", TypeError),
    ],
)
def test_configuration_rejects_invalid_update_rate(update_rate, exception: type[Exception]) -> None:
    config = BlueSkyMobilityConfiguration(home=HOME, update_rate=update_rate)

    with pytest.raises(exception, match="update_rate"):
        BlueSkyMobilityHandler(config)


def test_configuration_is_passive_and_explicit_validator_accepts_valid_config(tmp_path) -> None:
    config = BlueSkyMobilityConfiguration(base_workdir=tmp_path, home=HOME)

    assert config.base_workdir == tmp_path
    assert validate_bluesky_mobility_config(config) is None


@pytest.mark.parametrize("visualization", [None, 0, 1, "yes"])
def test_configuration_rejects_non_boolean_visualization(visualization) -> None:
    config = BlueSkyMobilityConfiguration(
        home=HOME,
        visualization=visualization,
    )

    with pytest.raises(TypeError, match="visualization"):
        BlueSkyMobilityHandler(config)


def test_handler_validates_path_and_home_when_consuming_configuration(tmp_path) -> None:
    invalid_home = BlueSkyMobilityConfiguration(home=(91.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="latitude"):
        BlueSkyMobilityHandler(invalid_home)

    string_path = BlueSkyMobilityConfiguration(base_workdir=str(tmp_path), home=HOME)
    with pytest.raises(TypeError, match="pathlib.Path"):
        BlueSkyMobilityHandler(string_path)

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("content")
    file_config = BlueSkyMobilityConfiguration(base_workdir=file_path, home=HOME)
    with pytest.raises(ValueError, match="directory"):
        BlueSkyMobilityHandler(file_config)

    missing_path = BlueSkyMobilityConfiguration(base_workdir=None, home=HOME)
    with pytest.raises(TypeError, match="pathlib.Path"):
        BlueSkyMobilityHandler(missing_path)


def test_configuration_validator_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="BlueSkyMobilityConfiguration"):
        validate_bluesky_mobility_config(object())


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"heading": float("nan")}, ValueError, "heading"),
        ({"heading": -1.0}, ValueError, "heading"),
        ({"heading": 360.0}, ValueError, "heading"),
        ({"speed": -1.0}, ValueError, "speed"),
        ({"speed": float("inf")}, ValueError, "speed"),
        ({"vertical_speed": float("nan")}, ValueError, "vertical_speed"),
        ({"vertical_speed": True}, TypeError, "vertical_speed"),
    ],
)
def test_command_rejects_invalid_values(
    kwargs: dict[str, float],
    exception: type[Exception],
    message: str,
) -> None:
    values = {"heading": 90.0, "speed": 250.0, "vertical_speed": 500.0}
    values.update(kwargs)

    with pytest.raises(exception, match=message):
        BlueSkyMobilityCommand(**values)


def test_command_accepts_omitted_fields() -> None:
    command = BlueSkyMobilityCommand()

    assert command.param_1 is None
    assert command.param_2 is None
    assert command.param_3 is None


def test_handler_initializes_bluesky_and_rejects_duplicate_registration(fake_handler) -> None:
    handler, _, _, stack, init_calls = fake_handler
    node = _node(1)

    assert init_calls == [("sim", handler._workdir, True, None)]
    assert handler._workdir.is_dir()
    assert stack.commands == ["DT 0.05"]

    handler.register_node(node)
    with pytest.raises(ValueError, match="already registered"):
        handler.register_node(node)


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"aircraft_type": ""}, ValueError, "aircraft_type"),
        ({"initial_latitude": float("nan")}, ValueError, "initial_latitude"),
        ({"initial_latitude": 91.0}, ValueError, "initial_latitude"),
        ({"initial_longitude": 181.0}, ValueError, "initial_longitude"),
        ({"initial_altitude": float("inf")}, ValueError, "initial_altitude"),
        ({"initial_heading": 360.0}, ValueError, "initial_heading"),
        ({"initial_speed": -1.0}, ValueError, "initial_speed"),
        ({"initial_speed": True}, TypeError, "initial_speed"),
    ],
)
def test_initialize_aircraft_rejects_invalid_values(
    fake_handler,
    overrides,
    exception: type[Exception],
    message: str,
) -> None:
    handler, traffic, _, _, _ = fake_handler
    node = _node(1)
    handler.register_node(node)
    values = {
        "node_id": 1,
        "aircraft_type": "A320",
        "initial_latitude": HOME[0],
        "initial_longitude": HOME[1],
        "initial_altitude": HOME[2],
        "initial_heading": 90.0,
        "initial_speed": 120.0,
    }
    values.update(overrides)

    with pytest.raises(exception, match=message):
        handler.initialize_aircraft(**values)

    assert traffic.id == []

def test_step_updates_two_nodes_sends_distinct_telemetry_and_reschedules(fake_handler) -> None:
    handler, _, simulation, _, _ = fake_handler
    first = _node(1)
    second = _node(2)
    handler.register_node(first)
    handler.register_node(second)
    handler.initialize_aircraft(1, "A320", HOME[0], HOME[1], HOME[2], 90.0, 120.0)
    handler.initialize_aircraft(2, "B738", HOME[0] + 0.01, HOME[1], HOME[2] + 100.0, 180.0, 140.0)

    event_loop = EventLoop()
    handler.inject(event_loop)
    step_event = event_loop.pop_event()
    step_event.callback()

    assert simulation.simt == pytest.approx(0.05)
    assert first.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
    assert second.position != first.position

    while event_loop.peek_event() is not None and event_loop.peek_event().timestamp == pytest.approx(0.05):
        event_loop.pop_event().callback()

    assert len(first.protocol_encapsulator.updates) == 1
    assert len(second.protocol_encapsulator.updates) == 1
    assert first.protocol_encapsulator.updates[0].geo_coords == pytest.approx(HOME)
    assert second.protocol_encapsulator.updates[0].geo_coords != pytest.approx(HOME)
    assert event_loop.peek_event().timestamp == pytest.approx(0.1)


def test_descent_command_uses_lower_target_and_positive_rate_magnitude(fake_handler) -> None:
    handler, _, _, stack, _ = fake_handler
    node = _node(1)
    handler.register_node(node)
    handler.initialize_aircraft(1, "A320", HOME[0], HOME[1], 1_000.0 * ft, 90.0, 120.0)

    handler.handle_command(BlueSkyMobilityCommand(180.0, 250.0, -500.0), node)

    assert stack.commands[-1] == "HDG AC1,180.0;SPD AC1,250.0;ALT AC1,500.0,500.0;"


@pytest.mark.parametrize(
    ("command", "expected_instruction"),
    [
        (BlueSkyMobilityCommand(heading=180.0), "HDG AC1,180.0;"),
        (BlueSkyMobilityCommand(speed=250.0), "SPD AC1,250.0;"),
        (BlueSkyMobilityCommand(vertical_speed=500.0), "ALT AC1,1500.0,500.0;"),
    ],
)
def test_handle_command_stacks_only_supplied_fields(fake_handler, command, expected_instruction: str) -> None:
    handler, _, _, stack, _ = fake_handler
    node = _node(1)
    handler.register_node(node)
    handler.initialize_aircraft(1, "A320", HOME[0], HOME[1], 1_000.0 * ft, 90.0, 120.0)

    handler.handle_command(command, node)

    assert stack.commands[-1] == expected_instruction


def test_empty_command_does_not_stack_an_instruction(fake_handler) -> None:
    handler, _, _, stack, _ = fake_handler
    node = _node(1)
    handler.register_node(node)
    handler.initialize_aircraft(1, "A320", HOME[0], HOME[1], HOME[2], 90.0, 120.0)
    command_count = len(stack.commands)

    handler.handle_command(BlueSkyMobilityCommand(), node)

    assert len(stack.commands) == command_count


def test_step_detects_bluesky_clock_drift(fake_handler) -> None:
    handler, _, simulation, _, _ = fake_handler
    event_loop = EventLoop()
    handler.inject(event_loop)
    simulation._update_rate = 0.1

    with pytest.raises(RuntimeError, match="not synchronized"):
        event_loop.pop_event().callback()


def test_visualization_connects_updates_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[object] = []
    stack = _FakeStack()
    simulation = _FakeSimulation(0.05)
    traffic = _FakeTraffic()
    visualization = SimpleNamespace(
        group_id=b"Sabc",
        recv_port=21000,
        send_port=21001,
        close=lambda: calls.append("visualization.close"),
    )
    network = SimpleNamespace(
        connect=lambda **kwargs: calls.append(("net.connect", kwargs)),
        update=lambda: calls.append("net.update"),
        close=lambda: calls.append("net.close"),
    )
    screen = SimpleNamespace(update=lambda: calls.append("screen.update"))

    monkeypatch.setattr(
        mobility.BlueSkyVisualization,
        "start",
        lambda **kwargs: calls.append(("visualization.start", kwargs)) or visualization,
    )
    monkeypatch.setattr(
        mobility.bs,
        "init",
        lambda mode, **kwargs: calls.append(("bs.init", mode, kwargs)),
    )
    monkeypatch.setattr(mobility.bs, "stack", stack)
    monkeypatch.setattr(mobility.bs, "sim", simulation, raising=False)
    monkeypatch.setattr(mobility.bs, "traf", traffic, raising=False)
    monkeypatch.setattr(mobility.bs, "net", network, raising=False)
    monkeypatch.setattr(mobility.bs, "scr", screen, raising=False)
    monkeypatch.setattr(
        mobility.Timer,
        "update_timers",
        lambda: calls.append("timer.update"),
    )

    handler = BlueSkyMobilityHandler(
        BlueSkyMobilityConfiguration(
            base_workdir=tmp_path,
            home=HOME,
            visualization=True,
        )
    )
    event_loop = EventLoop()
    handler.inject(event_loop)
    event_loop.pop_event().callback()
    handler.finalize()

    assert calls[0][0] == "visualization.start"
    assert calls[1] == (
        "bs.init",
        "sim",
        {
            "workdir": handler._workdir,
            "detached": False,
            "group_id": b"Sabc",
        },
    )
    assert calls[2] == (
        "net.connect",
        {"hostname": "127.0.0.1", "recv_port": 21000, "send_port": 21001},
    )
    assert stack.commands == [
        "DT 0.05",
        f"PAN {HOME[0]},{HOME[1]};ZOOM 1",
    ]
    assert calls[3:6] == ["timer.update", "net.update", "screen.update"]
    assert calls[-2:] == ["net.close", "visualization.close"]
    assert simulation.quit_calls == 1


def test_handle_command_rejects_wrong_command_and_uninitialized_node(fake_handler) -> None:
    handler, _, _, _, _ = fake_handler
    node = _node(1)

    with pytest.raises(ValueError, match="BlueSkyMobilityCommand"):
        handler.handle_command(SetSpeedMobilityCommand(10.0), node)
    with pytest.raises(ValueError, match="not initialized"):
        handler.handle_command(BlueSkyMobilityCommand(90.0, 250.0, 0.0), node)
