from gradysim.protocol.interface import IProtocol

from d_atc.environment.mobility import BlueSkyTelemetry, BlueSkyMobilityCommand


def _normalize(angle: float) -> float:
    return angle % 360

def _heading_difference(heading1: float, heading2: float) -> float:
    """Measures the angle between two headings, in degrees. The result is always between 0 and 180 degrees."""
    return min(abs(heading1 - heading2), 360 - abs(heading1 - heading2))

class AircraftCircleProtocol(IProtocol):
    """Protocol that causes an aircraft to circle around"""
    _target_heading: float | None

    def __init__(self):
        self._target_heading = None

    def initialize(self) -> None:
        pass

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: BlueSkyTelemetry) -> None:
        old_heading = self._target_heading

        if self._target_heading is None:
            self._target_heading = _normalize(telemetry.heading + 10)
        else:
            # If we are close to our target heading, increment again
            if _heading_difference(telemetry.heading, self._target_heading) <= 5:
                self._target_heading = _normalize(self._target_heading + 10)

        # If heading changed, send a command to update
        if old_heading != self._target_heading:
            command = BlueSkyMobilityCommand(heading=self._target_heading)
            self.provider.send_mobility_command(command)

    def finish(self) -> None:
        pass