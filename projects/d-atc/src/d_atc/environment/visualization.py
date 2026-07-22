from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bluesky.network.common import GROUPID_SIM, bin2hex, genid, hex2bin


_HOST = "127.0.0.1"
_STARTUP_TIMEOUT_S = 5.0
_READ_ONLY_MESSAGE = "This BlueSky view is read-only; simulation commands are disabled."


@dataclass(slots=True)
class BlueSkyVisualization:
    """Managed BlueSky server and QtGL observer processes."""

    server_process: subprocess.Popen
    client_process: subprocess.Popen
    recv_port: int
    send_port: int
    group_id: bytes

    @classmethod
    def start(
        cls,
        *,
        workdir: Path,
    ) -> BlueSkyVisualization:
        recv_port, send_port = _reserve_ports()
        group_id = genid(GROUPID_SIM)[:-1]
        common_args = [
            "--workdir", str(workdir),
            "--recv-port", str(recv_port),
            "--send-port", str(send_port),
            "--group-id", bin2hex(group_id),
        ]
        launcher = str(Path(__file__).resolve())
        server_process = subprocess.Popen(
            [sys.executable, launcher, "server", *common_args],
        )
        try:
            _wait_for_server(server_process, recv_port, send_port)
            client_process = subprocess.Popen(
                [
                    sys.executable,
                    launcher,
                    "client",
                    *common_args,
                ],
            )
        except BaseException:
            _stop_process(server_process)
            raise

        return cls(
            server_process=server_process,
            client_process=client_process,
            recv_port=recv_port,
            send_port=send_port,
            group_id=group_id,
        )

    def close(self) -> None:
        _stop_process(self.client_process)
        _stop_process(self.server_process)


def _reserve_ports() -> tuple[int, int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(2):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind((_HOST, 0))
            sockets.append(listener)
        return tuple(listener.getsockname()[1] for listener in sockets)
    finally:
        for listener in sockets:
            listener.close()


def _wait_for_server(
    process: subprocess.Popen,
    recv_port: int,
    send_port: int,
) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    pending = {recv_port, send_port}
    while pending and time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"BlueSky visualization server exited during startup with code {return_code}."
            )
        for port in tuple(pending):
            try:
                with socket.create_connection((_HOST, port), timeout=0.05):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.05)
    if pending:
        raise RuntimeError("Timed out while starting the BlueSky visualization server.")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _install_read_only_forward(
    clientstack,
    stack_module,
    stackbase,
    *,
    echo_ready=lambda: True,
) -> None:
    def reject_simulation_command(*unused_args, **unused_kwargs) -> None:
        if echo_ready():
            clientstack.echo(_READ_ONLY_MESSAGE)

    clientstack.forward = reject_simulation_command
    stack_module.forward = reject_simulation_command
    stackbase.forward = reject_simulation_command


def _safe_stack_receiver(receiver):
    def receive(data=None) -> None:
        if data not in (None, ""):
            receiver(data)

    return receive


def _install_safe_stack_subscription() -> None:
    from bluesky.network.subscriber import Subscription
    from bluesky.stack.stackbase import on_stack_received

    subscription = Subscription("STACK")
    subscription.deferred_subs = [
        callback
        for callback in subscription.deferred_subs
        if callback != on_stack_received
    ]
    subscription.connect(_safe_stack_receiver(on_stack_received))


def _run_server(args: argparse.Namespace) -> int:
    import bluesky as bs

    bs.init("server", workdir=Path(args.workdir), discoverable=False)
    bs.settings.recv_port = args.recv_port
    bs.settings.send_port = args.send_port
    bs.server.server_id = genid(hex2bin(args.group_id), seqidx=0)
    # This process is only the broker for the simulator owned by GrADyS.
    bs.server.addnodes = lambda *unused_args, **unused_kwargs: None
    bs.server.run()
    return 0


def _run_client(args: argparse.Namespace) -> int:
    import bluesky as bs

    bs.init("client", gui="qtgl", workdir=Path(args.workdir))
    bs.settings.recv_port = args.recv_port
    bs.settings.send_port = args.send_port

    from bluesky.stack import clientstack, stackbase

    _install_read_only_forward(
        clientstack,
        bs.stack,
        stackbase,
        echo_ready=lambda: bs.scr is not None,
    )
    _install_safe_stack_subscription()

    from bluesky.ui import qtgl

    qtgl.start(hostname=_HOST)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("server", "client"))
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--recv-port", required=True, type=int)
    parser.add_argument("--send-port", required=True, type=int)
    parser.add_argument("--group-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "server":
        return _run_server(args)
    return _run_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
