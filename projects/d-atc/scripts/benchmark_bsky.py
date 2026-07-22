from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Sequence

from d_atc.circle_simulation import (
    DEFAULT_CENTER,
    CircleSimulationConfiguration,
    build_circle_simulation,
)


DEFAULT_BENCHMARK_WORKDIR = Path(__file__).resolve().parents[1] / ".bluesky"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _run_worker(payload: dict[str, object]) -> dict[str, float | int]:
    config = CircleSimulationConfiguration(
        center=payload["center"],
        aircraft_count=payload["aircraft"],
        placement_radius_m=payload["radius_m"],
        altitude_spread_m=payload["altitude_spread_m"],
        initial_speed_mps=payload["speed_mps"],
        aircraft_type=payload["aircraft_type"],
        duration_s=payload["duration_s"],
        update_rate_s=payload["update_rate_s"],
        seed=payload["seed"],
        bluesky_workdir=Path(payload["workdir"]),
    )

    started = time.perf_counter()
    simulation = build_circle_simulation(config)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    simulation.run()
    rollout_seconds = time.perf_counter() - started

    simulated_seconds = float(simulation.mobility_handler.simulation.simt)
    steps = round(simulated_seconds / config.update_rate_s)
    return {
        "build_seconds": build_seconds,
        "rollout_seconds": rollout_seconds,
        "simulated_seconds": simulated_seconds,
        "steps": steps,
    }


def _run_trial(args: argparse.Namespace, aircraft: int, workers: int, trial: int) -> dict[str, float]:
    payloads = [
        {
            "center": (args.latitude, args.longitude, args.altitude_m),
            "aircraft": aircraft,
            "radius_m": args.radius_m,
            "altitude_spread_m": args.altitude_spread_m,
            "speed_mps": args.speed_mps,
            "aircraft_type": args.aircraft_type,
            "duration_s": args.duration_s,
            "update_rate_s": args.update_rate_s,
            "seed": args.seed + trial * workers + worker,
            "workdir": str(args.workdir.resolve()),
        }
        for worker in range(workers)
    ]

    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = list(executor.map(_run_worker, payloads))
    wall_seconds = time.perf_counter() - started

    rollout_steps_per_second = sum(
        result["steps"] / result["rollout_seconds"] for result in results
    )
    simulation_speed = sum(
        result["simulated_seconds"] / result["rollout_seconds"] for result in results
    )
    total_steps = sum(result["steps"] for result in results)
    return {
        "wall_seconds": wall_seconds,
        "build_seconds": max(result["build_seconds"] for result in results),
        "rollout_seconds": max(result["rollout_seconds"] for result in results),
        "steps_per_second": rollout_steps_per_second,
        "aircraft_updates_per_second": rollout_steps_per_second * aircraft,
        "simulation_speed": simulation_speed,
        "end_to_end_steps_per_second": total_steps / wall_seconds,
    }


def _run_case(args: argparse.Namespace, aircraft: int, workers: int) -> dict[str, object]:
    trials = [_run_trial(args, aircraft, workers, trial) for trial in range(args.trials)]
    return {
        "aircraft": aircraft,
        "workers": workers,
        "trials": args.trials,
        "duration_s": args.duration_s,
        "update_rate_s": args.update_rate_s,
        "steps_per_second": statistics.median(trial["steps_per_second"] for trial in trials),
        "aircraft_updates_per_second": statistics.median(
            trial["aircraft_updates_per_second"] for trial in trials
        ),
        "simulation_speed": statistics.median(trial["simulation_speed"] for trial in trials),
        "end_to_end_steps_per_second": statistics.median(
            trial["end_to_end_steps_per_second"] for trial in trials
        ),
        "build_seconds": statistics.median(trial["build_seconds"] for trial in trials),
        "rollout_seconds": statistics.median(trial["rollout_seconds"] for trial in trials),
        "wall_seconds": statistics.median(trial["wall_seconds"] for trial in trials),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the GrADyS/BlueSky circle simulation.",
    )
    parser.add_argument(
        "--aircraft",
        nargs="+",
        type=_positive_int,
        default=[100],
        help="Aircraft counts to benchmark (default: 100)",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=_positive_int,
        default=[1],
        help="Parallel simulator process counts (default: 1)",
    )
    parser.add_argument("--trials", type=_positive_int, default=1)
    parser.add_argument("--duration-s", type=_positive_float, default=10.0)
    parser.add_argument("--update-rate-s", type=_positive_float, default=0.05)
    parser.add_argument("--latitude", type=float, default=DEFAULT_CENTER[0])
    parser.add_argument("--longitude", type=float, default=DEFAULT_CENTER[1])
    parser.add_argument("--altitude-m", type=float, default=DEFAULT_CENTER[2])
    parser.add_argument("--radius-m", type=_positive_float, default=5_000.0)
    parser.add_argument("--altitude-spread-m", type=float, default=250.0)
    parser.add_argument("--speed-mps", type=_positive_float, default=120.0)
    parser.add_argument("--aircraft-type", default="A320")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_BENCHMARK_WORKDIR)
    parser.add_argument("--json-output", type=Path)
    return parser


def _print_result(result: dict[str, object]) -> None:
    print(
        f"{result['aircraft']:>7} aircraft/env | {result['workers']:>2} workers | "
        f"{result['steps_per_second']:>9,.1f} steps/s | "
        f"{result['aircraft_updates_per_second']:>14,.0f} aircraft-updates/s | "
        f"{result['simulation_speed']:>8,.1f}x simulated time | "
        f"{result['wall_seconds']:>7.2f}s wall"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        f"BlueSky circle benchmark on {platform.system()} {platform.machine()} "
        f"({os.cpu_count()} logical CPUs)"
    )
    print("Rollout rates exclude construction; end-to-end rates include process startup and construction.")

    results = []
    for aircraft in args.aircraft:
        for workers in args.workers:
            result = _run_case(args, aircraft, workers)
            results.append(result)
            _print_result(result)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "system": {
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
            },
            "results": results,
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
