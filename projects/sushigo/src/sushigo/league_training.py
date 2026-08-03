"""Bounded subprocess launcher for reproducible league training repetitions."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import subprocess
import sys
import threading
from typing import Callable, Sequence

from .league import TRAINING_PRESETS, discover_training_runs


@dataclass(frozen=True, slots=True)
class LeagueJob:
    preset: str
    repetition: int

    @property
    def label(self) -> str:
        return f"{self.preset}/repetition_{self.repetition}"

    def command(
        self,
        *,
        train_script: Path,
        params_path: Path,
        tracking_uri: str,
        experiment_name: str,
        device: str,
    ) -> list[str]:
        return [
            sys.executable,
            str(train_script),
            "--params",
            str(params_path),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            experiment_name,
            "--preset",
            self.preset,
            "--repetition",
            str(self.repetition),
            "--device",
            device,
        ]


def build_jobs(
    presets: Sequence[str], repetitions: int
) -> list[LeagueJob]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    unknown = set(presets).difference(TRAINING_PRESETS)
    if unknown:
        raise ValueError(f"Unknown presets: {sorted(unknown)}")
    return [
        LeagueJob(preset, repetition)
        for preset in presets
        for repetition in range(1, repetitions + 1)
    ]


def pending_jobs(
    jobs: Sequence[LeagueJob],
    *,
    tracking_uri: str,
    experiment_name: str,
) -> tuple[list[LeagueJob], list[LeagueJob]]:
    completed_runs = discover_training_runs(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        presets=tuple({job.preset for job in jobs}),
    )
    completed_keys = {
        (run.preset, run.repetition) for run in completed_runs
    }
    pending = [
        job
        for job in jobs
        if (job.preset, job.repetition) not in completed_keys
    ]
    completed = [job for job in jobs if job not in pending]
    return pending, completed


def run_job(command: Sequence[str], stop: threading.Event) -> bool:
    if stop.is_set():
        return False
    process = subprocess.Popen(command)
    while process.poll() is None:
        if stop.wait(0.2):
            process.terminate()
            process.wait()
            return False
    return process.returncode == 0


def run_jobs(
    jobs: Sequence[LeagueJob],
    *,
    parallelism: int,
    command_factory: Callable[[LeagueJob], Sequence[str]],
    runner: Callable[[Sequence[str], threading.Event], bool] = run_job,
) -> bool:
    if parallelism < 1:
        raise ValueError("parallelism must be at least 1")
    stop = threading.Event()
    remaining = iter(jobs)
    executor = ThreadPoolExecutor(max_workers=parallelism)
    futures = {
        executor.submit(runner, command_factory(job), stop): job
        for job in islice(remaining, parallelism)
    }
    try:
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                job = futures.pop(future)
                try:
                    succeeded = future.result()
                except Exception:
                    succeeded = False
                if not succeeded:
                    stop.set()
                    for queued in futures:
                        queued.cancel()
                    return False
                next_job = next(remaining, None)
                if next_job is not None:
                    futures[
                        executor.submit(
                            runner, command_factory(next_job), stop
                        )
                    ] = next_job
        return True
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
