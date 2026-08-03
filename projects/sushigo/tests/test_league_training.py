import threading

from sushigo.league_training import build_jobs, run_jobs


def test_job_matrix_and_bounded_runner():
    jobs = build_jobs(("fixed_2p", "variable_2_4"), 2)
    assert len(jobs) == 4
    active = 0
    maximum = 0
    lock = threading.Lock()

    def runner(command, stop):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        with lock:
            active -= 1
        return True

    assert run_jobs(
        jobs,
        parallelism=2,
        command_factory=lambda job: [job.label],
        runner=runner,
    )
    assert maximum <= 2


def test_runner_stops_after_failure():
    jobs = build_jobs(("fixed_2p",), 3)
    attempts = []

    def runner(command, stop):
        attempts.append(command[0])
        return False

    assert not run_jobs(
        jobs,
        parallelism=1,
        command_factory=lambda job: [job.label],
        runner=runner,
    )
    assert attempts == ["fixed_2p/repetition_1"]
