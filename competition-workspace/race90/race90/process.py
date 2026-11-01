from pathlib import Path
import os
import signal
import subprocess
import time


def terminate(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=1)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    # The leader may exit on SIGTERM while a descendant ignores it. Always
    # terminate the remaining group, including normal-return orphan children.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_command(argv, deadline, log=None, env=None, cwd=None, cancel=None, capture=False):
    if time.time() >= deadline or (cancel is not None and cancel.is_set()):
        raise TimeoutError("No remaining command budget")
    log_file = None
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log, "ab")
    proc = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True,
                            stdout=subprocess.PIPE if capture else log_file,
                            stderr=subprocess.PIPE if capture else subprocess.STDOUT)
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=min(0.5, max(0.01, deadline - time.time())))
                break
            except subprocess.TimeoutExpired:
                if time.time() >= deadline or (cancel is not None and cancel.is_set()):
                    raise TimeoutError(f"Command stopped at deadline: {argv[0]}")
        if proc.returncode:
            message = err.decode(errors="replace")[-3000:] if capture and err else f"See log: {log}"
            raise RuntimeError(f"{argv[0]} exited {proc.returncode}: {message}")
        return out.decode(errors="replace") if capture and out is not None else ""
    finally:
        terminate(proc)
        if log_file is not None:
            log_file.close()
