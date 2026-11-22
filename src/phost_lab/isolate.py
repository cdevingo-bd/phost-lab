"""Run a graph in a worker process, with a timeout and cancellation. Failures come back as data, not exceptions."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
from typing import Callable, Mapping, Sequence

from .graph import GraphSpec
from .runner import RunConfig
from .worker import PREFIX, PROGRESS


def run_isolated(graph: GraphSpec, scene: str, config: RunConfig, *, timeout: float = 900.0,
                 plugins: Sequence[str] = (), scene_modules: Sequence[str] = (), scene_dirs: Sequence[str] = (),
                 python: str | None = None, on_progress: Callable[[dict], None] | None = None,
                 cancel: threading.Event | None = None, env: Mapping[str, str] | None = None,
                 live: Mapping[str, int] | None = None) -> dict:
    """Returns the runner's result dict, or ``{"error": ..., "pass": False}`` if the worker failed.

    ``on_progress`` receives the worker's progress events as they happen. Setting ``cancel`` kills the worker.
    ``env`` adds environment variables for the worker alone. ``live`` (``steps``, ``every``, ``max_frames``) makes it a
    live pass (see ``live.py``) instead of a timed run.
    """
    request = dict(graph=graph.to_dict(), scene=scene, plugins=list(plugins), scene_modules=list(scene_modules),
                   scene_dirs=list(scene_dirs), config=dataclasses.asdict(config), progress=on_progress is not None,
                   mode="live" if live is not None else "run", live=dict(live or {}))
    request["config"]["worlds"] = list(config.worlds)
    request["config"]["calls"] = list(config.calls)
    # the worker sees the same modules as this process, so it runs the code that was asked for
    worker_env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), **(env or {}))
    process = subprocess.Popen([python or sys.executable, "-m", "phost_lab.worker"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env,
                               start_new_session=True)
    stopped: list[str] = []
    done = threading.Event()

    def stop(reason: str) -> None:
        if not stopped:
            stopped.append(reason)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def watch() -> None:
        # one thread both enforces the timeout and notices a cancel request
        waited = 0.0
        while not done.wait(0.2):
            waited += 0.2
            if cancel is not None and cancel.is_set():
                return stop("cancelled")
            if waited > timeout:
                return stop("timeout")

    tail: list[str] = []

    def drain() -> None:
        for line in process.stderr:
            tail.append(line)
            del tail[:-60]

    threads = [threading.Thread(target=watch, daemon=True), threading.Thread(target=drain, daemon=True)]
    for thread in threads:
        thread.start()
    result_line = None
    try:
        try:
            process.stdin.write(json.dumps(request))
            process.stdin.close()
        except BrokenPipeError:
            pass
        for line in process.stdout:
            if line.startswith(PROGRESS):
                if on_progress is not None:
                    on_progress(json.loads(line[len(PROGRESS):]))
            elif line.startswith(PREFIX):
                result_line = line
        process.wait()
    except BaseException:
        # a failing callback must not leave the worker running
        stop("aborted")
        raise
    finally:
        done.set()
        process.stdout.close()
        for thread in threads:
            thread.join(timeout=2)
    stderr = "".join(tail)[-2000:]
    return _result(result_line, stopped, process.returncode, timeout, stderr)


def _result(result_line: str | None, stopped: list[str], code: int, timeout: float, stderr: str) -> dict:
    if stopped and stopped[0] in ("timeout", "cancelled"):
        reason = f"timed out after {timeout:.0f} s" if stopped[0] == "timeout" else "cancelled"
        return {"error": reason, "pass": False, "stderr": stderr}
    if result_line is None:
        return {"error": f"worker exited with code {code} and no result", "pass": False, "stderr": stderr}
    result = json.loads(result_line[len(PREFIX):])
    if "error" in result:
        result["pass"] = False
    return result
