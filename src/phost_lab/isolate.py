"""Run a graph in a worker process, with a timeout. Failures come back as data, not exceptions."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import Sequence

from .graph import GraphSpec
from .runner import RunConfig
from .worker import PREFIX


def run_isolated(graph: GraphSpec, scene: str, config: RunConfig, *, timeout: float = 900.0,
                 plugins: Sequence[str] = (), scene_modules: Sequence[str] = (), python: str | None = None) -> dict:
    """Returns the runner's result dict, or ``{"error": ..., "pass": False}`` if the worker failed."""
    request = dict(graph=graph.to_dict(), scene=scene, plugins=list(plugins), scene_modules=list(scene_modules),
                   config=dataclasses.asdict(config))
    request["config"]["worlds"] = list(config.worlds)
    # the worker sees the same modules as this process, so it runs the code that was asked for
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    try:
        done = subprocess.run([python or sys.executable, "-m", "phost_lab.worker"], input=json.dumps(request),
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as expired:
        tail = expired.stderr or ""
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        return {"error": f"timed out after {timeout:.0f} s", "pass": False, "stderr": tail[-2000:]}
    lines = [line for line in done.stdout.splitlines() if line.startswith(PREFIX)]
    if not lines:
        return {"error": f"worker exited with code {done.returncode} and no result", "pass": False,
                "stderr": done.stderr[-2000:]}
    result = json.loads(lines[-1][len(PREFIX):])
    if "error" in result:
        result["pass"] = False
    return result
