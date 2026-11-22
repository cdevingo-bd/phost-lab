"""Run a graph on a scene: capture, time, and compare against a native control. Needs mujoco, mujoco_warp, CUDA.

Per world count the runner builds four things on the same start state: native, native2 (the control, to
measure the noise floor), and the graph under test. It reports step time, speedup, accuracy against native
relative to the noise floor, and the per-stage metrics each implementation emits, plus a verdict.
"""

from __future__ import annotations

import contextlib
import datetime
import gc
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import phost
from phost import SwapClass
from .driver import get_driver
from .graph import GraphSpec
from .metrics import accuracy_ratio
from .scene import Scene
from .settle import settled


@dataclass(frozen=True)
class RunConfig:
    worlds: tuple[int, ...] = (1, 8)
    steps: int = 20                  # steps replayed for the accuracy comparison
    launches: int = 5                # graph launches per timing sample
    repeats: int = 11                # timing samples (median reported)
    max_accuracy_ratio: float = 10.0
    max_native_spread: float = 0.10  # a cell noisier than this is reported as unreliable
    frame_every: int = 0             # keep world 0's qpos every this many replayed steps, for playback (0: none)
    max_frames: int = 60
    calls: tuple[int, ...] = ()      # steps per call to time, e.g. (1, 5, 20); () skips the per-call curve
    driver: str = "graph"            # how the host is stepped while timed (see driver.py): "graph", "eager", or a plugin's

    def __post_init__(self):
        object.__setattr__(self, "calls", tuple(int(n) for n in self.calls))
        if any(n < 1 for n in self.calls):
            raise ValueError("calls must be positive step counts")


def other_gpu_processes() -> int:
    """Processes other than this one using the GPU, or -1 when it cannot be told."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return sum(1 for pid in out if pid.isdigit() and int(pid) != os.getpid())
    except Exception:
        return -1


def run_graph(graph: GraphSpec, scene: Scene, config: RunConfig, registry: phost.Registry | None = None,
              on_progress: Callable[[dict], None] | None = None) -> dict:
    """Run in this process. Use ``isolate.run_isolated`` to run in a worker process.

    ``on_progress`` is called with ``{"event": "start", ...}`` once and ``{"event": "cell", "cell": {...}}`` as each
    world count finishes, so a caller can show results while the rest runs.
    """
    emit = on_progress or (lambda event: None)
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from phost.hosts.mjwarp import MJWarpHost

    registry = registry or phost.default
    graph.validate(registry)
    xml = scene.xml()
    model_cpu = mujoco.MjModel.from_xml_string(xml)
    cpu = settled(model_cpu, xml, scene.settle_steps)

    driver = get_driver(config.driver)

    def step_ms(advance):
        samples = []
        for _ in range(config.repeats):
            t = time.perf_counter()
            for _ in range(config.launches):
                advance()
            wp.synchronize()
            samples.append(1e3 * (time.perf_counter() - t) / config.launches)
        return statistics.median(samples)

    def call_curve(advance):
        """Median wall time of one call of n steps plus a sync, for each n in ``config.calls``."""
        curve = {}
        for n in config.calls:
            samples = []
            for _ in range(config.repeats):
                t = time.perf_counter()
                for _ in range(n):
                    advance()
                wp.synchronize()
                samples.append(1e3 * (time.perf_counter() - t))
            curve[n] = statistics.median(samples)
        return curve

    def reset(d, w):
        for field_name in ("qpos", "qvel"):
            getattr(d, field_name).assign(np.tile(getattr(cpu, field_name), (w, 1)).astype(np.float32))
        d.time.zero_()

    result: dict[str, Any] = dict(
        graph=graph.to_dict(), scene=scene.id, scene_hidden=scene.hidden, stage_classes=stage_classes(graph, registry),
        config=dict(worlds=list(config.worlds), steps=config.steps, driver=config.driver,
                    **({"calls": list(config.calls)} if config.calls else {})),
        meta=dict(date=datetime.datetime.now().isoformat(timespec="seconds"), mujoco=mujoco.__version__,
                  warp=wp.config.version, device=wp.get_device("cuda:0").name, nv=int(model_cpu.nv),
                  nefc_start=int(cpu.nefc), timestep=float(model_cpu.opt.timestep),
                  other_gpu_processes=other_gpu_processes()),
        cells=[])
    emit(dict(event="start", scene=scene.id, graph=graph.name, worlds=list(config.worlds), meta=result["meta"]))
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(model_cpu)
        for w in config.worlds:
            others = other_gpu_processes()
            timings, finals, graphs, keep, frames, curves = {}, {}, {}, [], [], {}
            for name in ("native", "graph", "native2"):
                d = mjw.put_data(model_cpu, cpu, nworld=w, nconmax=scene.nconmax(w), njmax=scene.constraints)
                session = None
                if name == "graph" and graph.stages:
                    session = phost.Session(MJWarpHost(model, d, cpu_model=model_cpu), graph.selections(), registry)
                with session if session is not None else contextlib.nullcontext():      # installed for as long as it is stepped
                    advance = driver.prepare(wp, lambda: mjw.step(model, d))
                    reset(d, w)
                    timings[name] = step_ms(advance)
                    if config.calls:
                        reset(d, w)
                        curves[name] = call_curve(advance)
                    reset(d, w)
                    if session is not None:
                        session.reset_metrics()
                    for step in range(config.steps):
                        advance()
                        if name == "graph" and config.frame_every and step % config.frame_every == 0 \
                                and len(frames) < config.max_frames:
                            frames.append(d.qpos.numpy()[0].astype(np.float64).tolist())     # reading syncs the device
                    wp.synchronize()
                    finals[name] = d.qpos.numpy().astype(np.float64)
                    if name == "graph":
                        stage_metrics = session.metrics() if session is not None else {}
                        advisories = list(session.advisories) if session is not None else []
                keep.append((d, advance, session))
            native_ms = (timings["native"] + timings["native2"]) / 2
            rms = lambda a, b: float(np.sqrt(np.mean((finals[a] - finals[b]) ** 2)))
            control, diff = rms("native", "native2"), rms("graph", "native")
            ratio = accuracy_ratio(diff, control)
            cell = dict(worlds=w, step_ms=timings["graph"], native_ms=native_ms,
                        speedup=native_ms / timings["graph"],
                        native_spread=abs(timings["native"] - timings["native2"]) / native_ms,
                        rms_dqpos=diff, control_rms_dqpos=control, accuracy_ratio=ratio,
                        stage_metrics=stage_metrics, advisories=advisories, other_gpu_processes=others)
            if config.calls:
                cell["per_call"] = per_call(curves)
            if frames:
                cell["frames"] = frames
            cell["verdict"] = _verdict(cell, finals["graph"], registry, graph, config)
            result["cells"].append(cell)
            emit(dict(event="cell", cell=cell))
            del keep, graphs, finals
            gc.collect()
    result["comparable_to_native"] = comparable_to_native(graph, registry)
    result["pass"] = all(c["verdict"]["ok"] for c in result["cells"])
    return result


def per_call(curves: dict) -> dict:
    """What a rollout loop pays per call, from the time of calls of n steps.

    Each variant's time for a call is modelled as ``fixed_ms + n * per_step_ms`` (a least-squares line through the
    measured points; two points or more are needed to separate the two, one point reports the whole as per-step).
    ``fixed_ms`` is the launch and synchronization cost of a call from Python: the floor under any wrapper that adds
    its own state conversion. ``ms`` and ``speedup`` are lists aligned with ``steps_per_call``; ``speedup`` is native
    over graph for a call of each length.
    """
    import numpy as np

    native = {n: (curves["native"][n] + curves["native2"][n]) / 2 for n in curves["native"]}
    out: dict[str, Any] = dict(steps_per_call=sorted(native), native={}, graph={}, speedup=[])
    for label, curve in (("native", native), ("graph", curves["graph"])):
        ns = sorted(curve)
        if len(ns) >= 2:
            per_step, fixed = np.polyfit(ns, [curve[n] for n in ns], 1)
        else:
            per_step, fixed = curve[ns[0]] / ns[0], 0.0
        out[label] = dict(ms=[curve[n] for n in ns], fixed_ms=max(float(fixed), 0.0), per_step_ms=float(per_step))
    out["speedup"] = [native[n] / curves["graph"][n] for n in sorted(native)]
    return out


def stage_classes(graph: GraphSpec, registry: phost.Registry) -> dict[str, str]:
    """Stage name -> swap class value, so a result says how it must be judged without needing the registry."""
    classes = {}
    for stage, node in graph.stages.items():
        impl = registry.impl(node.impl)
        classes[stage] = registry.contract(stage, impl.stage_version).swap_class.value
    return classes


def comparable_to_native(graph: GraphSpec, registry: phost.Registry) -> bool:
    """False when any selected stage changes the model: such a graph has no native answer to match."""
    for node in graph.stages.values():
        impl = registry.impl(node.impl)
        if registry.contract(impl.stage, impl.stage_version).swap_class is SwapClass.MODEL_CHANGING:
            return False
    return True


def _verdict(cell: dict, final_qpos, registry, graph: GraphSpec, config: RunConfig) -> dict:
    """``ok`` and the reasons, plus ``gates``: which kind of check passed (None when it did not apply). A scorer reads
    ``gates``; the messages are for people."""
    problems, notes = [], []
    finite = bool(np.isfinite(final_qpos).all())
    if not finite:
        problems.append("non-finite qpos")
    ratio = cell["accuracy_ratio"]
    changing = comparable_to_native(graph, registry) is False
    accuracy = None
    if changing:
        notes.append("a model-changing stage is selected: the trajectory is expected to differ from native, so accuracy "
                     "against native is reported but not gated (each stage is judged against its own float64 law)")
    elif ratio is None:
        notes.append("accuracy ratio not defined: the native control is at rounding-noise level")
    else:
        accuracy = bool(ratio <= config.max_accuracy_ratio)
        if not accuracy:
            problems.append(f"accuracy ratio {ratio:.2f} exceeds {config.max_accuracy_ratio}")
    if cell["native_spread"] > config.max_native_spread:
        notes.append(f"timing unreliable: the two native runs differ by {cell['native_spread']:.1%}")
    if cell["other_gpu_processes"] not in (0,):
        notes.append(f"{cell['other_gpu_processes']} other GPU processes were running (-1: unknown)")
    stage_problems = []
    for stage, metrics in cell["stage_metrics"].items():
        contract = registry.contract(stage, registry.impl(graph.stages[stage].impl).stage_version)
        by_name = {m.name: m for m in contract.metrics}
        for name, value in metrics.items():
            spec = by_name.get(name)
            if spec is not None and spec.kind == "bitmask" and value != 0:
                stage_problems.append(f"{stage}.{name} = {value} (must be 0)")
        for gate in contract.tolerances:
            if gate.metric in metrics and not metrics[gate.metric] <= gate.max:
                stage_problems.append(f"{stage}.{gate.metric} = {metrics[gate.metric]:.3g} exceeds {gate.max}")
    problems += stage_problems
    return dict(ok=not problems, problems=problems, notes=notes,
                gates=dict(finite=finite, accuracy=accuracy, stage=not stage_problems))
