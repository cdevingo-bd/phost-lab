"""Where the time goes in one MuJoCo Warp step, by phase, with and without a graph installed.

Each phase of ``mjw.step`` is captured as its own CUDA graph and replayed on the state real steps leave behind, so the
numbers match how a graph-replayed rollout runs (an eager trace is dominated by launch gaps). Per-phase graphs add a
small gap each, so the phase sum sits a little above the full-step graph; the shares are what matter.

The phases are read from MuJoCo Warp's private modules, so this is a diagnostic tied to the build it runs on. Under the
discrete integrator it also times the effective-metric solve on its own: that call sits inside the acceleration phase
and is the one place the ``effective.solve`` stage applies.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import phost

from .graph import GraphSpec
from .scene import Scene
from .settle import settled

PHASES = ("kinematics", "crb / armature", "collision", "make_constraint", "velocity+actuation+accel", "solver", "integrate")


def _capture(wp, fn):
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        fn()
    wp.synchronize()
    return cap.graph


def _replay_ms(wp, graph, launches: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(launches):
            wp.capture_launch(graph)
        wp.synchronize()
        samples.append(1e3 * (time.perf_counter() - start) / launches)
    return statistics.median(samples)


def _state_restorer(wp, d):
    """A callable that puts the integrated state (qpos, qvel, act, time) back to what it is now.

    ``integrate`` is the one phase that is not idempotent: replayed on its own it advances the state by one step of a
    stale acceleration, hundreds of times, and the state ends up NaN.
    """
    saved = {name: wp.clone(getattr(d, name)) for name in ("qpos", "qvel", "act", "time") if getattr(d, name).size}

    def restore():
        for name, source in saved.items():
            wp.copy(getattr(d, name), source)
    return restore


def _phases(mjw, model, d, restore) -> dict[str, Any]:
    """Name -> callable for each phase of one step. Each goes through the module attribute, so an installed hook applies."""
    from mujoco_warp._src import collision_driver, constraint, forward, smooth, solver
    from mujoco_warp._src.types import IntegratorType

    def integrate():
        integrator = model.opt.integrator
        if integrator == IntegratorType.EULER:
            forward.euler(model, d)
        elif integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
            forward.implicit(model, d)
        else:
            forward.discrete(model, d)
        restore()

    def inertia():
        smooth.crb(model, d)
        smooth.tendon_armature(model, d)

    def dynamics():
        forward.fwd_velocity(model, d)
        forward.fwd_actuation(model, d)
        forward.fwd_acceleration(model, d, factorize=True)

    return {"kinematics": lambda: forward.fwd_kinematics(model, d), "crb / armature": inertia,
            "collision": lambda: collision_driver.collision(model, d), "make_constraint": lambda: constraint.make_constraint(model, d),
            "velocity+actuation+accel": dynamics, "solver": lambda: solver.solve(model, d), "integrate": integrate}


def _effective_call(model, d):
    """The effective-metric solve as the host calls it, or None when this model does not make that call."""
    from mujoco_warp._src import derivative
    from mujoco_warp._src.types import IntegratorType
    if model.opt.integrator != IntegratorType.DISCRETE or int(model.nefmK) == 0:
        return None
    return lambda: derivative.eff_solve(model, d, d.qacc_smooth)


def profile_scene(graph: GraphSpec, scene: Scene, worlds: tuple[int, ...] = (1,), registry: phost.Registry | None = None,
                  launches: int = 10, repeats: int = 9) -> dict:
    """Phase times for the native step and, when ``graph`` selects any stage, for the step with it installed.

    Returns ``{"scene", "nv", "device", "cells": [{"worlds", "variants": {"native": {...}, "graph": {...}}}]}`` where each
    variant has ``full_ms`` (the whole step as one graph), ``phases`` (name -> ms), ``phase_sum_ms`` and, under the
    discrete integrator, ``effective_ms`` (the effective solve alone, inside ``velocity+actuation+accel``). The
    ``integrate`` phase is replayed with the state restored after it, and the cost of that restore is subtracted.
    """
    from .engine import require_engine

    require_engine(scene, "mjwarp", "the phase profile", "the phases of mjw.step")
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from phost.hosts.mjwarp import MJWarpHost

    registry = registry or phost.default
    graph.validate(registry)
    xml = scene.xml()
    model_cpu = mujoco.MjModel.from_xml_string(xml)
    cpu = settled(model_cpu, xml, scene.settle_steps)
    out: dict[str, Any] = dict(scene=scene.id, nv=int(model_cpu.nv), device=wp.get_device("cuda:0").name,
                               integrator=mujoco.mjtIntegrator(model_cpu.opt.integrator).name.replace("mjINT_", "").lower(),
                               timestep=float(model_cpu.opt.timestep), cells=[])
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(model_cpu)
        for w in worlds:
            variants = {}
            for name in ("native", "graph") if graph.stages else ("native",):
                def fresh():
                    d = mjw.put_data(model_cpu, cpu, nworld=w, nconmax=scene.nconmax(w), njmax=scene.constraints)
                    for _ in range(3):
                        mjw.step(model, d)                       # live contacts and constraints, hooks warmed
                    wp.synchronize()
                    return d

                d = mjw.put_data(model_cpu, cpu, nworld=w, nconmax=scene.nconmax(w), njmax=scene.constraints)
                session = None
                if name == "graph":
                    session = phost.Session(MJWarpHost(model, d, cpu_model=model_cpu), graph.selections(), registry)
                    session.__enter__()
                try:
                    # The whole step is timed on its own data: replaying single phases leaves the data in a state the
                    # real step does not start from (it then reports flex buffer overflows), so the two never share it.
                    for _ in range(3):
                        mjw.step(model, d)
                    wp.synchronize()
                    full = _replay_ms(wp, _capture(wp, lambda: mjw.step(model, d)), launches, repeats)
                    del d
                    d = fresh()
                    restore = _state_restorer(wp, d)
                    phases = {n: _replay_ms(wp, _capture(wp, fn), launches, repeats) for n, fn in _phases(mjw, model, d, restore).items()}
                    phases["integrate"] = max(0.0, phases["integrate"] - _replay_ms(wp, _capture(wp, restore), launches, repeats))
                    call = _effective_call(model, d)
                    entry = dict(full_ms=full, phases=phases, phase_sum_ms=sum(phases.values()),
                                 effective_ms=None if call is None else _replay_ms(wp, _capture(wp, call), launches, repeats))
                    if session is not None:
                        entry["stage_metrics"] = session.metrics()
                    variants[name] = entry
                finally:
                    if session is not None:
                        session.__exit__(None, None, None)
                    del d
            out["cells"].append(dict(worlds=w, variants=variants))
    return out


def table(result: dict) -> str:
    """Plain text: one block per world count, native next to the installed graph."""
    lines = [f"scene {result['scene']}  nv={result['nv']}  {result['integrator']} integrator, dt={result['timestep']}  {result['device']}"]
    for cell in result["cells"]:
        variants = cell["variants"]
        names = list(variants)
        lines.append(f"\nworlds={cell['worlds']}")
        lines.append(f"  {'phase':<26}" + "".join(f"{n + ' ms':>14}" for n in names))
        for phase in PHASES:
            lines.append(f"  {phase:<26}" + "".join(f"{variants[n]['phases'][phase]:>14.3f}" for n in names))
        if any(variants[n]["effective_ms"] is not None for n in names):
            lines.append(f"  {'  of which effective solve':<26}" + "".join(
                f"{'' if variants[n]['effective_ms'] is None else format(variants[n]['effective_ms'], '.3f'):>14}" for n in names))
        lines.append(f"  {'phase sum':<26}" + "".join(f"{variants[n]['phase_sum_ms']:>14.3f}" for n in names))
        lines.append(f"  {'full step (one graph)':<26}" + "".join(f"{variants[n]['full_ms']:>14.3f}" for n in names))
    return "\n".join(lines)
