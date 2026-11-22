"""Where the time goes in one step, by phase, with and without a graph installed.

The phases belong to the engine: an engine that offers the ``profile`` capability gives, for an instance, named functions that
each run one phase of its step on that instance's data (``Loaded.profile``). Each is replayed as its own captured graph where the
device can capture one, on the state real steps leave behind, so the numbers match how a replayed rollout runs (an eager trace
is dominated by launch gaps). Per-phase graphs add a small gap each, so the phase sum sits a little above the full-step graph;
the shares are what matter.

For MuJoCo Warp the phases are read from its private modules, so that is a diagnostic tied to the build it runs on. Under the
discrete integrator it also times the effective-metric solve on its own: that call sits inside the acceleration phase and is the
one place the ``effective.solve`` stage applies. The chain engine's phases are its own functions (passive force, actuation,
assembly, the banded solve, integration).
"""

from __future__ import annotations

from typing import Any

import phost

from .engine import engine_for, require_capability
from .graph import GraphSpec
from .scene import Scene


def profile_scene(graph: GraphSpec, scene: Scene, worlds: tuple[int, ...] = (1,), registry: phost.Registry | None = None,
                  launches: int = 10, repeats: int = 9) -> dict:
    """Phase times for the native step and, when ``graph`` selects any stage, for the step with it installed.

    Returns ``{"scene", "nv", "device", "integrator", "timestep", "cells": [{"worlds", "variants": {"native": {...}, "graph":
    {...}}}]}`` where each variant has ``full_ms`` (the whole step as one replay), ``phases`` (name -> ms, in step order),
    ``phase_sum_ms`` and, when the engine reports one, ``effective_ms`` (the effective solve alone, inside a larger phase). A phase
    that advances the state is replayed with the state restored after it, and the cost of that restore is subtracted.
    """
    require_capability(scene, "profile", "the phase profile")
    registry = registry or phost.default
    graph.validate(registry)
    engine = engine_for(scene, graph)
    with engine.load(scene) as loaded:
        device = loaded.device
        out: dict[str, Any] = dict(scene=scene.id, nv=int(loaded.meta["nv"]), device=device.name, engine=engine.name,
                                   integrator=loaded.meta.get("integrator", ""), timestep=float(loaded.meta["timestep"]), cells=[])
        for w in worlds:
            variants = {}
            for name in ("native", "graph") if graph.stages else ("native",):
                session = None
                whole = loaded.instance(w)
                if name == "graph":
                    session = phost.Session(whole.host, graph.selections(), registry)
                    session.__enter__()
                try:
                    # The whole step is timed on its own data: replaying single phases leaves the data in a state the real step
                    # does not start from (it then reports flex buffer overflows), so the two never share it.
                    for _ in range(3):
                        whole.step()
                    device.synchronize()
                    full = device.replay_ms(whole.step, launches, repeats)
                    del whole
                    fresh = loaded.instance(w)
                    for _ in range(3):
                        fresh.step()                                    # live contacts and constraints, hooks warmed
                    device.synchronize()
                    plan = loaded.profile(fresh)
                    phases = {n: device.replay_ms(fn, launches, repeats) for n, fn in plan["phases"].items()}
                    if plan.get("restore") is not None:
                        cost = device.replay_ms(plan["restore"], launches, repeats)
                        for n in plan.get("restored", ()):
                            phases[n] = max(0.0, phases[n] - cost)
                    extra = plan.get("extra", {})
                    entry = dict(full_ms=full, phases=phases, phase_sum_ms=sum(phases.values()),
                                 effective_ms=device.replay_ms(extra["effective"], launches, repeats) if "effective" in extra else None)
                    if session is not None:
                        entry["stage_metrics"] = session.metrics()
                    variants[name] = entry
                finally:
                    if session is not None:
                        session.__exit__(None, None, None)
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
        for phase in variants[names[0]]["phases"]:
            lines.append(f"  {phase:<26}" + "".join(f"{variants[n]['phases'][phase]:>14.3f}" for n in names))
        if any(variants[n]["effective_ms"] is not None for n in names):
            lines.append(f"  {'  of which effective solve':<26}" + "".join(
                f"{'' if variants[n]['effective_ms'] is None else format(variants[n]['effective_ms'], '.3f'):>14}" for n in names))
        lines.append(f"  {'phase sum':<26}" + "".join(f"{variants[n]['phase_sum_ms']:>14.3f}" for n in names))
        lines.append(f"  {'full step (one replay)':<26}" + "".join(f"{variants[n]['full_ms']:>14.3f}" for n in names))
    return "\n".join(lines)
