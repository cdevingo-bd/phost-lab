"""Watching a run: native and the custom graph stepped side by side from the same starting state, with their poses
streamed as they are produced.

Three copies of the scene start from the same state: ``native``, ``control`` (native again) and ``graph``. They are
stepped in lockstep, and every ``every`` steps the poses of ``native`` and ``graph`` are emitted for drawing, with how
far ``graph`` and ``control`` have each moved away from ``native``. MJWarp is not bitwise repeatable, so the control
is the yardstick: a graph that stays as close to native as native does to itself is indistinguishable from it (an engine whose
steps repeat exactly has a control at zero, and the drift is simply the distance from native).

Any engine that offers the ``live`` capability can be watched: it gives ``Loaded.layout()`` (what to draw once) and
``Instance.pose()`` (what to draw each frame). The chain engine draws its particles as spheres.

This pass is for looking, not timing. Timing is the lab's ordinary run, taken without any of this in the loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import phost

from .driver import resolve_driver
from .engine import engine_for, require_capability
from .engines.mjwarp import PLANE, SPHERE, layout, pose          # noqa: F401  (where they used to live)
from .graph import GraphSpec
from .scene import Scene


@dataclass(frozen=True)
class LiveConfig:
    steps: int = 300
    every: int = 6                    # steps between frames
    max_frames: int = 200


def run_live(graph: GraphSpec, scene: Scene, config: LiveConfig = LiveConfig(), registry: phost.Registry | None = None,
             on_progress: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] = lambda: False) -> dict:
    """Step the three copies and emit ``layout`` once, then ``live`` events. Returns a summary."""
    import numpy as np

    emit = on_progress or (lambda event: None)
    if scene.hidden:
        raise phost.ContractError("a held-out scene is never drawn")
    require_capability(scene, "live", "the live view")
    registry = registry or phost.default
    graph.validate(registry)
    engine = engine_for(scene, graph)
    with engine.load(scene) as loaded:
        device = loaded.device
        driver, _ = resolve_driver("graph", device)
        timestep = float(loaded.meta["timestep"])
        emit(dict(event="layout", scene=scene.id, graph=graph.name, timestep=timestep, every=config.every, **loaded.layout()))
        native, control, custom = loaded.instance(1), loaded.instance(1), loaded.instance(1)
        session = phost.Session(custom.host, graph.selections(), registry) if graph.stages else None
        with session if session is not None else contextlib.nullcontext():      # installed for as long as anything is stepped
            advance = {name: driver.prepare(device, inst.step) for name, inst in
                       (("native", native), ("control", control), ("graph", custom))}
            for inst in (native, control, custom):                 # the warm-up and capture steps moved them; start over
                inst.reset()
            if session is not None:
                session.reset_metrics()
            frames, drift, noise = 0, 0.0, 0.0
            for step in range(1, config.steps + 1):
                for name in ("native", "control", "graph"):
                    advance[name]()
                if step % config.every == 0:
                    device.synchronize()
                    a, b, c = (np.asarray(x.frame(), np.float64) for x in (native, control, custom))
                    drift, noise = float(np.sqrt(np.mean((c - a) ** 2))), float(np.sqrt(np.mean((b - a) ** 2)))
                    frames += 1
                    emit(dict(event="live", step=step, t=round(step * timestep, 6), native=native.pose(), graph=custom.pose(),
                              drift=drift, noise=noise, finite=bool(np.isfinite(c).all())))
                    if frames >= config.max_frames or should_stop():      # no point stepping what nobody will see
                        break
            device.synchronize()
            out = dict(scene=scene.id, graph=graph.to_dict(), steps=step, frames=frames, every=config.every, timestep=timestep,
                       drift=drift, noise=noise, stage_metrics=session.metrics() if session is not None else {},
                       advisories=list(session.advisories) if session is not None else [],
                       meta=dict(device=loaded.meta["device"], nv=int(loaded.meta["nv"]),
                                 **({"ngeom": int(loaded.meta["ngeom"])} if "ngeom" in loaded.meta else {})))
    return out
