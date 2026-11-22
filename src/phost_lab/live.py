"""Watching a run: native and the custom graph stepped side by side from the same starting state, with their poses
streamed as they are produced.

Three copies of the scene start from the same state: ``native``, ``control`` (native again) and ``graph``. They are
stepped in lockstep, and every ``every`` steps the poses of ``native`` and ``graph`` are emitted for drawing, with how
far ``graph`` and ``control`` have each moved away from ``native``. MJWarp is not bitwise repeatable, so the control
is the yardstick: a graph that stays as close to native as native does to itself is indistinguishable from it.

This pass is for looking, not timing. Timing is the lab's ordinary run, taken without any of this in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import phost

from .engine import require_engine
from .graph import GraphSpec
from .scene import Scene
from .settle import settled

SPHERE, PLANE = 2, 0                  # mjtGeom values whose orientation does not show


@dataclass(frozen=True)
class LiveConfig:
    steps: int = 300
    every: int = 6                    # steps between frames
    max_frames: int = 200


def layout(model, flex: list[dict] | None = None) -> dict:
    """What a viewer needs to draw the scene once: each geom's type, size, colour and body, and each flex's triangles."""
    geoms = [dict(type=int(model.geom_type[i]), size=[round(float(v), 5) for v in model.geom_size[i]],
                  rgba=[round(float(v), 3) for v in model.geom_rgba[i]], body=int(model.geom_bodyid[i]),
                  turns=bool(model.geom_type[i] not in (SPHERE, PLANE)))
             for i in range(model.ngeom)]
    flexes = []
    for f in range(model.nflex):
        dim, count = int(model.flex_dim[f]), int(model.flex_elemnum[f])
        start = int(model.flex_elemdataadr[f])
        elems = model.flex_elem[start:start + count * (dim + 1)]
        flexes.append(dict(dim=dim, vertadr=int(model.flex_vertadr[f]), vertnum=int(model.flex_vertnum[f]),
                           rgba=[round(float(v), 3) for v in model.flex_rgba[f]],
                           tri=np.asarray(elems).reshape(-1, dim + 1).tolist() if dim == 2 else []))
    return dict(geoms=geoms, flexes=flexes, bodies=int(model.nbody))


def pose(data, turns: np.ndarray, nflexvert: int) -> dict:
    """World 0's geom positions, the orientations of the geoms that turn, and flex vertices; compact (4 decimals)."""
    def compact(a) -> list[float]:           # rounded as float64, or float32 noise prints as 0.2410999983549118
        return np.round(np.asarray(a, np.float64), 4).ravel().tolist()

    out: dict[str, Any] = dict(x=compact(data.geom_xpos.numpy()[0]), r=compact(data.geom_xmat.numpy()[0][turns]))
    if nflexvert:
        out["f"] = compact(data.flexvert_xpos.numpy()[0])
    return out


def run_live(graph: GraphSpec, scene: Scene, config: LiveConfig = LiveConfig(), registry: phost.Registry | None = None,
             on_progress: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] = lambda: False) -> dict:
    """Step the three copies and emit ``layout`` once, then ``live`` events. Returns a summary."""
    emit = on_progress or (lambda event: None)
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from phost.hosts.mjwarp import MJWarpHost

    if scene.hidden:
        raise phost.ContractError("a held-out scene is never drawn")
    require_engine(scene, "mjwarp", "the live view", "MuJoCo geometry to draw")
    registry = registry or phost.default
    graph.validate(registry)
    xml = scene.xml()
    model_cpu = mujoco.MjModel.from_xml_string(xml)
    cpu = settled(model_cpu, xml, scene.settle_steps)
    turns = np.array([t != SPHERE and t != PLANE for t in model_cpu.geom_type], bool)
    emit(dict(event="layout", scene=scene.id, graph=graph.name, timestep=float(model_cpu.opt.timestep),
              every=config.every, **layout(model_cpu)))

    def capture(fn):
        fn(); wp.synchronize()
        with wp.ScopedCapture() as cap:
            fn()
        wp.synchronize()
        return cap.graph

    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(model_cpu)
        make = lambda: mjw.put_data(model_cpu, cpu, nworld=1, nconmax=scene.nconmax(1), njmax=scene.constraints)
        native, control, custom = make(), make(), make()
        graphs = {}
        for name, d in (("native", native), ("control", control)):
            graphs[name] = capture(lambda d=d: mjw.step(model, d))
        session = None
        if graph.stages:
            session = phost.Session(MJWarpHost(model, custom, cpu_model=model_cpu), graph.selections(), registry)
            with session:
                graphs["graph"] = capture(lambda: mjw.step(model, custom))
        else:
            graphs["graph"] = capture(lambda: mjw.step(model, custom))
        for d in (native, control, custom):                   # the warm-up and capture steps moved them; start over
            d.qpos.assign(np.tile(cpu.qpos, (1, 1)).astype(np.float32))
            d.qvel.assign(np.tile(cpu.qvel, (1, 1)).astype(np.float32))
            d.time.zero_()
        if session is not None:
            session.reset_metrics()
        frames, drift, noise = 0, 0.0, 0.0
        for step in range(1, config.steps + 1):
            for name in ("native", "control", "graph"):
                wp.capture_launch(graphs[name])
            if step % config.every == 0:
                a, b, c = (x.qpos.numpy()[0].astype(np.float64) for x in (native, control, custom))
                drift, noise = float(np.sqrt(np.mean((c - a) ** 2))), float(np.sqrt(np.mean((b - a) ** 2)))
                frames += 1
                emit(dict(event="live", step=step, t=round(step * float(model_cpu.opt.timestep), 6),
                          native=pose(native, turns, model_cpu.nflexvert), graph=pose(custom, turns, model_cpu.nflexvert),
                          drift=drift, noise=noise, finite=bool(np.isfinite(c).all())))
                if frames >= config.max_frames or should_stop():      # no point stepping what nobody will see
                    break
        wp.synchronize()
        out = dict(scene=scene.id, graph=graph.to_dict(), steps=step, frames=frames, every=config.every,
                   timestep=float(model_cpu.opt.timestep), drift=drift, noise=noise,
                   stage_metrics=session.metrics() if session is not None else {},
                   advisories=list(session.advisories) if session is not None else [],
                   meta=dict(device=wp.get_device("cuda:0").name, nv=int(model_cpu.nv), ngeom=int(model_cpu.ngeom)))
    return out
