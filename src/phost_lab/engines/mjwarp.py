"""MuJoCo Warp as an engine: MuJoCo parses and settles on the CPU, MuJoCo Warp steps on a CUDA device. Needs CUDA to load a scene.

Everything that names MuJoCo or Warp in the lab lives here: ``profile.py`` and ``live.py`` are generic over engines and ask
this one for its phases (``Loaded.profile``) and its drawing (``Loaded.layout``, ``Instance.pose``).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time
from typing import Any

import numpy as np

SCOPE = "cuda:0"
SPHERE, PLANE = 2, 0                  # mjtGeom values whose orientation does not show
# the phases of ``mjw.step`` that ``Loaded.profile`` times, in order
PHASES = ("kinematics", "crb / armature", "collision", "make_constraint", "velocity+actuation+accel", "solver", "integrate")


def other_gpu_processes() -> int:
    """Processes other than this one using the GPU, or -1 when it cannot be told."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return sum(1 for pid in out if pid.isdigit() and int(pid) != os.getpid())
    except Exception:
        return -1


class WarpDevice:
    """A Warp device. Anything this class does not define is the Warp module's, so a driver written against ``wp`` still works."""

    can_capture = True

    def __init__(self, wp, name: str = SCOPE):
        self.wp, self.name = wp, name

    def __getattr__(self, attribute):
        return getattr(self.wp, attribute)

    def synchronize(self) -> None:
        self.wp.synchronize()

    def capture(self, step):
        wp = self.wp
        with wp.ScopedCapture() as capture:
            step()
        wp.synchronize()
        graph = capture.graph
        return lambda: wp.capture_launch(graph)

    def busy(self) -> int:
        return other_gpu_processes()

    def replay_ms(self, step, launches: int, repeats: int) -> float:
        step()
        self.wp.synchronize()
        replay = self.capture(step)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(launches):
                replay()
            self.wp.synchronize()
            samples.append(1e3 * (time.perf_counter() - start) / launches)
        return statistics.median(samples)

    def floor_ms(self, launches: int, repeats: int) -> float:
        from .._floor import touch

        x = self.wp.zeros(1, dtype=float)
        return self.replay_ms(lambda: self.wp.launch(touch, dim=1, inputs=[x]), launches, repeats)


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

    functions = {"kinematics": lambda: forward.fwd_kinematics(model, d), "crb / armature": inertia,
                 "collision": lambda: collision_driver.collision(model, d), "make_constraint": lambda: constraint.make_constraint(model, d),
                 "velocity+actuation+accel": dynamics, "solver": lambda: solver.solve(model, d), "integrate": integrate}
    return {name: functions[name] for name in PHASES}


def _effective_call(model, d):
    """The effective-metric solve as the host calls it, or None when this model does not make that call."""
    from mujoco_warp._src import derivative
    from mujoco_warp._src.types import IntegratorType
    if model.opt.integrator != IntegratorType.DISCRETE or int(model.nefmK) == 0:
        return None
    return lambda: derivative.eff_solve(model, d, d.qacc_smooth)


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


class _Instance:
    def __init__(self, loaded: "_Loaded", worlds: int):
        mjw = loaded.mjw
        self._loaded, self.worlds = loaded, worlds
        self.data = mjw.put_data(loaded.model_cpu, loaded.cpu, nworld=worlds, nconmax=loaded.scene.nconmax(worlds),
                                 njmax=loaded.scene.constraints)
        self._host = None

    @property
    def host(self):
        if self._host is None:                 # only a run that installs stages needs one
            from phost.hosts.mjwarp import MJWarpHost
            self._host = MJWarpHost(self._loaded.model, self.data, cpu_model=self._loaded.model_cpu)
        return self._host

    def step(self) -> None:
        self._loaded.mjw.step(self._loaded.model, self.data)

    def reset(self) -> None:
        cpu = self._loaded.cpu
        for name in ("qpos", "qvel"):
            getattr(self.data, name).assign(np.tile(getattr(cpu, name), (self.worlds, 1)).astype(np.float32))
        self.data.time.zero_()

    def state(self) -> np.ndarray:
        return self.data.qpos.numpy().astype(np.float64)

    def frame(self) -> list[float]:
        return self.data.qpos.numpy()[0].astype(np.float64).tolist()

    def pose(self) -> dict:
        return pose(self.data, self._loaded.turns, self._loaded.nflexvert)


class _Loaded:
    def __init__(self, scene, settle: bool):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from ..settle import settled

        self.scene, self.mjw, self._wp = scene, mjw, wp
        xml = scene.text()
        self.model_cpu = mujoco.MjModel.from_xml_string(xml)
        self.cpu = settled(self.model_cpu, xml, scene.settle_steps) if settle else mujoco.MjData(self.model_cpu)
        self.device = WarpDevice(wp)
        self.turns = np.array([t != SPHERE and t != PLANE for t in self.model_cpu.geom_type], bool)
        self.nflexvert = int(self.model_cpu.nflexvert)
        self.model = None
        self._scope = None
        self.meta = dict(mujoco=mujoco.__version__, warp=wp.config.version, device=wp.get_device(SCOPE).name,
                         nv=int(self.model_cpu.nv), ngeom=int(self.model_cpu.ngeom), nefc_start=int(self.cpu.nefc), timestep=float(self.model_cpu.opt.timestep),
                         integrator=mujoco.mjtIntegrator(self.model_cpu.opt.integrator).name.replace("mjINT_", "").lower(),
                         repeatable=False)       # contact-heavy scenes differ between identical runs: the lab measures that

    def __enter__(self):
        self._scope = self._wp.ScopedDevice(SCOPE)
        self._scope.__enter__()
        self.model = self.mjw.put_model(self.model_cpu)
        return self

    def __exit__(self, *exc):
        scope, self._scope = self._scope, None
        return scope.__exit__(*exc) if scope is not None else None

    def instance(self, worlds: int) -> _Instance:
        return _Instance(self, int(worlds))

    def layout(self) -> dict:
        """What a viewer needs to draw the scene once."""
        return layout(self.model_cpu)

    def profile(self, instance: _Instance) -> dict:
        """The phases of ``mjw.step`` as capturable functions on this instance's data, for ``profile``.

        ``phases`` is name -> function in step order; ``restore`` puts the integrated state back and ``restored`` names the phases
        after which it is needed (a phase that advances the state would otherwise run away); ``extra`` is timed alone and
        reported beside the phases (the effective solve, under the discrete integrator).
        """
        restore = _state_restorer(self._wp, instance.data)
        extra = {}
        call = _effective_call(self.model, instance.data)
        if call is not None:
            extra["effective"] = call
        return dict(phases=_phases(self.mjw, self.model, instance.data, restore), restore=restore, restored=("integrate",),
                    extra=extra)


class MJWarpEngine:
    name = "mjwarp"
    suffix = ".xml"
    capabilities = frozenset({"profile", "live", "render"})
    device_kind = "cuda"          # where it steps: "cpu" or "cuda" (what a tool may run alongside it)
    doc = "MuJoCo Warp on a CUDA device; scenes are MJCF"

    def unavailable(self) -> str | None:
        import importlib.util
        missing = [m for m in ("mujoco", "mujoco_warp", "warp") if importlib.util.find_spec(m) is None]
        return f"{', '.join(missing)} not installed" if missing else None

    def load(self, scene, *, settle: bool = True) -> _Loaded:
        return _Loaded(scene, settle)
