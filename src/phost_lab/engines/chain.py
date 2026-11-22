"""A spring-mass chain, stepped in NumPy, as an engine: the lab's second simulator, and not a MuJoCo-shaped one.

A scene is a small JSON text (``chain_text``): how many particles, how many are driven, the springs and the damping,
the timestep, and where the chain is plucked. ``settle_steps`` run the chain with nothing installed before a run starts
from it, as MuJoCo scenes are settled on the CPU. The host is ``phost.hosts.chain.ChainHost``; its stages are the ones a
chain has (passive force, control law, probe, and the banded effective solve).

Stepping is deterministic, so a native run agrees with itself exactly. The runner then judges accuracy against a rounding
floor instead of against the difference between two native runs (see ``runner.ROUNDING_FLOOR``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from phost import ContractError

from ..engine import CpuDevice

KEYS = {"n", "nu", "timestep", "stiffness", "damping", "gain", "seed", "pluck_at", "pluck_size"}
SPACING, LIFT, RADIUS = 0.1, 0.5, 0.03      # how the chain is drawn: particles on a line, displacement shown vertically
SHOWN = 2.0                                  # displacement is drawn this many times larger than it is


def chain_text(n: int = 32, nu: int = 2, *, timestep: float = 0.01, stiffness: float = 400.0, damping: float = 0.2,
               gain: float = 5.0, seed: int = 0, pluck_at: int | None = None, pluck_size: float = 0.1) -> str:
    """The scene text for a chain of ``n`` particles. ``pluck_at`` defaults to a quarter of the way along."""
    return json.dumps(dict(n=n, nu=nu, timestep=timestep, stiffness=stiffness, damping=damping, gain=gain, seed=seed,
                           pluck_at=n // 4 if pluck_at is None else pluck_at, pluck_size=pluck_size), sort_keys=True)


def parse(text: str) -> dict[str, Any]:
    try:
        spec = json.loads(text)
    except ValueError as error:
        raise ContractError(f"a chain scene is a JSON object: {error}") from None
    if not isinstance(spec, dict) or set(spec) - KEYS or "n" not in spec:
        raise ContractError(f"a chain scene is a JSON object with n and any of {sorted(KEYS - {'n'})}")
    return spec


class _Instance:
    def __init__(self, loaded: "_Loaded", worlds: int):
        self._loaded, self.worlds = loaded, worlds
        self.host = loaded.make_host(worlds)
        self.step = self.make_step()
        self.reset()

    def make_step(self):
        return self.host.step

    @property
    def data(self):
        return self.host.data

    def reset(self) -> None:
        self.host.load(self._loaded.x, self._loaded.v)

    def state(self) -> np.ndarray:
        return np.asarray(self.host.data.x, np.float64)

    def frame(self) -> list[float]:
        return np.asarray(self.host.data.x, np.float64)[0].tolist()

    def pose(self) -> dict:
        """World 0's particles as sphere positions (the format a viewer draws): along a line, displaced vertically."""
        x = np.asarray(self.host.data.x, np.float64)[0]
        points = np.stack([np.arange(x.size) * SPACING, np.zeros(x.size), LIFT + SHOWN * x], axis=1)
        return dict(x=np.round(points, 4).ravel().tolist(), r=[])


def _phase_functions(instance) -> dict:
    """The chain's step as named functions on one instance's data, through ``host.ns`` so an installed stage applies."""
    host = instance.host
    m, d, ns = host.model, host.data, host.ns
    saved = (d.x.copy(), d.v.copy(), d.t.copy())

    def restore():
        d.x[:], d.v[:], d.t[:] = saved

    def integrate():
        h = np.float32(m.timestep)
        d.v += h * d.a
        d.x += h * d.v
        d.t += h

    phases = {"passive force": lambda: ns.passive(m, d), "actuation": lambda: ns.actuate(m, d),
              "assemble": lambda: ns.assemble(m, d), "effective solve": lambda: ns.effective_solve(m, d, d.a),
              "integrate": integrate}
    return dict(phases=phases, restore=restore, restored=("integrate",), extra={})


class _Loaded:
    def __init__(self, scene, settle: bool):
        from phost.hosts.chain import ChainHost, make_model

        spec = parse(scene.text())
        n = int(spec["n"])
        self.model = make_model(n, int(spec.get("nu", 2)), timestep=float(spec.get("timestep", 0.01)),
                                stiffness=float(spec.get("stiffness", 400.0)), damping=float(spec.get("damping", 0.2)),
                                gain=float(spec.get("gain", 5.0)), seed=int(spec.get("seed", 0)))
        self.x, self.v = np.zeros(n), np.zeros(n)
        at = int(spec.get("pluck_at", n // 4))
        if not 0 <= at < n:
            raise ContractError(f"pluck_at {at} is outside a chain of {n} particles")
        self.x[at] = float(spec.get("pluck_size", 0.1))
        if settle and scene.settle_steps > 0:
            host = ChainHost(self.model, worlds=1)
            host.load(self.x, self.v)
            for _ in range(scene.settle_steps):
                host.step()
            self.x, self.v = host.data.x[0].astype(np.float64), host.data.v[0].astype(np.float64)
        self.device = CpuDevice("cpu (numpy)")
        self.meta = dict(numpy=np.__version__, device=self.device.name, nv=n, nu=self.model.nu, nefc_start=0,
                         timestep=float(self.model.timestep), integrator="linearly-implicit-euler", repeatable=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def make_host(self, worlds: int):
        from phost.hosts.chain import ChainHost
        return ChainHost(self.model, worlds=worlds)

    instance_class = _Instance

    def instance(self, worlds: int):
        return self.instance_class(self, int(worlds))

    def layout(self) -> dict:
        n, nu = self.model.nv, self.model.nu
        geoms = [dict(type=2, size=[RADIUS, 0, 0], rgba=[0.9, 0.45, 0.2, 1.0] if i < nu else [0.25, 0.5, 0.9, 1.0],
                      body=i + 1, turns=False) for i in range(n)]
        return dict(geoms=geoms, flexes=[], bodies=n + 1)

    def profile(self, instance: _Instance) -> dict:
        return _phase_functions(instance)


class ChainEngine:
    name = "chain"
    suffix = ".chain"
    capabilities = frozenset({"profile", "live"})
    device_kind = "cpu"          # where it steps: "cpu" or "cuda" (what a tool may run alongside it)
    doc = "a damped spring-mass chain stepped in NumPy: no GPU, no MuJoCo"

    def load(self, scene, *, settle: bool = True) -> _Loaded:
        return _Loaded(scene, settle)


def scenes(registry) -> None:
    """Add the chain scenes to a ``SceneRegistry``."""
    from ..scene import Scene

    registry.add(Scene("chain32:pluck", lambda: chain_text(32, 2), 0,
                       "32 particles joined by springs, plucked a quarter of the way along; nv=32, nu=2. A NumPy simulator "
                       "with no GPU and no MuJoCo", ("chain", "cpu", "banded"), engine="chain"))
    registry.add(Scene("chain256:ringing", lambda: chain_text(256, 4, damping=0.05, stiffness=900.0), 200,
                       "256 particles, 2 s into a lightly damped ring-down; nv=256, nu=4; the effective system is a band of "
                       "half-width 1", ("chain", "cpu", "banded"), engine="chain"))
