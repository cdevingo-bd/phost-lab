"""MuJoCo Warp as an engine: MuJoCo parses and settles on the CPU, MuJoCo Warp steps on a CUDA device. Needs CUDA to load a scene.

Everything that names MuJoCo or Warp in the lab lives here (and in ``profile.py`` and ``live.py``, which are tools for this
engine's phases and drawing).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time

import numpy as np

SCOPE = "cuda:0"


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
        self.model = None
        self._scope = None
        self.meta = dict(mujoco=mujoco.__version__, warp=wp.config.version, device=wp.get_device(SCOPE).name,
                         nv=int(self.model_cpu.nv), nefc_start=int(self.cpu.nefc), timestep=float(self.model_cpu.opt.timestep),
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


class MJWarpEngine:
    name = "mjwarp"
    doc = "MuJoCo Warp on a CUDA device; scenes are MJCF"

    def load(self, scene, *, settle: bool = True) -> _Loaded:
        return _Loaded(scene, settle)
