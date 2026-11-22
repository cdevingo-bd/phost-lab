"""The chain over JAX arrays as an engine: a simulator whose arrays cannot be written in place, and whose step can be compiled.

The scene is the chain's (``phost_lab.engines.chain``), settled the same way, so the same scene on ``chain`` and on
``jaxchain`` starts from the same state and a run on one can be compared with a run on the other. The host is
``phost.hosts.jaxchain.JaxChainHost``, whose arrays are ``Ref`` cells; its stages are the chain's.

* **Capture is compilation.** ``graph`` asks the device to capture one call of the step. Here a step knows how to compile itself
  (``compile()``): the installed stages are traced into one XLA program, ahead of time, and a replay is a call of that program.
  ``eager`` runs the step as Python, one JAX operation at a time, which is what a rollout loop that is not jitted pays.
* **Synchronizing is waiting on the last result**, because JAX returns as soon as work is enqueued.

Needs ``jax`` (``unavailable()`` says so when it is missing). The CPU build is what this was written against; a GPU build of
JAX changes the device name and nothing else.
"""

from __future__ import annotations

import importlib.util
import statistics
import time

import numpy as np

from phost import ContractError

from . import chain


def _jax():
    try:
        import jax
    except ImportError:
        raise ContractError("the jaxchain engine needs jax: pip install jax") from None
    return jax


class Compilable:
    """One call of the host's own Python (a step, or a phase of one) that can also be compiled into an XLA program."""

    def __init__(self, host, device: "JaxDevice", function):
        self.host, self.device, self.function = host, device, function

    def __call__(self) -> None:
        self.function()
        self.device.pending = self.host.data.x.value

    def compile(self):
        """Trace ``function`` with whatever stages are installed now and compile it; return a function that replays it."""
        jax, host, device = _jax(), self.host, self.device
        state = host.get_state()
        try:
            compiled = jax.jit(host.pure(self.function)).lower(state).compile()
        except Exception as error:                         # a stage whose Python cannot be traced
            raise ContractError(f"the step cannot be compiled by jax ({type(error).__name__}: {(str(error).splitlines() or [""])[0][:160]}); use the "
                                "'eager' driver") from None

        def replay() -> None:
            new = compiled(host.get_state())
            host.set_state(new)
            device.pending = new
        return replay


class JaxDevice:
    can_capture = True

    def __init__(self):
        jax = _jax()
        self.name = f"{jax.devices()[0].platform} (jax {jax.__version__})"
        self.pending = None

    def synchronize(self) -> None:
        if self.pending is not None:
            _jax().block_until_ready(self.pending)

    def capture(self, step):
        compile = getattr(step, "compile", None)
        return compile() if compile is not None else None

    def busy(self) -> int:
        return 0

    def replay_ms(self, step, launches: int, repeats: int) -> float:
        step()
        self.synchronize()
        replay = self.capture(step) or step
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(launches):
                replay()
            self.synchronize()
            samples.append(1e3 * (time.perf_counter() - start) / launches)
        return statistics.median(samples)

    def floor_ms(self, launches: int, repeats: int) -> float:
        jax = _jax()
        import jax.numpy as jnp
        x = jnp.zeros(1)
        compiled = jax.jit(lambda v: v + 1.0).lower(x).compile()
        box = [x]

        def tick() -> None:
            box[0] = compiled(box[0])
            self.pending = box[0]
        return self.replay_ms(tick, launches, repeats)


class _Instance(chain._Instance):
    def make_step(self):
        return Compilable(self.host, self._loaded.device, self.host.step)


class _Loaded(chain._Loaded):
    instance_class = _Instance

    def __init__(self, scene, settle: bool):
        _jax()
        super().__init__(scene, settle)
        self.device = JaxDevice()
        self.meta = dict(self.meta, device=self.device.name, jax=_jax().__version__)

    def make_host(self, worlds: int):
        from phost.hosts.jaxchain import JaxChainHost
        return JaxChainHost(self.model, worlds=worlds)

    def profile(self, instance: _Instance) -> dict:
        """The step's phases as functions of the cells, each compilable on its own."""
        host, device = instance.host, self._device_of(instance)
        m, d, ns = host.model, host.data, host.ns
        from phost.hosts import jaxchain as hosts
        from phost.ports import unwrap

        saved = {name: unwrap(getattr(d, name)) for name in ("x", "v", "t")}

        def restore():
            for name, value in saved.items():
                getattr(d, name)[...] = value

        def integrate():
            h = m.timestep
            d.v[...] = unwrap(d.v) + h * unwrap(d.a)
            d.x[...] = unwrap(d.x) + h * unwrap(d.v)
            d.t[...] = unwrap(d.t) + h

        wrap = lambda function: Compilable(host, device, function)
        phases = {"passive force": wrap(lambda: ns.passive(m, d)), "actuation": wrap(lambda: ns.actuate(m, d)),
                  "assemble": wrap(lambda: ns.assemble(m, d)), "effective solve": wrap(lambda: ns.effective_solve(m, d, d.a)),
                  "integrate": wrap(integrate)}
        return dict(phases=phases, restore=wrap(restore), restored=("integrate",), extra={})

    @staticmethod
    def _device_of(instance):
        return instance._loaded.device


class JaxChainEngine:
    name = "jaxchain"
    suffix = ".jaxchain"
    capabilities = frozenset({"profile", "live"})
    doc = "the damped spring-mass chain over JAX arrays: a step that compiles, and arrays that cannot be written in place"

    @property
    def device_kind(self) -> str:
        """JAX steps on whatever its default device is: the CPU build is "cpu", an accelerator build is not."""
        try:
            import jax
            return "cpu" if jax.devices()[0].platform == "cpu" else "cuda"
        except Exception:
            return "unknown"

    def unavailable(self) -> str | None:
        return None if importlib.util.find_spec("jax") is not None else "jax is not installed"

    def load(self, scene, *, settle: bool = True) -> _Loaded:
        return _Loaded(scene, settle)


def scenes(registry) -> None:
    """Add the JAX chain scenes to a ``SceneRegistry``: the chain's scenes, on the other engine."""
    from ..scene import Scene

    registry.add(Scene("jaxchain32:pluck", lambda: chain.chain_text(32, 2), 0,
                       "the 32-particle chain on JAX arrays, step compiled by XLA; same scene and start state as chain32:pluck",
                       ("chain", "jax", "banded"), engine="jaxchain"))
