"""The boundary between the lab and the thing that advances a simulation.

A run needs a short list of things from a simulator, and none of them is MuJoCo: load a scene and settle it to a start
state, make instances of it with some number of worlds, step one, put it back at the start, read its state, say what
device it runs on and how that device is synchronized and (if it can) captured. An ``Engine`` provides them; the runner,
the opportunity reports and the drivers speak only this, so a new simulator is one engine and one registration.

    engine.load(scene)        -> Loaded     parse and settle on the CPU; entering it opens the device scope
    loaded.instance(worlds)   -> Instance   the data for that many worlds, and a ``phost.Host`` over it
    instance.step()                         one step of every world, through whatever the host has installed
    instance.reset()                        back to the loaded start state, time zero
    instance.state()          -> (worlds, n) float64: the coordinates runs are compared on
    loaded.device                           synchronize, capture, how busy, the cost of launching nothing

and, only if it has something to say about them, three optional tools (``Engine.capabilities`` names them):

    loaded.check_host(worlds) -> Host       a host a contract check can drive, when the engine's ordinary host cannot be (``drive``)
    engine.render(scene, frames, w, h)      frames drawn as (h, w, 3) uint8 images; ``render_unavailable()`` says why not here
    loaded.profile(instance), loaded.layout() and instance.pose()      the phases of a step, and something to draw

Three ship: ``mjwarp`` (MuJoCo Warp on a CUDA device; ``phost_lab.engines.mjwarp``), ``chain`` (a spring-mass chain
stepped in NumPy; ``phost_lab.engines.chain``) and ``jaxchain`` (the same chain over JAX arrays, compiled by XLA;
``phost_lab.engines.jaxchain``); the chains exist to show the boundary is not MuJoCo's. A scene names its engine
(``Scene.engine``) and a graph names the host its stages are for (``GraphSpec.host``): they must agree, because stages
are bound to a host by name.

A plugin adds an engine by calling ``register_engine`` from its ``register(registry)``; an engine for a framework that
steps in JAX would give its device a ``capture`` that returns a jitted replay and leave the rest alone.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from phost import ContractError


class Device(Protocol):
    """What the lab asks of the thing a step runs on."""

    name: str
    can_capture: bool            # False: the ``graph`` driver cannot be used here, and a run says it stepped eagerly

    def synchronize(self) -> None:
        """Block until every step launched so far has finished."""

    def capture(self, step: Callable[[], None]) -> Callable[[], None] | None:
        """Record one call of ``step`` and return a function that replays it (without waiting), or None if this
        device cannot capture. ``step`` has been run once already."""

    def busy(self) -> int:
        """Other processes using the device, or -1 when it cannot be told. 0 for a device nobody else shares."""

    def floor_ms(self, launches: int, repeats: int) -> float:
        """What replaying one trivial piece of work costs: the fixed price of timing anything on this device."""

    def replay_ms(self, step: Callable[[], None], launches: int, repeats: int) -> float:
        """Median milliseconds to replay ``step`` once (captured when the device can, as it is otherwise)."""


class Instance(Protocol):
    """One copy of a loaded scene with some number of worlds."""

    host: Any                    # a phost.Host over this instance's data: what a Session installs stages on
    worlds: int

    def step(self) -> None: ...
    def reset(self) -> None: ...
    def state(self) -> np.ndarray: ...
    def frame(self) -> list[float]:
        """World 0's coordinates as a list, for playback."""


class Loaded(Protocol):
    """A scene parsed and settled, usable as a context manager that holds the device scope open while instances are used."""

    device: Device
    meta: Mapping[str, Any]      # facts for a result: device, nv, timestep, versions, ``repeatable``

    def instance(self, worlds: int) -> Instance: ...
    def __enter__(self) -> "Loaded": ...
    def __exit__(self, *exc) -> None: ...


class Engine(Protocol):
    name: str                    # the host name its stages are bound to (``GraphSpec.host``)
    doc: str
    suffix: str                  # the file suffix of a scene for this engine in a scene directory (".xml" for MJCF)
    capabilities: frozenset      # optional tools it offers: "profile", "live", "render" (see ``supports``)
    device_kind: str             # optional, where it steps: "cpu" (several runs may share the machine) or "cuda"; absent: unknown,
                                 # which a caller treats as one at a time

    def load(self, scene, *, settle: bool = True) -> Loaded:
        """Parse ``scene`` and (if ``settle``) run it to its start state. ``settle=False`` is for a report that needs the
        model and not a state: applicability and structure."""

    # optional, with the "render" capability:
    #   render(scene, frames, width, height) -> iterator of (height, width, 3) uint8 arrays, one per frame (a coordinate vector)
    #   render_unavailable() -> str | None    why frames cannot be drawn on this machine (a missing renderer), or None


class CpuDevice:
    """A device with nothing to capture and nothing to wait for: plain Python or NumPy, stepped directly."""

    name = "cpu"
    can_capture = False

    def __init__(self, name: str = "cpu"):
        self.name = name

    def synchronize(self) -> None:
        return None

    def capture(self, step):
        return None

    def busy(self) -> int:
        return 0

    def floor_ms(self, launches: int, repeats: int) -> float:
        return 0.0

    def replay_ms(self, step, launches: int, repeats: int) -> float:
        step()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(launches):
                step()
            samples.append(1e3 * (time.perf_counter() - start) / launches)
        return statistics.median(samples)


ENGINES: dict[str, Engine] = {}


def register_engine(engine: Engine) -> Engine:
    """Make ``engine`` available by its ``name`` (``Scene.engine``). Registering the same kind again is a no-op."""
    existing = ENGINES.get(engine.name)
    if existing is not None and type(existing) is not type(engine):
        raise ContractError(f"engine {engine.name!r} is already registered by {type(existing).__name__}")
    return ENGINES.setdefault(engine.name, engine)


def get_engine(name: str) -> Engine:
    _register_builtin()
    try:
        return ENGINES[name]
    except KeyError:
        raise ContractError(f"no engine {name!r}; known {sorted(ENGINES)}") from None


def engine_for(scene, graph=None) -> Engine:
    """The engine a scene runs on. With ``graph``, refuse a graph whose stages are bound to a different host: a stage
    cannot be installed on a simulator it was not written for."""
    engine = get_engine(scene.engine)
    if graph is not None and graph.stages and graph.host != engine.name:
        raise ContractError(f"graph {graph.name!r} is for host {graph.host!r} but scene {scene.id!r} runs on engine "
                            f"{engine.name!r}; set the graph's host to {engine.name!r}")
    return engine


_builtin_registered = False


def supports(scene, capability: str) -> bool:
    """Whether the engine a scene runs on offers an optional tool: ``profile`` (``Loaded.profile``: the phases of its step),
    ``live`` (``Loaded.layout`` and ``Instance.pose``: something to draw) or ``render`` (``Engine.render``: frames drawn as images)."""
    return capability in getattr(get_engine(scene.engine), "capabilities", frozenset())


def can_render() -> bool:
    """Whether any registered engine that offers ``render`` can draw frames on this machine."""
    return any("render" in getattr(e, "capabilities", frozenset()) and getattr(e, "render_unavailable", lambda: None)() is None
               for e in engines().values())


def render_frames(scene, frames, width: int, height: int):
    """The frames of a run on ``scene`` drawn by its engine: an iterator of ``(height, width, 3)`` uint8 arrays. Refused, naming
    the engines that can, when the scene's engine does not draw."""
    require_capability(scene, "render", "drawing frames")
    engine = get_engine(scene.engine)
    why = getattr(engine, "render_unavailable", lambda: None)()
    if why:
        raise ContractError(f"the {engine.name!r} engine cannot draw frames here: {why}")
    return engine.render(scene, frames, int(width), int(height))


def check_host(loaded, worlds: int):
    """A host over ``worlds`` worlds of a loaded scene that ``phost.testing.contract.check_impl`` can drive (it has ``drive``).

    An engine whose ordinary host can be driven (the chains) needs to say nothing; one that needs more than its ordinary host
    gives ``Loaded.check_host(worlds)`` (MuJoCo Warp adds ``drive`` over a state with contacts). Call it inside ``with loaded:``."""
    make = getattr(loaded, "check_host", None)
    host = make(int(worlds)) if make is not None else loaded.instance(int(worlds)).host
    if not hasattr(host, "drive"):
        raise ContractError(f"a {type(host).__name__} cannot be driven by a contract check (it has no drive()); the engine "
                            f"gives a drivable host with Loaded.check_host(worlds)")
    return host


def require_capability(scene, capability: str, tool: str) -> None:
    """Refuse a tool the scene's engine does not offer, naming the engines that do."""
    if not supports(scene, capability):
        offering = sorted(n for n, e in engines().items() if capability in getattr(e, "capabilities", frozenset()))
        raise ContractError(f"{tool} is not offered by the {scene.engine!r} engine (scene {scene.id!r}); engines that offer it: "
                            f"{', '.join(offering) or 'none'}")


_builtin_registered = False


def require_engine(scene, name: str, tool: str, why: str) -> None:
    """Refuse a tool that only one engine's scenes make sense for (``why`` says what it needs)."""
    if scene.engine != name:
        raise ContractError(f"{tool} needs {why}, which only the {name!r} engine has; scene {scene.id!r} runs on {scene.engine!r}")


def _register_builtin() -> None:
    """The shipped engines, on first use (their modules import this one, so this cannot run when it is imported)."""
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True
    from .engines import chain, jaxchain, mjwarp
    register_engine(mjwarp.MJWarpEngine())
    register_engine(chain.ChainEngine())
    register_engine(jaxchain.JaxChainEngine())


def unavailable(scene) -> str | None:
    """Why the engine a scene runs on cannot be used on this machine (its simulator is not installed), or None."""
    check = getattr(get_engine(scene.engine), "unavailable", None)
    return check() if check is not None else None


def engines() -> dict[str, Engine]:
    """Every registered engine by name."""
    _register_builtin()
    return dict(ENGINES)
