"""How a host is stepped while it is timed: the boundary between a run and the thing that advances the simulation.

A run times and compares steps; it does not care how one step is launched. The way a step is launched changes what a
step costs: replaying a captured CUDA graph pays one launch for the whole step, tracing it from Python pays a launch per
kernel, and a rollout driven from another framework pays that framework's per-call overhead on top. A ``Driver`` names
one of those ways. Everything that times or advances a host (the per-step time, the steps-per-call curve, the accuracy
replay) goes through ``advance()``, so a new way of stepping is one class and a registration, not a change to the runner.

* ``graph``: capture the step once as a CUDA graph and replay it. The default, and what the lab measured until now.
* ``eager``: run the step from Python every time, as an un-captured rollout loop does.

A driver is handed the engine's ``Device`` (``phost_lab.engine``) and one call of the host's own step. It asks the device for
what it needs (``synchronize``, ``capture``) and never names a simulator, so the same driver serves every engine whose device
can do it. ``graph`` needs a device that can capture; on one that cannot (NumPy on the CPU) a run steps eagerly and its
result says so (``resolve_driver``).

A plugin adds one by calling ``register_driver`` from its ``register(registry)`` (a JAX-wrapped step is the case this was
made for). The session that installs stages stays active for as long as ``advance`` is used, so a driver that re-runs Python
sees the installed stages, and a driver that captured them still replays them.
"""

from __future__ import annotations

from typing import Callable, Protocol

from phost.errors import ContractError


class Driver(Protocol):
    name: str
    doc: str

    def prepare(self, device, step: Callable[[], None]) -> Callable[[], None]:
        """Warm ``step`` up on this host and return ``advance``: a function that performs one step of the host.

        ``step`` is one call of the host's own step on its data. ``advance`` returns without waiting for the device; the
        caller synchronizes. ``device`` is the engine's ``Device``: ``synchronize()``, ``capture(step)``, ``can_capture``.
        (For the MuJoCo Warp engine it also forwards every other attribute to the Warp module, so a driver written
        against ``wp`` keeps working.)
        """


class GraphDriver:
    name = "graph"
    doc = "capture the step once as a CUDA graph; every step replays it"

    def prepare(self, device, step):
        step()
        device.synchronize()
        replay = device.capture(step)
        if replay is None:
            raise ContractError(f"the device {device.name!r} cannot capture a step; use the 'eager' driver")
        device.synchronize()
        return replay


class EagerDriver:
    name = "eager"
    doc = "run the step from Python every time: every kernel is launched from the host thread"

    def prepare(self, device, step):
        step()
        device.synchronize()
        return step


DRIVERS: dict[str, Driver] = {}


def register_driver(driver: Driver) -> Driver:
    """Make ``driver`` available by its ``name`` (``--driver``, ``RunConfig.driver``). Registering the same one again is a no-op."""
    existing = DRIVERS.get(driver.name)
    if existing is not None and type(existing) is not type(driver):
        raise ContractError(f"driver {driver.name!r} is already registered by {type(existing).__name__}")
    return DRIVERS.setdefault(driver.name, driver)


def get_driver(name: str) -> Driver:
    try:
        return DRIVERS[name]
    except KeyError:
        raise ContractError(f"no driver {name!r}; known {sorted(DRIVERS)}") from None


def resolve_driver(name: str, device) -> tuple[Driver, str | None]:
    """The driver to use for ``name`` on ``device``, and a note when it is not the one asked for.

    ``graph`` is the default, so a device that cannot capture gets ``eager`` and the note says so: a run on the CPU should
    work without a flag, and its result must not claim it replayed a graph.
    """
    driver = get_driver(name)
    if name == "graph" and not getattr(device, "can_capture", True):
        return get_driver("eager"), f"device {device.name!r} cannot capture a step, so it was stepped eagerly"
    return driver, None


register_driver(GraphDriver())
register_driver(EagerDriver())
