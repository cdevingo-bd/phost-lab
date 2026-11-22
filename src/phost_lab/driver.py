"""How a host is stepped while it is timed: the boundary between a run and the thing that advances the simulation.

A run times and compares steps; it does not care how one step is launched. The way a step is launched changes what a
step costs: replaying a captured CUDA graph pays one launch for the whole step, tracing it from Python pays a launch per
kernel, and a rollout driven from another framework pays that framework's per-call overhead on top. A ``Driver`` names
one of those ways. Everything that times or advances a host (the per-step time, the steps-per-call curve, the accuracy
replay) goes through ``advance()``, so a new way of stepping is one class and a registration, not a change to the runner.

* ``graph``: capture the step once as a CUDA graph and replay it. The default, and what the lab measured until now.
* ``eager``: run the step from Python every time, as an un-captured rollout loop does.

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

    def prepare(self, wp, step: Callable[[], None]) -> Callable[[], None]:
        """Warm ``step`` up on this host and return ``advance``: a function that performs one step of the host.

        ``step`` is one call of the host's own step on its data. ``advance`` returns without waiting for the device; the
        caller synchronizes. ``wp`` is the Warp module, passed so a driver does not import it at module level.
        """


class GraphDriver:
    name = "graph"
    doc = "capture the step once as a CUDA graph; every step replays it"

    def prepare(self, wp, step):
        step()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            step()
        wp.synchronize()
        graph = capture.graph
        return lambda: wp.capture_launch(graph)


class EagerDriver:
    name = "eager"
    doc = "run the step from Python every time: every kernel is launched from the host thread"

    def prepare(self, wp, step):
        step()
        wp.synchronize()
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


register_driver(GraphDriver())
register_driver(EagerDriver())
