"""``Driver``: how a host is stepped while timed. No GPU needed here; the end-to-end runs are in test_gpu.py."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from phost.errors import ContractError
from phost_lab import driver as drivers
from phost_lab.engine import CpuDevice
from phost_lab.engines.mjwarp import WarpDevice
from phost_lab.runner import RunConfig


class FakeWarp:
    """Just enough of Warp to see what a driver asks of it."""

    def __init__(self):
        self.log = []

    def synchronize(self):
        self.log.append("sync")

    @contextmanager
    def ScopedCapture(self):
        self.log.append("capture begin")
        yield SimpleNamespace(graph="GRAPH")
        self.log.append("capture end")

    def capture_launch(self, graph):
        self.log.append(f"launch {graph}")


def test_the_two_shipped_drivers_are_registered_and_the_default_is_the_graph():
    assert {"graph", "eager"} <= set(drivers.DRIVERS)
    assert RunConfig().driver == "graph"
    assert drivers.get_driver("graph").doc and drivers.get_driver("eager").doc


def test_an_unknown_driver_is_refused_with_the_known_ones():
    with pytest.raises(ContractError, match="no driver 'jax'.*eager.*graph"):
        drivers.get_driver("jax")


def test_the_graph_driver_traces_once_inside_a_capture_and_every_step_replays_it():
    wp, calls = FakeWarp(), []
    advance = drivers.get_driver("graph").prepare(WarpDevice(wp), lambda: calls.append("step"))
    assert calls == ["step", "step"]                      # one warm-up, one inside the capture
    advance(); advance()
    assert calls == ["step", "step"]                      # replays do not run Python
    assert wp.log.count("launch GRAPH") == 2 and "capture begin" in wp.log


def test_the_eager_driver_runs_the_step_from_python_every_time():
    wp, calls = FakeWarp(), []
    advance = drivers.get_driver("eager").prepare(WarpDevice(wp), lambda: calls.append("step"))
    assert calls == ["step"]                              # the warm-up
    advance(); advance()
    assert calls == ["step"] * 3 and "capture begin" not in wp.log


def test_a_plugin_driver_registers_by_name_and_a_different_driver_cannot_take_its_name():
    class Counting:
        name, doc = "test-counting", "counts the steps it runs"

        def prepare(self, device, step):
            return step

    try:
        first = drivers.register_driver(Counting())
        assert drivers.register_driver(Counting()) is first                    # the same kind again is a no-op
        assert drivers.get_driver("test-counting") is first

        class Impostor(Counting):
            pass

        with pytest.raises(ContractError, match="already registered"):
            drivers.register_driver(Impostor())
    finally:
        drivers.DRIVERS.pop("test-counting", None)


def test_a_driver_written_against_the_warp_module_still_works_on_the_warp_device():
    """The device forwards everything it does not define to Warp, so ``wp.ScopedCapture`` and friends are still there."""
    wp = FakeWarp()
    device = WarpDevice(wp)
    with device.ScopedCapture() as capture:
        pass
    assert capture.graph == "GRAPH" and device.can_capture


def test_the_graph_driver_refuses_a_device_that_cannot_capture_and_says_what_to_use():
    with pytest.raises(ContractError, match="cannot capture.*eager"):
        drivers.get_driver("graph").prepare(CpuDevice("cpu"), lambda: None)


def test_the_default_driver_falls_back_to_eager_on_such_a_device_and_says_so():
    driver, note = drivers.resolve_driver("graph", CpuDevice("cpu (numpy)"))
    assert driver.name == "eager" and "cannot capture" in note
    driver, note = drivers.resolve_driver("graph", WarpDevice(FakeWarp()))
    assert driver.name == "graph" and note is None
    driver, note = drivers.resolve_driver("eager", CpuDevice("cpu"))                 # an explicit choice is not second-guessed
    assert driver.name == "eager" and note is None
