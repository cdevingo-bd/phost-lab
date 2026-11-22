"""End to end on a real GPU with MJWarp. Skipped when mujoco_warp or CUDA is not available."""

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

from phost_lab import NATIVE, default_scenes
from phost_lab.runner import RunConfig, run_graph

try:
    wp.init()
    HAVE_CUDA = wp.is_cuda_available()
except Exception:                                   # pragma: no cover
    HAVE_CUDA = False
pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")

FAST = RunConfig(worlds=(1,), steps=5, launches=2, repeats=3)


def test_native_graph_matches_itself(registry):
    out = run_graph(NATIVE, default_scenes().get("trampoline8:impact"), FAST, registry)
    cell, = out["cells"]
    assert out["pass"] and cell["stage_metrics"] == {}
    assert 0.5 < cell["speedup"] < 2.0               # native against native: timing noise only


def test_progress_events_frames_and_gates_on_a_real_run(registry):
    events = []
    config = RunConfig(worlds=(1,), steps=6, launches=2, repeats=3, frame_every=2, max_frames=10)
    out = run_graph(NATIVE, default_scenes().get("trampoline8:impact"), config, registry, on_progress=events.append)
    assert [e["event"] for e in events] == ["start", "cell"]
    assert events[0]["meta"]["timestep"] > 0 and events[0]["worlds"] == [1]
    cell, = out["cells"]
    assert events[1]["cell"]["worlds"] == cell["worlds"]
    import mujoco
    nq = mujoco.MjModel.from_xml_string(default_scenes().get("trampoline8:impact").xml()).nq
    assert len(cell["frames"]) == 3                          # steps 0, 2 and 4 of 6
    assert all(len(frame) == nq for frame in cell["frames"])
    assert cell["frames"][0] != cell["frames"][-1]            # the scene is moving
    assert cell["verdict"]["gates"]["finite"] is True


def test_frames_are_capped_and_off_by_default(registry):
    scene = default_scenes().get("trampoline8:impact")
    off = run_graph(NATIVE, scene, FAST, registry)
    assert "frames" not in off["cells"][0]
    capped = run_graph(NATIVE, scene, RunConfig(worlds=(1,), steps=8, launches=2, repeats=3, frame_every=1, max_frames=4),
                       registry)
    assert len(capped["cells"][0]["frames"]) == 4


def test_the_real_worker_streams_progress_before_the_result(tmp_path):
    from phost_lab.isolate import run_isolated
    (tmp_path / "held.xml").write_text(default_scenes().get("trampoline8:impact").xml())
    (tmp_path / "held.json").write_text('{"hidden": true}')
    events = []
    out = run_isolated(NATIVE, "held", FAST, scene_dirs=[str(tmp_path)], on_progress=events.append, timeout=300)
    assert [e["event"] for e in events] == ["start", "cell"]
    assert out["pass"] and out["scene"] == "held" and out["scene_hidden"] is True


def test_a_live_pass_streams_a_layout_then_frames_of_native_and_graph(registry):
    import math
    from phost_lab.live import LiveConfig, run_live
    events = []
    out = run_live(NATIVE, default_scenes().get("trampoline8:impact"), LiveConfig(steps=12, every=3), registry,
                   on_progress=events.append)
    assert [e["event"] for e in events] == ["layout", "live", "live", "live", "live"]
    layout, frames = events[0], events[1:]
    assert [f["step"] for f in frames] == [3, 6, 9, 12] and out["frames"] == 4 and out["steps"] == 12
    assert len(layout["geoms"]) == out["meta"]["ngeom"] and layout["flexes"]
    for frame in frames:
        assert len(frame["native"]["x"]) == 3 * out["meta"]["ngeom"] == len(frame["graph"]["x"])
        assert frame["finite"] is True and math.isfinite(frame["drift"]) and math.isfinite(frame["noise"])
    # the graph is the native step again, so it can only differ by the device's own run-to-run noise
    assert frames[-1]["native"]["x"] != frames[0]["native"]["x"] and out["drift"] < 1e-3
    assert out["stage_metrics"] == {}


def test_live_frames_stop_at_the_cap(registry):
    from phost_lab.live import LiveConfig, run_live
    events = []
    out = run_live(NATIVE, default_scenes().get("trampoline8:impact"), LiveConfig(steps=20, every=2, max_frames=3),
                   registry, on_progress=events.append)
    assert out["frames"] == 3 and sum(e["event"] == "live" for e in events) == 3
    assert out["steps"] == 6                                  # it stopped stepping at the last frame it could show
