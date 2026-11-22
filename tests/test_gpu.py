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


def test_profile_times_every_phase_and_finds_the_effective_solve_only_under_the_discrete_integrator():
    from phost_lab.engines.mjwarp import PHASES
    from phost_lab.profile import profile_scene, table
    scenes = default_scenes()
    flat = profile_scene(NATIVE, scenes.get("trampoline8:impact"), (1,), launches=2, repeats=3)
    discrete = profile_scene(NATIVE, scenes.get("trampoline8:discrete"), (1,), launches=2, repeats=3)
    for result, expected in ((flat, "euler"), (discrete, "discrete")):
        assert result["integrator"] == expected
        entry = result["cells"][0]["variants"]["native"]
        assert list(entry["phases"]) == list(PHASES) and all(v >= 0 for v in entry["phases"].values())
        assert entry["full_ms"] > 0 and entry["phase_sum_ms"] > 0
    assert flat["cells"][0]["variants"]["native"]["effective_ms"] is None
    assert discrete["cells"][0]["variants"]["native"]["effective_ms"] > 0
    text = table(discrete)
    assert "of which effective solve" in text and "trampoline8:discrete" in text and "native ms" in text


def test_steps_per_call_reports_the_fixed_cost_of_a_call_and_survives_the_worker_round_trip():
    from phost_lab.cli import _table
    from phost_lab.isolate import run_isolated
    config = RunConfig(worlds=(1,), steps=3, launches=2, repeats=3, calls=(1, 4, 8))
    out = run_isolated(NATIVE, "trampoline8:impact", config, timeout=300)
    assert "error" not in out, out
    assert out["config"]["calls"] == [1, 4, 8]
    pc = out["cells"][0]["per_call"]
    assert pc["steps_per_call"] == [1, 4, 8]
    for label in ("native", "graph"):
        assert len(pc[label]["ms"]) == 3 and pc[label]["ms"][2] > pc[label]["ms"][0] > 0
        assert pc[label]["fixed_ms"] >= 0 and pc[label]["per_step_ms"] > 0
    assert len(pc["speedup"]) == 3
    assert "fixed ms" in _table(out)


def _full_registry():
    from phost import Registry
    from phost_lab.bootstrap import load_registry
    return load_registry([], Registry())


def test_applicability_matrix_is_read_from_what_the_host_refuses():
    from phost_lab.opportunity import applicability, applicability_table
    result = applicability(_full_registry(), default_scenes(), ["trampoline8:impact", "trampoline8:discrete", "arm7:swing"])
    cells = {scene: data["stages"] for scene, data in result["scenes"].items()}
    assert not cells["trampoline8:impact"]["effective.solve"]["applies"]
    assert "discrete integrator" in cells["trampoline8:impact"]["effective.solve"]["reason"]
    assert cells["trampoline8:discrete"]["effective.solve"]["applies"]
    assert cells["trampoline8:discrete"]["effective.solve"]["reason"] is None
    for scene in cells:                                              # a stage that attaches anywhere applies everywhere
        assert cells[scene]["body.wrench"]["applies"] and cells[scene]["newton.linear_solve"]["applies"]
    assert "no actuators" in cells["trampoline8:impact"]["control.law"]["impls"]["control.law/pd"]
    assert "bypasses this stage" in cells["trampoline8:discrete"]["dof.passive_force"]["impls"]["dof.passive_force/spring-damper"]
    text = applicability_table(result)
    assert "effective.solve" in text and "why not:" in text and "no actuators" in text


def test_scene_for_is_the_first_scene_where_the_stage_is_accepted():
    from phost_lab.opportunity import scene_for
    registry, scenes = _full_registry(), default_scenes()
    assert scene_for("newton.linear_solve", scenes, registry) == "trampoline8:impact"
    assert scene_for("effective.solve", scenes, registry) == "trampoline8:discrete"
    assert scene_for("body.wrench", scenes, registry) == "trampoline8:impact"      # only a native implementation: the host's acceptance decides
    assert scene_for("no.such.stage", scenes, registry) is None


def test_opportunities_give_each_stage_a_share_and_a_ceiling_only_where_a_swap_is_a_speed_swap():
    from phost_lab.opportunity import opportunities, opportunities_table
    scenes = default_scenes()
    flat = opportunities(scenes.get("trampoline8:impact"), (1,), launches=2, repeats=3)["cells"][0]
    assert not flat["stages"]["effective.solve"]["applies"]
    assert flat["stages"]["effective.solve"]["native_ms"] is None                  # n/a here, so nothing was timed
    discrete = opportunities(scenes.get("trampoline8:discrete"), (1,), launches=2, repeats=3)["cells"][0]
    eff = discrete["stages"]["effective.solve"]
    assert 0 < eff["share"] < 1 and eff["ceiling"] >= 1.0 and eff["enclosing"] is None
    assert eff["ceiling"] == pytest.approx(1.0 / (1.0 - eff["share"]))
    newton = discrete["stages"]["newton.linear_solve"]
    assert newton["enclosing"] and "solver.solve" in newton["enclosing"] and newton["ceiling"] is None   # an upper bound is not a ceiling
    assert discrete["stages"]["probe.state"]["native_ms"] is None                  # an observer has nothing to time alone
    assert discrete["stages"]["body.wrench"]["ceiling"] is None                    # model changing: no speed swap
    assert discrete["launch_floor_ms"] > 0 and discrete["step_ms"] > discrete["launch_floor_ms"]
    text = opportunities_table(dict(scene="trampoline8:discrete", cells=[discrete]))
    assert "upper bound" in text and "effective.solve" in text and "x" in text


def test_every_step_of_a_run_goes_through_the_driver():
    """The runner times, replays and compares through ``advance`` and nothing else, so a new way of stepping is one class."""
    from phost_lab import driver as drivers

    class Counting(drivers.GraphDriver):
        name, doc, advanced = "test-counting", "the graph driver, counting", 0

        def prepare(self, wp, step):
            advance = super().prepare(wp, step)

            def counted():
                Counting.advanced += 1
                advance()
            return counted

    drivers.register_driver(Counting())
    try:
        out = run_graph(NATIVE, default_scenes().get("trampoline8:impact"), RunConfig(worlds=(1,), steps=5, launches=2, repeats=3,
                                                                                    driver="test-counting"), None)
        assert out["config"]["driver"] == "test-counting"
        assert Counting.advanced == 3 * (2 * 3 + 5)          # native, graph, native2: timing samples plus the accuracy replay
    finally:
        drivers.DRIVERS.pop("test-counting")


def test_the_eager_driver_runs_a_whole_comparison_and_the_choice_survives_the_worker_round_trip():
    from phost_lab.isolate import run_isolated
    out = run_isolated(NATIVE, "trampoline8:impact", RunConfig(worlds=(1,), steps=3, launches=1, repeats=3, driver="eager"),
                       timeout=300)
    cell, = out["cells"]
    assert out["config"]["driver"] == "eager" and out["pass"]
    assert cell["step_ms"] > 0 and 0.3 < cell["speedup"] < 3.0       # native against native, stepped from Python


def test_the_structure_report_says_what_a_structure_exploiting_solve_could_save_on_the_flex_scene():
    from phost_lab.opportunity import structure, structure_table
    scenes = default_scenes()
    flat = structure(scenes.get("trampoline8:impact"))
    assert flat["stages"]["effective.solve"]["structure"] is None and "discrete integrator" in flat["stages"]["effective.solve"]["reason"]
    s = structure(scenes.get("trampoline12:discrete"))["stages"]["effective.solve"]["structure"]
    assert s["n"] == 330 and s["coupled"] == 300 and s["density"] < 0.1
    assert s["bandwidth_rcm"] <= s["bandwidth"] < s["n"]
    assert s["banded_flops"] < 0.2 * s["dense_flops"] and s["coupled_flops"] > 0.5 * s["dense_flops"]     # coupled rows barely help here
    text = structure_table(dict(scene="trampoline12:discrete", nv=330, integrator="discrete", stages=dict(
        {"effective.solve": dict(applies=True, reason=None, structure=s)})))
    assert "reverse Cuthill-McKee" in text and "flops are not time" in text


def test_the_reports_run_in_a_worker_and_come_back_as_data():
    from phost_lab.isolate import run_isolated
    out = run_isolated(NATIVE, "trampoline8:discrete", RunConfig(), timeout=300, plugins=[],
                       report=dict(kinds=["opportunities", "structure", "applicability"], worlds=[1]))
    assert "error" not in out, out
    assert out["opportunities"]["cells"][0]["stages"]["effective.solve"]["share"] > 0
    assert out["structure"]["stages"]["effective.solve"]["structure"]["n"] > 0
    assert out["applicability"]["scenes"]["trampoline8:discrete"]["stages"]["effective.solve"]["applies"]
    profiled = run_isolated(NATIVE, "trampoline8:impact", RunConfig(), timeout=300, report=dict(kinds=["profile"], worlds=[1]))
    variants = profiled["profile"]["cells"][0]["variants"]
    assert list(variants) == ["native"] and variants["native"]["full_ms"] > 0 and variants["native"]["phases"]["solver"] > 0
    bad = run_isolated(NATIVE, "trampoline8:discrete", RunConfig(), timeout=300, report=dict(kinds=["nope"]))
    assert "no report 'nope'" in bad["error"]
