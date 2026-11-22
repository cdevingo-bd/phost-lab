"""The engine boundary, exercised end to end on the chain engine: a NumPy simulator, no GPU, no MuJoCo.

Everything here runs the real runner, the real reports and the real stage hooks. If the lab still depended on MuJoCo Warp
anywhere on this path, these would fail on a machine without CUDA.
"""

import json
import subprocess
import sys
import os

import pytest

import phost
from phost import ContractError, Impl, ImplKind
from phost.stages import effective_solve as es
from phost.testing import runtimes

from phost_lab import GraphSpec, NATIVE, Node, Scene, SceneRegistry, default_scenes
from phost_lab.bootstrap import load_registry
from phost_lab.engine import CpuDevice, ENGINES, engine_for, engines, get_engine, register_engine, require_capability, supports
from phost_lab.engines import chain as chain_engine
from phost_lab.opportunity import applicability, opportunities, opportunities_table, scene_for, structure, structure_table
from phost_lab.live import LiveConfig, run_live
from phost_lab.profile import profile_scene, table as profile_table
from phost_lab.runner import RunConfig, run_graph

QUICK = RunConfig(worlds=(1, 2), steps=10, launches=2, repeats=3)


@pytest.fixture
def registry():
    reg = load_registry(registry=phost.Registry())
    for name, cls in (("ref", runtimes.EffectiveReferenceRuntime), ("wrong", runtimes.WrongEffectiveRuntime)):
        reg.add_impl(Impl(id=f"test/effective-{name}", stage=es.NAME, stage_version=1, version="1", title=name,
                          kind=ImplKind.BUILTIN, factory=lambda host, params, cls=cls: cls(host, params)))
    return reg


@pytest.fixture
def scene():
    return Scene("chain-test", lambda: chain_engine.chain_text(24, 2), 5, "a small chain", ("chain",), engine="chain")


def graph(impl, host="chain"):
    return GraphSpec("g", {es.NAME: Node(f"test/effective-{impl}")}, host=host)


# ---- engines and scenes -----------------------------------------------------------------------------

def test_the_two_shipped_engines_are_registered_and_an_unknown_one_is_refused_with_the_known():
    assert {"mjwarp", "chain"} <= set(engines())
    assert get_engine("chain").doc and get_engine("mjwarp").doc
    with pytest.raises(ContractError, match="no engine 'jax'.*chain.*mjwarp"):
        get_engine("jax")


def test_a_different_engine_cannot_take_a_registered_name():
    class Impostor:
        name, doc = "chain", "not the chain"

    with pytest.raises(ContractError, match="already registered"):
        register_engine(Impostor())


def test_the_default_scenes_say_which_engine_runs_them():
    scenes = default_scenes()
    by_id = {s["id"]: s for s in scenes.describe()}
    assert by_id["chain32:pluck"]["engine"] == "chain" and by_id["chain256:ringing"]["engine"] == "chain"
    assert by_id["arm7:swing"]["engine"] == "mjwarp" and by_id["trampoline8:impact"]["engine"] == "mjwarp"


def test_a_held_out_scene_does_not_say_what_it_runs_on():
    hidden = Scene("held-out", lambda: "{}", hidden=True, engine="chain")
    assert hidden.public() == dict(id="held-out", hidden=True, description="", settle_steps=0, tags=[])


def test_a_graph_for_one_host_is_refused_on_a_scene_of_another(scene):
    with pytest.raises(ContractError, match="host 'mjwarp'.*engine 'chain'"):
        engine_for(scene, graph("ref", host="mjwarp"))
    assert engine_for(scene, graph("ref")).name == "chain"
    assert engine_for(scene, NATIVE).name == "chain"                 # a graph with no stages is for any host


def test_a_chain_scene_is_validated_when_it_is_loaded():
    bad = Scene("bad", lambda: json.dumps(dict(n=8, wobble=1)), engine="chain")
    with pytest.raises(ContractError, match="JSON object"):
        get_engine("chain").load(bad)
    out_of_range = Scene("far", lambda: chain_engine.chain_text(8, 1, pluck_at=9), engine="chain")
    with pytest.raises(ContractError, match="pluck_at 9"):
        get_engine("chain").load(out_of_range)


def test_the_chain_engine_settles_a_scene_and_resets_to_it(scene):
    with get_engine("chain").load(scene) as loaded:
        instance = loaded.instance(3)
        start = instance.state().copy()
        for _ in range(4):
            instance.step()
        assert abs(instance.state() - start).max() > 0
        instance.reset()
        assert (instance.state() == start).all() and instance.worlds == 3 and len(instance.frame()) == 24
        assert loaded.meta["nv"] == 24 and loaded.meta["repeatable"] is True
    unsettled = get_engine("chain").load(scene, settle=False)
    assert abs(unsettled.instance(1).state()[0] - start[0]).max() > 0          # five settle steps moved it


# ---- a run, end to end ------------------------------------------------------------------------------

def test_the_native_graph_runs_on_the_chain_and_is_stepped_eagerly_with_the_reason_recorded(registry, scene):
    result = run_graph(NATIVE, scene, QUICK, registry)
    assert result["pass"] and [c["worlds"] for c in result["cells"]] == [1, 2]
    assert result["config"]["driver"] == "eager" and result["config"]["driver_requested"] == "graph"
    assert "cannot capture" in result["meta"]["driver_note"]
    assert result["meta"]["engine"] == "chain" and result["meta"]["device"] == "cpu (numpy)" and result["meta"]["nv"] == 24
    cell = result["cells"][0]
    assert cell["control_rms_dqpos"] == 0.0 and cell["rms_dqpos"] == 0.0         # native agrees with itself exactly
    assert cell["accuracy_basis"] == "rounding floor" and cell["accuracy_ratio"] == 0.0


def test_a_correct_exact_swap_passes_against_the_rounding_floor(registry, scene):
    result = run_graph(graph("ref"), scene, QUICK, registry)
    assert result["pass"], [c["verdict"] for c in result["cells"]]
    cell = result["cells"][0]
    assert cell["accuracy_ratio"] < 1.0 and result["stage_classes"][es.NAME] == "exact"
    assert cell["stage_metrics"][es.NAME]["status"] == 0
    assert cell["verdict"]["gates"] == dict(finite=True, accuracy=True, stage=True)


def test_a_wrong_exact_swap_is_caught_even_though_native_has_no_noise_floor(registry, scene):
    """With no control noise the ratio used to be undefined and the accuracy gate silently off. It is not off now."""
    result = run_graph(graph("wrong"), scene, QUICK, registry)
    assert not result["pass"]
    problems = result["cells"][0]["verdict"]["problems"]
    assert any("accuracy ratio" in p for p in problems), problems
    assert result["cells"][0]["verdict"]["gates"]["accuracy"] is False


def test_the_frames_of_a_run_are_the_chains_coordinates(registry, scene):
    result = run_graph(NATIVE, scene, RunConfig(worlds=(1,), steps=6, launches=1, repeats=2, frame_every=2), registry)
    frames = result["cells"][0]["frames"]
    assert len(frames) == 3 and len(frames[0]) == 24


def test_the_per_call_curve_works_without_a_device_to_capture_on(registry, scene):
    result = run_graph(NATIVE, scene, RunConfig(worlds=(1,), steps=4, launches=1, repeats=2, calls=(1, 3)), registry)
    assert result["cells"][0]["per_call"]["steps_per_call"] == [1, 3]


def test_a_run_on_the_wrong_host_fails_before_anything_is_loaded(registry, scene):
    with pytest.raises(ContractError, match="engine 'chain'"):
        run_graph(graph("ref", host="mjwarp"), scene, QUICK, registry)


# ---- the reports ------------------------------------------------------------------------------------

def test_applicability_reads_a_chain_scene_through_its_engine(registry, scene):
    scenes = SceneRegistry()
    scenes.add(scene)
    out = applicability(registry, scenes, ["chain-test"])
    cell = out["scenes"]["chain-test"]
    assert cell["nv"] == 24 and cell["integrator"] == "linearly-implicit-euler"
    assert sorted(cell["stages"]) == ["control.law", "dof.passive_force", "effective.solve", "probe.state"]
    assert all(s["applies"] for s in cell["stages"].values())


def test_structure_reports_the_chains_band(scene):
    out = structure(scene)
    banded = out["stages"][es.NAME]["structure"]
    assert banded["n"] == 24 and banded["bandwidth"] == 1 and banded["nnz"] == 24 + 2 * 23
    assert "band: 1" in structure_table(out)


def test_opportunities_runs_on_the_cpu_and_times_the_stages_the_host_can_run_alone(scene):
    out = opportunities(scene, (1,), launches=3, repeats=5)
    cell = out["cells"][0]
    assert cell["launch_floor_ms"] == 0.0 and cell["step_ms"] > 0 and cell["nv"] == 24
    stages = cell["stages"]
    assert set(stages) == {"control.law", "dof.passive_force", "effective.solve", "probe.state"}
    solve = stages["effective.solve"]
    assert solve["native_ms"] > 0 and 0 < solve["share"] <= 1 and solve["ceiling"] > 1       # an exact swap has a ceiling
    assert stages["control.law"]["ceiling"] is None and stages["dof.passive_force"]["ceiling"] is None   # they change the physics
    assert stages["probe.state"]["native_ms"] is None                                          # an observer: nothing to time alone
    assert "attaches after the whole step" in opportunities_table(out)


def test_scene_for_finds_the_first_scene_whose_host_accepts_the_stage(registry):
    scenes = SceneRegistry()
    chain_engine.scenes(scenes)
    assert scene_for(es.NAME, scenes, registry) == "chain32:pluck"
    assert scene_for("body.wrench", scenes, registry) is None             # a chain has no bodies


def test_an_engine_declares_which_optional_tools_it_offers(registry, scene):
    assert supports(scene, "profile") and supports(scene, "live") and not supports(scene, "render")      # drawn by the dashboard: MuJoCo only
    assert supports(Scene("m", lambda: "<mujoco/>"), "render")
    bare = Scene("bare", scene.xml, engine="test-bare")

    class Bare:
        name, doc, suffix, capabilities = "test-bare", "no tools", ".bare", frozenset()

    try:
        register_engine(Bare())
        with pytest.raises(ContractError, match="the live view is not offered by the 'test-bare' engine.*chain, mjwarp"):
            require_capability(bare, "live", "the live view")
    finally:
        ENGINES.pop("test-bare", None)


# ---- the tools an engine offers ---------------------------------------------------------------------

def test_the_phase_profile_runs_on_the_chains_own_phases(registry, scene):
    out = profile_scene(graph("ref"), scene, (1,), registry, launches=2, repeats=3)
    assert out["engine"] == "chain" and out["integrator"] == "linearly-implicit-euler" and out["nv"] == 24
    for variant in out["cells"][0]["variants"].values():
        assert list(variant["phases"]) == ["passive force", "actuation", "assemble", "effective solve", "integrate"]
        assert variant["full_ms"] > 0 and variant["effective_ms"] is None and variant["phase_sum_ms"] > 0
    assert set(out["cells"][0]["variants"]) == {"native", "graph"}
    assert "effective solve" in profile_table(out) and "full step (one replay)" in profile_table(out)


def test_the_profile_restores_the_state_the_integrate_phase_would_otherwise_run_away_with(registry, scene):
    out = profile_scene(NATIVE, scene, (2,), registry, launches=5, repeats=3)
    phases = out["cells"][0]["variants"]["native"]["phases"]
    assert all(v >= 0 and v == v for v in phases.values())                     # finite, non-negative after the restore cost


def test_the_live_view_draws_the_chains_particles_and_measures_how_far_a_graph_drifts(registry, scene):
    events = []
    out = run_live(graph("ref"), scene, LiveConfig(steps=30, every=5), registry, on_progress=events.append)
    layout = events[0]
    assert layout["event"] == "layout" and len(layout["geoms"]) == 24 and layout["flexes"] == [] and layout["timestep"] == 0.01
    frames = [e for e in events if e["event"] == "live"]
    assert len(frames) == 6 and len(frames[0]["native"]["x"]) == 72 and frames[0]["finite"]
    assert out["noise"] == 0.0 and out["drift"] < 1e-5                          # native repeats exactly; a correct solve stays on it
    assert out["meta"]["device"] == "cpu (numpy)" and out["stage_metrics"][es.NAME]["status"] == 0
    wrong = run_live(graph("wrong"), scene, LiveConfig(steps=30, every=5), registry)
    assert wrong["drift"] > 1e-4


def test_a_held_out_chain_scene_is_never_drawn(registry):
    hidden = Scene("held", lambda: chain_engine.chain_text(8, 1), engine="chain", hidden=True)
    with pytest.raises(ContractError, match="never drawn"):
        run_live(NATIVE, hidden, registry=registry)


# ---- scene directories --------------------------------------------------------------------------------

def test_a_scene_directory_carries_scenes_for_every_engine(tmp_path, registry):
    (tmp_path / "ring.chain").write_text(chain_engine.chain_text(12, 1, pluck_at=3))
    (tmp_path / "ring.json").write_text(json.dumps(dict(hidden=True, settle_steps=3, tags=["held"])))
    (tmp_path / "ball.xml").write_text("<mujoco/>")
    scenes = SceneRegistry()
    assert scenes.add_directory(tmp_path) == ["ball", "ring"]
    ring, ball = scenes.get("ring"), scenes.get("ball")
    assert ring.engine == "chain" and ring.hidden and ring.settle_steps == 3 and ball.engine == "mjwarp"
    assert scenes.describe()[1] == dict(id="ring", hidden=True, description="", settle_steps=0, tags=[])     # nothing shown
    result = run_graph(NATIVE, ring, RunConfig(worlds=(1,), steps=3, launches=1, repeats=2), registry)
    assert result["pass"] and result["meta"]["nv"] == 12 and result["scene_hidden"] is True


def test_a_scene_directory_refuses_a_scene_in_two_formats_and_a_metadata_engine_that_disagrees(tmp_path):
    (tmp_path / "x.chain").write_text(chain_engine.chain_text(8, 1))
    (tmp_path / "x.xml").write_text("<mujoco/>")
    with pytest.raises(ContractError, match="'x' is in both"):
        SceneRegistry().add_directory(tmp_path)
    (tmp_path / "x.xml").unlink()
    (tmp_path / "x.json").write_text(json.dumps(dict(engine="mjwarp")))
    with pytest.raises(ContractError, match="engine 'mjwarp', but x.chain is a scene for 'chain'"):
        SceneRegistry().add_directory(tmp_path)


# ---- a third party's engine -------------------------------------------------------------------------

def test_a_plugin_engine_is_used_through_the_protocol_alone(registry, scene):
    """An engine that wraps another and counts what the runner asks of it: the runner reaches nothing but the protocol."""
    calls = []

    class Counting:
        name, doc = "test-counting", "the chain engine, watched"

        def load(self, scene, *, settle=True):
            inner = get_engine("chain").load(scene, settle=settle)
            calls.append("load")

            class Watched:
                device = CpuDevice("watched cpu")
                meta = dict(inner.meta, device="watched cpu")

                def __enter__(self):
                    calls.append("enter")
                    return self

                def __exit__(self, *exc):
                    calls.append("exit")

                def instance(self, worlds):
                    calls.append(f"instance {worlds}")
                    return inner.instance(worlds)

            return Watched()

    try:
        register_engine(Counting())
        watched = Scene("watched", scene.xml, 0, engine="test-counting")
        result = run_graph(NATIVE, watched, RunConfig(worlds=(1,), steps=3, launches=1, repeats=2), registry)
        assert result["pass"] and result["meta"]["engine"] == "test-counting" and result["meta"]["device"] == "watched cpu"
        assert calls[:2] == ["load", "enter"] and calls.count("instance 1") == 3 and calls[-1] == "exit"
    finally:
        ENGINES.pop("test-counting", None)


def test_the_lab_imports_without_mujoco_or_warp():
    code = ("import sys, phost_lab, phost_lab.runner, phost_lab.opportunity, phost_lab.engines.chain; "
            "bad = [m for m in ('mujoco', 'mujoco_warp', 'warp') if m in sys.modules]; print(bad); raise SystemExit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
    assert done.returncode == 0, done.stdout + done.stderr
