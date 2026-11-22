"""Held-out scenes, scenes loaded from a directory, progress streaming, cancellation and verdict gates."""

import json
import stat
import threading
import time

import pytest

from phost import ContractError
from phost_lab import GraphSpec, SceneRegistry
from phost_lab.bootstrap import load_scenes
from phost_lab.isolate import run_isolated
from phost_lab.runner import RunConfig, _verdict
from phost_lab.scene import Scene
from phost_lab.worker import PREFIX, PROGRESS

from test_runner_cli_isolate import CONFIG, GRAPH, STAGE, cell, plugin, script, verdict

XML = "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>"


# ---- hidden scenes ----

def test_a_hidden_scene_shows_its_id_and_nothing_else():
    scene = Scene("held", lambda: XML, settle_steps=40, description="secret details", tags=("a",), hidden=True)
    assert scene.public() == dict(id="held", hidden=True, description="", settle_steps=0, tags=[])
    assert scene.xml() == XML                                     # still runnable


def test_describe_never_leaks_a_hidden_scene():
    reg = SceneRegistry()
    reg.add(Scene("open", lambda: XML, 3, "visible", ("t",)))
    reg.add(Scene("held", lambda: XML, 40, "secret details", ("a",), hidden=True))
    shown = json.dumps(reg.describe())
    assert "visible" in shown and "secret" not in shown and '"settle_steps": 40' not in shown
    assert [s["hidden"] for s in reg.describe()] == [False, True]


# ---- scenes from a directory ----

def test_add_directory_registers_xml_files_with_optional_metadata(tmp_path):
    (tmp_path / "a.xml").write_text(XML)
    (tmp_path / "b.xml").write_text(XML)
    (tmp_path / "b.json").write_text(json.dumps(dict(settle_steps=7, description="d", tags=["x"], hidden=True)))
    (tmp_path / "ignored.txt").write_text("not a scene")
    reg = SceneRegistry()
    assert reg.add_directory(tmp_path) == ["a", "b"]
    a, b = reg.get("a"), reg.get("b")
    assert (a.settle_steps, a.hidden, a.tags) == (0, False, ())
    assert (b.settle_steps, b.hidden, b.tags, b.description) == (7, True, ("x",), "d")
    assert b.xml() == XML


def test_add_directory_rejects_unknown_metadata_keys(tmp_path):
    (tmp_path / "a.xml").write_text(XML)
    (tmp_path / "a.json").write_text(json.dumps(dict(settle=3)))
    with pytest.raises(ContractError, match="unknown keys"):
        SceneRegistry().add_directory(tmp_path)


def test_add_directory_rejects_an_id_that_is_already_taken(tmp_path):
    (tmp_path / "trampoline8:impact.xml").write_text(XML)
    with pytest.raises(ContractError, match="already registered"):
        load_scenes(directories=[tmp_path])


def test_load_scenes_includes_directories(tmp_path):
    (tmp_path / "extra.xml").write_text(XML)
    scenes = load_scenes(directories=[tmp_path])
    assert "extra" in scenes.ids() and "trampoline8:impact" in scenes.ids()


# ---- verdict gates ----

def test_gates_say_which_check_passed(registry):
    g = verdict(registry, cell())["gates"]
    assert g == dict(finite=True, accuracy=True, stage=True)


def test_gates_report_a_failed_accuracy_check_separately(registry):
    v = verdict(registry, cell(accuracy_ratio=CONFIG.max_accuracy_ratio * 1.01))
    assert not v["ok"] and v["gates"] == dict(finite=True, accuracy=False, stage=True)


def test_an_accuracy_ratio_exactly_at_the_limit_passes(registry):
    assert verdict(registry, cell(accuracy_ratio=CONFIG.max_accuracy_ratio))["gates"]["accuracy"] is True


def test_gates_report_a_failed_stage_check_separately(registry):
    bad = cell(stage_metrics={STAGE: dict(status=1, max_relative_residual=1e-7, max_invariant_error=0.0, fallbacks=0)})
    assert verdict(registry, bad)["gates"] == dict(finite=True, accuracy=True, stage=False)


def test_gates_report_an_undefined_accuracy_as_none(registry):
    assert verdict(registry, cell(accuracy_ratio=None))["gates"]["accuracy"] is None


def test_gates_report_non_finite_qpos(registry):
    import numpy as np
    assert verdict(registry, cell(), qpos=np.array([np.nan]))["gates"]["finite"] is False


# ---- progress streaming and cancellation (a fake worker, no GPU) ----

def emit(event):
    return f"echo '{PROGRESS}{json.dumps(event)}'"


def result(**fields):
    return f"echo '{PREFIX}{json.dumps(dict(pass_=True, **fields))}'"


def test_progress_events_arrive_in_order_and_the_result_follows(tmp_path):
    events = []
    body = "\n".join([emit(dict(event="start", n=1)), emit(dict(event="cell", cell=dict(worlds=1))),
                      result(done=True)])
    out = run_isolated(GraphSpec("g"), "s", CONFIG, python=script(tmp_path, "cat >/dev/null\n" + body),
                       on_progress=events.append)
    assert [e["event"] for e in events] == ["start", "cell"] and events[1]["cell"] == dict(worlds=1)
    assert out["done"] is True


def test_progress_is_only_requested_when_there_is_a_listener(tmp_path):
    seen = tmp_path / "request.json"
    py = script(tmp_path, f"cat > {seen}\n{result()}")
    run_isolated(GraphSpec("g"), "s", CONFIG, python=py)
    assert json.loads(seen.read_text())["progress"] is False
    run_isolated(GraphSpec("g"), "s", CONFIG, python=py, on_progress=lambda e: None)
    assert json.loads(seen.read_text())["progress"] is True


def test_the_request_carries_scene_dirs(tmp_path):
    seen = tmp_path / "request.json"
    run_isolated(GraphSpec("g"), "s", CONFIG, scene_dirs=["/some/dir"], python=script(tmp_path, f"cat > {seen}\n{result()}"))
    assert json.loads(seen.read_text())["scene_dirs"] == ["/some/dir"]


def test_cancel_kills_the_worker_promptly(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    out = run_isolated(GraphSpec("g"), "s", CONFIG, python=script(tmp_path, "sleep 30"), cancel=cancel, timeout=60)
    assert out["pass"] is False and out["error"] == "cancelled" and time.monotonic() - started < 5


def test_cancel_kills_children_of_the_worker_too(tmp_path):
    marker = tmp_path / "child_alive"
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    # the child would create the marker after 2 s if it survived
    py = script(tmp_path, f"(sleep 2; touch {marker}) &\nsleep 30")
    run_isolated(GraphSpec("g"), "s", CONFIG, python=py, cancel=cancel, timeout=60)
    time.sleep(2.5)
    assert not marker.exists()


def test_a_failing_progress_callback_kills_the_worker_and_propagates(tmp_path):
    marker = tmp_path / "kept_running"
    body = f"{emit(dict(event='start'))}\nsleep 2\ntouch {marker}\n{result()}"

    def boom(event):
        raise RuntimeError("listener failed")

    with pytest.raises(RuntimeError, match="listener failed"):
        run_isolated(GraphSpec("g"), "s", CONFIG, python=script(tmp_path, "cat >/dev/null\n" + body), on_progress=boom)
    time.sleep(2.5)
    assert not marker.exists()


def test_a_timeout_after_progress_still_reports_a_timeout(tmp_path):
    events = []
    py = script(tmp_path, f"cat >/dev/null\n{emit(dict(event='start'))}\nsleep 10")
    out = run_isolated(GraphSpec("g"), "s", CONFIG, python=py, timeout=1.0, on_progress=events.append)
    assert "timed out" in out["error"] and [e["event"] for e in events] == ["start"]


def test_the_stderr_tail_is_kept_when_the_worker_dies(tmp_path):
    out = run_isolated(GraphSpec("g"), "s", CONFIG, python=script(tmp_path, "cat >/dev/null\necho oops >&2\nexit 7"))
    assert "code 7" in out["error"] and "oops" in out["stderr"]



def test_the_command_line_does_not_abbreviate_options(plugin, tmp_path):
    from phost_lab.cli import main
    graph = tmp_path / "g.json"
    graph.write_text(GraphSpec("g").to_json())
    for argv in (["run", str(graph), "--scene", "x", "--no-iso"], ["--scene", "list"], ["--scene-d", "x", "list"]):
        with pytest.raises(SystemExit):
            main(argv)
