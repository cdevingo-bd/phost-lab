import json
import os
import stat
import sys

import numpy as np
import pytest

from phost_lab import GraphSpec, Node
from phost_lab.bootstrap import load_registry
from phost_lab.cli import main
from phost_lab.isolate import run_isolated
from phost_lab.runner import RunConfig, _verdict

STAGE = "newton.linear_solve"
CONFIG = RunConfig()
GRAPH = GraphSpec("g", {STAGE: Node("test/reference")})


def cell(**over):
    base = dict(accuracy_ratio=1.0, native_spread=0.01, other_gpu_processes=0,
                stage_metrics={STAGE: dict(status=0, max_relative_residual=1e-7, max_invariant_error=0.0, fallbacks=0)})
    base.update(over)
    return base


def verdict(registry, c, qpos=np.zeros(3)):
    return _verdict(c, qpos, registry, GRAPH, CONFIG)


def test_verdict_ok_on_a_clean_cell(registry):
    v = verdict(registry, cell())
    assert v["ok"] and v["problems"] == [] and v["notes"] == []


def test_verdict_fails_on_non_finite_qpos(registry):
    assert "non-finite qpos" in verdict(registry, cell(), qpos=np.array([0.0, np.nan]))["problems"]


def test_verdict_fails_on_accuracy_ratio(registry):
    v = verdict(registry, cell(accuracy_ratio=11.0))
    assert not v["ok"] and "accuracy ratio" in v["problems"][0]


def test_verdict_notes_an_undefined_ratio_without_failing(registry):
    v = verdict(registry, cell(accuracy_ratio=None))
    assert v["ok"] and any("not defined" in n for n in v["notes"])


def test_verdict_fails_on_a_nonzero_status_bitmask(registry):
    metrics = {STAGE: dict(status=4, max_relative_residual=1e-7, max_invariant_error=0.0, fallbacks=0)}
    v = verdict(registry, cell(stage_metrics=metrics))
    assert not v["ok"] and any("status" in p for p in v["problems"])


def test_verdict_fails_on_a_tolerance_breach(registry):
    metrics = {STAGE: dict(status=0, max_relative_residual=1.0, max_invariant_error=0.0, fallbacks=0)}
    v = verdict(registry, cell(stage_metrics=metrics))
    assert not v["ok"] and any("max_relative_residual" in p for p in v["problems"])


def test_verdict_notes_noise_and_shared_gpu(registry):
    v = verdict(registry, cell(native_spread=0.5, other_gpu_processes=2))
    assert v["ok"] and len(v["notes"]) == 2
    assert any("unknown" in n for n in verdict(registry, cell(other_gpu_processes=-1))["notes"])


# ---- plugins and the command line (no GPU needed) ----

@pytest.fixture
def plugin(tmp_path, monkeypatch):
    (tmp_path / "labplugin.py").write_text(
        "from phost import Impl, ImplKind\n"
        "from phost.testing import runtimes\n"
        "def register(registry):\n"
        "    registry.add_impl(Impl(id='plug/reference', stage='newton.linear_solve', stage_version=1, version='1',\n"
        "        title='r', kind=ImplKind.BUILTIN, factory=lambda host, params: runtimes.ReferenceRuntime(host, params)))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "labplugin"


def write_graph(tmp_path, impl):
    path = tmp_path / "g.json"
    path.write_text(GraphSpec("g", {STAGE: Node(impl)}, plugins=("labplugin",)).to_json())
    return str(path)


def test_load_registry_registers_builtin_contract_and_plugin(plugin):
    from phost import Registry
    reg = load_registry([plugin], Registry())
    assert reg.contract(STAGE, 1) and reg.impl("plug/reference")
    assert load_registry([plugin], reg) is reg                 # idempotent


def test_cli_validate_ok_and_error(plugin, tmp_path, capsys):
    assert main(["validate", write_graph(tmp_path, "plug/reference")]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["validate", write_graph(tmp_path, "plug/missing")]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_list_prints_scenes_and_stages(plugin, capsys):
    assert main(["--plugin", plugin, "list"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert any(s["id"] == "trampoline12:impact" for s in out["scenes"])
    assert STAGE in json.dumps(out["registry"])


def test_cli_run_unknown_scene_is_an_error_not_a_traceback(plugin, tmp_path, capsys):
    assert main(["run", write_graph(tmp_path, "plug/reference"), "--scene", "nope"]) == 2
    assert "no scene" in capsys.readouterr().err


# ---- isolation: failures come back as data ----

def script(tmp_path, body):
    path = tmp_path / "fakepython"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_isolated_worker_reports_an_unknown_scene_as_an_error(plugin):
    out = run_isolated(GraphSpec("g", {STAGE: Node("plug/reference")}), "nope", CONFIG, plugins=[plugin], timeout=120)
    assert out["pass"] is False and "no scene" in out["error"]


def test_isolated_crash_without_a_result_is_reported(tmp_path):
    out = run_isolated(GraphSpec("g"), "trampoline8:impact", CONFIG, python=script(tmp_path, "echo boom >&2; exit 3"))
    assert out["pass"] is False and "code 3" in out["error"] and "boom" in out["stderr"]


def test_isolated_timeout_is_reported(tmp_path):
    out = run_isolated(GraphSpec("g"), "trampoline8:impact", CONFIG, python=script(tmp_path, "sleep 10"), timeout=0.5)
    assert out["pass"] is False and "timed out" in out["error"]


# ---- model-changing graphs are not compared with native ----

@pytest.fixture
def full_registry():
    from phost import Registry
    return load_registry([], Registry())


def graph_of(stage, impl):
    return GraphSpec("g", {stage: Node(impl)})


def test_load_registry_has_every_shipped_stage(full_registry):
    names = {c.name for c in full_registry.contracts()}
    assert names == {"newton.linear_solve", "dof.passive_force", "control.law", "probe.state"}
    assert full_registry.impl("control.law/pd") and full_registry.impl("probe.state/peak")


def test_only_model_changing_stages_make_a_graph_incomparable(full_registry):
    from phost_lab.runner import comparable_to_native
    assert comparable_to_native(GraphSpec("n"), full_registry)
    assert comparable_to_native(graph_of("probe.state", "probe.state/peak"), full_registry)
    assert comparable_to_native(graph_of(STAGE, "newton.linear_solve/native"), full_registry)
    assert not comparable_to_native(graph_of("dof.passive_force", "dof.passive_force/spring-damper"), full_registry)
    assert not comparable_to_native(graph_of("control.law", "control.law/pd"), full_registry)


def metrics_for(stage):
    return {stage: dict(calls=1)}


def test_a_model_changing_graph_is_not_failed_for_differing_from_native(full_registry):
    g = graph_of("control.law", "control.law/pd")
    c = cell(accuracy_ratio=1e9, stage_metrics=metrics_for("control.law"))
    v = _verdict(c, np.zeros(3), full_registry, g, CONFIG)
    assert v["ok"] and any("model-changing" in n for n in v["notes"])


def test_an_observer_that_changes_the_trajectory_is_failed(full_registry):
    g = graph_of("probe.state", "probe.state/peak")
    c = cell(accuracy_ratio=1e9, stage_metrics=metrics_for("probe.state"))
    v = _verdict(c, np.zeros(3), full_registry, g, CONFIG)
    assert not v["ok"] and "accuracy ratio" in v["problems"][0]


def test_actuated_scene_has_a_motor_per_joint():
    mujoco = pytest.importorskip("mujoco")
    from phost_lab import default_scenes
    model = mujoco.MjModel.from_xml_string(default_scenes().get("arm7:swing").xml())
    assert (model.nv, model.nu) == (7, 7)
