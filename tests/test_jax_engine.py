"""The JAX chain engine: a step that compiles, and a run compared with the NumPy chain from the same scene."""

import pytest

jax = pytest.importorskip("jax", reason="needs jax")

import phost
from phost import Impl, ImplKind
from phost.stages import effective_solve as es

from phost_lab import GraphSpec, NATIVE, Node, Scene, SceneRegistry, default_scenes
from phost_lab.bootstrap import load_registry
from phost_lab.driver import resolve_driver
from phost_lab.engine import engine_for, get_engine, supports, unavailable
from phost_lab.engines import chain as chain_engine
from phost_lab.live import LiveConfig, run_live
from phost_lab.opportunity import applicability, opportunities, scene_for
from phost_lab.profile import profile_scene
from phost_lab.runner import RunConfig, run_graph

QUICK = RunConfig(worlds=(1, 2), steps=10, launches=2, repeats=3)


class JaxSolve:
    """A direct solve written for JAX arrays, so it can be traced into the compiled step."""

    factor = 1.0

    def __init__(self, host, params):
        pass

    def solve(self, s):
        import jax.numpy as jnp
        from phost.ports import unwrap

        n = int(s.qacc.shape[1])
        upper = jnp.triu(unwrap(s.h)[:, :n, :n])
        a = upper + jnp.swapaxes(upper, 1, 2) - jnp.einsum("wi,ij->wij", jnp.einsum("wii->wi", upper), jnp.eye(n))
        s.qacc[...] = self.factor * jnp.linalg.solve(a, unwrap(s.rhs)[:, :n, None])[..., 0]

    def metrics(self):
        return {"status": 0}


class JaxWrong(JaxSolve):
    factor = 1.01


@pytest.fixture
def registry():
    reg = load_registry(registry=phost.Registry())
    for name, cls in (("ref", JaxSolve), ("wrong", JaxWrong)):
        reg.add_impl(Impl(id=f"test/effective-{name}", stage=es.NAME, stage_version=1, version="1", title=name,
                          kind=ImplKind.BUILTIN, factory=lambda host, params, cls=cls: cls(host, params)))
    return reg


@pytest.fixture
def scene():
    return Scene("jax-test", lambda: chain_engine.chain_text(24, 2), 5, "a small chain", ("chain",), engine="jaxchain")


def graph(impl, host="jaxchain"):
    return GraphSpec("g", {es.NAME: Node(f"test/effective-{impl}")}, host=host)


def test_the_engine_is_registered_available_and_has_its_scene():
    assert get_engine("jaxchain").unavailable() is None
    assert {s["id"]: s["engine"] for s in default_scenes().describe()}["jaxchain32:pluck"] == "jaxchain"


def test_the_graph_driver_compiles_the_step_and_a_replay_runs_the_compiled_program(scene):
    with get_engine("jaxchain").load(scene) as loaded:
        device = loaded.device
        assert device.can_capture and "jax" in device.name
        eager, compiled = loaded.instance(2), loaded.instance(2)
        driver, note = resolve_driver("graph", device)
        assert driver.name == "graph" and note is None
        advance = driver.prepare(device, compiled.step)
        compiled.reset()
        for _ in range(30):
            advance()
            eager.step()
        device.synchronize()
        eager.reset()
        for _ in range(30):
            eager.step()
        assert abs(compiled.state() - eager.state()).max() < 1e-6


def test_a_run_on_the_jax_chain_replays_a_compiled_step_and_a_wrong_swap_is_still_caught(registry, scene):
    result = run_graph(graph("ref"), scene, QUICK, registry)
    assert result["pass"], [c["verdict"] for c in result["cells"]]
    assert result["config"]["driver"] == "graph" and "driver_requested" not in result["config"]
    assert result["meta"]["engine"] == "jaxchain" and result["cells"][0]["stage_metrics"][es.NAME]["calls"] > 0
    wrong = run_graph(graph("wrong"), scene, QUICK, registry)
    assert not wrong["pass"] and any("accuracy ratio" in p for p in wrong["cells"][0]["verdict"]["problems"])


def test_a_stage_that_cannot_be_traced_is_refused_by_the_graph_driver_and_runs_eagerly(registry, scene):
    from phost.testing import runtimes
    from phost import ContractError
    registry.add_impl(Impl(id="test/effective-numpy", stage=es.NAME, stage_version=1, version="1", title="numpy",
                           kind=ImplKind.BUILTIN, factory=lambda host, params: runtimes.EffectiveReferenceRuntime(host, params)))
    host_python = GraphSpec("g", {es.NAME: Node("test/effective-numpy")}, host="jaxchain")
    with pytest.raises(ContractError, match="cannot be compiled by jax.*'eager' driver"):
        run_graph(host_python, scene, QUICK, registry)
    out = run_graph(host_python, scene, RunConfig(worlds=(1,), steps=5, launches=1, repeats=2, driver="eager"), registry)
    assert out["pass"]


def test_the_stage_call_count_in_a_compiled_run_counts_executions_not_traces(registry, scene):
    cfg = RunConfig(worlds=(1,), steps=7, launches=1, repeats=1)
    calls = run_graph(graph("ref"), scene, cfg, registry)["cells"][0]["stage_metrics"][es.NAME]["calls"]
    assert calls == 7                                                       # reset_metrics, then exactly the 7 replayed steps


def test_the_eager_driver_runs_the_step_as_python_and_agrees(registry, scene):
    graph_run = run_graph(graph("ref"), scene, QUICK, registry)
    eager = run_graph(graph("ref"), scene, RunConfig(worlds=(1,), steps=10, launches=2, repeats=3, driver="eager"), registry)
    assert eager["pass"] and eager["config"]["driver"] == "eager"
    assert abs(eager["cells"][0]["rms_dqpos"] - graph_run["cells"][0]["rms_dqpos"]) < 1e-5


def test_the_same_scene_on_the_numpy_chain_and_the_jax_chain_starts_from_the_same_state(scene):
    numpy_scene = Scene("n", scene.xml, 5, engine="chain")
    with get_engine("chain").load(numpy_scene) as a, get_engine("jaxchain").load(scene) as b:
        assert abs(a.instance(1).state() - b.instance(1).state()).max() == 0.0
        one, two = a.instance(1), b.instance(1)
        for _ in range(40):
            one.step()
            two.step()
        assert abs(one.state() - two.state()).max() < 1e-6


def test_the_tools_work_on_the_jax_chain(registry, scene):
    assert supports(scene, "profile") and supports(scene, "live") and not supports(scene, "render")
    prof = profile_scene(graph("ref"), scene, (1,), registry, launches=2, repeats=3)
    for variant in prof["cells"][0]["variants"].values():
        assert list(variant["phases"]) == ["passive force", "actuation", "assemble", "effective solve", "integrate"]
        assert variant["full_ms"] > 0 and all(v >= 0 for v in variant["phases"].values())
    events = []
    live = run_live(graph("ref"), scene, LiveConfig(steps=20, every=5), registry, on_progress=events.append)
    assert events[0]["event"] == "layout" and len(events[0]["geoms"]) == 24 and live["drift"] < 1e-5 and live["noise"] == 0.0
    out = opportunities(scene, (1,), launches=2, repeats=3)
    assert out["cells"][0]["stages"][es.NAME]["native_ms"] is not None and out["cells"][0]["launch_floor_ms"] >= 0


def test_a_scene_for_an_engine_whose_simulator_is_missing_is_skipped_not_fatal(registry, monkeypatch):
    engine = get_engine("jaxchain")
    monkeypatch.setattr(type(engine), "unavailable", lambda self: "jax is not installed")
    scenes = SceneRegistry()
    chain_engine.scenes(scenes)
    from phost_lab.engines import jaxchain
    jaxchain.scenes(scenes)
    assert unavailable(scenes.get("jaxchain32:pluck")) == "jax is not installed"
    assert "jaxchain32:pluck" not in applicability(registry, scenes)["scenes"]
    assert scene_for("body.wrench", scenes, registry) is None


def test_a_jaxchain_scene_directory_file_is_found_by_its_suffix(tmp_path):
    (tmp_path / "held.jaxchain").write_text(chain_engine.chain_text(8, 1))
    scenes = SceneRegistry()
    assert scenes.add_directory(tmp_path) == ["held"] and scenes.get("held").engine == "jaxchain"
