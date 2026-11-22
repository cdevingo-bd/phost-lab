import json

import pytest

from phost import ContractError
from phost_lab import NATIVE, GraphSpec, Node, SceneRegistry, default_scenes
from phost_lab.metrics import NOISE_FLOOR, accuracy_ratio

STAGE = "newton.linear_solve"


def graph(impl="test/reference", stage=STAGE, params=None):
    return GraphSpec("g", {stage: Node(impl, params or {})}, plugins=("some.plugin",), note="n")


def test_round_trip_is_lossless():
    g = graph(params={"a": 1})
    assert GraphSpec.from_json(g.to_json()) == g
    assert GraphSpec.from_dict(json.loads(g.to_json())).to_dict() == g.to_dict()


def test_native_graph_has_no_stages_and_validates(registry):
    NATIVE.validate(registry)
    assert NATIVE.selections() == []


@pytest.mark.parametrize("bad", [dict(), dict(version=2, name="x"), dict(version=1), dict(version=1, name="x", stages={"s": 3})])
def test_malformed_graph_is_a_contract_error(bad):
    with pytest.raises(ContractError):
        GraphSpec.from_dict(bad)


def test_validate_accepts_a_registered_impl(registry):
    graph().validate(registry)


def test_validate_rejects_unknown_impl(registry):
    with pytest.raises(ContractError):
        graph(impl="nope/none").validate(registry)


def test_validate_rejects_impl_on_the_wrong_stage(registry):
    with pytest.raises(ContractError, match="is for"):
        graph(stage="some.other_stage").validate(registry)


def test_validate_rejects_unknown_params(registry):
    with pytest.raises(ContractError):
        graph(params={"not_a_param": 1}).validate(registry)


def test_default_scenes_and_lookup():
    scenes = default_scenes()
    assert {"trampoline12:impact", "trampoline12:rest"} <= set(scenes.ids())
    assert scenes.get("trampoline12:impact").settle_steps < scenes.get("trampoline12:rest").settle_steps
    with pytest.raises(ContractError, match="no scene"):
        scenes.get("nope")


def test_discrete_trampolines_are_the_euler_ones_with_the_discrete_integrator():
    mujoco = pytest.importorskip("mujoco")
    scenes = default_scenes()
    assert {"trampoline8:discrete", "trampoline12:discrete", "trampoline14:discrete"} <= set(scenes.ids())
    discrete = mujoco.MjModel.from_xml_string(scenes.get("trampoline12:discrete").xml())
    euler = mujoco.MjModel.from_xml_string(scenes.get("trampoline12:impact").xml())
    assert euler.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert discrete.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
    assert discrete.nv == euler.nv == 330                      # the description says nv=330
    assert "discrete" in scenes.get("trampoline12:discrete").tags


def test_duplicate_scene_is_rejected():
    reg = SceneRegistry()
    scene = default_scenes().get("trampoline8:impact")
    reg.add(scene)
    with pytest.raises(ContractError):
        reg.add(scene)


def test_scene_xml_is_a_mujoco_model():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(default_scenes().get("trampoline8:impact").xml())
    assert model.nv == 138                   # the description says nv=138


def test_accuracy_ratio_is_undefined_when_the_control_is_noise():
    assert accuracy_ratio(1e-3, NOISE_FLOOR / 10) is None
    assert accuracy_ratio(2e-6, 1e-6) == pytest.approx(2.0)
