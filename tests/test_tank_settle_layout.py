"""The tank scene, the settled-state cache and the viewer layout. No GPU needed."""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from phost_lab import SceneRegistry, default_scenes
from phost_lab.live import layout
from phost_lab.settle import cache_dir, key, settled
from phost_lab.tank import SAIL_GRID, tank_xml, water_count

BALL = "<mujoco><worldbody><body pos='0 0 1'><freejoint/><geom size='.1' mass='1'/></body></worldbody></mujoco>"


# ---- the scene ----

def test_the_tank_is_registered_and_says_what_its_water_is():
    scene = default_scenes().get("tank:boat")
    assert scene.settle_steps == 500 and scene.contacts == 1500 and scene.constraints == 3000
    assert "stand-in" in scene.description and "no fluid solver" in scene.description


def test_the_tank_has_the_parts_that_were_asked_for():
    model = mujoco.MjModel.from_xml_string(tank_xml())
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert {"tank", "boat", "mast", "crate_a", "crate_b", "crate_c"} <= names
    assert model.nflex == 1 and model.flex_vertnum[0] == SAIL_GRID ** 2         # a cloth sail
    assert water_count() == 150 and model.nv == 1 + 3 * 150 + 6 + 3 * (SAIL_GRID ** 2 - 2 * SAIL_GRID + 1) + 3 * 6


def test_the_tank_starts_without_overlaps():
    """Every part must start clear of every other, or the first steps blow the scene up (the sail once started inside the
    hull and the crates)."""
    model = mujoco.MjModel.from_xml_string(tank_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, sorted(c.dist for c in data.contact[:data.ncon])[:3]


def test_the_boat_starts_just_above_the_water_and_the_crates_on_its_deck():
    model = mujoco.MjModel.from_xml_string(tank_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = lambda n: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
    boat = body("boat")
    deck_top = boat[2] + 0.05
    for crate, half in (("crate_a", 0.025), ("crate_b", 0.02), ("crate_c", 0.02)):
        assert deck_top < body(crate)[2] - half < deck_top + 0.005
        assert abs(body(crate)[0] - boat[0]) < 0.19 and abs(body(crate)[1] - boat[1]) < 0.085       # inside the rails


def test_the_sail_starts_beside_the_mast_above_the_deck():
    model = mujoco.MjModel.from_xml_string(tank_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boat = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "boat")]
    sail = data.flexvert_xpos - boat
    assert sail[:, 0].min() > -0.001 and sail[:, 0].max() < 0.2 and sail[:, 2].min() > 0.05     # forward of the mast, on deck


def test_the_water_is_frictionless_and_the_water_option_changes_how_much_there_is():
    model = mujoco.MjModel.from_xml_string(tank_xml(layers=2))
    assert water_count(layers=2) == 100 and model.nv < mujoco.MjModel.from_xml_string(tank_xml()).nv
    first = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE][0]
    assert tuple(model.geom_friction[first]) == (0, 0, 0) and model.geom_condim[first] == 1 and model.geom_priority[first] == 1


def test_a_short_run_stays_finite_and_the_boat_stays_up():
    model = mujoco.MjModel.from_xml_string(tank_xml(layers=2))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    data = mujoco.MjData(model)
    boat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "boat")
    start = data.xpos[boat][2]
    for _ in range(120):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.abs(data.qacc).max() < 1e4
    assert data.xpos[boat][2] > start - 0.1            # it settled onto the water, it did not fall through


def test_the_sail_can_be_left_out():
    model = mujoco.MjModel.from_xml_string(tank_xml(sail=False))
    assert model.nflex == 0


# ---- the settled-state cache ----

def test_the_first_settle_is_computed_and_stored_the_second_is_read(tmp_path, monkeypatch):
    model = mujoco.MjModel.from_xml_string(BALL)
    first = settled(model, BALL, 100, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    calls = []
    real = mujoco.mj_step
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: (calls.append(1), real(*a))[1])
    second = settled(model, BALL, 100, tmp_path)
    assert calls == [] and np.array_equal(first.qpos, second.qpos) and np.array_equal(first.qvel, second.qvel)
    assert second.time == first.time > 0 and second.xpos[1][2] == first.xpos[1][2]      # forward ran on the loaded state


def test_a_cached_state_equals_a_fresh_one_exactly(tmp_path):
    model = mujoco.MjModel.from_xml_string(BALL)
    fresh = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, fresh)
    mujoco.mj_forward(model, fresh)
    settled(model, BALL, 100, tmp_path)
    cached = settled(model, BALL, 100, tmp_path)
    assert np.array_equal(cached.qpos, fresh.qpos) and np.array_equal(cached.qvel, fresh.qvel)


def test_zero_steps_touch_no_cache(tmp_path):
    settled(mujoco.MjModel.from_xml_string(BALL), BALL, 0, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_the_key_depends_on_model_steps_and_nothing_else_you_could_forget():
    assert key(BALL, 100) == key(BALL, 100)
    assert len({key(BALL, 100), key(BALL, 101), key(BALL.replace("mass='1'", "mass='2'"), 100)}) == 3


@pytest.mark.parametrize("damage", ["garbage", "wrong_shape", "nan"])
def test_a_cache_file_that_does_not_fit_is_ignored_and_rewritten(tmp_path, damage):
    model = mujoco.MjModel.from_xml_string(BALL)
    good = settled(model, BALL, 50, tmp_path)
    path = next(tmp_path.glob("*.npz"))
    if damage == "garbage":
        path.write_bytes(b"not an npz")
    elif damage == "wrong_shape":
        np.savez(path, qpos=np.zeros(3), qvel=np.zeros(6), act=np.zeros(0), time=0.0)
    else:
        np.savez(path, qpos=np.full(7, np.nan), qvel=np.zeros(6), act=np.zeros(0), time=0.0)
    again = settled(model, BALL, 50, tmp_path)
    assert np.array_equal(again.qpos, good.qpos)
    with np.load(path) as saved:
        assert np.array_equal(saved["qpos"], good.qpos)                     # rewritten with the right state


def test_an_unwritable_cache_does_not_stop_a_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    data = settled(mujoco.MjModel.from_xml_string(BALL), BALL, 20, blocker / "sub")        # cannot mkdir under a file
    assert data.time > 0


def test_the_cache_folder_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOST_LAB_CACHE", str(tmp_path / "a"))
    assert cache_dir() == tmp_path / "a"
    monkeypatch.delenv("PHOST_LAB_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert cache_dir() == tmp_path / "x" / "phost-lab"


# ---- scene folders can size their buffers ----

def test_a_scene_folder_can_set_the_buffers(tmp_path):
    (tmp_path / "big.xml").write_text(BALL)
    (tmp_path / "big.json").write_text(json.dumps(dict(contacts=500, constraints=900)))
    (tmp_path / "plain.xml").write_text(BALL)
    reg = SceneRegistry()
    reg.add_directory(tmp_path)
    assert (reg.get("big").contacts, reg.get("big").constraints) == (500, 900)
    assert reg.get("plain").contacts is None and reg.get("plain").nconmax(8) == 16 * 8 + 64     # the original sizing
    assert reg.get("big").nconmax(8) == 500


# ---- the layout a viewer draws from ----

def test_layout_describes_every_geom_and_the_cloth_triangles():
    model = mujoco.MjModel.from_xml_string(tank_xml())
    out = layout(model)
    assert len(out["geoms"]) == model.ngeom and out["bodies"] == model.nbody
    assert sum(g["turns"] for g in out["geoms"]) == 15                      # hull, rails, crates, mast, boom: not the water
    flex, = out["flexes"]
    assert flex["vertnum"] == 64 and len(flex["tri"]) == 2 * (SAIL_GRID - 1) ** 2
    assert max(max(t) for t in flex["tri"]) == 63 and all(len(set(t)) == 3 for t in flex["tri"])
    json.dumps(out)


def test_layout_of_a_scene_without_cloth_has_no_triangles():
    out = layout(mujoco.MjModel.from_xml_string(default_scenes().get("arm7:swing").xml()))
    assert out["flexes"] == [] and all(g["turns"] for g in out["geoms"])
