import json

import pytest

from autocam import config as config_mod
from autocam.config import ConfigError


def test_default_config_loads_and_validates():
    cfg = config_mod.load()
    assert cfg.schema_version == config_mod.CONFIG_SCHEMA_VERSION
    assert cfg.control.zeta == 1.0
    assert cfg.centroid.min_players == 4
    assert cfg.detect.detect_hz == 10.0


def test_hash_is_deterministic_and_value_sensitive():
    a = config_mod.load()
    b = config_mod.load()
    assert a.hash() == b.hash()
    c = config_mod.load(overrides={"control": {"omega_n": 0.9}})
    assert c.hash() != a.hash()


def test_detection_fingerprint_ignores_control_tuning():
    """Re-tuning the control law must never invalidate the detection cache."""
    base = config_mod.load()
    tuned = config_mod.load(overrides={"control": {"omega_n": 1.4, "deadzone": 0.05}})
    assert tuned.detection_fingerprint() == base.detection_fingerprint()

    rethreshold = config_mod.load(overrides={"detect": {"conf_min": 0.5}})
    assert rethreshold.detection_fingerprint() != base.detection_fingerprint()


def test_partial_override_keeps_other_defaults():
    cfg = config_mod.load(overrides={"control": {"deadzone": 0.01}})
    default = config_mod.load()
    assert cfg.control.deadzone == 0.01
    assert cfg.control.omega_n == default.control.omega_n


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_mod.load(overrides={"control": {"omega_nn": 1.0}})


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level"):
        config_mod.load(overrides={"controls": {}})


@pytest.mark.parametrize(
    "override",
    [
        {"control": {"zeta": 0.0}},
        {"control": {"deadzone": 0.9}},
        {"detect": {"conf_min": 1.5}},
        {"detect": {"detect_hz": 0.0}},
        {"detect": {"bbox_area_min": 0.5, "bbox_area_max": 0.1}},
        {"centroid": {"min_players": 0}},
        {"centroid": {"spread_metric": "variance"}},
        {"render": {"crop_y_mode": "sideways"}},
        {"render": {"crf": 99}},
    ],
)
def test_out_of_range_values_are_rejected(override):
    with pytest.raises(ConfigError):
        config_mod.load(overrides=override)


def test_schema_version_mismatch_is_rejected():
    raw = json.loads(config_mod.DEFAULT_CONFIG_PATH.read_text())
    raw["schema_version"] = 99
    with pytest.raises(ConfigError, match="schema_version"):
        config_mod.from_dict(raw)


def test_dump_roundtrip(tmp_path):
    cfg = config_mod.load(overrides={"control": {"omega_n": 1.1}})
    out = tmp_path / "tuned.json"
    config_mod.dump(cfg, out)
    reloaded = config_mod.from_dict(json.loads(out.read_text()))
    assert reloaded.hash() == cfg.hash()


def test_user_config_file_layers_over_defaults(tmp_path):
    path = tmp_path / "user.json"
    path.write_text(json.dumps({"control": {"omega_n": 1.25}}))
    cfg = config_mod.load(path)
    assert cfg.control.omega_n == 1.25
    assert cfg.control.max_pan_vel == config_mod.load().control.max_pan_vel


def test_phase2_constants_live_in_the_same_file():
    """Cross-cutting requirement: one source of truth for both phases."""
    cfg = config_mod.load()
    assert cfg.phase2.deadzone_scale == 2.0
    assert 0 < cfg.phase2.fov_h_deg < 180


def test_default_config_can_be_relocated(tmp_path, monkeypatch):
    """Phase 2 bundles the same file; installs off the source tree need a hook."""
    raw = json.loads(config_mod.DEFAULT_CONFIG_PATH.read_text())
    raw["control"]["omega_n"] = 2.0
    relocated = tmp_path / "elsewhere.json"
    relocated.write_text(json.dumps(raw))

    monkeypatch.setenv("AUTOCAM_DEFAULT_CONFIG", str(relocated))
    assert config_mod.default_config_path() == relocated
    assert config_mod.load().control.omega_n == 2.0


def test_missing_default_config_is_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCAM_DEFAULT_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setattr(config_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "_PACKAGE_DIR", tmp_path)
    with pytest.raises(ConfigError, match="AUTOCAM_DEFAULT_CONFIG"):
        config_mod.load()
