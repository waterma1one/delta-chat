"""Config loader + .env.example security tests (plan §12, this section's portion)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from delta_chat import config as config_mod
from delta_chat.config import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_loads_and_key_reachability():
    cfg = load_config()
    assert isinstance(cfg, Config)
    # nothing downstream should need a hardcoded constant — assert reachability
    assert isinstance(cfg.get("obs.capture_payloads"), bool)
    assert set(cfg.get("delta.weights")) == {"text", "spatial", "type", "region"}
    assert isinstance(cfg.get("delta.tau"), (int, float))
    assert cfg.get("delta.thresholds.unchanged_text_sim") == pytest.approx(0.97)


def test_missing_key_raises_clear_error_naming_the_path():
    cfg = load_config()
    with pytest.raises(ConfigError) as exc:
        cfg.get("delta.does_not_exist")
    assert "delta.does_not_exist" in str(exc.value)
    # a default suppresses the error
    assert cfg.get("delta.does_not_exist", 123) == 123


def test_env_override_takes_effect(monkeypatch):
    monkeypatch.setenv("DELTA_CHAT__DELTA__TAU", "0.42")
    monkeypatch.setenv("DELTA_CHAT__OBS__OTLP_ENABLED", "true")
    cfg = load_config()
    assert cfg.get("delta.tau") == pytest.approx(0.42)  # YAML-parsed to float
    assert cfg.get("obs.otlp_enabled") is True           # YAML-parsed to bool


def test_dotenv_overrides_reach_config(tmp_path, monkeypatch):
    """Regression: a DELTA_CHAT__ override living only in the repo-root .env (not the shell env)
    must reach the config. It previously did not — litellm loaded .env lazily, on first import,
    *after* load_config() had already frozen provider/model, so the .env layer silently no-op'd
    and provider auto-detection then picked the wrong provider. load_config now loads .env itself
    before reading env overrides.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("DELTA_CHAT__LLM__PROVIDER=fromdotenv\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)  # allow a fresh load in-process
    monkeypatch.delenv("DELTA_CHAT__LLM__PROVIDER", raising=False)  # only the .env provides it

    cfg = load_config()

    assert cfg.get("llm.provider") == "fromdotenv"


def test_shell_env_wins_over_dotenv(tmp_path, monkeypatch):
    """override=False semantics: an exported shell var beats the .env file value for the same key."""
    (tmp_path / ".env").write_text("DELTA_CHAT__LLM__PROVIDER=fromdotenv\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)
    monkeypatch.setenv("DELTA_CHAT__LLM__PROVIDER", "fromshell")

    cfg = load_config()

    assert cfg.get("llm.provider") == "fromshell"


def test_env_files_contains_no_real_secret():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    # sk-... / sk-ant-... real-key shapes and long high-entropy tokens must be absent.
    real_key = re.compile(r"sk-(?:ant-)?[A-Za-z0-9]{16,}")
    aws = re.compile(r"AKIA[0-9A-Z]{16}")
    assert not real_key.search(text), "possible real API key in .env.example"
    assert not aws.search(text), "possible AWS key in .env.example"
    # every value should be a placeholder or a localhost URL
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        _, val = line.split("=", 1)
        assert ("REPLACE" in val) or val.startswith("http://localhost"), line
