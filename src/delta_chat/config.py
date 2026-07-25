"""Config loader. One source of truth (``config/default.yaml``) with env-var overrides.

Env override convention: ``DELTA_CHAT__<SECTION>__<KEY>=value`` (double underscore nests). The value
is YAML-parsed, so ``DELTA_CHAT__DELTA__TAU=0.4`` sets a float and ``...__OTLP_ENABLED=true`` a bool.
Missing keys raise a clear, traceable error naming the requested path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "DELTA_CHAT__"
_MISSING = object()
_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load the repo-root ``.env`` into ``os.environ`` exactly once, before config reads overrides.

    litellm also loads ``.env`` — but only lazily, on its first import, which on the chat path
    happens *after* ``load_config()`` has already frozen provider/model (grounding imports ``llm``
    lazily inside its functions). That timing made two things silently wrong: ``DELTA_CHAT__*``
    overrides in ``.env`` never reached the config, and ``ANTHROPIC_API_KEY`` leaked into the env
    late — after cfg was built with ``provider: auto`` — so provider auto-detection picked anthropic
    over the intended provider. Loading ``.env`` here, deterministically and before
    ``_apply_env_overrides``, closes both gaps regardless of import order.

    ``override=False`` keeps real exported shell vars authoritative over the ``.env`` file, per the
    usual convention (and so test-set env vars are never clobbered).
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - python-dotenv is a declared dep; guard is defensive only
        return
    env_path = _repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:  # fall back to find_dotenv's cwd-upward walk when the repo layout isn't preserved
        load_dotenv(override=False)


class ConfigError(KeyError):
    """Raised when a requested config key does not exist."""


def _repo_root() -> Path:
    # src/delta_chat/config.py -> repo root is two parents up from the package dir.
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return _repo_root() / "config" / "default.yaml"


class Config:
    """Dotted-path accessor over the loaded config mapping."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        cur: Any = self._data
        parts = dotted.split(".")
        for i, part in enumerate(parts):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                if default is not _MISSING:
                    return default
                reached = ".".join(parts[: i + 1])
                raise ConfigError(
                    f"Missing config key '{dotted}' (no value at '{reached}')"
                )
        return cur

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    for env_key, raw_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        path = [p.lower() for p in env_key[len(_ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        try:
            value = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            value = raw_val
        node = data
        for part in path[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[path[-1]] = value
    return data


def _resolve_config_path() -> Path:
    """Locate config/default.yaml robustly: explicit env var, then repo layout, then cwd.

    This survives a wheel/Docker install where the source checkout layout isn't preserved.
    """
    env = os.environ.get("DELTA_CHAT_CONFIG")
    if env:
        return Path(env)
    repo_default = default_config_path()
    if repo_default.exists():
        return repo_default
    return Path.cwd() / "config" / "default.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``config/default.yaml`` (or ``path``), apply env overrides, return a ``Config``."""
    _load_dotenv_once()
    cfg_path = Path(path) if path is not None else _resolve_config_path()
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return Config(_apply_env_overrides(data))
