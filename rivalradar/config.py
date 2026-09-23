"""Configuration loading.

Non-secret settings come from ``config.yaml``. Secrets come from environment
variables only, never from config. They can be kept in a ``.env`` file in the
repository folder (gitignored), which is loaded wherever the command is run
from; variables already set in the real environment take precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The repository folder, next to the ``rivalradar`` package.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

try:  # optional convenience; real env still wins
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:  # pragma: no cover - dotenv is optional
    pass


DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


@dataclass
class Secrets:
    """Secrets pulled from the environment. Values may be ``None`` if unset."""

    groq_api_key: str | None = None
    epo_ops_key: str | None = None
    epo_ops_secret: str | None = None
    # Optional: OpenAlex rate-limits anonymous search under load; a free key
    # avoids that.
    openalex_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            epo_ops_key=os.environ.get("EPO_OPS_KEY"),
            epo_ops_secret=os.environ.get("EPO_OPS_SECRET"),
            openalex_api_key=os.environ.get("OPENALEX_API_KEY"),
        )

    def require(self, name: str) -> str:
        """Return a secret by attribute name or raise if it is unset."""
        value = getattr(self, name, None)
        if not value:
            env_name = name.upper()
            raise ConfigError(
                f"Missing required secret: set the {env_name} environment variable."
            )
        return value


@dataclass
class Config:
    """Parsed, non-secret configuration plus environment secrets."""

    raw: dict[str, Any]
    path: Path
    secrets: Secrets = field(default_factory=Secrets.from_env)

    # --- convenience accessors -------------------------------------------

    @property
    def target(self) -> dict[str, Any]:
        """The target startup. Not read from the file: the CLI sets it each run."""
        return self.raw.get("target", {}) or {}

    @property
    def contact_email(self) -> str:
        return str(self.raw.get("contact_email") or "").strip()

    @property
    def seed_rivals(self) -> list[dict[str, Any]]:
        return list(self.raw.get("seed_rivals", []) or [])

    @property
    def max_patents(self) -> int:
        return int(self.raw.get("small_startup_filter", {}).get("max_patents", 50))

    @property
    def lookback_days(self) -> int:
        return int(self.raw.get("lookback_days", 7))

    @property
    def sources(self) -> dict[str, bool]:
        return dict(self.raw.get("sources", {}) or {})

    @property
    def llm(self) -> dict[str, Any]:
        return dict(self.raw.get("llm", {}) or {})

    @property
    def arxiv(self) -> dict[str, Any]:
        return dict(self.raw.get("arxiv", {}) or {})

    @property
    def max_join_age_years(self) -> int:
        return int(self.raw.get("hiring", {}).get("max_join_age_years", 1))

    @property
    def discovery(self) -> dict[str, Any]:
        return dict(self.raw.get("discovery", {}) or {})

    @property
    def output(self) -> dict[str, Any]:
        return dict(self.raw.get("output", {}) or {})

    @property
    def db_path(self) -> Path:
        return Path(self.raw.get("storage", {}).get("db_path", "rivalradar.sqlite"))

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.raw.get("storage", {}).get("snapshot_dir", "snapshots"))

    def source_enabled(self, name: str) -> bool:
        return bool(self.sources.get(name, False))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from a YAML file and the environment."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}.")

    return Config(raw=raw, path=config_path)
