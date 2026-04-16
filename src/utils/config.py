"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_optional_path(config_dir: Path, value: Any) -> Any:
    """Resolve relative path-like config values against the config file directory."""

    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value

    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((config_dir / path).resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain Python dictionary."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a dictionary-like YAML config in {path}")

    config_dir = path.parent
    project_cfg = config.get("project")
    if isinstance(project_cfg, dict):
        project_cfg["output_dir"] = _resolve_optional_path(
            config_dir,
            project_cfg.get("output_dir"),
        )

    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        data_cfg["root_dir"] = _resolve_optional_path(config_dir, data_cfg.get("root_dir"))
        data_cfg["split_manifest_path"] = _resolve_optional_path(
            config_dir,
            data_cfg.get("split_manifest_path"),
        )

    return config
