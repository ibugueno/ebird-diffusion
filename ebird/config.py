from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {config_path}")
    return _expand(config)


def require_sections(config: dict[str, Any], *sections: str) -> None:
    missing = [section for section in sections if section not in config]
    if missing:
        raise KeyError(f"Missing configuration sections: {', '.join(missing)}")
