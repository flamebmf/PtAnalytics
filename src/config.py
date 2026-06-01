# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import yaml
from loguru import logger


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)


def load_settings(config_dir: Path) -> dict:
    return _read_yaml(config_dir / "settings.yaml")


def load_cameras(config_dir: Path) -> dict:
    return _read_yaml(config_dir / "cameras.yaml")


def load_triggers(config_dir: Path) -> dict:
    return _read_yaml(config_dir / "triggers.yaml")
