from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        # default to config.yaml if exists
        default = "config.yaml"
        if os.path.exists(default):
            path = default
        else:
            path = "config.example.yaml"

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_slack_webhook(cfg: Dict[str, Any]) -> str | None:
    node = cfg.get("notifier", {}).get("slack", {})
    if not node or not node.get("enabled", False):
        return None
    env_key = node.get("webhook_env", "SLACK_WEBHOOK_URL")
    return os.environ.get(env_key)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
