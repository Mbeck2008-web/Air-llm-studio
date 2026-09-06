"""Application paths and settings."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _default_data_dir() -> Path:
    from airllm_studio.runtime_env import apply_sandbox_environment

    return apply_sandbox_environment()


@dataclass
class AppConfig:
    data_dir: Path = field(default_factory=_default_data_dir)
    hf_token: str = ""
    default_temperature: float = 0.7
    default_max_tokens: int = 512
    default_top_p: float = 0.9
    tools_enabled: bool = False
    web_search_enabled: bool = False
    compression: Optional[str] = None  # None | "4bit" | "8bit"
    delete_original_after_split: bool = False
    demo_mode: bool = False  # force mock engine even if airllm is installed
    # Master switch (chat header toggle): use AirLLM layer streaming when loading
    airllm_enabled: bool = True
    system_prompt: str = ""

    @property
    def models_path(self) -> Path:
        return self.data_dir / "models.json"

    @property
    def chats_dir(self) -> Path:
        return self.data_dir / "chats"

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def shards_dir(self) -> Path:
        return self.data_dir / "layer_shards"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        if "data_dir" in data and data["data_dir"]:
            cfg.data_dir = Path(data["data_dir"])
        for key in (
            "hf_token",
            "default_temperature",
            "default_max_tokens",
            "default_top_p",
            "tools_enabled",
            "web_search_enabled",
            "compression",
            "delete_original_after_split",
            "demo_mode",
            "airllm_enabled",
            "system_prompt",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg

    def save(self) -> None:
        self.ensure_dirs()
        payload = self.to_dict()
        # Don't write absolute secrets unnecessarily; still persist token if set
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls()
        if cfg.settings_path.exists():
            try:
                with open(cfg.settings_path, encoding="utf-8") as f:
                    data = json.load(f)
                cfg = cls.from_dict(data)
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        cfg.ensure_dirs()
        return cfg


_config: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    global _config
    if _config is None or reload:
        _config = AppConfig.load()
    return _config


def update_config(**kwargs: Any) -> AppConfig:
    cfg = get_config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.save()
    return cfg
