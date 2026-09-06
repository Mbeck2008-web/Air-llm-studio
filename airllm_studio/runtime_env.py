"""Sandbox-legal data/cache locations for a packaged / App Store build."""

from __future__ import annotations

import os
from pathlib import Path


def is_sandboxed() -> bool:
    return bool(os.environ.get("APP_SANDBOX_CONTAINER_ID"))


def application_support_root() -> Path:
    home = Path.home()
    if os.uname().sysname == "Darwin":
        return home / "Library" / "Application Support" / "AirLLMStudio"
    return home / ".airllm_studio"


def apply_sandbox_environment(data_dir: Path | None = None) -> Path:
    """
    Point Hugging Face / transformers caches at Application Support.

    Under App Sandbox, ~/Library/Application Support is remapped into the
    container. ~/.cache is *not* writable without extra entitlements.
    """
    root = Path(data_dir) if data_dir is not None else application_support_root()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    hf = cache / "huggingface"
    hf.mkdir(parents=True, exist_ok=True)
    (root / "chats").mkdir(parents=True, exist_ok=True)
    (root / "layer_shards").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf / "hub"))
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    return root


def assert_data_dir_writable(data_dir: Path | None = None) -> Path:
    """Fail clearly when the sandbox/container cannot write user data."""
    root = apply_sandbox_environment(data_dir)
    probe = root / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise PermissionError(
            "AirLLM Studio cannot write its Application Support folder. "
            "The App Sandbox entitlement set is missing user-data access, "
            f"or the container is not writable ({root}): {exc}"
        ) from exc
    return root
