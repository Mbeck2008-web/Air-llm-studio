"""Mac App Store bundle identity and shipped packaging metadata.

These helpers read the same Info.plist / entitlements files copied into
the .app — tests must call these functions, not a parallel fixture copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BUNDLE_IDENTIFIER = "ai.airllm.studio"
APP_DISPLAY_NAME = "AirLLM Studio"
APP_EXECUTABLE = "AirLLMStudio"


def macos_asset_dir() -> Path:
    return Path(__file__).resolve().parent / "macos"


def info_plist_path() -> Path:
    return macos_asset_dir() / "Info.plist"


def entitlements_path() -> Path:
    return macos_asset_dir() / "AirLLMStudio.entitlements"


def signing_config_path() -> Path:
    return macos_asset_dir() / "signing.json"


def _load_plist(path: Path) -> Dict[str, Any]:
    import plistlib

    with path.open("rb") as fh:
        data = plistlib.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Plist at {path} is not a dict")
    return data


def load_info_plist() -> Dict[str, Any]:
    return _load_plist(info_plist_path())


def load_entitlements() -> Dict[str, Any]:
    return _load_plist(entitlements_path())


def load_signing_config() -> Dict[str, Any]:
    with signing_config_path().open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("signing.json must be an object")
    return data


def bundle_identity() -> Dict[str, str]:
    info = load_info_plist()
    return {
        "bundle_identifier": str(info.get("CFBundleIdentifier") or ""),
        "display_name": str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or ""),
        "version": str(info.get("CFBundleShortVersionString") or ""),
        "build": str(info.get("CFBundleVersion") or ""),
        "executable": str(info.get("CFBundleExecutable") or ""),
    }
