"""Build the double-clickable AirLLM Studio.app from shipped metadata."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

from airllm_studio.bundle import (
    APP_EXECUTABLE,
    entitlements_path,
    info_plist_path,
    macos_asset_dir,
    signing_config_path,
)

APP_BUNDLE_NAME = "AirLLM Studio.app"


def default_dist_app() -> Path:
    return Path(__file__).resolve().parents[1] / "dist" / APP_BUNDLE_NAME


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_app(dest: Path | None = None, *, release: bool = False) -> Path:
    """Create Contents/MacOS launcher + embed airllm_studio (no sibling .venv required).

    When release=True, the launcher exports AIRLLM_STUDIO_RELEASE=1 so Dev Unlock
    is disabled in customer / MAS builds.
    """
    dest = dest or default_dist_app()
    if dest.exists():
        shutil.rmtree(dest)
    contents = dest / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    app_py = resources / "pythonpath"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    shutil.copy2(info_plist_path(), contents / "Info.plist")
    shutil.copy2(entitlements_path(), resources / "AirLLMStudio.entitlements")
    shutil.copy2(signing_config_path(), resources / "signing.json")
    storekit = macos_asset_dir() / "Products.storekit"
    if storekit.is_file():
        shutil.copy2(storekit, resources / "Products.storekit")
    pkg = Path(__file__).resolve().parent
    shutil.copytree(pkg, app_py / "airllm_studio", dirs_exist_ok=True)

    py = Path(sys.prefix) / "bin" / "python3"
    if not py.is_file():
        py = Path(sys.executable).resolve()
    (resources / "runtime.txt").write_text(str(py) + "\n", encoding="utf-8")

    import site

    extra = []
    for p in list(site.getsitepackages() or []) + [site.getusersitepackages()]:
        if p and Path(p).is_dir():
            extra.append(str(Path(p).resolve()))
    (resources / "site-packages.txt").write_text("\n".join(extra) + "\n", encoding="utf-8")

    release_export = (
        "export AIRLLM_STUDIO_RELEASE=1\n" if release else ""
    )
    launcher = macos / APP_EXECUTABLE
    _write_executable(
        launcher,
        f"""#!/bin/bash
set -euo pipefail
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
RES="$CONTENTS/Resources"
PP="$RES/pythonpath"
if [[ -f "$RES/site-packages.txt" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" && -d "$line" ]] && PP="$PP:$line"
  done < "$RES/site-packages.txt"
fi
export PYTHONPATH="$PP${{PYTHONPATH:+:$PYTHONPATH}}"
export AIRLLM_STUDIO_BUNDLE=1
{release_export}PY=""
if [[ -x "$RES/python/bin/python3" ]]; then
  PY="$RES/python/bin/python3"
elif [[ -x "{py}" ]]; then
  PY="{py}"
else
  PY="$(command -v python3 || true)"
fi
if [[ -z "${{PY}}" ]]; then
  echo "AirLLM Studio: no Python 3 runtime found." >&2
  exit 1
fi
exec "$PY" -m airllm_studio.launch "$@"
""",
    )

    codesign = shutil.which("codesign")
    if codesign:
        subprocess.run(
            [
                codesign,
                "--force",
                "--deep",
                "--options",
                "runtime",
                "--entitlements",
                str(resources / "AirLLMStudio.entitlements"),
                "-s",
                "-",
                str(dest),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    return dest
