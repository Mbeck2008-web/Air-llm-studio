#!/usr/bin/env python3
"""CLI wrapper: python scripts/build_macos_app.py [--release] [dest.app]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airllm_studio.packaging import APP_BUNDLE_NAME, build_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AirLLM Studio.app")
    parser.add_argument(
        "dest",
        nargs="?",
        type=Path,
        default=ROOT / "dist" / APP_BUNDLE_NAME,
        help="Destination .app path",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Export AIRLLM_STUDIO_RELEASE=1 (disables Dev Unlock).",
    )
    args = parser.parse_args()
    print(build_app(args.dest, release=bool(args.release)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
