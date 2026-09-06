#!/usr/bin/env python3
"""CLI wrapper: python scripts/build_macos_app.py [dest.app]"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airllm_studio.packaging import APP_BUNDLE_NAME, build_app  # noqa: E402


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / APP_BUNDLE_NAME
    print(build_app(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
