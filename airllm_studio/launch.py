"""Packaged desktop entry used by the .app binary and `python -m airllm_studio.launch`.

`--help` / `--smoke` must work without a display, pywebview, or a sibling .venv.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from airllm_studio import __app_name__, __version__
from airllm_studio.bundle import APP_DISPLAY_NAME, BUNDLE_IDENTIFIER, bundle_identity
from airllm_studio.runtime_env import apply_sandbox_environment, assert_data_dir_writable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="AirLLMStudio",
        description=f"{APP_DISPLAY_NAME} — local large-model desktop app for Apple Silicon.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Import the packaged host, apply sandbox paths, print identity, exit.",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Start the native desktop window (default when no flags are given).",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Print version and bundle identifier.",
    )
    return p


def smoke_report() -> str:
    apply_sandbox_environment()
    ident = bundle_identity()
    lines = [
        f"{APP_DISPLAY_NAME} smoke OK",
        f"name={__app_name__}",
        f"version={__version__}",
        f"bundle_id={ident['bundle_identifier'] or BUNDLE_IDENTIFIER}",
        f"display_name={ident['display_name']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        ident = bundle_identity()
        print(f"{APP_DISPLAY_NAME} {__version__} ({ident['bundle_identifier']})")
        return 0

    if args.smoke:
        assert_data_dir_writable()
        print(smoke_report())
        return 0

    # Default: GUI (also --gui). Apply sandbox env before heavy imports.
    apply_sandbox_environment()
    assert_data_dir_writable()
    from airllm_studio.__main__ import main as app_main

    app_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
