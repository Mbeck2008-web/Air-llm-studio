#!/usr/bin/env bash
# Create venv and install UI deps (and optionally AirLLM stack)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer Homebrew Python 3.12 (ships usable Tcl/Tk via python-tk@3.12).
# Apple /usr/bin/python3 uses Tk 8.5.9 which leaves CustomTkinter blank white.
pick_python() {
  local candidates=(
    "/opt/homebrew/bin/python3.12"
    "/usr/local/bin/python3.12"
    "/opt/homebrew/bin/python3.11"
    "/usr/local/bin/python3.11"
    "/opt/homebrew/bin/python3"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -x "$p" ]]; then
      if "$p" -c "import tkinter as tk; r=tk.Tk(); v=r.tk.call('info','patchlevel'); r.destroy(); assert not str(v).startswith('8.5')" 2>/dev/null; then
        echo "$p"
        return 0
      fi
    fi
  done
  # Fallback: whatever python3 is, may still be broken on macOS
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "No python3 found" >&2
  return 1
}

PY="$(pick_python)"
echo "→ Using Python: $PY"
"$PY" -c "import tkinter as tk; r=tk.Tk(); print('  Tk', r.tk.call('info','patchlevel')); r.destroy()" 2>/dev/null || {
  echo ""
  echo "ERROR: This Python cannot open a modern Tk window."
  echo "On Apple Silicon install:"
  echo "  brew install python@3.12 python-tk@3.12"
  echo "Then re-run: ./scripts/setup.sh"
  exit 1
}

echo "→ Creating virtualenv at .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
echo "→ Installing UI + AirLLM / MLX / Torch stack (this may take a few minutes)…"
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "  Launch:  ./scripts/run.sh"
echo "  Verify:  .venv/bin/python -c 'from airllm import AutoModel; import mlx, torch; print(\"ok\")'"
