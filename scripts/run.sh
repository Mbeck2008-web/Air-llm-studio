#!/usr/bin/env bash
# Launch AirLLM Studio as a desktop app
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TK_SILENCE_DEPRECATION=1
# WebKit UI is default. Classic: AIRLLM_STUDIO_UI=tk
export AIRLLM_STUDIO_UI="${AIRLLM_STUDIO_UI:-web}"
# MoE defaults to CPU — MPS layer thrashing can SIGABRT (MPSGraph). Override: AIRLLM_MOE_DEVICE=mps
export AIRLLM_MOE_DEVICE="${AIRLLM_MOE_DEVICE:-cpu}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

# Prefer the venv interpreter explicitly (avoid PATH shadowing)
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fail fast if Tk is the broken Apple 8.5 system build
"$PY" - <<'PY' || {
  echo ""
  echo "Tk looks broken or too old (blank white window risk)."
  echo "Run: ./scripts/setup.sh   (requires: brew install python@3.12 python-tk@3.12)"
  exit 1
}
import tkinter as tk
r = tk.Tk()
v = str(r.tk.call("info", "patchlevel"))
r.destroy()
print(f"Tk {v}")
if v.startswith("8.5"):
    raise SystemExit(1)
PY

exec "$PY" -m airllm_studio
