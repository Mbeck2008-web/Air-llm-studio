#!/bin/bash
# Double-clickable launcher on macOS (opens Terminal then the app)
cd "$(dirname "$0")/.." || exit 1
export TK_SILENCE_DEPRECATION=1

# MoE on MPS frequently SIGABRTs (MPSGraph weak-ref) during layer thrashing.
# Default compute device for MoE is CPU; override with AIRLLM_MOE_DEVICE=mps if desired.
export AIRLLM_MOE_DEVICE="${AIRLLM_MOE_DEVICE:-cpu}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
# Reduce background Metal work during Python exit / layer free
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# Guard against Apple system Tk 8.5 blank window
if ! "$PY" -c "import tkinter as tk; r=tk.Tk(); v=str(r.tk.call('info','patchlevel')); r.destroy(); raise SystemExit(v.startswith('8.5'))" 2>/dev/null; then
  echo "ERROR: Python is using an old/broken Tk (blank white window)."
  echo "Fix:  brew install python@3.12 python-tk@3.12"
  echo "Then: ./scripts/setup.sh && open this file again"
  echo ""
  read -r -p "Press Enter to close…"
  exit 1
fi

exec "$PY" -m airllm_studio
