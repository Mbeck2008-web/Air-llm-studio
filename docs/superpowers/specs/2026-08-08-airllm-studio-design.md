# AirLLM Studio — Design Spec (v1)

**Date:** 2026-08-08  
**Status:** Approved  
**Stack:** CustomTkinter + Python + AirLLM (MLX path on Apple Silicon)

## Goal

Native-feeling macOS desktop app for running very large LLMs locally via AirLLM layer/expert streaming on Apple Silicon. Focus: reliable load/prepare, clear layer/MoE info, solid chat, basic tools.

## Architecture

```
UI (CustomTkinter, main thread)
  ↔ queue/events
Core services (worker threads)
  ModelManager | InferenceEngine | ChatStore | ToolRegistry | ModelMeta
```

- UI never blocks on download, prepare, or generate.
- Progress and tokens delivered via `queue.Queue`; UI drains with `after()`.
- App data: `~/Library/Application Support/AirLLMStudio/`

## Features (v1)

1. **Model library** — HF repo ID search/download, prepare/split with disk warnings, local library, load/unload.
2. **Model info panel** — params, layers, dense vs MoE, experts, estimated AirLLM memory, MoE routing diagram.
3. **Chat** — multi-chat, history, generation controls, progressive UI updates.
4. **Tools** — extensible registry; built-in optional web search; simple tool-call loop.
5. **AirLLM feedback** — preparing layers, streaming layer N/M, expert activity status.
6. **Demo mode** — full UI without AirLLM installed.

## Out of scope

Fine-tuning, remote/multi-GPU, notarized .app distribution, plugin marketplace.

## Success criteria

- Launches as desktop window (not browser).
- Download → prepare → load → chat path works on Apple Silicon when AirLLM deps present.
- MoE models show expert structure.
- Tools can be enabled/disabled; new tools are one registration.
