# AirLLM Studio

Local large-model **desktop app for Apple Silicon**. Chat-first. AirLLM streams layers (and MoE experts) through unified memory so models that would not fit can still run.

Not a browser tab. A native window (WebKit via pywebview) talking to the same Python engine that prepares Hugging Face weights and runs inference.

## What you do

1. Open the app  
2. **Library** → prepare a model (download + split shards)  
3. **Load** it  
4. Talk  

Search is **off** unless you tick it on a single message. Greetings never hit the web.

## Quick start (macOS)

```bash
brew install python@3.12 python-tk@3.12
cd ~/Projects/airllm-studio
./scripts/setup.sh
./scripts/run.sh
```

Or double-click `scripts/AirLLM Studio.command`.

**Mac App Store–ready `.app`** (unsigned / ad-hoc signed):

```bash
python3 scripts/build_macos_app.py
open "dist/AirLLM Studio.app"
```

See [PACKAGING.md](PACKAGING.md) for ad-hoc / Developer ID / Mac App Store signing, and [design/APP_STORE.md](design/APP_STORE.md) for Connect / review steps.

Classic CustomTkinter UI (if you need it):

```bash
AIRLLM_STUDIO_UI=tk ./scripts/run.sh
```

MoE defaults to **CPU** (stable). To try Metal (can crash on beta macOS):

```bash
AIRLLM_MOE_DEVICE=mps ./scripts/run.sh
```

## Layout

```
airllm_studio/
  core/        engine, MoE routing, downloads, session
  tools/       optional web_search
  web/         chat-first UI
  desktop.py   native WebKit host
  ui/          legacy CustomTkinter (AIRLLM_STUDIO_UI=tk)
```

Data: `~/Library/Application Support/AirLLMStudio/`

## Notes

- Dense models use AirLLM’s MLX path. MoE models use per-expert streaming (safetensors).
- First prepare is slow and disk-heavy. Chat after that streams only the active layer / routed experts.
- Set a Hugging Face token in Settings for gated models.

## Free / Pro

Local chat stays free. Pro (web search/tools, higher token limit, HF library search)
unlocks via App Store IAP — monthly `$7.99`, yearly `$49.99`, lifetime `$59.99`.
Use **Restore Purchases** on the Pro tab after reinstall. Release builds
(`python3 scripts/build_macos_app.py --release`) hide Dev Unlock; StoreKit Testing
+ TestStore still exercise Free/Pro.

## Privacy / Terms

- In-app: Pro tab links (bundled `airllm_studio/web/legal/`).
- Public HTTPS (GitHub Pages): [Privacy](https://mbeck2008-web.github.io/Air-llm-studio/legal/privacy.html) · [Terms](https://mbeck2008-web.github.io/Air-llm-studio/legal/terms.html)

## License

App source: Apache-2.0 (`LICENSE`). Copyright © 2026 Michael Beck. All rights reserved.
Third-party notes: `NOTICE`. AirLLM and model weights stay under their own licenses.
