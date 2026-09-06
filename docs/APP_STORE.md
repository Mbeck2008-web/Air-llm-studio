# Shipping AirLLM Studio on the Mac App Store

This repo ships **App Store–ready packaging**: sandbox + Hardened Runtime entitlements, bundle identity, and a double-clickable `.app`. Uploading and review still need an Apple Developer Program membership.

## What is already in the tree

| Asset | Location |
|--------|----------|
| Bundle ID / version / display name | `airllm_studio/macos/Info.plist` (`ai.airllm.studio`) |
| App Sandbox + network + user files | `airllm_studio/macos/AirLLMStudio.entitlements` |
| Hardened Runtime (`codesign --options runtime`) | `airllm_studio/macos/signing.json` + `airllm_studio/packaging.py` |
| Packaged entry | `python -m airllm_studio.launch` (`--help`, `--smoke`, default GUI) |
| App builder | `python scripts/build_macos_app.py` → `dist/AirLLM Studio.app` |

User data and Hugging Face caches go under Application Support (`~/Library/Application Support/AirLLMStudio/`, remapped into the sandbox container when the app is signed with App Sandbox). Model **weights are not** inside the `.app`.

## Build the `.app` (unsigned / ad-hoc)

```bash
cd /path/to/airllm-studio
python3 scripts/build_macos_app.py
open "dist/AirLLM Studio.app"
```

The launcher embeds the `airllm_studio` package and prefers `Contents/Resources/python/bin/python3` if you later copy a relocatable CPython there. Until then it uses the interpreter that ran the build script (or `python3` on `PATH`).

Ad-hoc `codesign -s - --options runtime` is attempted automatically so entitlements can be inspected:

```bash
codesign -dv --entitlements :- "dist/AirLLM Studio.app"
```

## Remaining steps (Apple Developer identity required)

These **cannot** be finished without a paid Apple Developer Program team:

1. **Certificates** — create a *Mac App Distribution* certificate and a *Mac App Store* provisioning profile for `ai.airllm.studio` in the developer portal.
2. **Re-sign** the built app (and any nested Python/dylibs) with that certificate, keeping `--options runtime` and the shipped entitlements file.
3. **Product page** — App Store Connect: screenshots, privacy nutrition label, category, age rating, support URL.
4. **Review notes** — explain local-only inference, Hugging Face downloads, optional web search, JIT for PyTorch/MLX (`allow-jit` / `allow-unsigned-executable-memory`). Apple sometimes rejects interpreter + JIT stacks; be prepared to discuss or drop Metal/JIT.
5. **Upload** — Transporter or `xcrun altool` / `notarytool` is *not* a substitute; use App Store Connect / Transporter for **Mac App Store** (not Developer ID notarization).
6. **Sandbox QA** — launch the *signed* sandbox build and confirm downloads land in the container, not `~/.cache/huggingface`.

## In-App Purchase products

The shipped catalog lives in `airllm_studio/billing/catalog.py`. Create the same product IDs in App Store Connect:

| Product ID | Type | USD list price | Period |
|------------|------|----------------|--------|
| `ai.airllm.studio.pro.monthly` | Auto-renewable subscription | $7.99 | Month |
| `ai.airllm.studio.pro.yearly` | Auto-renewable subscription | $49.99 | Year |
| `ai.airllm.studio.pro.lifetime` | Non-consumable | $59.99 | Lifetime |

Free tier (no purchase required): add/prepare/load a local model and send a chat, including the demo/mock engine when ML dependencies are absent. Generation defaults stay at 512 tokens.

Pro unlocks: web search / tool calling, higher token limit (4096), Hugging Face library search.

Unlocks go only through StoreKit (or the documented local test store when StoreKit cannot run). License keys are refused. Restore Purchases re-grants entitlement from prior transactions.

Subscriptions auto-renew until cancelled in System Settings → Apple ID → Subscriptions. The in-app paywall links to bundled Privacy Policy and Terms (`airllm_studio/web/legal/`). Host the same text at a public HTTPS URL on the App Store Connect product page.

## Privacy / review talking points

- No analytics by default.
- Optional Hugging Face token stored locally.
- Optional DuckDuckGo search only when the user enables tools.
- Large files: user-initiated model downloads; user-selected file entitlement for imports.

## Known App Store risk

Embedding CPython + PyTorch/MLX requires JIT / unsigned executable memory and typically `disable-library-validation`. Reviewers may still refuse that combination. The entitlements here match what the app actually does; they do not guarantee approval.
