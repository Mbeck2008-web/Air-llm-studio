# Packaging AirLLM Studio (macOS only)

Build a double-clickable `AirLLM Studio.app`, then sign for local use,
Developer ID distribution, or Mac App Store upload.

## Prerequisites

- Apple Silicon Mac
- Python 3.9+ (3.12 recommended) with project deps installed
- Xcode Command Line Tools (`codesign`, `productbuild` as needed)

```bash
cd /path/to/Air-llm-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 scripts/build_macos_app.py
open "dist/AirLLM Studio.app"
```

Data and IAP entitlement files live under:

`~/Library/Application Support/AirLLMStudio/`

(including `iap/transactions.json` for the local test store).

## 1. Ad-hoc (local / CI)

The build script already runs:

```bash
codesign --force --deep --options runtime \
  --entitlements airllm_studio/macos/AirLLMStudio.entitlements \
  -s - \
  "dist/AirLLM Studio.app"
```

Use this for smoke tests. Gatekeeper may still block first launch;
right-click → Open, or clear quarantine:

```bash
xattr -dr com.apple.quarantine "dist/AirLLM Studio.app"
```

Ad-hoc builds do **not** set `AIRLLM_STUDIO_RELEASE=1`, so the in-app
**Dev Unlock Pro** control remains available for QA.

## 2. Developer ID (outside the App Store)

1. Create a *Developer ID Application* certificate in the Apple Developer portal.
2. Re-sign the app (and nested Python/dylibs) with that identity, keeping
   Hardened Runtime and the shipped entitlements:

```bash
codesign --force --deep --options runtime \
  --entitlements airllm_studio/macos/AirLLMStudio.entitlements \
  -s "Developer ID Application: YOUR NAME (TEAMID)" \
  "dist/AirLLM Studio.app"
```

3. Notarize and staple (`notarytool` / `stapler`) before distributing DMGs.

Assumptions: you embed or ship a relocatable CPython under
`Contents/Resources/python/` for machines without your build interpreter;
until then the launcher uses the Python recorded at build time.

## 3. Mac App Store (MAS)

1. Create *Mac App Distribution* certificate + provisioning profile for
   `ai.airllm.studio`.
2. Build, then re-sign for App Store distribution with the same entitlements.
3. Export `AIRLLM_STUDIO_RELEASE=1` in the launcher environment (or rebuild
   with `python3 scripts/build_macos_app.py --release`) so Dev Unlock is
   disabled in customer builds.
4. Create IAP products matching `airllm_studio/billing/catalog.py` and the
   StoreKit configuration at `airllm_studio/macos/Products.storekit`.
5. Upload with Transporter / App Store Connect. See [docs/APP_STORE.md](docs/APP_STORE.md).

## Bundle identity

| Key | Value |
|-----|--------|
| Bundle ID | `ai.airllm.studio` |
| Executable | `AirLLMStudio` |
| Display name | AirLLM Studio |
| Entitlements | `airllm_studio/macos/AirLLMStudio.entitlements` |

## Free / Pro gate (no browser required)

The Pro tab inside the native WebKit window is the paywall. Free tier keeps
local models/chat (512 token cap). Pro unlocks web search/tools, 4096 tokens,
and Hugging Face library search. Purchases use StoreKit when available;
otherwise the documented file-backed test store under Application Support.
