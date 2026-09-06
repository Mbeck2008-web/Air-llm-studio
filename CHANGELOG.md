# Changelog

## Unreleased — packaging, paywall, IP

### Packaged

- Double-clickable `AirLLM Studio.app` via `scripts/build_macos_app.py`
  (`ai.airllm.studio`, sandbox entitlements, Hardened Runtime ad-hoc sign).
- Application Support data root: `~/Library/Application Support/AirLLMStudio/`
  (IAP receipts under `iap/`).
- StoreKit local config: `airllm_studio/macos/Products.storekit` (monthly /
  yearly / lifetime SKUs at $7.99 / $49.99 / $59.99).
- `PACKAGING.md` for ad-hoc, Developer ID, and Mac App Store signing paths.

### Gated (Free / Pro)

- Free: add/prepare/load local models, chat (demo/mock allowed), max 512 tokens;
  no web search, no tools, no Hugging Face library search.
- Pro: web search + tools, up to 4096 tokens, HF library search.
- Paywall lives in the native window (Pro tab): Buy, Restore Purchases,
  Manage subscription copy, Privacy/Terms opened in-app (no external browser).
- Entitlement store persists under Application Support; license keys refused.
- Dev Unlock Pro available only when `AIRLLM_STUDIO_RELEASE` is not set
  (disabled for Release / MAS customer builds).

### IP

- `LICENSE` (Apache-2.0 for app source), `NOTICE` (third-party not relicensed).
- About / copyright: Copyright © 2026 Michael Beck. All rights reserved.
- Terms + Privacy remain at `airllm_studio/web/legal/` and are linked from Pro.

### Leftover Apple steps (manual)

- Paid Apple Developer Program team, Mac App Distribution cert + profile.
- Re-sign nested Python/dylibs; App Store Connect product page, privacy
  nutrition label, IAP products, screenshots, review notes.
- Sandbox QA of a fully signed MAS build; host public HTTPS Privacy/Terms URLs
  on the Connect listing.
- Optional: embed relocatable CPython under `Contents/Resources/python/`.
