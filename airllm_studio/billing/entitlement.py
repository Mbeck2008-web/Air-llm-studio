"""Pure entitlement policy. No StoreKit I/O lives here."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from airllm_studio.billing.catalog import (
    FREE_MAX_TOKENS,
    PAID_MAX_TOKENS,
    PRODUCTS,
    get_product,
)


FREE_CAPABILITIES = frozenset(
    {
        "local_chat",
        "add_model",
        "prepare_model",
        "load_model",
    }
)

PAID_CAPABILITIES = frozenset(
    {
        "web_search",
        "tools",
        "high_token_limit",
        "hf_search",
    }
)


class LicenseKeyRefused(Exception):
    """Digital unlocks must go through In-App Purchase (App Store 3.1.1)."""


def refuse_license_key(code: str) -> None:
    """Always refuse. There is no license-key unlock path."""
    raise LicenseKeyRefused(
        "License keys are not accepted. Unlock Pro with an In-App Purchase."
    )


def entitled_from_receipts(
    receipts: Sequence[Any],
    products: Iterable[Any] = PRODUCTS,
) -> bool:
    """True when any receipt matches a shipped paid SKU."""
    sku_ids = {getattr(p, "product_id", None) for p in products}
    sku_ids.discard(None)
    for receipt in receipts:
        pid = receipt.get("product_id") if isinstance(receipt, dict) else getattr(
            receipt, "product_id", None
        )
        if pid in sku_ids and get_product(str(pid)) is not None:
            return True
    return False


def allows(capability: str, entitled: bool) -> bool:
    """Gate a named capability. Unknown names stay free so core chat cannot lock."""
    if capability in FREE_CAPABILITIES:
        return True
    if capability in PAID_CAPABILITIES:
        return bool(entitled)
    return True


def clamp_generation(
    max_tokens: int,
    use_search: bool,
    entitled: bool,
) -> Tuple[int, bool, List[str]]:
    notes: List[str] = []
    tokens = int(max_tokens)
    search = bool(use_search)
    if allows("high_token_limit", entitled):
        tokens = min(max(tokens, 1), PAID_MAX_TOKENS)
    else:
        if tokens > FREE_MAX_TOKENS:
            notes.append(f"Token limit is {FREE_MAX_TOKENS} on the free tier.")
        tokens = min(max(tokens, 1), FREE_MAX_TOKENS)
    if search and not allows("web_search", entitled):
        search = False
        notes.append("Web search is a Pro feature. Chat continues without it.")
    return tokens, search, notes


def sanitize_settings(data: Dict[str, Any], entitled: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Drop paid settings the user is not entitled to enable."""
    patch = dict(data)
    notes: List[str] = []
    if not allows("tools", entitled) or not allows("web_search", entitled):
        wanted_tools = bool(patch.get("tools_enabled"))
        wanted_search = bool(patch.get("web_search_enabled"))
        if wanted_tools or wanted_search:
            patch["tools_enabled"] = False
            patch["web_search_enabled"] = False
            notes.append("Tools and web search require Pro.")
    if "default_max_tokens" in patch and not allows("high_token_limit", entitled):
        try:
            asked = int(patch["default_max_tokens"])
        except (TypeError, ValueError):
            asked = FREE_MAX_TOKENS
        if asked > FREE_MAX_TOKENS:
            notes.append(f"Max tokens stay at {FREE_MAX_TOKENS} on the free tier.")
        patch["default_max_tokens"] = min(asked, FREE_MAX_TOKENS)
    return patch, notes
