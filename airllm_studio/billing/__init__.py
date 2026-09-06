"""In-App Purchase catalog, entitlement policy, and StoreKit-facing store."""

from __future__ import annotations

from typing import Any, Dict, List

from airllm_studio.billing.catalog import (
    FREE_MAX_TOKENS,
    FREE_TIER_KEEPS,
    PAID_MAX_TOKENS,
    PAID_UNLOCKS,
    PRODUCTS,
    catalog_dict,
    get_product,
    legal_page_path,
    legal_page_url,
)
from airllm_studio.billing.entitlement import (
    FREE_CAPABILITIES,
    PAID_CAPABILITIES,
    allows,
    clamp_generation,
    entitled_from_receipts,
    refuse_license_key,
    sanitize_settings,
)
from airllm_studio.billing.store import (
    LicenseKeyRefused,
    PurchaseError,
    Receipt,
    StoreKitStore,
    TestStore,
    make_store,
)

__all__ = [
    "FREE_CAPABILITIES",
    "FREE_MAX_TOKENS",
    "FREE_TIER_KEEPS",
    "LicenseKeyRefused",
    "PAID_CAPABILITIES",
    "PAID_MAX_TOKENS",
    "PAID_UNLOCKS",
    "PRODUCTS",
    "PurchaseError",
    "Receipt",
    "StoreKitStore",
    "TestStore",
    "allows",
    "catalog_dict",
    "clamp_generation",
    "entitled_from_receipts",
    "get_product",
    "legal_page_path",
    "legal_page_url",
    "make_store",
    "paywall_payload",
    "refuse_license_key",
    "sanitize_settings",
]


def paywall_payload(store: StoreKitStore) -> Dict[str, Any]:
    """Snapshot the shipped catalog plus current entitlement for the UI."""
    payload = catalog_dict()
    receipts: List[Dict[str, Any]] = [r.to_dict() for r in store.receipts()]
    payload["entitled"] = bool(store.is_entitled())
    payload["receipts"] = receipts
    return payload
