"""In-App Purchase catalog, entitlement policy, and StoreKit-facing store."""

from __future__ import annotations

from typing import Any, Dict, List

from airllm_studio.billing.catalog import (
    FREE_MAX_TOKENS,
    FREE_TIER_KEEPS,
    PAID_MAX_TOKENS,
    PAID_UNLOCKS,
    PRODUCTS,
    PUBLIC_LEGAL_BASE,
    catalog_dict,
    get_product,
    legal_page_file_url,
    legal_page_path,
    legal_page_url,
    public_legal_url,
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
from airllm_studio.billing.release import (
    dev_unlock_available,
    is_release_build,
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
    "dev_unlock_available",
    "is_release_build",
    "clamp_generation",
    "entitled_from_receipts",
    "get_product",
    "PUBLIC_LEGAL_BASE",
    "legal_page_file_url",
    "legal_page_path",
    "legal_page_url",
    "public_legal_url",
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
    payload["dev_unlock_available"] = bool(dev_unlock_available())
    payload["release_build"] = bool(is_release_build())
    return payload
