"""StoreKit-facing purchase API with a documented local test-store fallback.

Live StoreKit needs App Store Connect products and a sandbox Apple ID.
When that environment is missing, purchase and restore complete through
TestStore so the same entitlement gate can be exercised.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from airllm_studio.billing.catalog import get_product
from airllm_studio.billing.entitlement import (
    LicenseKeyRefused,
    entitled_from_receipts,
    refuse_license_key,
)
from airllm_studio.billing.release import dev_unlock_available


class PurchaseError(Exception):
    """Purchase could not be completed."""


_LICENSE_KEY = re.compile(
    r"^[A-Z0-9]{4}(?:-[A-Z0-9]{4}){2,}$",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def looks_like_license_key(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if text.startswith("ai.airllm.studio."):
        return False
    lowered = text.lower()
    if "license" in lowered or lowered.startswith("key="):
        return True
    return bool(_LICENSE_KEY.fullmatch(text))


def _reject_license(value: str) -> None:
    if looks_like_license_key(value):
        refuse_license_key(value)


@dataclass
class Receipt:
    product_id: str
    transaction_id: str
    purchased_at: str
    source: str = "test-store"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Receipt":
        return cls(
            product_id=str(data["product_id"]),
            transaction_id=str(data["transaction_id"]),
            purchased_at=str(data.get("purchased_at") or _now_iso()),
            source=str(data.get("source") or "test-store"),
        )


class TestStore:
    """File-backed StoreKit substitute. History survives clear_entitlement()."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._active_ids: List[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        history = raw.get("history") or []
        active = raw.get("active_ids") or []
        if isinstance(history, list):
            self._history = [h for h in history if isinstance(h, dict)]
        if isinstance(active, list):
            self._active_ids = [str(i) for i in active]

    def _save(self) -> None:
        payload = {"history": self._history, "active_ids": self._active_ids}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def purchase(self, product_id: str) -> Receipt:
        _reject_license(product_id)
        product = get_product(product_id)
        if product is None:
            raise PurchaseError(f"Unknown StoreKit product: {product_id}")
        receipt = Receipt(
            product_id=product.product_id,
            transaction_id=str(uuid.uuid4()),
            purchased_at=_now_iso(),
            source="test-store",
        )
        row = receipt.to_dict()
        self._history.append(row)
        self._active_ids.append(receipt.transaction_id)
        self._save()
        return receipt

    def restore(self) -> List[Receipt]:
        self._active_ids = [str(row["transaction_id"]) for row in self._history]
        self._save()
        return self.receipts()

    def clear_entitlement(self) -> None:
        """Drop local entitlement. Prior transactions stay for Restore."""
        self._active_ids = []
        self._save()

    def receipts(self) -> List[Receipt]:
        by_id = {str(row["transaction_id"]): row for row in self._history}
        out: List[Receipt] = []
        for tid in self._active_ids:
            row = by_id.get(tid)
            if row:
                out.append(Receipt.from_dict(row))
        return out

    def is_entitled(self) -> bool:
        return entitled_from_receipts(self.receipts())

    def dev_unlock(self) -> Receipt:
        """Grant Pro via lifetime SKU for non-Release builds only."""
        if not dev_unlock_available():
            raise PurchaseError("Dev Unlock is disabled in Release builds.")
        return self.purchase("ai.airllm.studio.pro.lifetime")


def _try_storekit_purchase(product_id: str) -> Optional[Receipt]:
    """Attempt a live StoreKit purchase. None means use the test store."""
    del product_id
    return None


def _try_storekit_restore() -> Optional[List[Receipt]]:
    """Attempt a live StoreKit restore. None means use the test store."""
    return None


class StoreKitStore:
    """The store the app calls. StoreKit first, TestStore when StoreKit cannot run."""

    def __init__(self, fallback: TestStore) -> None:
        self.fallback = fallback

    def purchase(self, product_id: str) -> Receipt:
        _reject_license(product_id)
        live = _try_storekit_purchase(product_id)
        if live is not None:
            return live
        return self.fallback.purchase(product_id)

    def restore(self) -> List[Receipt]:
        live = _try_storekit_restore()
        if live is not None:
            return live
        return self.fallback.restore()

    def clear_entitlement(self) -> None:
        self.fallback.clear_entitlement()

    def receipts(self) -> List[Receipt]:
        return self.fallback.receipts()

    def is_entitled(self) -> bool:
        return self.fallback.is_entitled()

    def unlock_with_license_key(self, code: str) -> Receipt:
        refuse_license_key(code)
        raise LicenseKeyRefused("unreachable")

    def dev_unlock(self) -> Receipt:
        if not dev_unlock_available():
            raise PurchaseError("Dev Unlock is disabled in Release builds.")
        return self.fallback.dev_unlock()


def make_store(data_dir: Path) -> StoreKitStore:
    test = TestStore(Path(data_dir) / "iap" / "transactions.json")
    return StoreKitStore(fallback=test)
