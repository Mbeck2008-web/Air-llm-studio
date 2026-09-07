"""Shipped Mac App Store IAP catalog.

Prices are USD list prices shown on the paywall. Live App Store Connect
SKUs must use the same product identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IAPProduct:
    product_id: str
    display_name: str
    iap_type: str  # auto-renewable | non-consumable
    usd: float
    period: Optional[str]  # month | year | None (lifetime)
    auto_renew: bool
    description: str

    def period_plain(self) -> str:
        if self.iap_type == "non-consumable" or not self.auto_renew:
            return "One-time purchase. Does not auto-renew."
        if self.period == "month":
            return (
                "Billed every month. Auto-renews until you cancel in "
                "System Settings → Apple ID → Subscriptions."
            )
        if self.period == "year":
            return (
                "Billed every year. Auto-renews until you cancel in "
                "System Settings → Apple ID → Subscriptions."
            )
        return "Subscription. Auto-renews until you cancel in App Store settings."

    def price_display(self) -> str:
        return f"${self.usd:.2f}"


PRODUCTS: Tuple[IAPProduct, ...] = (
    IAPProduct(
        product_id="ai.airllm.studio.pro.monthly",
        display_name="Pro Monthly",
        iap_type="auto-renewable",
        usd=7.99,
        period="month",
        auto_renew=True,
        description="Unlock Pro extras for one month.",
    ),
    IAPProduct(
        product_id="ai.airllm.studio.pro.yearly",
        display_name="Pro Yearly",
        iap_type="auto-renewable",
        usd=49.99,
        period="year",
        auto_renew=True,
        description="Unlock Pro extras for one year.",
    ),
    IAPProduct(
        product_id="ai.airllm.studio.pro.lifetime",
        display_name="Pro Lifetime",
        iap_type="non-consumable",
        usd=59.99,
        period=None,
        auto_renew=False,
        description="Unlock Pro extras forever with a one-time purchase.",
    ),
)

FREE_TIER_KEEPS: Tuple[str, ...] = (
    "Add, prepare, and load local models",
    "Chat with a loaded model or the demo/mock engine",
    "Generation defaults up to 512 tokens",
)

PAID_UNLOCKS: Tuple[str, ...] = (
    "Web search tool and tool calling",
    "Higher generation limit (up to 4096 tokens)",
    "Hugging Face library search",
)

FREE_MAX_TOKENS = 512
PAID_MAX_TOKENS = 4096

MANAGE_SUBSCRIPTION_COPY = (
    "Manage or cancel an auto-renewing subscription in "
    "System Settings → Apple ID → Subscriptions. "
    "Pro Lifetime does not auto-renew."
)

ABOUT_COPYRIGHT = "Copyright © 2026 Michael Beck. All rights reserved."

_LEGAL_DIR = Path(__file__).resolve().parents[1] / "web" / "legal"

# GitHub Pages (docs/ on main). Same HTML as airllm_studio/web/legal/*.html.
PUBLIC_LEGAL_BASE = "https://mbeck2008-web.github.io/Air-llm-studio/legal"


def get_product(product_id: str) -> Optional[IAPProduct]:
    for product in PRODUCTS:
        if product.product_id == product_id:
            return product
    return None


def legal_page_path(kind: str) -> Path:
    name = "privacy.html" if kind == "privacy" else "terms.html"
    return _LEGAL_DIR / name


def public_legal_url(kind: str) -> str:
    """Stable HTTPS URL for App Store Connect and About / paywall metadata."""
    name = "privacy.html" if kind == "privacy" else "terms.html"
    return f"{PUBLIC_LEGAL_BASE}/{name}"


def legal_page_url(kind: str) -> str:
    """Stable public HTTPS URL (same as public_legal_url). Bundled file:// via legal_page_path()."""
    return public_legal_url(kind)


def legal_page_file_url(kind: str) -> str:
    """file:// URI for the bundled copy (in-app WebKit fallback)."""
    return legal_page_path(kind).resolve().as_uri()


def catalog_dict() -> Dict[str, Any]:
    """The structure the paywall and tests import. Not a parallel fixture."""
    products: List[Dict[str, Any]] = []
    for product in PRODUCTS:
        row = asdict(product)
        row["price_display"] = product.price_display()
        row["period_plain"] = product.period_plain()
        products.append(row)
    has_subscription = any(p.iap_type == "auto-renewable" for p in PRODUCTS)
    return {
        "products": products,
        "free_tier": list(FREE_TIER_KEEPS),
        "paid_unlocks": list(PAID_UNLOCKS),
        "free_max_tokens": FREE_MAX_TOKENS,
        "paid_max_tokens": PAID_MAX_TOKENS,
        "has_subscription": has_subscription,
        "privacy_policy_url": public_legal_url("privacy"),
        "terms_url": public_legal_url("terms"),
        "privacy_policy_href": "legal/privacy.html",
        "terms_href": "legal/terms.html",
        "manage_subscription_copy": MANAGE_SUBSCRIPTION_COPY,
        "about_copyright": ABOUT_COPYRIGHT,
    }
