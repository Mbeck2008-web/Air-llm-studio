"""Drive the shipped IAP catalog, entitlement gate, purchase, and restore."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from airllm_studio.billing import (
    LicenseKeyRefused,
    PRODUCTS,
    PurchaseError,
    allows,
    catalog_dict,
    entitled_from_receipts,
    legal_page_path,
    make_store,
)
from airllm_studio.billing.store import looks_like_license_key
from airllm_studio.core.config import AppConfig
from airllm_studio.core import config as cfgmod
from airllm_studio.core.session import StudioSession


def _wait_gen_done(session: StudioSession, timeout: float = 12.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in session.drain_events():
            if event.get("kind") == "gen_done":
                return event
        time.sleep(0.05)
    raise AssertionError("session.send never emitted gen_done")


class TestShippedCatalog(unittest.TestCase):
    def test_catalog_has_reasonable_usd_skus(self) -> None:
        cat = catalog_dict()
        products = cat["products"]
        self.assertGreaterEqual(len(products), 2)
        self.assertEqual(len(products), len(PRODUCTS))
        monthly = [p for p in PRODUCTS if p.period == "month"]
        yearly_or_life = [
            p
            for p in PRODUCTS
            if p.period == "year" or p.iap_type == "non-consumable"
        ]
        self.assertTrue(monthly, "need a monthly SKU")
        self.assertTrue(yearly_or_life, "need a yearly or lifetime SKU")
        self.assertTrue(all(p.usd <= 9.99 for p in monthly))
        self.assertTrue(any(p.usd <= 59.99 for p in yearly_or_life))
        for product in PRODUCTS:
            self.assertTrue(product.product_id)
            self.assertIn(product.iap_type, ("auto-renewable", "non-consumable"))
            self.assertGreater(product.usd, 0)
        self.assertTrue(cat["free_tier"])
        self.assertTrue(cat["paid_unlocks"])
        self.assertTrue(cat["has_subscription"])
        self.assertTrue(cat["privacy_policy_url"])
        self.assertTrue(cat["terms_url"])

    def test_legal_pages_and_paywall_markup_exist(self) -> None:
        cat = catalog_dict()
        privacy = legal_page_path("privacy")
        terms = legal_page_path("terms")
        self.assertTrue(privacy.is_file(), privacy)
        self.assertTrue(terms.is_file(), terms)
        privacy_text = privacy.read_text(encoding="utf-8")
        terms_text = terms.read_text(encoding="utf-8")
        self.assertIn("Privacy", privacy_text)
        self.assertIn("auto-renew", terms_text.lower())
        web = Path(__file__).resolve().parents[1] / "airllm_studio" / "web"
        html = (web / "index.html").read_text(encoding="utf-8")
        js = (web / "app.js").read_text(encoding="utf-8")
        self.assertIn("Restore Purchases", html)
        self.assertIn("legal/privacy.html", html)
        self.assertIn("legal/terms.html", html)
        self.assertIn('data-tab="pro"', html)
        self.assertIn("Manage or cancel", html)
        self.assertIn("Michael Beck", html)
        self.assertIn("btn-dev-unlock", html)
        self.assertNotIn('target="_blank"', html)
        self.assertIn("purchase", js)
        self.assertIn("restore_purchases", js)
        self.assertIn("dev_unlock", js)
        self.assertTrue(cat.get("manage_subscription_copy"))
        self.assertIn("Michael Beck", cat.get("about_copyright", ""))

    def test_license_key_is_not_a_catalog_sku(self) -> None:
        self.assertTrue(looks_like_license_key("ABCD-EFGH-IJKL-MNOP"))
        self.assertFalse(
            entitled_from_receipts(
                [
                    {
                        "product_id": "ABCD-EFGH-IJKL-MNOP",
                        "transaction_id": "x",
                    }
                ]
            )
        )
        for product in PRODUCTS:
            self.assertFalse(looks_like_license_key(product.product_id))


class TestStorePurchaseRestore(unittest.TestCase):
    def test_lock_purchase_restore_on_storekit_store(self) -> None:
        with TemporaryDirectory() as tmp:
            store = make_store(Path(tmp))
            sku = next(p.product_id for p in PRODUCTS if p.period == "month")
            self.assertFalse(store.is_entitled())
            self.assertTrue(allows("local_chat", store.is_entitled()))
            self.assertFalse(allows("web_search", store.is_entitled()))
            receipt = store.purchase(sku)
            self.assertEqual(receipt.product_id, sku)
            self.assertTrue(store.is_entitled())
            self.assertTrue(allows("web_search", store.is_entitled()))
            store.clear_entitlement()
            self.assertFalse(store.is_entitled())
            restored = store.restore()
            self.assertTrue(restored)
            self.assertTrue(store.is_entitled())
            self.assertTrue(allows("web_search", store.is_entitled()))

    def test_unknown_product_and_license_key_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            store = make_store(Path(tmp))
            with self.assertRaises(LicenseKeyRefused):
                store.purchase("ABCD-EFGH-IJKL-MNOP")
            with self.assertRaises(LicenseKeyRefused):
                store.unlock_with_license_key("LICENSE-KEY-0001")
            with self.assertRaises(PurchaseError):
                store.purchase("ai.airllm.studio.not.a.sku")
            self.assertFalse(store.is_entitled())


class TestSessionGateAndFreeChat(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._prev = cfgmod._config
        cfg = AppConfig(data_dir=Path(self._tmp.name), demo_mode=True)
        cfg.ensure_dirs()
        cfgmod._config = cfg
        self.session = StudioSession(cfg=cfg)

    def tearDown(self) -> None:
        cfgmod._config = self._prev
        self._tmp.cleanup()

    def test_unentitled_chat_works_and_paid_stays_locked(self) -> None:
        session = self.session
        self.assertTrue(session.can_use("local_chat"))
        self.assertFalse(session.can_use("web_search"))
        self.assertFalse(session.can_use("hf_search"))
        self.assertFalse(session.can_use("high_token_limit"))

        hf = session.search_hf("llama")
        self.assertFalse(hf.get("ok"))

        snap = session.save_settings(
            {
                "tools_enabled": True,
                "web_search_enabled": True,
                "default_max_tokens": 4096,
            }
        )
        self.assertFalse(snap["config"]["tools_enabled"])
        self.assertFalse(snap["config"]["web_search_enabled"])
        self.assertLessEqual(
            int(snap["config"]["default_max_tokens"]),
            catalog_dict()["free_max_tokens"],
        )
        self.assertFalse(snap["billing"]["entitled"])

        session.engine.load("demo/tiny")
        sent = session.send("hello from the free tier", use_search=True, max_new_tokens=4096)
        self.assertTrue(sent.get("ok"))
        done = _wait_gen_done(session)
        self.assertIn("kind", done)
        chat = session.store.get(session.active_chat_id)
        self.assertIsNotNone(chat)
        user_msgs = [m for m in chat.messages if m.role == "user"]
        self.assertTrue(user_msgs)
        self.assertIn("hello from the free tier", user_msgs[0].content)

    def test_purchase_then_restore_unlocks_paid_on_session(self) -> None:
        session = self.session
        sku = next(p.product_id for p in PRODUCTS if p.period == "year")
        self.assertFalse(session.can_use("web_search"))
        bought = session.purchase(sku)
        self.assertTrue(bought.get("ok"))
        self.assertTrue(session.can_use("web_search"))
        self.assertTrue(session.can_use("hf_search"))
        snap = session.save_settings(
            {"tools_enabled": True, "web_search_enabled": True, "default_max_tokens": 2048}
        )
        self.assertTrue(snap["config"]["tools_enabled"])
        self.assertTrue(snap["config"]["web_search_enabled"])
        self.assertEqual(int(snap["config"]["default_max_tokens"]), 2048)

        session.iap.clear_entitlement()
        self.assertFalse(session.can_use("web_search"))
        restored = session.restore_purchases()
        self.assertTrue(restored.get("ok"))
        self.assertTrue(session.can_use("web_search"))
        self.assertTrue(session.snapshot()["billing"]["entitled"])

    def test_session_refuses_license_key(self) -> None:
        out = self.session.unlock_with_license_key("ABCD-EFGH-IJKL-MNOP")
        self.assertFalse(out.get("ok"))
        self.assertEqual(out.get("error"), "license_key_refused")
        self.assertFalse(self.session.can_use("web_search"))
        bought = self.session.purchase("ABCD-EFGH-IJKL-MNOP")
        self.assertFalse(bought.get("ok"))
        self.assertEqual(bought.get("error"), "license_key_refused")



class TestDevUnlock(unittest.TestCase):
    def test_dev_unlock_works_outside_release(self) -> None:
        import os

        from airllm_studio.billing import dev_unlock_available, make_store

        prev = os.environ.pop("AIRLLM_STUDIO_RELEASE", None)
        try:
            self.assertTrue(dev_unlock_available())
            with TemporaryDirectory() as tmp:
                store = make_store(Path(tmp))
                receipt = store.dev_unlock()
                self.assertEqual(receipt.product_id, "ai.airllm.studio.pro.lifetime")
                self.assertTrue(store.is_entitled())
        finally:
            if prev is not None:
                os.environ["AIRLLM_STUDIO_RELEASE"] = prev

    def test_dev_unlock_blocked_in_release(self) -> None:
        import os

        from airllm_studio.billing import PurchaseError, make_store
        from airllm_studio.billing.release import is_release_build

        prev = os.environ.get("AIRLLM_STUDIO_RELEASE")
        os.environ["AIRLLM_STUDIO_RELEASE"] = "1"
        try:
            self.assertTrue(is_release_build())
            with TemporaryDirectory() as tmp:
                store = make_store(Path(tmp))
                with self.assertRaises(PurchaseError):
                    store.dev_unlock()
                self.assertFalse(store.is_entitled())
        finally:
            if prev is None:
                os.environ.pop("AIRLLM_STUDIO_RELEASE", None)
            else:
                os.environ["AIRLLM_STUDIO_RELEASE"] = prev


if __name__ == "__main__":
    unittest.main()
