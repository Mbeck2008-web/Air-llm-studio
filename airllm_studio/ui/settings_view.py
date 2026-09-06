"""Settings screen."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import customtkinter as ctk

from airllm_studio.core.config import AppConfig
from airllm_studio.ui.theme import COLORS, font


class SettingsView(ctk.CTkFrame):
    def __init__(
        self,
        master,
        config: AppConfig,
        on_save: Callable[[dict], None],
        billing: Optional[Dict[str, Any]] = None,
        on_purchase: Optional[Callable[[str], None]] = None,
        on_restore: Optional[Callable[[], None]] = None,
        on_open_legal: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=COLORS["bg"], corner_radius=0, **kwargs)
        self.config = config
        self.on_save = on_save
        self.billing = billing or {}
        self.on_purchase = on_purchase
        self.on_restore = on_restore
        self.on_open_legal = on_open_legal
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_elevated"], height=48, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header,
            text="Settings",
            font=font(14, "bold"),
            text_color=COLORS["text"],
        ).pack(side="left", padx=16, pady=12)

        body = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=16)

        def section(title: str):
            ctk.CTkLabel(
                body,
                text=title,
                font=font(12, "bold"),
                text_color=COLORS["text_dim"],
            ).pack(anchor="w", pady=(16, 6))

        section("HUGGING FACE")
        ctk.CTkLabel(
            body,
            text="Access token (for gated models)",
            font=font(12),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")
        self.hf_token = ctk.CTkEntry(body, show="•", height=36, font=font(13))
        self.hf_token.pack(fill="x", pady=(4, 0))
        if self.config.hf_token:
            self.hf_token.insert(0, self.config.hf_token)

        section("GENERATION DEFAULTS")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Temperature", width=120, anchor="w").pack(side="left")
        self.temp = ctk.CTkEntry(row, width=80)
        self.temp.insert(0, str(self.config.default_temperature))
        self.temp.pack(side="left")

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.pack(fill="x", pady=4)
        ctk.CTkLabel(row2, text="Max tokens", width=120, anchor="w").pack(side="left")
        self.max_tok = ctk.CTkEntry(row2, width=80)
        self.max_tok.insert(0, str(self.config.default_max_tokens))
        self.max_tok.pack(side="left")

        section("AIRLLM")
        ctk.CTkLabel(
            body,
            text="Compression (empty / 4bit / 8bit)",
            font=font(12),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")
        self.compression = ctk.CTkEntry(body, height=36)
        self.compression.pack(fill="x", pady=(4, 8))
        if self.config.compression:
            self.compression.insert(0, self.config.compression)

        self.delete_original = ctk.CTkCheckBox(
            body,
            text="Delete original HF weights after split (saves disk)",
            font=font(12),
        )
        if self.config.delete_original_after_split:
            self.delete_original.select()
        self.delete_original.pack(anchor="w", pady=4)

        self.demo_mode = ctk.CTkCheckBox(
            body,
            text="Force demo mode (no real AirLLM load)",
            font=font(12),
        )
        if self.config.demo_mode:
            self.demo_mode.select()
        self.demo_mode.pack(anchor="w", pady=4)

        section("TOOLS")
        self.tools_enabled = ctk.CTkCheckBox(
            body, text="Enable tool calling", font=font(12)
        )
        if self.config.tools_enabled:
            self.tools_enabled.select()
        self.tools_enabled.pack(anchor="w", pady=4)

        self.web_search = ctk.CTkCheckBox(
            body, text="Enable web_search tool", font=font(12)
        )
        if self.config.web_search_enabled:
            self.web_search.select()
        self.web_search.pack(anchor="w", pady=4)

        section("PATHS")
        ctk.CTkLabel(
            body,
            text=f"Data directory:\n{self.config.data_dir}",
            font=font(11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", pady=4)

        section("PRO")
        self._paywall_status = ctk.CTkLabel(
            body,
            text="",
            font=font(12),
            text_color=COLORS["accent"],
            justify="left",
            wraplength=520,
        )
        self._paywall_status.pack(anchor="w", pady=(0, 8))
        ctk.CTkLabel(
            body,
            text="Free tier keeps:\n" + "\n".join(f"• {x}" for x in self.billing.get("free_tier") or []),
            font=font(12),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", pady=4)
        ctk.CTkLabel(
            body,
            text="Payment unlocks:\n" + "\n".join(f"• {x}" for x in self.billing.get("paid_unlocks") or []),
            font=font(12),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", pady=4)
        for product in self.billing.get("products") or []:
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=4)
            label = (
                f"{product.get('display_name')} — {product.get('price_display')} USD"
            )
            if product.get("auto_renew"):
                label += f"  ·  {product.get('period_plain')}"
            else:
                label += f"  ·  {product.get('period_plain')}"
            ctk.CTkLabel(row, text=label, font=font(12), anchor="w", wraplength=420).pack(
                side="left", fill="x", expand=True
            )
            pid = product.get("product_id") or ""
            ctk.CTkButton(
                row,
                text="Buy",
                width=72,
                command=lambda p=pid: self.on_purchase and self.on_purchase(p),
            ).pack(side="right", padx=4)
        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", pady=8)
        ctk.CTkButton(
            actions,
            text="Restore Purchases",
            command=lambda: self.on_restore and self.on_restore(),
        ).pack(side="left", padx=(0, 8))
        if self.billing.get("has_subscription"):
            ctk.CTkButton(
                actions,
                text="Privacy Policy",
                fg_color="transparent",
                command=lambda: self.on_open_legal and self.on_open_legal("privacy"),
            ).pack(side="left", padx=4)
            ctk.CTkButton(
                actions,
                text="Terms of Use",
                fg_color="transparent",
                command=lambda: self.on_open_legal and self.on_open_legal("terms"),
            ).pack(side="left", padx=4)
        self._refresh_paywall_status()

        ctk.CTkButton(
            body,
            text="Save settings",
            font=font(13, "bold"),
            height=40,
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=self._save,
        ).pack(anchor="w", pady=24)

        self.status = ctk.CTkLabel(body, text="", font=font(12), text_color=COLORS["success"])
        self.status.pack(anchor="w")

    def _save(self) -> None:
        try:
            temp = float(self.temp.get().strip())
        except ValueError:
            temp = 0.7
        try:
            max_tok = int(self.max_tok.get().strip())
        except ValueError:
            max_tok = 512
        comp = self.compression.get().strip() or None
        if comp not in (None, "4bit", "8bit"):
            comp = None

        data = {
            "hf_token": self.hf_token.get().strip(),
            "default_temperature": temp,
            "default_max_tokens": max_tok,
            "compression": comp,
            "delete_original_after_split": bool(self.delete_original.get()),
            "demo_mode": bool(self.demo_mode.get()),
            "tools_enabled": bool(self.tools_enabled.get()),
            "web_search_enabled": bool(self.web_search.get()),
        }
        self.on_save(data)
        self.status.configure(text="Saved.")

    def refresh_billing(self, billing: Dict[str, Any]) -> None:
        self.billing = billing or {}
        self._refresh_paywall_status()

    def _refresh_paywall_status(self) -> None:
        if not hasattr(self, "_paywall_status"):
            return
        if self.billing.get("entitled"):
            self._paywall_status.configure(text="Pro is unlocked on this Mac.")
        else:
            self._paywall_status.configure(
                text="You are on the free tier. Local chat stays available."
            )
