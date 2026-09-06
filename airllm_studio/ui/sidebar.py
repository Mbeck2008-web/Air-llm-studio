"""Left navigation: chats + section switch."""

from __future__ import annotations

from typing import Callable, List, Optional

import customtkinter as ctk

from airllm_studio.core.chat_store import Chat
from airllm_studio.ui.theme import COLORS, font


class Sidebar(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_nav: Callable[[str], None],
        on_new_chat: Callable[[], None],
        on_select_chat: Callable[[str], None],
        on_delete_chat: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color=COLORS["bg_elevated"], corner_radius=0, width=220, **kwargs)
        self.on_nav = on_nav
        self.on_new_chat = on_new_chat
        self.on_select_chat = on_select_chat
        self.on_delete_chat = on_delete_chat
        self.pack_propagate(False)
        self._chat_buttons: List[ctk.CTkButton] = []
        self._build()

    def _build(self) -> None:
        # Brand
        brand = ctk.CTkFrame(self, fg_color="transparent")
        brand.pack(fill="x", padx=14, pady=(18, 8))
        ctk.CTkLabel(
            brand,
            text="AirLLM Studio",
            font=font(16, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="Local · Layer-streamed",
            font=font(11),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w")

        # Nav buttons
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=10, pady=(12, 4))

        self._nav_btns = {}
        for key, label in (
            ("chat", "Chat"),
            ("library", "Model Library"),
            ("settings", "Settings"),
        ):
            b = ctk.CTkButton(
                nav,
                text=label,
                font=font(13),
                fg_color="transparent",
                hover_color=COLORS["border"],
                anchor="w",
                height=34,
                command=lambda k=key: self.on_nav(k),
            )
            b.pack(fill="x", pady=2)
            self._nav_btns[key] = b

        self.set_active_nav("chat")

        # Chats header
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(16, 4))
        ctk.CTkLabel(
            row,
            text="CHATS",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(side="left")
        ctk.CTkButton(
            row,
            text="+",
            width=28,
            height=28,
            font=font(16),
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["accent_dim"],
            command=self.on_new_chat,
        ).pack(side="right")

        self.chat_list = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
        )
        self.chat_list.pack(fill="both", expand=True, padx=8, pady=(4, 12))

    def set_active_nav(self, key: str) -> None:
        for k, b in self._nav_btns.items():
            if k == key:
                b.configure(fg_color=COLORS["bg_panel"], text_color=COLORS["accent"])
            else:
                b.configure(fg_color="transparent", text_color=COLORS["text"])

    def refresh_chats(self, chats: List[Chat], active_id: Optional[str] = None) -> None:
        for w in self.chat_list.winfo_children():
            w.destroy()
        self._chat_buttons.clear()

        for chat in chats:
            row = ctk.CTkFrame(self.chat_list, fg_color="transparent")
            row.pack(fill="x", pady=1)
            is_active = chat.id == active_id
            btn = ctk.CTkButton(
                row,
                text=chat.title or "New Chat",
                font=font(12),
                fg_color=COLORS["bg_panel"] if is_active else "transparent",
                hover_color=COLORS["border"],
                anchor="w",
                height=32,
                command=lambda cid=chat.id: self.on_select_chat(cid),
            )
            btn.pack(side="left", fill="x", expand=True)
            del_btn = ctk.CTkButton(
                row,
                text="×",
                width=28,
                height=28,
                font=font(14),
                fg_color="transparent",
                hover_color=COLORS["danger"],
                text_color=COLORS["text_dim"],
                command=lambda cid=chat.id: self.on_delete_chat(cid),
            )
            del_btn.pack(side="right", padx=(2, 0))
            self._chat_buttons.append(btn)
