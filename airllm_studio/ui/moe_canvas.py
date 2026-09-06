"""MoE / dense architecture visualization (CustomTkinter-safe)."""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk

from airllm_studio.core.model_meta import ModelInfo
from airllm_studio.ui.theme import COLORS, font


class MoEDiagram(ctk.CTkFrame):
    """
    Compact structured view of dense layers or sparse MoE routing.
    Inspired by ultra-sparse MoE diagrams: tokens → router → few active experts.
    """

    def __init__(self, master, width: int = 270, height: int = 190, **kwargs):
        super().__init__(
            master,
            width=width,
            height=height,
            fg_color=COLORS["bg_elevated"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=8,
            **kwargs,
        )
        self._width = width
        self._height = height
        self.grid_propagate(False)
        self.pack_propagate(False)
        self._inner = ctk.CTkFrame(self, fg_color="transparent")
        self._inner.pack(fill="both", expand=True, padx=10, pady=8)
        self.draw_placeholder()

    def _clear(self) -> None:
        for w in self._inner.winfo_children():
            w.destroy()

    def draw_placeholder(self) -> None:
        self._clear()
        ctk.CTkLabel(
            self._inner,
            text="Select a model to view\narchitecture diagram",
            font=font(12),
            text_color=COLORS["text_dim"],
            justify="center",
        ).pack(expand=True)

    def draw_dense(self, info: ModelInfo) -> None:
        self._clear()
        layers = info.num_layers or 12
        ctk.CTkLabel(
            self._inner,
            text="Dense Transformer",
            font=font(12, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 6))

        stack = ctk.CTkFrame(self._inner, fg_color="transparent")
        stack.pack(fill="x")
        show = min(layers, 10)
        for i in range(show):
            ctk.CTkFrame(
                stack,
                height=10,
                fg_color=COLORS["accent_dim"],
                corner_radius=3,
            ).pack(fill="x", pady=1)

        if layers > show:
            ctk.CTkLabel(
                self._inner,
                text=f"… {layers} layers total",
                font=font(10),
                text_color=COLORS["text_muted"],
            ).pack(anchor="w", pady=(6, 0))

        ctk.CTkLabel(
            self._inner,
            text="AirLLM streams one layer at a time",
            font=font(10),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", pady=(8, 0))

    def draw_moe(self, info: ModelInfo) -> None:
        self._clear()
        n_experts = info.num_experts or 8
        active = info.experts_per_token or 2
        n_experts = max(n_experts, 2)
        active = min(active, n_experts)

        ctk.CTkLabel(
            self._inner,
            text=f"Sparse MoE  ·  {active} of {n_experts} active",
            font=font(12, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 8))

        # Tokens → Router → Experts
        flow = ctk.CTkFrame(self._inner, fg_color="transparent")
        flow.pack(fill="x")

        left = ctk.CTkFrame(flow, fg_color="transparent")
        left.pack(side="left", padx=(0, 6))
        ctk.CTkLabel(left, text="tokens", font=font(9), text_color=COLORS["text_dim"]).pack()
        for _ in range(3):
            ctk.CTkFrame(
                left, width=18, height=18, corner_radius=9, fg_color=COLORS["moe_token"]
            ).pack(pady=2)

        ctk.CTkLabel(
            flow, text="→", font=font(16), text_color=COLORS["text_muted"]
        ).pack(side="left", padx=4)

        mid = ctk.CTkFrame(
            flow,
            width=40,
            height=40,
            corner_radius=8,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["accent"],
        )
        mid.pack(side="left", padx=4)
        mid.pack_propagate(False)
        ctk.CTkLabel(
            mid, text="R", font=font(14, "bold"), text_color=COLORS["accent"]
        ).pack(expand=True)

        ctk.CTkLabel(
            flow, text="→", font=font(16), text_color=COLORS["text_muted"]
        ).pack(side="left", padx=4)

        right = ctk.CTkFrame(flow, fg_color="transparent")
        right.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ctk.CTkLabel(
            right, text="experts", font=font(9), text_color=COLORS["text_dim"]
        ).pack(anchor="w")

        grid = ctk.CTkFrame(right, fg_color="transparent")
        grid.pack(anchor="w")
        show = min(n_experts, 16)
        cols = 4
        for i in range(show):
            r, c = divmod(i, cols)
            is_active = i < active
            cell = ctk.CTkFrame(
                grid,
                width=22,
                height=22,
                corner_radius=4,
                fg_color=COLORS["moe_active"] if is_active else COLORS["moe_idle"],
            )
            cell.grid(row=r, column=c, padx=2, pady=2)
            cell.grid_propagate(False)

        if n_experts > show:
            ctk.CTkLabel(
                right,
                text=f"+{n_experts - show} more",
                font=font(9),
                text_color=COLORS["text_dim"],
            ).pack(anchor="w", pady=(4, 0))

        footer = "AirLLM streams only routed experts"
        if info.shared_experts:
            footer += f"  ·  {info.shared_experts} shared"
        ctk.CTkLabel(
            self._inner,
            text=footer,
            font=font(10),
            text_color=COLORS["text_dim"],
            wraplength=240,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def render(self, info: Optional[ModelInfo]) -> None:
        if info is None:
            self.draw_placeholder()
        elif info.is_moe:
            self.draw_moe(info)
        else:
            self.draw_dense(info)
