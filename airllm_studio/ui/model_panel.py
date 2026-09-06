"""Right-hand model information panel."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from airllm_studio.core.model_meta import (
    ModelInfo,
    can_load_without_airllm,
    estimate_full_load_memory_gb,
    format_params,
)
from airllm_studio.ui.moe_canvas import MoEDiagram
from airllm_studio.ui.theme import COLORS, font


class ModelPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_load: Optional[Callable[[], None]] = None,
        on_unload: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=COLORS["bg_panel"], corner_radius=0, **kwargs)
        self.on_load = on_load
        self.on_unload = on_unload
        self._info: Optional[ModelInfo] = None
        self._status = "—"
        self._airllm_enabled = True
        self._load_blocked = False
        self._build()

    def _build(self) -> None:
        pad = {"padx": 14, "pady": 4}

        ctk.CTkLabel(
            self,
            text="MODEL",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=14, pady=(16, 4))

        self.title_lbl = ctk.CTkLabel(
            self,
            text="No model selected",
            font=font(15, "bold"),
            text_color=COLORS["text"],
            wraplength=260,
            justify="left",
        )
        self.title_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        self.badge = ctk.CTkLabel(
            self,
            text="—",
            font=font(11),
            text_color=COLORS["accent"],
            fg_color=COLORS["bg_elevated"],
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.badge.pack(anchor="w", padx=14, pady=(0, 10))

        # Stats grid
        self.stats = ctk.CTkTextbox(
            self,
            height=140,
            font=font(12),
            fg_color=COLORS["bg_elevated"],
            text_color=COLORS["text"],
            border_color=COLORS["border"],
            border_width=1,
            wrap="word",
            activate_scrollbars=True,
        )
        self.stats.pack(fill="x", padx=14, pady=4)
        self.stats.insert("1.0", "Select a model from the library.")
        self.stats.configure(state="disabled")

        ctk.CTkLabel(
            self,
            text="ARCHITECTURE",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=14, pady=(14, 4))

        self.diagram = MoEDiagram(self, width=270, height=190)
        self.diagram.pack(padx=14, pady=4)

        # Memory callout
        self.mem_lbl = ctk.CTkLabel(
            self,
            text="",
            font=font(12),
            text_color=COLORS["warning"],
            wraplength=260,
            justify="left",
        )
        self.mem_lbl.pack(anchor="w", padx=14, pady=(10, 4))

        # Warning when AirLLM is required
        self.warn_lbl = ctk.CTkLabel(
            self,
            text="",
            font=font(11),
            text_color=COLORS["danger"],
            wraplength=260,
            justify="left",
        )
        self.warn_lbl.pack(anchor="w", padx=14, pady=(4, 4))

        # Actions
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(12, 4))

        self.load_btn = ctk.CTkButton(
            btn_row,
            text="Load",
            font=font(13, "bold"),
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=self._load,
            height=36,
        )
        self.load_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.unload_btn = ctk.CTkButton(
            btn_row,
            text="Unload",
            font=font(13),
            fg_color=COLORS["bg_elevated"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            command=self._unload,
            height=36,
        )
        self.unload_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.status_lbl = ctk.CTkLabel(
            self,
            text="Engine: unloaded",
            font=font(11),
            text_color=COLORS["text_muted"],
            wraplength=260,
            justify="left",
        )
        self.status_lbl.pack(anchor="w", padx=14, pady=(8, 16))

    def _load(self) -> None:
        if self._load_blocked:
            return
        if self.on_load:
            self.on_load()

    def _unload(self) -> None:
        if self.on_unload:
            self.on_unload()

    def set_airllm_enabled(self, enabled: bool) -> None:
        self._airllm_enabled = enabled
        self._refresh_memory_gate()

    def set_model(self, info: Optional[ModelInfo], library_status: str = "") -> None:
        self._info = info
        if info is None:
            self.title_lbl.configure(text="No model selected")
            self.badge.configure(text="—")
            self._set_stats("Select a model from the library.")
            self.diagram.draw_placeholder()
            self.mem_lbl.configure(text="")
            self.warn_lbl.configure(text="")
            self._load_blocked = False
            self.load_btn.configure(state="normal", text="Load")
            return

        self.title_lbl.configure(text=info.display_name or info.repo_id)
        kind = "MoE" if info.is_moe else "Dense"
        self.badge.configure(text=f"{kind}  ·  {info.architecture}")

        lines = []
        lines.append(f"Repo: {info.repo_id}")
        if info.total_params is not None:
            lines.append(f"Parameters: {format_params(info.total_params)}")
        if info.num_layers is not None:
            lines.append(f"Layers: {info.num_layers}")
        if info.hidden_size:
            lines.append(f"Hidden size: {info.hidden_size}")
        if info.is_moe:
            lines.append(f"Experts: {info.num_experts or '?'}")
            lines.append(f"Active / token: {info.experts_per_token or '?'}")
            if info.shared_experts:
                lines.append(f"Shared experts: {info.shared_experts}")
        full_gb = estimate_full_load_memory_gb(info)
        lines.append(f"Full load (no AirLLM): ~{full_gb:.0f} GB")
        if info.estimated_disk_gb:
            lines.append(f"Disk (weights est.): ~{info.estimated_disk_gb:.1f} GB")
        if library_status:
            lines.append(f"Library status: {library_status}")
        if info.notes:
            lines.append("")
            lines.append("Notes:")
            for n in info.notes:
                lines.append(f"  • {n}")

        self._set_stats("\n".join(lines))
        self.diagram.render(info)

        if info.estimated_airllm_memory_gb is not None:
            self.mem_lbl.configure(
                text=(
                    f"Est. AirLLM peak: ~{info.estimated_airllm_memory_gb:.1f} GB\n"
                    f"Full load (AirLLM off): ~{full_gb:.0f} GB\n"
                    f"(streams 1 layer"
                    f"{' / active experts' if info.is_moe else ''}"
                    f" — not the full model)"
                )
            )
        else:
            self.mem_lbl.configure(text=f"Full load (AirLLM off): ~{full_gb:.0f} GB")

        self._refresh_memory_gate()

    def _refresh_memory_gate(self) -> None:
        """Block Load when AirLLM is off and model won't fit in RAM."""
        info = self._info
        if info is None:
            self.warn_lbl.configure(text="")
            self._load_blocked = False
            self.load_btn.configure(state="normal", text="Load")
            return

        if self._airllm_enabled:
            self.warn_lbl.configure(text="")
            self._load_blocked = False
            self.load_btn.configure(
                state="normal",
                text="Load",
                fg_color=COLORS["accent_dim"],
                hover_color=COLORS["accent"],
            )
            return

        ok, msg, need, total = can_load_without_airllm(info)
        if ok:
            self.warn_lbl.configure(
                text=f"AirLLM off — full load ~{need:.0f} GB (fits on {total:.0f} GB).",
                text_color=COLORS["text_muted"],
            )
            self._load_blocked = False
            self.load_btn.configure(
                state="normal",
                text="Load",
                fg_color=COLORS["accent_dim"],
                hover_color=COLORS["accent"],
            )
        else:
            self.warn_lbl.configure(
                text=(
                    f"⚠ Too large without AirLLM\n"
                    f"Needs ~{need:.0f} GB · Mac has {total:.0f} GB\n"
                    f"Turn AirLLM On to load."
                ),
                text_color=COLORS["danger"],
            )
            self._load_blocked = True
            self.load_btn.configure(
                state="disabled",
                text="Needs AirLLM",
                fg_color=COLORS["border"],
                hover_color=COLORS["border"],
            )

    def set_engine_status(self, text: str) -> None:
        self.status_lbl.configure(text=text)

    def _set_stats(self, text: str) -> None:
        self.stats.configure(state="normal")
        self.stats.delete("1.0", "end")
        self.stats.insert("1.0", text)
        self.stats.configure(state="disabled")
