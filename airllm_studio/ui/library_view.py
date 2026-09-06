"""Model library: featured one-click downloads, HF search, local library.

Performance notes
-----------------
CustomTkinter is expensive per-widget. We use compact one-row cards and
build the catalog progressively (1–2 rows per idle tick) so the app stays
responsive. Local library only lists downloads in progress / finished —
not every catalog entry again.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import customtkinter as ctk

from airllm_studio.core.catalog import FEATURED_MODELS, SIZE_TIERS, CatalogModel, format_eta
from airllm_studio.core.model_manager import LibraryEntry
from airllm_studio.core.model_meta import format_params
from airllm_studio.core.variants import ModelVariant, default_variant, variant_dropdown_labels
from airllm_studio.ui.theme import COLORS, font

# Small batches + enough delay that macOS can redraw between ticks
_ROWS_PER_TICK = 3
_TICK_MS = 30
# First paint shows this many rows synchronously so the page isn't empty
_SYNC_FIRST_ROWS = 4


class LibraryView(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_search: Callable[[str], None],
        on_add_repo: Callable[[str], None],
        on_prepare: Callable[..., None],
        on_select: Callable[[str], None],
        on_fetch_info: Callable[[str], None],
        on_pause: Optional[Callable[[str], None]] = None,
        on_resume: Optional[Callable[[str], None]] = None,
        on_delete_download: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=COLORS["bg"], corner_radius=0, **kwargs)
        self.on_search = on_search
        self.on_add_repo = on_add_repo
        self.on_prepare = on_prepare
        self.on_select = on_select
        self.on_fetch_info = on_fetch_info
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_delete_download = on_delete_download
        self._active_repo: Optional[str] = None
        self._card_progress: dict = {}  # base_repo_id -> (bar, status_lbl)
        self._variant_maps: Dict[str, Dict[str, ModelVariant]] = {}
        self._variant_menus: Dict[str, ctk.CTkOptionMenu] = {}
        self._disk_hint: Dict[str, ctk.CTkLabel] = {}
        self._featured_built = False
        self._featured_building = False
        self._build_queue: List[Tuple[str, object]] = []
        self._loading_lbl: Optional[ctk.CTkLabel] = None
        self._lib_cache_key: Optional[tuple] = None
        self._paint_token = 0
        self._build_shell()

    def _build_shell(self) -> None:
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_elevated"], height=48, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header,
            text="Model Library",
            font=font(14, "bold"),
            text_color=COLORS["text"],
        ).pack(side="left", padx=16, pady=12)

        prog_wrap = ctk.CTkFrame(self, fg_color="transparent")
        prog_wrap.pack(fill="x", padx=16, pady=(8, 0))

        self.progress_lbl = ctk.CTkLabel(
            prog_wrap,
            text="Select a model for one-click download, or search Hugging Face.",
            font=font(12),
            text_color=COLORS["accent"],
            anchor="w",
        )
        self.progress_lbl.pack(fill="x")

        self.eta_lbl = ctk.CTkLabel(
            prog_wrap,
            text="",
            font=font(11),
            text_color=COLORS["text_muted"],
            anchor="w",
        )
        self.eta_lbl.pack(fill="x", pady=(2, 0))

        self.progress = ctk.CTkProgressBar(prog_wrap, height=8)
        self.progress.pack(fill="x", pady=(4, 4))
        self.progress.set(0)

        # Transport controls for active download
        ctl = ctk.CTkFrame(prog_wrap, fg_color="transparent")
        ctl.pack(fill="x", pady=(2, 4))
        self.pause_btn = ctk.CTkButton(
            ctl,
            text="⏸ Pause",
            width=90,
            height=28,
            font=font(11),
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["warning"],
            command=self._pause_active,
        )
        self.pause_btn.pack(side="left", padx=(0, 6))
        self.resume_btn = ctk.CTkButton(
            ctl,
            text="▶ Resume",
            width=90,
            height=28,
            font=font(11),
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=self._resume_active,
        )
        self.resume_btn.pack(side="left", padx=(0, 6))
        self.delete_btn = ctk.CTkButton(
            ctl,
            text="🗑 Delete download",
            width=130,
            height=28,
            font=font(11),
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["danger"],
            command=self._delete_active,
        )
        self.delete_btn.pack(side="left")

        body = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=6)
        self._body = body

        # Active / paused jobs list
        ctk.CTkLabel(
            body,
            text="DOWNLOADS  ·  pause / resume survives app quit & reboot",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=8, pady=(2, 4))
        self.jobs_frame = ctk.CTkFrame(body, fg_color=COLORS["bg_elevated"], corner_radius=8)
        self.jobs_frame.pack(fill="x", padx=4, pady=(0, 8))
        ctk.CTkLabel(
            self.jobs_frame,
            text="No active or paused downloads.",
            font=font(11),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=10, pady=8)

        ctk.CTkLabel(
            body,
            text="ONE-CLICK  ·  DENSE (MLX) + MoE (EXPERT STREAMING)",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=8, pady=(2, 0))
        ctk.CTkLabel(
            body,
            text="Dense → AirLLM MLX. MoE (Mixtral / DeepSeek / Qwen-MoE) → per-expert streaming on MPS/CPU. ★ = recommended.",
            font=font(11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.featured_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.featured_frame.pack(fill="x", padx=2)

        self._loading_lbl = ctk.CTkLabel(
            self.featured_frame,
            text="Loading model catalog…",
            font=font(12),
            text_color=COLORS["text_muted"],
        )
        self._loading_lbl.pack(anchor="w", padx=8, pady=8)

        ctk.CTkLabel(
            body,
            text="HUGGING FACE SEARCH",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=8, pady=(12, 4))

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 6))

        self.repo_entry = ctk.CTkEntry(
            row,
            placeholder_text="Repo ID or search — e.g. Qwen/Qwen2.5-72B-Instruct",
            font=font(13),
            height=34,
        )
        self.repo_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.repo_entry.bind("<Return>", lambda e: self._search())

        ctk.CTkButton(
            row,
            text="Search",
            width=72,
            height=34,
            font=font(12),
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["border"],
            command=self._search,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            row,
            text="Add",
            width=64,
            height=34,
            font=font(12, "bold"),
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=self._add,
        ).pack(side="left")

        self.search_frame = ctk.CTkFrame(body, fg_color=COLORS["bg_elevated"], corner_radius=8)
        self.search_frame.pack(fill="x", padx=8, pady=(0, 6))

        ctk.CTkLabel(
            body,
            text="YOUR DOWNLOADS",
            font=font(11, "bold"),
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=8, pady=(10, 4))

        self.list_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.list_frame.pack(fill="x", padx=2, pady=(0, 12))

    # ── Progressive featured catalog ───────────────────────

    def ensure_featured_loaded(self) -> None:
        if self._featured_built or self._featured_building:
            return
        self._featured_building = True
        self._paint_token += 1
        token = self._paint_token
        self._build_queue = self._plan_featured_queue()
        total = sum(1 for k, _ in self._build_queue if k == "card")
        if self._loading_lbl is not None:
            self._loading_lbl.configure(text=f"Loading catalog 0/{total}…")
        # Immediate first rows so library feels instant; rest progressive
        self._paint_featured_batch(token, max_rows=_SYNC_FIRST_ROWS)
        if self._build_queue:
            self.after(_TICK_MS, lambda: self._paint_featured_tick(token))
        else:
            self._featured_built = True
            self._featured_building = False

    def _plan_featured_queue(self) -> List[Tuple[str, object]]:
        by_tier: dict = {}
        for m in FEATURED_MODELS:
            by_tier.setdefault(m.size_tier, []).append(m)
        queue: List[Tuple[str, object]] = []
        for tier, subtitle in SIZE_TIERS:
            models = by_tier.get(tier) or []
            if not models:
                continue
            queue.append(("tier", (tier, subtitle)))
            for m in models:
                queue.append(("card", m))
        return queue

    def _paint_featured_batch(self, token: int, max_rows: int) -> None:
        if token != self._paint_token or not self.winfo_exists():
            return
        rows = 0
        while self._build_queue and rows < max_rows:
            kind, payload = self._build_queue.pop(0)
            if kind == "tier":
                tier, subtitle = payload  # type: ignore[misc]
                self._add_tier_header(tier, subtitle)
            else:
                self._featured_row(payload)  # type: ignore[arg-type]
                rows += 1

        if self._loading_lbl is not None:
            done = len(self._card_progress)
            total = len(FEATURED_MODELS)
            if self._build_queue:
                self._loading_lbl.configure(text=f"Loading catalog {done}/{total}…")
            else:
                self._loading_lbl.destroy()
                self._loading_lbl = None

    def _paint_featured_tick(self, token: int) -> None:
        if token != self._paint_token or not self.winfo_exists():
            return
        self._paint_featured_batch(token, max_rows=_ROWS_PER_TICK)
        if self._build_queue:
            t = token
            self.after(_TICK_MS, lambda: self._paint_featured_tick(t))
        else:
            self._featured_built = True
            self._featured_building = False

    def _add_tier_header(self, tier: str, subtitle: str) -> None:
        head = ctk.CTkFrame(self.featured_frame, fg_color="transparent")
        if self._loading_lbl is not None:
            head.pack(fill="x", padx=4, pady=(12, 2), before=self._loading_lbl)
        else:
            head.pack(fill="x", padx=4, pady=(12, 2))
        ctk.CTkLabel(
            head, text=tier, font=font(13, "bold"), text_color=COLORS["accent"]
        ).pack(side="left")
        ctk.CTkLabel(
            head,
            text=f"  —  {subtitle}",
            font=font(11),
            text_color=COLORS["text_dim"],
        ).pack(side="left")

    def _featured_row(self, m: CatalogModel) -> None:
        """Compact row with precision/quantization dropdown."""
        border = COLORS["accent"] if m.recommended else COLORS["border"]
        row = ctk.CTkFrame(
            self.featured_frame,
            fg_color=COLORS["bg_elevated"],
            border_width=2 if m.recommended else 1,
            border_color=border,
            corner_radius=8,
        )
        if self._loading_lbl is not None:
            row.pack(fill="x", pady=3, padx=2, before=self._loading_lbl)
        else:
            row.pack(fill="x", pady=3, padx=2)

        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(10, 6), pady=8)

        title = ("★ " if m.recommended else "") + m.label
        ctk.CTkLabel(
            left,
            text=title,
            font=font(13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).pack(fill="x")

        bits = [
            m.role_label(),
            "MoE" if m.is_moe else "Dense",
            format_params(m.params_b),
            f"~{m.disk_gb or m.params_b * 2:.0f} GB full",
        ]
        if m.gated:
            bits.append("Gated")
        sub = " · ".join(bits)
        if m.blurb:
            sub += f"  —  {m.blurb}"
        ctk.CTkLabel(
            left,
            text=sub,
            font=font(10),
            text_color=COLORS["text_muted"],
            anchor="w",
            wraplength=420,
            justify="left",
        ).pack(fill="x")

        # Precision / quant dropdown
        labels, mapping = variant_dropdown_labels(m.repo_id)
        self._variant_maps[m.repo_id] = mapping

        prec_row = ctk.CTkFrame(left, fg_color="transparent")
        prec_row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(
            prec_row,
            text="Precision",
            font=font(10),
            text_color=COLORS["text_dim"],
        ).pack(side="left", padx=(0, 6))

        menu = ctk.CTkOptionMenu(
            prec_row,
            values=labels,
            width=200,
            height=26,
            font=font(11),
            fg_color=COLORS["bg_panel"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["accent_dim"],
            dropdown_fg_color=COLORS["bg_elevated"],
            command=lambda _choice, rid=m.repo_id: self._on_variant_change(rid),
        )
        menu.pack(side="left")
        menu.set(labels[0])
        self._variant_menus[m.repo_id] = menu

        hint = ctk.CTkLabel(
            prec_row,
            text=self._variant_disk_hint(m, mapping[labels[0]]),
            font=font(10),
            text_color=COLORS["text_muted"],
        )
        hint.pack(side="left", padx=(8, 0))
        self._disk_hint[m.repo_id] = hint

        bar = ctk.CTkProgressBar(left, height=4, width=200)
        bar.pack(anchor="w", pady=(4, 0))
        bar.set(0)
        status = ctk.CTkLabel(
            left, text="", font=font(9), text_color=COLORS["text_dim"], anchor="w"
        )
        status.pack(fill="x")
        self._card_progress[m.repo_id] = (bar, status, status)

        btns = ctk.CTkFrame(row, fg_color="transparent")
        btns.pack(side="right", padx=10, pady=8)
        ctk.CTkButton(
            btns,
            text="⬇ Download",
            width=110,
            height=30,
            font=font(12, "bold"),
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=lambda rid=m.repo_id: self._one_click(rid),
        ).pack(pady=(0, 4))
        ctk.CTkButton(
            btns,
            text="Select",
            width=110,
            height=26,
            font=font(11),
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["border"],
            command=lambda rid=m.repo_id: self._select_with_variant(rid),
        ).pack()

    def _variant_disk_hint(self, m: CatalogModel, v: ModelVariant) -> str:
        base = m.disk_gb or m.params_b * 2.0
        dl = v.download_gb_for(base)
        final = v.final_gb_for(base)
        if v.source == "airllm" and abs(dl - final) > 0.5:
            # AirLLM still downloads full weights, then compresses
            return f"↓~{dl:.0f} GB then ~{final:.0f} GB · {v.precision}"
        return f"↓~{dl:.0f} GB · {v.precision}"

    def _get_variant(self, base_repo_id: str) -> ModelVariant:
        menu = self._variant_menus.get(base_repo_id)
        mapping = self._variant_maps.get(base_repo_id) or {}
        if menu is None or not mapping:
            return default_variant(base_repo_id)
        label = menu.get()
        return mapping.get(label) or default_variant(base_repo_id)

    def _on_variant_change(self, base_repo_id: str) -> None:
        v = self._get_variant(base_repo_id)
        # Update disk hint
        from airllm_studio.core.catalog import get_catalog_model

        cat = get_catalog_model(base_repo_id)
        if cat and base_repo_id in getattr(self, "_disk_hint", {}):
            self._disk_hint[base_repo_id].configure(
                text=self._variant_disk_hint(cat, v)
            )

    # ── Actions ────────────────────────────────────────────

    def _one_click(self, base_repo_id: str) -> None:
        v = self._get_variant(base_repo_id)
        self._active_repo = base_repo_id
        self.on_prepare(
            v.repo_id,
            compression=v.compression,
            variant_label=v.label,
            base_repo_id=base_repo_id,
        )

    def _select_with_variant(self, base_repo_id: str) -> None:
        v = self._get_variant(base_repo_id)
        # Select the concrete repo that will be loaded
        self.on_select(v.repo_id)

    def _pause_active(self) -> None:
        rid = self._active_repo
        if rid and self.on_pause:
            self.on_pause(rid)

    def _resume_active(self) -> None:
        rid = self._active_repo
        if rid and self.on_resume:
            self.on_resume(rid)

    def _delete_active(self) -> None:
        rid = self._active_repo
        if rid and self.on_delete_download:
            self.on_delete_download(rid)

    def refresh_jobs(self, jobs: list) -> None:
        """Render paused / active download jobs with controls."""
        for w in self.jobs_frame.winfo_children():
            w.destroy()
        # Only show interesting jobs
        jobs = [
            j
            for j in jobs
            if getattr(j, "status", "")
            in ("downloading", "preparing", "paused", "error", "queued")
        ]
        if not jobs:
            ctk.CTkLabel(
                self.jobs_frame,
                text="No active or paused downloads. Progress is saved if you quit.",
                font=font(11),
                text_color=COLORS["text_dim"],
            ).pack(anchor="w", padx=10, pady=8)
            return

        for j in jobs:
            row = ctk.CTkFrame(self.jobs_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=4)
            title = j.repo_id
            if j.variant_label:
                title += f"  ·  {j.variant_label}"
            pct = ""
            if j.total_gb > 0:
                pct = f"  ·  {j.downloaded_gb:.1f}/{j.total_gb:.1f} GB"
            ctk.CTkLabel(
                row,
                text=f"{j.status.upper()}{pct}\n{title}",
                font=font(11),
                text_color=COLORS["text"],
                justify="left",
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            btns = ctk.CTkFrame(row, fg_color="transparent")
            btns.pack(side="right")
            rid = j.repo_id

            if j.status in ("downloading", "preparing"):
                ctk.CTkButton(
                    btns,
                    text="Pause",
                    width=70,
                    height=26,
                    font=font(11),
                    command=lambda r=rid: self.on_pause and self.on_pause(r),
                ).pack(side="left", padx=2)
            if j.status in ("paused", "error", "queued"):
                ctk.CTkButton(
                    btns,
                    text="Resume",
                    width=70,
                    height=26,
                    font=font(11, "bold"),
                    fg_color=COLORS["accent_dim"],
                    hover_color=COLORS["accent"],
                    text_color=COLORS["bg"],
                    command=lambda r=rid: self.on_resume and self.on_resume(r),
                ).pack(side="left", padx=2)
            ctk.CTkButton(
                btns,
                text="Delete",
                width=70,
                height=26,
                font=font(11),
                fg_color=COLORS["bg_panel"],
                hover_color=COLORS["danger"],
                command=lambda r=rid: self.on_delete_download and self.on_delete_download(r),
            ).pack(side="left", padx=2)
            # Track last interacted job as active for top buttons
            ctk.CTkButton(
                btns,
                text="Focus",
                width=60,
                height=26,
                font=font(10),
                fg_color="transparent",
                command=lambda r=rid: setattr(self, "_active_repo", r),
            ).pack(side="left", padx=2)

    def _search(self) -> None:
        q = self.repo_entry.get().strip()
        if q:
            self.on_search(q)

    def _add(self) -> None:
        q = self.repo_entry.get().strip()
        if q and "/" in q:
            self.on_add_repo(q)
        elif q:
            self.on_search(q)

    def set_progress(
        self,
        message: str,
        value: float = -1,
        meta: Optional[dict] = None,
        repo_id: Optional[str] = None,
    ) -> None:
        meta = meta or {}
        self.progress_lbl.configure(text=message)

        parts = []
        eta = meta.get("eta_seconds")
        if eta is not None and value is not None and 0 <= value < 1:
            parts.append(f"ETA {format_eta(eta)}")
        pred = meta.get("predicted_eta_seconds")
        if pred is not None and (eta is None or value < 0.05):
            parts.append(f"Predicted ~{format_eta(pred)} @ 50 Mbps")
        speed = meta.get("speed_mbps")
        if speed:
            parts.append(f"{speed:.0f} Mbps")
        dl = meta.get("downloaded_gb")
        tot = meta.get("total_gb")
        if dl is not None and tot is not None:
            parts.append(f"{dl:.1f} / {tot:.0f} GB")
        elif tot is not None and value >= 0:
            parts.append(f"~{tot:.0f} GB total")
        elapsed = meta.get("elapsed_seconds")
        if elapsed and value >= 1:
            parts.append(f"took {format_eta(elapsed)}")
        self.eta_lbl.configure(text="  ·  ".join(parts) if parts else "")

        if value < 0:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(min(max(value, 0.0), 1.0))

        rid = repo_id or self._active_repo
        if rid and rid in self._card_progress:
            bar, eta_lbl, status_lbl = self._card_progress[rid]
            if value >= 0:
                bar.set(min(max(value, 0.0), 1.0))
            status_lbl.configure(
                text=(message[:80] + (" · " + " · ".join(parts) if parts else ""))[:120]
            )

    def show_search_results(self, results: List[dict]) -> None:
        for w in self.search_frame.winfo_children():
            w.destroy()
        if not results:
            ctk.CTkLabel(
                self.search_frame,
                text="No results (check network / query).",
                text_color=COLORS["text_dim"],
                font=font(12),
            ).pack(anchor="w", padx=12, pady=8)
            return
        for r in results[:20]:
            repo = r["repo_id"]
            row = ctk.CTkFrame(self.search_frame, fg_color="transparent")
            row.pack(fill="x", pady=2, padx=8)
            info = repo
            if r.get("downloads"):
                info += f"  ·  ↓{r['downloads']:,}"
            ctk.CTkLabel(
                row, text=info, font=font(12), text_color=COLORS["text"], anchor="w"
            ).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row,
                text="Download",
                width=84,
                height=26,
                font=font(11, "bold"),
                fg_color=COLORS["accent_dim"],
                hover_color=COLORS["accent"],
                text_color=COLORS["bg"],
                command=lambda rid=repo: self._one_click(rid),
            ).pack(side="right", padx=(4, 0))

    def refresh_library(
        self,
        entries: List[LibraryEntry],
        selected: Optional[str] = None,
    ) -> None:
        # Only downloads / in-progress — catalog lives in the featured section
        entries = [
            e
            for e in entries
            if e.status not in ("listed",) or e.local_path or e.shards_path
        ]
        key = tuple((e.repo_id, e.status) for e in entries) + (selected,)
        if key == self._lib_cache_key and self.list_frame.winfo_children():
            return
        self._lib_cache_key = key

        for w in self.list_frame.winfo_children():
            w.destroy()

        if not entries:
            ctk.CTkLabel(
                self.list_frame,
                text="No downloads yet — use One-Click above.",
                text_color=COLORS["text_dim"],
                font=font(12),
            ).pack(anchor="w", padx=8, pady=8)
            return

        for e in entries:
            info = e.model_info()
            title = info.display_name if info else e.repo_id.split("/")[-1]
            is_sel = e.repo_id == selected
            card = ctk.CTkFrame(
                self.list_frame,
                fg_color=COLORS["bg_panel"] if is_sel else COLORS["bg_elevated"],
                border_width=1,
                border_color=COLORS["accent"] if is_sel else COLORS["border"],
                corner_radius=6,
            )
            card.pack(fill="x", pady=2, padx=2)

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=6)
            ctk.CTkLabel(
                top, text=title, font=font(12, "bold"), text_color=COLORS["text"]
            ).pack(side="left")
            status_color = {
                "ready": COLORS["success"],
                "loaded": COLORS["accent"],
                "error": COLORS["danger"],
                "preparing": COLORS["warning"],
                "downloading": COLORS["warning"],
            }.get(e.status, COLORS["text_dim"])
            ctk.CTkLabel(
                top,
                text=e.status.upper(),
                font=font(10, "bold"),
                text_color=status_color,
            ).pack(side="right")

            ctk.CTkLabel(
                card,
                text=e.repo_id,
                font=font(10),
                text_color=COLORS["text_muted"],
                anchor="w",
            ).pack(fill="x", padx=8)

            actions = ctk.CTkFrame(card, fg_color="transparent")
            actions.pack(fill="x", padx=8, pady=(2, 6))
            ctk.CTkButton(
                actions,
                text="Select",
                width=64,
                height=24,
                font=font(11),
                command=lambda rid=e.repo_id: self.on_select(rid),
            ).pack(side="left", padx=(0, 4))
            if e.status not in ("ready", "loaded"):
                ctk.CTkButton(
                    actions,
                    text="Download",
                    width=80,
                    height=24,
                    font=font(11, "bold"),
                    fg_color=COLORS["accent_dim"],
                    hover_color=COLORS["accent"],
                    text_color=COLORS["bg"],
                    command=lambda rid=e.repo_id: self._one_click(rid),
                ).pack(side="left")
