"""Main application window — wires UI to core services."""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from airllm_studio import __app_name__, __version__
from airllm_studio.billing import (
    LicenseKeyRefused,
    PurchaseError,
    allows,
    clamp_generation,
    make_store,
    paywall_payload,
    sanitize_settings,
)
from airllm_studio.core.chat_store import ChatStore, Message
from airllm_studio.core.config import get_config, update_config
from airllm_studio.core.engine import (
    EngineEvent,
    GenerationConfig,
    InferenceEngine,
    extract_tool_calls,
)
from airllm_studio.core.tools_policy import (
    CONSERVATIVE_TOOL_PROMPT,
    should_offer_tools,
)
from airllm_studio.core.model_manager import ModelManager
from airllm_studio.core.model_meta import ModelInfo
from airllm_studio.tools.registry import get_default_registry
from airllm_studio.ui.chat_view import ChatView
from airllm_studio.ui.library_view import LibraryView
from airllm_studio.ui.model_panel import ModelPanel
from airllm_studio.ui.settings_view import SettingsView
from airllm_studio.ui.sidebar import Sidebar
from airllm_studio.ui.theme import COLORS, apply_theme, font


class AirLLMStudioApp(ctk.CTk):
    def __init__(self) -> None:
        # Theme must be set before the first CTk window is created
        apply_theme()
        super().__init__()
        self.cfg = get_config()
        self.cfg.ensure_dirs()

        self.title(f"{__app_name__}  ·  v{__version__}")
        self.geometry("1280x820")
        self.minsize(1000, 640)
        self.configure(fg_color=COLORS["bg"])
        # Force a first paint on macOS Aqua
        try:
            self.update_idletasks()
        except Exception:
            pass

        # Services
        self.store = ChatStore()
        self.models = ModelManager()
        self.engine = InferenceEngine()
        self.iap = make_store(self.cfg.data_dir)
        self.tools = get_default_registry(
            web_search=bool(
                allows("web_search", self.iap.is_entitled())
                and self.cfg.web_search_enabled
                and self.cfg.tools_enabled
            )
        )

        self._event_q: queue.Queue = queue.Queue()
        self._selected_repo: Optional[str] = None
        self._active_chat_id: Optional[str] = None
        self._busy = False

        self._build()
        self._bootstrap_chat()
        self._poll_events()

        self._set_status(self._backend_status_line())

    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(
            self,
            on_nav=self._nav,
            on_new_chat=self._new_chat,
            on_select_chat=self._select_chat,
            on_delete_chat=self._delete_chat,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Center stack
        self.center = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.center.grid(row=0, column=1, sticky="nsew")
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        self.chat_view = ChatView(
            self.center,
            on_send=self._on_send,
            on_stop=self._on_stop,
            on_model_pick=self._on_model_dropdown,
            on_airllm_toggle=self._on_airllm_toggle,
            airllm_enabled=self.cfg.airllm_enabled and not self.cfg.demo_mode,
        )
        # Library + settings are built lazily on first open (big UI = beach ball if eager)
        self.library_view: Optional[LibraryView] = None
        self.settings_view: Optional[SettingsView] = None
        self._catalog_seeded = False

        self.chat_view.grid(row=0, column=0, sticky="nsew")
        self._show_view("chat")

        # Right panel
        self.model_panel = ModelPanel(
            self,
            on_load=self._on_load_model,
            on_unload=self._on_unload_model,
        )
        self.model_panel.configure(width=300)
        self.model_panel.grid(row=0, column=2, sticky="nsew")
        self.model_panel.grid_propagate(False)

        # Bottom status
        self.status_bar = ctk.CTkLabel(
            self,
            text="",
            font=font(11),
            text_color=COLORS["text_dim"],
            anchor="w",
            fg_color=COLORS["bg_elevated"],
            height=26,
        )
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=0)

    def _ensure_library(self) -> LibraryView:
        if self.library_view is None:
            self.library_view = LibraryView(
                self.center,
                on_search=self._on_hf_search,
                on_add_repo=self._on_add_repo,
                on_prepare=self._on_prepare,
                on_select=self._on_select_model,
                on_fetch_info=self._on_fetch_info,
                on_pause=self._on_pause_download,
                on_resume=self._on_resume_download,
                on_delete_download=self._on_delete_download,
            )
            self.library_view.grid(row=0, column=0, sticky="nsew")
            self.library_view.grid_remove()
        return self.library_view

    def _lib(self) -> LibraryView:
        """Always-valid library view (created on demand for progress UI)."""
        return self._ensure_library()

    def _ensure_settings(self) -> SettingsView:
        if self.settings_view is None:
            self.settings_view = SettingsView(
                self.center,
                config=self.cfg,
                on_save=self._on_save_settings,
                billing=paywall_payload(self.iap),
                on_purchase=self._on_purchase,
                on_restore=self._on_restore_purchases,
                on_open_legal=self._on_open_legal,
            )
            self.settings_view.grid(row=0, column=0, sticky="nsew")
            self.settings_view.grid_remove()
        return self.settings_view

    def _show_view(self, name: str) -> None:
        self.chat_view.grid_remove()
        if self.library_view is not None:
            self.library_view.grid_remove()
        if self.settings_view is not None:
            self.settings_view.grid_remove()

        if name == "chat":
            self.chat_view.grid()
            self._refresh_model_dropdown()
        elif name == "library":
            lib = self._ensure_library()
            lib.grid()
            # Cheap seed once; progressive cards; deferred list paint
            if not self._catalog_seeded:
                self.models.ensure_catalog_seeded()
                self._catalog_seeded = True
            lib.ensure_featured_loaded()
            lib.refresh_jobs(self.models.jobs.list_jobs())
            self.after(
                10,
                lambda: lib.refresh_library(
                    self.models.list_models(), self._selected_repo
                ),
            )
        elif name == "settings":
            self._ensure_settings().grid()
        self.sidebar.set_active_nav(name)

    def _refresh_model_dropdown(self) -> None:
        """Populate chat header with prepared / downloaded models."""
        entries = self.models.list_downloaded_models()
        options = []
        for e in entries:
            info = e.model_info()
            name = (info.display_name if info else None) or e.repo_id.split("/")[-1]
            tag = e.status
            if e.status == "ready":
                tag = "ready"
            elif e.status == "loaded":
                tag = "loaded"
            label = f"{name}  ·  {tag}"
            options.append((e.repo_id, label))
        self.chat_view.set_model_options(options, selected_repo=self._selected_repo)

    def _on_model_dropdown(self, repo_id: str) -> None:
        """User picked a model from the top chat dropdown."""
        self._on_select_model(repo_id)
        entry = self.models.get(repo_id)
        if entry and entry.status in ("ready", "loaded"):
            # Auto-load only if memory gate allows
            if self.engine.loaded_repo != repo_id:
                if self._memory_allows_load(show_warning=True):
                    self._on_load_model()
        else:
            self._set_status(f"Selected model: {repo_id}")

    def _backend_status_line(self) -> str:
        if self.cfg.demo_mode:
            mode = "demo (forced)"
        elif not self.cfg.airllm_enabled:
            mode = "demo (AirLLM off)"
        elif not self.engine._airllm_available():
            mode = "demo (airllm not installed)"
        else:
            mode = "AirLLM on"
        return f"Ready · {mode} · data={self.cfg.data_dir}"

    def _on_airllm_toggle(self, enabled: bool) -> None:
        """Chat-header switch: use AirLLM when loading the selected model."""
        from airllm_studio.core.config import update_config

        self.cfg = update_config(airllm_enabled=enabled)
        # Clear force-demo when user turns AirLLM on from the header
        if enabled and self.cfg.demo_mode:
            self.cfg = update_config(demo_mode=False)
        self.engine.cfg = self.cfg
        self.models.cfg = self.cfg
        self.model_panel.set_airllm_enabled(enabled and not self.cfg.demo_mode)

        was_loaded = self.engine.loaded_repo
        if was_loaded:
            # Turning AirLLM off under a large model: unload and refuse reload
            if not enabled and not self._memory_allows_load(show_warning=True):
                self.engine.unload()
                self.model_panel.set_engine_status(
                    "Engine: unloaded — model too large without AirLLM"
                )
                return
            self.engine.unload()
            self.model_panel.set_engine_status("Engine: unloaded — reloading…")
            self._set_status(
                f"AirLLM {'enabled' if enabled else 'disabled'} — reloading {was_loaded}…"
            )
            self._on_load_model()
        else:
            if not enabled and not self._memory_allows_load(show_warning=True):
                return
            tag = "ON — layer/expert streaming for next load" if enabled else "OFF — full-model load"
            if enabled and not self.engine._airllm_available():
                tag = "ON but airllm package not installed — install for real inference"
            self._set_status(f"AirLLM {tag}")
            self.model_panel.set_engine_status(
                f"AirLLM: {'on' if enabled and self.engine._airllm_available() else 'off'}"
            )

    def _selected_model_info(self) -> Optional[ModelInfo]:
        if not self._selected_repo:
            return None
        entry = self.models.get(self._selected_repo)
        if entry and entry.model_info():
            return entry.model_info()
        from airllm_studio.core.catalog import get_catalog_model

        cat = get_catalog_model(self._selected_repo)
        if cat:
            return cat.to_model_info()
        return ModelInfo(repo_id=self._selected_repo)

    def _memory_allows_load(self, show_warning: bool = True) -> bool:
        """
        If AirLLM is off, require the full model to fit in system memory.
        Always allowed when AirLLM is enabled (layer streaming).
        """
        # AirLLM On → streaming; any size allowed
        if self.cfg.airllm_enabled and not self.cfg.demo_mode:
            return True

        info = self._selected_model_info()
        if info is None:
            return True

        from airllm_studio.core.model_meta import can_load_without_airllm
        from tkinter import messagebox

        ok, msg, need, have = can_load_without_airllm(info)
        if ok:
            return True

        if show_warning:
            self._set_status(f"⚠ Load blocked: {msg}")
            self.model_panel.set_airllm_enabled(False)
            try:
                messagebox.showwarning(
                    "AirLLM required",
                    (
                        f"{info.display_name or info.repo_id}\n\n"
                        f"{msg}\n\n"
                        "Enable the AirLLM toggle (right of the model selector) "
                        "to load this model."
                    ),
                    parent=self,
                )
            except Exception:
                pass
        return False

    def _nav(self, key: str) -> None:
        self._show_view(key)

    def _set_status(self, text: str) -> None:
        self.status_bar.configure(text=f"  {text}")

    def _bootstrap_chat(self) -> None:
        # Don't block first paint with full catalog disk seed — do it idle
        chat = self.store.ensure_default()
        self._active_chat_id = chat.id
        self._refresh_sidebar()
        self.chat_view.set_title(chat.title)
        self.chat_view.render_messages(chat.messages)

        ready = self.models.list_downloaded_models()
        models = ready or self.models.list_models()
        if models:
            self._on_select_model(models[0].repo_id)
        self._refresh_model_dropdown()
        self.after(100, self._idle_seed_catalog)

    def _idle_seed_catalog(self) -> None:
        if self._catalog_seeded:
            return
        self.models.ensure_catalog_seeded()
        self._catalog_seeded = True

    def _refresh_sidebar(self) -> None:
        self.sidebar.refresh_chats(self.store.list_chats(), self._active_chat_id)

    # ── Chats ──────────────────────────────────────────────

    def _new_chat(self) -> None:
        chat = self.store.create()
        self._active_chat_id = chat.id
        self._refresh_sidebar()
        self.chat_view.set_title(chat.title)
        self.chat_view.clear_messages()
        self._show_view("chat")

    def _select_chat(self, chat_id: str) -> None:
        chat = self.store.get(chat_id)
        if not chat:
            return
        self._active_chat_id = chat_id
        self._refresh_sidebar()
        self.chat_view.set_title(chat.title)
        self.chat_view.render_messages(chat.messages)
        self._show_view("chat")

    def _delete_chat(self, chat_id: str) -> None:
        self.store.delete(chat_id)
        if self._active_chat_id == chat_id:
            chat = self.store.ensure_default()
            self._active_chat_id = chat.id
            self.chat_view.set_title(chat.title)
            self.chat_view.render_messages(chat.messages)
        self._refresh_sidebar()

    # ── Models ─────────────────────────────────────────────

    def _on_select_model(self, repo_id: str) -> None:
        self._selected_repo = repo_id
        entry = self.models.get(repo_id)
        if entry is None:
            # Ensure catalog entry exists for one-click models
            from airllm_studio.core.catalog import get_catalog_model

            cat = get_catalog_model(repo_id)
            if cat:
                self.models.upsert_info(cat.to_model_info(), status="listed")
                entry = self.models.get(repo_id)
        info = entry.model_info() if entry else ModelInfo(repo_id=repo_id)
        status = entry.status if entry else ""
        self.model_panel.set_airllm_enabled(
            self.cfg.airllm_enabled and not self.cfg.demo_mode
        )
        self.model_panel.set_model(info, library_status=status)
        eng = self.engine.loaded_repo
        if eng:
            self.model_panel.set_engine_status(f"Engine: loaded · {eng}")
        else:
            self.model_panel.set_engine_status("Engine: unloaded")
        if self.library_view is not None:
            self._lib().refresh_library(
                self.models.list_models(), self._selected_repo
            )
        self._refresh_model_dropdown()
        # Soft warning when AirLLM is off and model is large
        if not (self.cfg.airllm_enabled and not self.cfg.demo_mode):
            from airllm_studio.core.model_meta import can_load_without_airllm

            ok, msg, _, _ = can_load_without_airllm(info)
            if not ok:
                self._set_status(f"⚠ {msg}")

    def _on_hf_search(self, query: str) -> None:
        if not allows("hf_search", self.iap.is_entitled()):
            self._set_status("Hugging Face search is a Pro library action")
            self._lib().set_progress("Hugging Face search requires Pro", -1)
            return
        self._lib().set_progress(f"Searching Hugging Face for “{query}”…", -1)
        self._set_status(f"Searching HF: {query}")

        def work() -> None:
            try:
                results = self.models.search_hf(query)
                self._event_q.put(("hf_search", results, None))
            except Exception as exc:
                self._event_q.put(("hf_search", [], str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_add_repo(self, repo_id: str) -> None:
        repo_id = repo_id.strip()
        if not repo_id:
            return
        self._lib().set_progress(f"Fetching {repo_id}…", -1)

        def work() -> None:
            try:
                info = self.models.fetch_remote_info(
                    repo_id,
                    progress=lambda m, p: self._event_q.put(("progress", m, p)),
                )
                self._event_q.put(("info_ready", repo_id, info))
            except Exception as exc:
                # Still add shell entry from name heuristics
                from airllm_studio.core.model_meta import parse_model_info

                info = parse_model_info(repo_id, {})
                self.models.upsert_info(info, status="listed")
                self._event_q.put(("info_ready", repo_id, info))
                self._event_q.put(
                    ("progress", f"Added with limited metadata ({exc})", 1.0)
                )

        threading.Thread(target=work, daemon=True).start()

    def _on_fetch_info(self, repo_id: str) -> None:
        self._on_add_repo(repo_id)

    def _on_prepare(
        self,
        repo_id: str,
        compression: Optional[str] = None,
        variant_label: str = "",
        base_repo_id: Optional[str] = None,
    ) -> None:
        # Ensure entry + catalog metadata exist
        from airllm_studio.core.catalog import get_catalog_model
        from airllm_studio.core.model_meta import parse_model_info

        progress_key = base_repo_id or repo_id
        cat = get_catalog_model(base_repo_id or repo_id) or get_catalog_model(repo_id)
        if cat and not self.models.get(repo_id):
            info = cat.to_model_info()
            info.repo_id = repo_id
            if variant_label:
                info.display_name = f"{cat.label} ({variant_label})"
                info.notes = list(info.notes) + [f"Variant: {variant_label}"]
            self.models.upsert_info(info, status="listed")
        elif not self.models.get(repo_id):
            info = parse_model_info(repo_id, {})
            if variant_label:
                info.notes.append(f"Variant: {variant_label}")
            self.models.upsert_info(info, status="listed")
        elif cat and self.models.get(repo_id) and not self.models.get(repo_id).info:
            self.models.upsert_info(cat.to_model_info())

        self._lib()._active_repo = progress_key  # type: ignore[attr-defined]
        ok, msg = self.models.check_disk(repo_id)
        self._lib().set_progress(msg, 0.0, repo_id=progress_key)
        if not ok:
            self._set_status(msg)
            return

        cnote = f" [{variant_label}]" if variant_label else ""
        self._set_status(f"Downloading / preparing {repo_id}{cnote}…")
        self._show_view("library")

        def on_progress(message: str, p: float, meta=None) -> None:
            self._event_q.put(("progress", message, p, meta, progress_key))

        def on_done(entry) -> None:
            self._event_q.put(("prepare_done", entry.repo_id, entry))

        def on_error(err: str) -> None:
            self._event_q.put(("error", err, None))

        self.models.download_and_prepare_async(
            repo_id,
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_paused=lambda rid: self._event_q.put(("download_paused", rid, None)),
            compression=compression,
            variant_label=variant_label,
            base_repo_id=base_repo_id,
        )
        if self.library_view is not None:
            self._lib().refresh_jobs(self.models.jobs.list_jobs())

    def _on_pause_download(self, repo_id: str) -> None:
        self.models.pause_download(repo_id)
        self._set_status(
            f"Pausing {repo_id}… (partial files kept — Resume anytime, even after reboot)"
        )
        if self.library_view is not None:
            self._lib().refresh_jobs(self.models.jobs.list_jobs())
            self._lib().set_progress(
                f"Paused {repo_id} — click Resume to continue",
                -1,
                repo_id=repo_id,
            )

    def _on_resume_download(self, repo_id: str) -> None:
        job = self.models.jobs.get(repo_id)
        if not job:
            self._set_status(f"No paused job for {repo_id}")
            return
        self._lib()._active_repo = job.base_repo_id or repo_id
        self._set_status(f"Resuming {repo_id}… (uses cached partial download)")

        def on_progress(message: str, p: float, meta=None) -> None:
            key = job.base_repo_id or repo_id
            self._event_q.put(("progress", message, p, meta, key))

        def on_done(entry) -> None:
            self._event_q.put(("prepare_done", entry.repo_id, entry))

        def on_error(err: str) -> None:
            self._event_q.put(("error", err, None))

        def on_paused(rid: str) -> None:
            self._event_q.put(("download_paused", rid, None))

        self.models.resume_download_async(
            repo_id,
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_paused=on_paused,
        )
        if self.library_view is not None:
            self._lib().refresh_jobs(self.models.jobs.list_jobs())

    def _on_delete_download(self, repo_id: str) -> None:
        from tkinter import messagebox

        try:
            ok = messagebox.askyesno(
                "Delete download",
                (
                    f"Delete download data for:\n\n{repo_id}\n\n"
                    "This removes the Hugging Face cache for this model "
                    "(and any AirLLM shards). You can download again later.\n\n"
                    "Continue?"
                ),
                parent=self,
            )
        except Exception:
            ok = True
        if not ok:
            return
        self._set_status(f"Deleting download for {repo_id}…")

        def work() -> None:
            try:
                summary = self.models.delete_download(repo_id, delete_cache=True)
                self._event_q.put(("download_deleted", summary, None))
            except Exception as exc:
                self._event_q.put(("error", str(exc), None))

        threading.Thread(target=work, daemon=True, name=f"delete-{repo_id}").start()

    def _on_load_model(self) -> None:
        if not self._selected_repo:
            self._set_status("Select a model first.")
            return
        if not self._memory_allows_load(show_warning=True):
            return

        entry = self.models.get(self._selected_repo)
        info = entry.model_info() if entry else None
        shards = entry.shards_path if entry else None

        self._set_status(f"Loading {self._selected_repo}…")
        self.model_panel.set_engine_status("Engine: loading…")

        def on_event(ev: EngineEvent) -> None:
            self._event_q.put(("engine", ev, None))

        self.engine.load_async(
            self._selected_repo,
            info=info,
            shards_path=shards,
            on_event=on_event,
        )

    def _on_unload_model(self) -> None:
        self.engine.unload()
        self.model_panel.set_engine_status("Engine: unloaded")
        self._set_status("Model unloaded.")

    # ── Chat / generation ──────────────────────────────────

    def _on_stop(self) -> None:
        self.engine.stop_generation()
        self.chat_view.set_streaming(False)
        self.chat_view.set_stream_status("Stopped")
        self._set_status("Generation stopped.")

    def _on_send(self, text: str) -> None:
        if not self._active_chat_id:
            return
        if not self.engine.is_loaded and not self.engine.loaded_repo:
            # Auto-load selected model if memory gate allows
            if self._selected_repo:
                if not self._memory_allows_load(show_warning=True):
                    return
                entry = self.models.get(self._selected_repo)
                info = entry.model_info() if entry else ModelInfo(repo_id=self._selected_repo)

                def after_load(ev: EngineEvent) -> None:
                    self._event_q.put(("engine", ev, None))
                    if ev.kind == "done" and ev.message == "loaded":
                        self._event_q.put(("autoload_send", text, None))

                self.engine.load_async(
                    self._selected_repo, info=info, on_event=after_load
                )
                self._set_status("Auto-loading model for chat…")
                return
            self._set_status("Load a model before chatting.")
            return

        self._run_generation(text)

    def _run_generation(self, user_text: str) -> None:
        assert self._active_chat_id
        chat_id = self._active_chat_id

        self.store.add_message(chat_id, Message(role="user", content=user_text))
        self.chat_view.append_message("user", user_text)
        self.store.add_message(chat_id, Message(role="assistant", content=""))
        self.chat_view.append_message("assistant", "…")
        self._refresh_sidebar()

        params = self.chat_view.get_gen_params()
        tokens, search, notes = clamp_generation(
            int(params["max_new_tokens"]),
            bool(self.cfg.web_search_enabled),
            self.iap.is_entitled(),
        )
        for note in notes:
            self._set_status(note)
        gen_cfg = GenerationConfig(
            temperature=params["temperature"],
            max_new_tokens=tokens,
            top_p=params["top_p"],
        )
        self._offer_search = search

        self.chat_view.set_streaming(True)
        self.chat_view.set_stream_status("Generating…")
        self._busy = True

        def work() -> None:
            try:
                final = self._agent_loop(chat_id, gen_cfg)
                self._event_q.put(("gen_done", final, chat_id))
            except Exception as exc:
                self._event_q.put(("error", str(exc), None))
                self._event_q.put(("gen_done", f"Error: {exc}", chat_id))

        threading.Thread(target=work, daemon=True, name="chat-gen").start()

    def _agent_loop(self, chat_id: str, gen_cfg: GenerationConfig) -> str:
        """Generate, optionally run tools, continue once."""
        chat = self.store.get(chat_id)
        if not chat:
            return ""

        last_user = ""
        for m in reversed(chat.messages):
            if m.role == "user":
                last_user = m.content
                break
        offer = should_offer_tools(
            last_user,
            tools_enabled=bool(
                allows("tools", self.iap.is_entitled())
                and allows("web_search", self.iap.is_entitled())
                and self.cfg.tools_enabled
                and self.cfg.web_search_enabled
            ),
            use_search=bool(getattr(self, "_offer_search", self.cfg.web_search_enabled)),
        )

        messages: List[Dict[str, str]] = []
        if offer and self.tools.names():
            sys = CONSERVATIVE_TOOL_PROMPT + "\n\n" + self.tools.prompt_section()
            messages.append({"role": "system", "content": sys})

        for m in chat.messages:
            if m.role == "assistant" and m.content == "":
                continue
            messages.append({"role": m.role, "content": m.content})

        def on_event(ev: EngineEvent) -> None:
            self._event_q.put(("engine", ev, None))

        text = self.engine.generate(messages, gen_cfg=gen_cfg, on_event=on_event)

        # Tool loop (single hop) — only if we actually offered tools
        if offer:
            calls = extract_tool_calls(text)
            if calls:
                for call in calls[:2]:
                    name = call["name"]
                    if name not in self.tools.names():
                        continue
                    result = self.tools.run(name, call["arguments"])
                    tool_msg = Message(
                        role="tool",
                        content=result.output,
                        tool_name=name,
                    )
                    self.store.add_message(chat_id, tool_msg)
                    self._event_q.put(("tool_msg", tool_msg, None))

                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool {name} result:\n{result.output}\n\nContinue your answer.",
                        }
                    )

                text = self.engine.generate(
                    messages, gen_cfg=gen_cfg, on_event=on_event
                )

        self.store.update_last_assistant(chat_id, text)
        return text

    def _on_purchase(self, product_id: str) -> None:
        try:
            receipt = self.iap.purchase(product_id)
            self._set_status(f"Unlocked {receipt.product_id}")
        except LicenseKeyRefused as exc:
            self._set_status(str(exc))
        except PurchaseError as exc:
            self._set_status(str(exc))
        if self.settings_view is not None:
            self.settings_view.refresh_billing(paywall_payload(self.iap))

    def _on_restore_purchases(self) -> None:
        receipts = self.iap.restore()
        if receipts:
            self._set_status(f"Restored {len(receipts)} purchase(s)")
        else:
            self._set_status("No previous purchases to restore")
        if self.settings_view is not None:
            self.settings_view.refresh_billing(paywall_payload(self.iap))

    def _on_open_legal(self, kind: str) -> None:
        import webbrowser

        from airllm_studio.billing import legal_page_url

        webbrowser.open(legal_page_url("privacy" if kind == "privacy" else "terms"))

    def _on_save_settings(self, data: dict) -> None:
        data, notes = sanitize_settings(data, self.iap.is_entitled())
        self.cfg = update_config(**data)
        self.engine.cfg = self.cfg
        self.models.cfg = self.cfg
        self.tools = get_default_registry(
            web_search=bool(
                allows("web_search", self.iap.is_entitled())
                and self.cfg.web_search_enabled
                and self.cfg.tools_enabled
            )
        )
        for note in notes:
            self._set_status(note)
        # Keep header switch in sync with Settings → demo_mode / airllm
        if "demo_mode" in data and data["demo_mode"]:
            self.cfg = update_config(airllm_enabled=False)
            self.chat_view.set_airllm_enabled(False)
        elif "demo_mode" in data and not data["demo_mode"]:
            # Restore airllm switch visual from config
            self.chat_view.set_airllm_enabled(self.cfg.airllm_enabled)
        self._set_status("Settings saved. " + self._backend_status_line())

    # ── Event pump ─────────────────────────────────────────

    def _poll_events(self) -> None:
        try:
            while True:
                item = self._event_q.get_nowait()
                self._handle_event(item)
        except queue.Empty:
            pass
        # ~60fps so per-token stream feels live
        self.after(16, self._poll_events)

    def _handle_event(self, item: tuple) -> None:
        kind = item[0]
        if kind == "progress":
            # Support both old (3-tuple) and new (5-tuple) progress events
            message = item[1]
            p = item[2] if len(item) > 2 else -1
            meta = item[3] if len(item) > 3 else None
            repo_id = item[4] if len(item) > 4 else None
            if isinstance(meta, str):
                # old signature accident
                meta = None
            self._lib().set_progress(
                message,
                p if p is not None else -1,
                meta=meta if isinstance(meta, dict) else None,
                repo_id=repo_id,
            )
            self._set_status(message)
            eta = ""
            if isinstance(meta, dict) and meta.get("eta_seconds") is not None:
                from airllm_studio.core.catalog import format_eta

                eta = f" · ETA {format_eta(meta['eta_seconds'])}"
            self.chat_view.set_stream_status((message[:60] + eta)[:90])

        elif kind == "hf_search":
            _, results, err = item
            self._lib().set_progress(
                f"Found {len(results)} models" if not err else f"Search error: {err}",
                1.0,
            )
            self._lib().show_search_results(results or [])

        elif kind == "info_ready":
            _, repo_id, info = item
            self._lib().set_progress(f"Added: {repo_id}", 1.0, repo_id=repo_id)
            self._lib().refresh_library(
                self.models.list_models(), self._selected_repo
            )
            self._on_select_model(repo_id)

        elif kind == "prepare_done":
            _, repo_id, _entry = item
            self._lib().set_progress(
                f"Prepared & ready: {repo_id}",
                1.0,
                meta={"eta_seconds": 0},
                repo_id=repo_id,
            )
            self._lib().refresh_library(
                self.models.list_models(), self._selected_repo
            )
            self._lib().refresh_jobs(self.models.jobs.list_jobs())
            self._on_select_model(repo_id)
            self._refresh_model_dropdown()
            self._set_status(f"Model ready: {repo_id} — pick it from the top dropdown")

        elif kind == "download_paused":
            _, rid, _ = item
            self._set_status(
                f"Paused {rid} — Resume when ready (survives app quit / reboot)"
            )
            if self.library_view is not None:
                self._lib().refresh_jobs(self.models.jobs.list_jobs())
                self._lib().set_progress(
                    f"Paused {rid}",
                    -1,
                    repo_id=rid,
                )

        elif kind == "download_deleted":
            _, summary, _ = item
            rid = summary.get("repo_id", "")
            freed = summary.get("freed_gb", 0)
            self._set_status(
                f"Deleted {rid} — freed ~{freed:.1f} GB"
                if freed
                else f"Deleted download record for {rid}"
            )
            if self.library_view is not None:
                self._lib().refresh_jobs(self.models.jobs.list_jobs())
                self._lib().refresh_library(
                    self.models.list_models(), self._selected_repo
                )
                self._lib().set_progress(
                    f"Deleted {rid} (~{freed:.1f} GB freed)",
                    0,
                    repo_id=rid,
                )

        elif kind == "engine":
            _, ev, _ = item
            self._handle_engine_event(ev)

        elif kind == "tool_msg":
            _, msg, _ = item
            self.chat_view.append_message("tool", msg.content, tool_name=msg.tool_name)

        elif kind == "gen_done":
            _, final, chat_id = item
            self.chat_view.set_streaming(False)
            self.chat_view.set_stream_status("")
            self.chat_view.update_last_assistant(final)
            self._busy = False
            if chat_id == self._active_chat_id:
                chat = self.store.get(chat_id)
                if chat:
                    self.chat_view.set_title(chat.title)
            self._refresh_sidebar()
            self._set_status("Ready")
            if self.engine.is_loaded:
                self.model_panel.set_engine_status(
                    f"Engine: ready · {self.engine.loaded_repo}"
                )

        elif kind == "autoload_send":
            _, text, _ = item
            self._run_generation(text)

        elif kind == "error":
            _, err, _ = item
            self._set_status(f"Error: {str(err)[:200]}")
            self._lib().set_progress(f"Error: {str(err)[:120]}", 0)
            self.chat_view.set_streaming(False)

    def _handle_engine_event(self, ev: EngineEvent) -> None:
        if ev.kind == "token":
            self.chat_view.update_last_assistant(ev.message)
        elif ev.kind == "progress":
            msg = ev.message
            self.chat_view.set_stream_status(msg[:90])
            self._set_status(msg)
            st = ev.data.get("status")
            if st:
                layer = ev.data.get("layer")
                total = ev.data.get("total")
                if layer and total:
                    self.model_panel.set_engine_status(
                        f"Engine: streaming layer {layer}/{total}"
                    )
                else:
                    self.model_panel.set_engine_status(f"Engine: {st}")
        elif ev.kind == "status":
            self._set_status(ev.message)
            st = ev.data.get("status", "")
            if st == "ready" or "loaded" in ev.message.lower():
                self.model_panel.set_engine_status(
                    f"Engine: ready · {self.engine.loaded_repo or ''}"
                )
            else:
                self.model_panel.set_engine_status(f"Engine: {st or ev.message}")
        elif ev.kind == "done":
            if ev.message == "loaded":
                demo = ev.data.get("demo")
                tag = " (demo)" if demo else ""
                self.model_panel.set_engine_status(
                    f"Engine: ready{tag} · {self.engine.loaded_repo}"
                )
                self._set_status(f"Loaded {self.engine.loaded_repo}{tag}")
        elif ev.kind == "error":
            self.model_panel.set_engine_status("Engine: error")
            self._set_status(f"Engine error: {ev.message}")


def run_app() -> None:
    apply_theme()
    app = AirLLMStudioApp()
    try:
        app.lift()
        app.attributes("-topmost", True)
        app.after(200, lambda: app.attributes("-topmost", False))
        app.focus_force()
    except Exception:
        pass
    app.mainloop()
