"""UI-agnostic application session. Both the WebKit shell and CTk use this."""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional

from airllm_studio.billing import (
    LicenseKeyRefused,
    PurchaseError,
    allows,
    clamp_generation,
    make_store,
    paywall_payload,
    refuse_license_key,
    sanitize_settings,
)
from airllm_studio.core.chat_store import ChatStore, Message
from airllm_studio.core.config import AppConfig, get_config, update_config
from airllm_studio.core.engine import EngineEvent, GenerationConfig, InferenceEngine
from airllm_studio.core.engine import extract_tool_calls
from airllm_studio.core.model_manager import ModelManager
from airllm_studio.core.model_meta import ModelInfo, can_load_without_airllm
from airllm_studio.core.tools_policy import (
    CONSERVATIVE_TOOL_PROMPT,
    should_offer_tools,
)
from airllm_studio.tools.registry import get_default_registry


class StudioSession:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or get_config()
        self.cfg.ensure_dirs()
        self.store = ChatStore(chats_dir=self.cfg.chats_dir)
        self.models = ModelManager()
        self.engine = InferenceEngine()
        self.engine.cfg = self.cfg
        self.iap = make_store(self.cfg.data_dir)
        self.tools = get_default_registry(
            web_search=self._tools_allowed()
        )
        self.events: queue.Queue = queue.Queue()
        self.selected_repo: Optional[str] = None
        self.active_chat_id: Optional[str] = None
        self.busy = False
        self._catalog_seeded = False

        chat = self.store.ensure_default()
        self.active_chat_id = chat.id
        ready = self.models.list_downloaded_models()
        if ready:
            self.selected_repo = ready[0].repo_id

    # ── serialization ──────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        chat = self.store.get(self.active_chat_id) if self.active_chat_id else None
        return {
            "config": {
                "airllm_enabled": bool(self.cfg.airllm_enabled and not self.cfg.demo_mode),
                "demo_mode": bool(self.cfg.demo_mode),
                "tools_enabled": bool(self.cfg.tools_enabled),
                "web_search_enabled": bool(self.cfg.web_search_enabled),
                "hf_token_set": bool(self.cfg.hf_token),
                "default_temperature": self.cfg.default_temperature,
                "default_max_tokens": self.cfg.default_max_tokens,
                "default_top_p": self.cfg.default_top_p,
                "compression": self.cfg.compression or "",
                "system_prompt": self.cfg.system_prompt or "",
                "data_dir": str(self.cfg.data_dir),
            },
            "backend": self.backend_line(),
            "chats": [self._chat_summary(c) for c in self.store.list_chats()],
            "active_chat_id": self.active_chat_id,
            "messages": [m.to_dict() for m in (chat.messages if chat else [])],
            "chat_title": chat.title if chat else "New Chat",
            "models": [self._entry_dict(e) for e in self.models.list_models()],
            "ready_models": [self._entry_dict(e) for e in self.models.list_downloaded_models()],
            "jobs": [j.to_dict() for j in self.models.jobs.list_jobs()],
            "selected_repo": self.selected_repo,
            "selected_model": self._selected_model_payload(),
            "engine": {
                "loaded_repo": self.engine.loaded_repo,
                "status": getattr(self.engine.status, "value", str(self.engine.status)),
                "is_loaded": self.engine.is_loaded,
            },
            "busy": self.busy,
            "activity": self._activity_payload(),
            "billing": paywall_payload(self.iap),
        }

    def can_use(self, capability: str) -> bool:
        return allows(capability, self.iap.is_entitled())

    def _tools_allowed(self) -> bool:
        return bool(
            self.can_use("web_search")
            and self.can_use("tools")
            and self.cfg.web_search_enabled
            and self.cfg.tools_enabled
        )

    def _activity_payload(self) -> Dict[str, Any]:
        probe = getattr(self.engine, "_probe", None)
        if probe is None:
            return {"attached": False, "layers_total": 0, "experts_total": 0}
        return probe.snapshot()

    def drain_events(self, max_n: int = 500) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(max_n):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def _emit(self, kind: str, **data: Any) -> None:
        self.events.put({"kind": kind, **data})

    def _chat_summary(self, chat) -> Dict[str, Any]:
        return {
            "id": chat.id,
            "title": chat.title,
            "updated_at": chat.updated_at,
            "preview": next(
                (m.content[:80] for m in reversed(chat.messages) if m.role == "user"),
                "",
            ),
        }

    def _entry_dict(self, e) -> Dict[str, Any]:
        info = e.model_info()
        return {
            "repo_id": e.repo_id,
            "status": e.status,
            "shards_path": e.shards_path,
            "error": e.error,
            "updated_at": e.updated_at,
            "display_name": (info.display_name if info else None) or e.repo_id.split("/")[-1],
            "is_moe": bool(info.is_moe) if info else False,
            "info": info.to_dict() if info else None,
        }

    def _selected_model_payload(self) -> Optional[Dict[str, Any]]:
        if not self.selected_repo:
            return None
        entry = self.models.get(self.selected_repo)
        info = entry.model_info() if entry else ModelInfo(repo_id=self.selected_repo)
        return {
            "repo_id": self.selected_repo,
            "status": entry.status if entry else "listed",
            "display_name": info.display_name or self.selected_repo.split("/")[-1],
            "is_moe": info.is_moe,
            "num_layers": info.num_layers,
            "num_experts": info.num_experts,
            "experts_per_token": info.experts_per_token,
            "params": info.total_params,
            "hidden_size": info.hidden_size,
            "architecture": info.architecture,
            "airllm_gb": info.estimated_airllm_memory_gb,
            "disk_gb": info.estimated_disk_gb,
            "notes": info.notes,
        }

    def backend_line(self) -> str:
        if self.cfg.demo_mode:
            mode = "demo"
        elif not self.cfg.airllm_enabled:
            mode = "AirLLM off"
        elif not self.engine._airllm_available():
            mode = "airllm not installed"
        else:
            mode = "AirLLM ready"
        return mode

    def seed_catalog(self) -> Dict[str, Any]:
        if not self._catalog_seeded:
            self.models.ensure_catalog_seeded()
            self._catalog_seeded = True
        return self.snapshot()

    # ── chats ──────────────────────────────────────────────

    def new_chat(self) -> Dict[str, Any]:
        chat = self.store.create()
        self.active_chat_id = chat.id
        return self.snapshot()

    def select_chat(self, chat_id: str) -> Dict[str, Any]:
        if self.store.get(chat_id):
            self.active_chat_id = chat_id
        return self.snapshot()

    def delete_chat(self, chat_id: str) -> Dict[str, Any]:
        self.store.delete(chat_id)
        if self.active_chat_id == chat_id:
            chat = self.store.ensure_default()
            self.active_chat_id = chat.id
        return self.snapshot()

    # ── settings / model ───────────────────────────────────

    def save_settings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "hf_token",
            "default_temperature",
            "default_max_tokens",
            "default_top_p",
            "tools_enabled",
            "web_search_enabled",
            "compression",
            "demo_mode",
            "airllm_enabled",
            "delete_original_after_split",
            "system_prompt",
        }
        patch = {k: data[k] for k in allowed if k in data}
        if "default_temperature" in patch:
            patch["default_temperature"] = float(patch["default_temperature"])
        if "default_max_tokens" in patch:
            patch["default_max_tokens"] = int(patch["default_max_tokens"])
        if "default_top_p" in patch:
            patch["default_top_p"] = float(patch["default_top_p"])
        if patch.get("compression") in ("", "none", "None"):
            patch["compression"] = None
        patch, notes = sanitize_settings(patch, self.iap.is_entitled())
        self.cfg = update_config(**patch)
        self.engine.cfg = self.cfg
        self.models.cfg = self.cfg
        self.tools = get_default_registry(web_search=self._tools_allowed())
        for note in notes:
            self._emit("error", message=note)
        self._emit("status", message="Settings saved")
        return self.snapshot()

    def purchase(self, product_id: str) -> Dict[str, Any]:
        try:
            receipt = self.iap.purchase(product_id)
        except LicenseKeyRefused as exc:
            self._emit("error", message=str(exc))
            return {"ok": False, "error": "license_key_refused", **self.snapshot()}
        except PurchaseError as exc:
            self._emit("error", message=str(exc))
            return {"ok": False, "error": str(exc), **self.snapshot()}
        self._emit("status", message=f"Unlocked {receipt.product_id}")
        return {"ok": True, "receipt": receipt.to_dict(), **self.snapshot()}

    def restore_purchases(self) -> Dict[str, Any]:
        receipts = self.iap.restore()
        if receipts:
            self._emit("status", message=f"Restored {len(receipts)} purchase(s)")
        else:
            self._emit("status", message="No previous purchases to restore")
        return {
            "ok": True,
            "receipts": [r.to_dict() for r in receipts],
            **self.snapshot(),
        }

    def unlock_with_license_key(self, code: str) -> Dict[str, Any]:
        try:
            refuse_license_key(code)
        except LicenseKeyRefused as exc:
            self._emit("error", message=str(exc))
            return {"ok": False, "error": "license_key_refused", **self.snapshot()}
        return {"ok": False, "error": "license_key_refused", **self.snapshot()}

    def set_airllm(self, enabled: bool) -> Dict[str, Any]:
        self.cfg = update_config(airllm_enabled=bool(enabled))
        if enabled and self.cfg.demo_mode:
            self.cfg = update_config(demo_mode=False)
        self.engine.cfg = self.cfg
        was = self.engine.loaded_repo
        if was:
            self.engine.unload()
            ok, msg = self.memory_gate()
            if ok:
                self.load_model()
            else:
                self._emit("status", message=msg)
        return self.snapshot()

    def select_model(self, repo_id: str) -> Dict[str, Any]:
        self.selected_repo = repo_id
        entry = self.models.get(repo_id)
        if entry is None:
            from airllm_studio.core.catalog import get_catalog_model

            cat = get_catalog_model(repo_id)
            if cat:
                self.models.upsert_info(cat.to_model_info(), status="listed")
        return self.snapshot()

    def pick_model(self, repo_id: str) -> Dict[str, Any]:
        """Header dropdown: select a downloaded model and load it for chat."""
        self.select_model(repo_id)
        entry = self.models.get(repo_id)
        ready = bool(
            entry
            and (
                entry.status in ("ready", "loaded")
                or entry.shards_path
                or entry.local_path
            )
        )
        if ready and self.engine.loaded_repo != repo_id:
            return self.load_model()
        return self.snapshot()

    def memory_gate(self) -> tuple[bool, str]:
        if self.cfg.airllm_enabled and not self.cfg.demo_mode:
            return True, ""
        info = None
        if self.selected_repo:
            e = self.models.get(self.selected_repo)
            info = e.model_info() if e else ModelInfo(repo_id=self.selected_repo)
        if info is None:
            return True, ""
        ok, msg, _, _ = can_load_without_airllm(info)
        return ok, msg

    def load_model(self) -> Dict[str, Any]:
        if not self.selected_repo:
            self._emit("error", message="Select a model first")
            return self.snapshot()
        ok, msg = self.memory_gate()
        if not ok:
            self._emit("error", message=msg)
            return self.snapshot()
        entry = self.models.get(self.selected_repo)
        info = entry.model_info() if entry else None
        shards = entry.shards_path if entry else None
        self._emit("status", message=f"Loading {self.selected_repo}…")

        def on_event(ev: EngineEvent) -> None:
            self._engine_event(ev)

        self.engine.load_async(
            self.selected_repo, info=info, shards_path=shards, on_event=on_event
        )
        return self.snapshot()

    def unload_model(self) -> Dict[str, Any]:
        self.engine.unload()
        self._emit("status", message="Model unloaded")
        self._emit("engine", status="unloaded", loaded_repo=None)
        return self.snapshot()

    def _engine_event(self, ev: EngineEvent) -> None:
        payload: Dict[str, Any] = {
            "kind": "engine",
            "ev": ev.kind,
            "message": ev.message,
            "data": ev.data,
        }
        self.events.put(payload)

    # ── library ────────────────────────────────────────────

    def prepare(self, repo_id: str, compression: Optional[str] = None) -> Dict[str, Any]:
        from airllm_studio.core.catalog import get_catalog_model
        from airllm_studio.core.model_meta import parse_model_info

        cat = get_catalog_model(repo_id)
        if cat and not self.models.get(repo_id):
            self.models.upsert_info(cat.to_model_info(), status="listed")
        elif not self.models.get(repo_id):
            self.models.upsert_info(parse_model_info(repo_id, {}), status="listed")

        ok, msg = self.models.check_disk(repo_id)
        self._emit("progress", message=msg, progress=0.0, repo_id=repo_id)
        if not ok:
            self._emit("error", message=msg)
            return self.snapshot()

        def on_progress(message: str, p: float, meta=None) -> None:
            self.events.put(
                {
                    "kind": "progress",
                    "message": message,
                    "progress": p,
                    "meta": meta,
                    "repo_id": repo_id,
                }
            )

        def on_done(entry) -> None:
            self.events.put({"kind": "prepare_done", "repo_id": entry.repo_id})

        def on_error(err: str) -> None:
            self.events.put({"kind": "error", "message": err})

        self.models.download_and_prepare_async(
            repo_id,
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_paused=lambda rid: self.events.put({"kind": "download_paused", "repo_id": rid}),
            compression=compression or None,
        )
        return self.snapshot()

    def pause_download(self, repo_id: str) -> Dict[str, Any]:
        self.models.pause_download(repo_id)
        self._emit("status", message=f"Pausing {repo_id}")
        return self.snapshot()

    def resume_download(self, repo_id: str) -> Dict[str, Any]:
        job = self.models.jobs.get(repo_id)
        if not job:
            self._emit("error", message=f"No paused job for {repo_id}")
            return self.snapshot()

        def on_progress(message: str, p: float, meta=None) -> None:
            self.events.put(
                {
                    "kind": "progress",
                    "message": message,
                    "progress": p,
                    "meta": meta,
                    "repo_id": repo_id,
                }
            )

        self.models.resume_download_async(
            repo_id,
            on_progress=on_progress,
            on_done=lambda e: self.events.put({"kind": "prepare_done", "repo_id": e.repo_id}),
            on_error=lambda err: self.events.put({"kind": "error", "message": err}),
            on_paused=lambda rid: self.events.put({"kind": "download_paused", "repo_id": rid}),
        )
        return self.snapshot()

    def delete_download(self, repo_id: str) -> Dict[str, Any]:
        def work() -> None:
            try:
                summary = self.models.delete_download(repo_id, delete_cache=True)
                self.events.put({"kind": "download_deleted", "summary": summary})
            except Exception as exc:
                self.events.put({"kind": "error", "message": str(exc)})

        threading.Thread(target=work, daemon=True, name=f"delete-{repo_id}").start()
        return self.snapshot()

    def search_hf(self, query: str) -> Dict[str, Any]:
        if not self.can_use("hf_search"):
            self._emit("error", message="Hugging Face search is a Pro library action")
            return {"ok": False, "error": "hf_search requires Pro"}

        def work() -> None:
            try:
                results = self.models.search_hf(query)
                self.events.put({"kind": "hf_search", "results": results, "error": None})
            except Exception as exc:
                self.events.put({"kind": "hf_search", "results": [], "error": str(exc)})

        threading.Thread(target=work, daemon=True).start()
        self._emit("status", message=f"Searching Hugging Face: {query}")
        return {"ok": True}

    def add_repo(self, repo_id: str) -> Dict[str, Any]:
        repo_id = (repo_id or "").strip()
        if not repo_id:
            return self.snapshot()

        def work() -> None:
            try:
                info = self.models.fetch_remote_info(repo_id)
                self.events.put({"kind": "info_ready", "repo_id": repo_id, "info": info.to_dict()})
            except Exception as exc:
                from airllm_studio.core.model_meta import parse_model_info

                info = parse_model_info(repo_id, {})
                self.models.upsert_info(info, status="listed")
                self.events.put({"kind": "info_ready", "repo_id": repo_id, "info": info.to_dict()})
                self.events.put({"kind": "status", "message": f"Added with limited metadata ({exc})"})

        threading.Thread(target=work, daemon=True).start()
        return self.snapshot()

    # ── chat ───────────────────────────────────────────────

    def stop(self) -> Dict[str, Any]:
        self.engine.stop_generation()
        self.busy = False
        self._emit("status", message="Generation stopped")
        self._emit("gen_done", text="", stopped=True)
        return {"ok": True}

    def send(
        self,
        text: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        use_search: bool = False,
    ) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text or not self.active_chat_id:
            return {"ok": False, "error": "empty"}
        if self.busy:
            return {"ok": False, "error": "busy"}

        tokens, search, notes = clamp_generation(
            int(max_new_tokens if max_new_tokens is not None else self.cfg.default_max_tokens),
            bool(use_search),
            self.iap.is_entitled(),
        )
        for note in notes:
            self._emit("status", message=note)

        if not self.engine.is_loaded and not self.engine.loaded_repo:
            if self.selected_repo:
                ok, msg = self.memory_gate()
                if not ok:
                    self._emit("error", message=msg)
                    return {"ok": False, "error": msg}
                entry = self.models.get(self.selected_repo)
                info = entry.model_info() if entry else ModelInfo(repo_id=self.selected_repo)
                shards = entry.shards_path if entry else None

                def after_load(ev: EngineEvent) -> None:
                    self._engine_event(ev)
                    if ev.kind == "done" and ev.message == "loaded":
                        self._start_generation(
                            text, temperature, tokens, top_p, search
                        )

                self.engine.load_async(
                    self.selected_repo, info=info, shards_path=shards, on_event=after_load
                )
                self._emit("status", message="Loading model, then sending…")
                return {"ok": True, "loading": True}
            self._emit("error", message="Load a model before chatting")
            return {"ok": False, "error": "no model"}

        self._start_generation(text, temperature, tokens, top_p, search)
        return {"ok": True}

    def _start_generation(
        self,
        user_text: str,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
        top_p: Optional[float],
        use_search: bool,
    ) -> None:
        assert self.active_chat_id
        chat_id = self.active_chat_id
        self.store.add_message(chat_id, Message(role="user", content=user_text))
        self.store.add_message(chat_id, Message(role="assistant", content=""))
        self.busy = True
        self._emit("user", content=user_text)
        self._emit("assistant_start")

        gen_cfg = GenerationConfig(
            temperature=float(temperature if temperature is not None else self.cfg.default_temperature),
            max_new_tokens=int(max_new_tokens if max_new_tokens is not None else self.cfg.default_max_tokens),
            top_p=float(top_p if top_p is not None else self.cfg.default_top_p),
        )
        offer = should_offer_tools(
            user_text,
            tools_enabled=self._tools_allowed(),
            use_search=use_search,
        )

        def work() -> None:
            try:
                final = self._agent_loop(chat_id, gen_cfg, offer_tools=offer)
                self.store.update_last_assistant(chat_id, final)
                self.busy = False
                self._emit("gen_done", text=final, stopped=False, chat_id=chat_id)
            except Exception as exc:
                self.busy = False
                self._emit("error", message=str(exc))
                self._emit("gen_done", text=f"Error: {exc}", stopped=False, chat_id=chat_id)

        threading.Thread(target=work, daemon=True, name="chat-gen").start()

    def _agent_loop(self, chat_id: str, gen_cfg: GenerationConfig, offer_tools: bool) -> str:
        chat = self.store.get(chat_id)
        if not chat:
            return ""

        messages: List[Dict[str, str]] = []
        sys_bits: List[str] = []
        if (self.cfg.system_prompt or "").strip():
            sys_bits.append(self.cfg.system_prompt.strip())
        if offer_tools and self.tools.names():
            sys_bits.append(CONSERVATIVE_TOOL_PROMPT + "\n\n" + self.tools.prompt_section())
        if sys_bits:
            messages.append({"role": "system", "content": "\n\n".join(sys_bits)})

        for m in chat.messages:
            if m.role == "assistant" and m.content == "":
                continue
            messages.append({"role": m.role, "content": m.content})

        def on_event(ev: EngineEvent) -> None:
            self._engine_event(ev)

        text = self.engine.generate(messages, gen_cfg=gen_cfg, on_event=on_event)

        if offer_tools:
            calls = extract_tool_calls(text)
            if calls:
                # Don't keep the raw tool markup as the visible answer
                self._emit("token", message="", full_text="")
                for call in calls[:1]:
                    name = call["name"]
                    if name not in self.tools.names():
                        continue
                    result = self.tools.run(name, call["arguments"])
                    tool_msg = Message(role="tool", content=result.output, tool_name=name)
                    self.store.add_message(chat_id, tool_msg)
                    self._emit("tool", name=name, content=result.output)
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool {name} result:\n{result.output}\n\n"
                                "Answer the original question briefly using these results. "
                                "Do not dump the list verbatim."
                            ),
                        }
                    )
                text = self.engine.generate(messages, gen_cfg=gen_cfg, on_event=on_event)

        return text
