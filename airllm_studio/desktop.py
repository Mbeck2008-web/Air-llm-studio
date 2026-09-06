"""Native WebKit shell for AirLLM Studio (pywebview)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from airllm_studio.core.session import StudioSession

WEB_DIR = Path(__file__).resolve().parent / "web"


class Bridge:
    """Methods exposed to the frontend as window.pywebview.api.*"""

    def __init__(self, session: StudioSession) -> None:
        self.s = session

    def snapshot(self) -> dict:
        return self.s.snapshot()

    def poll(self) -> list:
        return self.s.drain_events()

    def seed_catalog(self) -> dict:
        return self.s.seed_catalog()

    def new_chat(self) -> dict:
        return self.s.new_chat()

    def select_chat(self, chat_id: str) -> dict:
        return self.s.select_chat(chat_id)

    def delete_chat(self, chat_id: str) -> dict:
        return self.s.delete_chat(chat_id)

    def save_settings(self, data: dict) -> dict:
        return self.s.save_settings(data or {})

    def set_airllm(self, enabled: bool) -> dict:
        return self.s.set_airllm(bool(enabled))

    def select_model(self, repo_id: str) -> dict:
        return self.s.select_model(repo_id)

    def pick_model(self, repo_id: str) -> dict:
        return self.s.pick_model(repo_id)

    def load_model(self) -> dict:
        return self.s.load_model()

    def unload_model(self) -> dict:
        return self.s.unload_model()

    def prepare(self, repo_id: str, compression: Optional[str] = None) -> dict:
        return self.s.prepare(repo_id, compression=compression or None)

    def pause_download(self, repo_id: str) -> dict:
        return self.s.pause_download(repo_id)

    def resume_download(self, repo_id: str) -> dict:
        return self.s.resume_download(repo_id)

    def delete_download(self, repo_id: str) -> dict:
        return self.s.delete_download(repo_id)

    def search_hf(self, query: str) -> dict:
        return self.s.search_hf(query)

    def add_repo(self, repo_id: str) -> dict:
        return self.s.add_repo(repo_id)

    def stop(self) -> dict:
        return self.s.stop()

    def send(
        self,
        text: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        use_search: bool = False,
    ) -> dict:
        return self.s.send(
            text,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
            top_p=float(top_p),
            use_search=bool(use_search),
        )

    def purchase(self, product_id: str) -> dict:
        return self.s.purchase(product_id)

    def restore_purchases(self) -> dict:
        return self.s.restore_purchases()

    def unlock_with_license_key(self, code: str) -> dict:
        return self.s.unlock_with_license_key(code)

    def dev_unlock(self) -> dict:
        return self.s.dev_unlock()

    def open_legal(self, kind: str) -> dict:
        """Return a file:// URL for in-app navigation (no external browser)."""
        from airllm_studio.billing import legal_page_path, legal_page_url

        which = "privacy" if kind == "privacy" else "terms"
        path = legal_page_path(which)
        if not path.is_file():
            return {"ok": False, "error": f"missing legal page: {path}"}
        return {
            "ok": True,
            "href": f"legal/{'privacy' if which == 'privacy' else 'terms'}.html",
            "url": legal_page_url(which),
        }


def run_desktop() -> None:
    try:
        import webview
    except ImportError:
        print("pywebview is required for the desktop UI.")
        print("  pip install pywebview")
        print("Or launch the legacy UI: AIRLLM_STUDIO_UI=tk python -m airllm_studio")
        sys.exit(1)

    index = WEB_DIR / "index.html"
    if not index.is_file():
        print(f"Missing UI: {index}")
        sys.exit(1)

    session = StudioSession()
    window = webview.create_window(
        "AirLLM Studio",
        url=index.as_uri(),
        js_api=Bridge(session),
        width=1280,
        height=840,
        min_size=(980, 640),
        background_color="#0b0c0d",
        text_select=True,
    )
    # Keep a reference so the session outlives create_window
    window._studio_session = session  # type: ignore[attr-defined]
    webview.start(debug=os.environ.get("AIRLLM_STUDIO_DEBUG") == "1")
