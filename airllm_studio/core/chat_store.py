"""Multi-chat conversation persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    timestamp: str = field(default_factory=_now_iso)
    tool_name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp") or _now_iso(),
            tool_name=data.get("tool_name"),
            meta=data.get("meta") or {},
        )


@dataclass
class Chat:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Message] = field(default_factory=list)
    model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chat":
        return cls(
            id=data["id"],
            title=data.get("title") or "New Chat",
            created_at=data.get("created_at") or _now_iso(),
            updated_at=data.get("updated_at") or _now_iso(),
            messages=[Message.from_dict(m) for m in data.get("messages") or []],
            model_id=data.get("model_id"),
        )

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def auto_title(self) -> None:
        if self.title not in ("New Chat", "", None):
            return
        for m in self.messages:
            if m.role == "user" and m.content.strip():
                t = m.content.strip().replace("\n", " ")
                self.title = (t[:48] + "…") if len(t) > 48 else t
                return


class ChatStore:
    def __init__(self, chats_dir: Optional[Path] = None) -> None:
        cfg = get_config()
        self.chats_dir = chats_dir or cfg.chats_dir
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.chats_dir / "index.json"
        self._cache: Dict[str, Chat] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, encoding="utf-8") as f:
                ids = json.load(f)
            for cid in ids:
                path = self.chats_dir / f"{cid}.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        self._cache[cid] = Chat.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, KeyError):
            self._cache = {}

    def _save_index(self) -> None:
        ordered = sorted(
            self._cache.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump([c.id for c in ordered], f, indent=2)

    def _save_chat(self, chat: Chat) -> None:
        path = self.chats_dir / f"{chat.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat.to_dict(), f, indent=2)
        self._save_index()

    def list_chats(self) -> List[Chat]:
        return sorted(
            self._cache.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )

    def get(self, chat_id: str) -> Optional[Chat]:
        return self._cache.get(chat_id)

    def create(self, title: str = "New Chat") -> Chat:
        now = _now_iso()
        chat = Chat(
            id=str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self._cache[chat.id] = chat
        self._save_chat(chat)
        return chat

    def delete(self, chat_id: str) -> None:
        self._cache.pop(chat_id, None)
        path = self.chats_dir / f"{chat_id}.json"
        if path.exists():
            path.unlink()
        self._save_index()

    def add_message(self, chat_id: str, message: Message) -> Chat:
        chat = self._cache[chat_id]
        chat.messages.append(message)
        chat.auto_title()
        chat.touch()
        self._save_chat(chat)
        return chat

    def update_last_assistant(self, chat_id: str, content: str) -> Chat:
        chat = self._cache[chat_id]
        for m in reversed(chat.messages):
            if m.role == "assistant":
                m.content = content
                break
        else:
            chat.messages.append(Message(role="assistant", content=content))
        chat.touch()
        self._save_chat(chat)
        return chat

    def ensure_default(self) -> Chat:
        chats = self.list_chats()
        if chats:
            return chats[0]
        return self.create()
