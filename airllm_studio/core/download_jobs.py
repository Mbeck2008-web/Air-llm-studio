"""Persistent download jobs — pause / resume across app restarts."""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DownloadJob:
    repo_id: str
    status: str = "queued"  # queued | downloading | paused | preparing | ready | error | cancelled
    compression: Optional[str] = None
    variant_label: str = ""
    base_repo_id: Optional[str] = None
    downloaded_gb: float = 0.0
    total_gb: float = 0.0
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def progress_key(self) -> str:
        return self.base_repo_id or self.repo_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadJob":
        return cls(
            repo_id=data["repo_id"],
            status=data.get("status") or "queued",
            compression=data.get("compression"),
            variant_label=data.get("variant_label") or "",
            base_repo_id=data.get("base_repo_id"),
            downloaded_gb=float(data.get("downloaded_gb") or 0),
            total_gb=float(data.get("total_gb") or 0),
            error=data.get("error"),
            created_at=data.get("created_at") or _now(),
            updated_at=data.get("updated_at") or _now(),
        )


class DownloadJobStore:
    """JSON-backed job registry under Application Support."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._jobs: Dict[str, DownloadJob] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("jobs") or []:
                j = DownloadJob.from_dict(item)
                # Interrupted mid-flight → paused so user can resume
                if j.status in ("downloading", "preparing", "queued"):
                    j.status = "paused"
                    j.updated_at = _now()
                self._jobs[j.repo_id] = j
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            self._jobs = {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "jobs": [j.to_dict() for j in self._jobs.values()],
                "updated_at": _now(),
            }
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.path)

    def get(self, repo_id: str) -> Optional[DownloadJob]:
        return self._jobs.get(repo_id)

    def list_jobs(self) -> List[DownloadJob]:
        order = {
            "downloading": 0,
            "preparing": 1,
            "paused": 2,
            "queued": 3,
            "error": 4,
            "ready": 5,
            "cancelled": 6,
        }
        return sorted(
            self._jobs.values(),
            key=lambda j: (order.get(j.status, 9), j.updated_at),
            reverse=False,
        )

    def list_resumable(self) -> List[DownloadJob]:
        return [j for j in self.list_jobs() if j.status in ("paused", "error", "queued")]

    def upsert(self, job: DownloadJob) -> DownloadJob:
        with self._lock:
            job.updated_at = _now()
            self._jobs[job.repo_id] = job
            self.save()
            return job

    def update(
        self,
        repo_id: str,
        *,
        status: Optional[str] = None,
        downloaded_gb: Optional[float] = None,
        total_gb: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Optional[DownloadJob]:
        with self._lock:
            j = self._jobs.get(repo_id)
            if not j:
                return None
            if status is not None:
                j.status = status
            if downloaded_gb is not None:
                j.downloaded_gb = downloaded_gb
            if total_gb is not None:
                j.total_gb = total_gb
            if error is not None:
                j.error = error
            j.updated_at = _now()
            self.save()
            return j

    def remove(self, repo_id: str) -> None:
        with self._lock:
            self._jobs.pop(repo_id, None)
            self.save()


def _hf_hub_root() -> Path:
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_dir_for(repo_id: str) -> Path:
    """Hugging Face hub cache folder for a model repo (honors HF_HOME)."""
    name = "models--" + repo_id.replace("/", "--")
    return _hf_hub_root() / name


def hf_lock_dir_for(repo_id: str) -> Path:
    name = "models--" + repo_id.replace("/", "--")
    return _hf_hub_root() / ".locks" / name


def delete_model_cache(
    repo_id: str,
    shards_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Delete HF hub cache + optional AirLLM shards for a model.
    Returns summary of what was removed.
    """
    removed: List[str] = []
    freed = 0

    def _rm(path: Path) -> None:
        nonlocal freed
        if not path.exists():
            return
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
            removed.append(str(path))
            return
        # du before delete
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    freed += p.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))

    _rm(hf_cache_dir_for(repo_id))
    _rm(hf_lock_dir_for(repo_id))

    if shards_dir is not None:
        shard = shards_dir / repo_id.replace("/", "__")
        _rm(shard)

    return {
        "repo_id": repo_id,
        "removed_paths": removed,
        "freed_gb": round(freed / (1024 ** 3), 2),
    }
