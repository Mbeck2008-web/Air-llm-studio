"""Local model library, Hugging Face download, and prepare orchestration."""

from __future__ import annotations

import json
import shutil
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .catalog import FEATURED_MODELS, CatalogModel, estimate_download_seconds, format_eta
from .config import get_config
from .download_jobs import (
    DownloadJob,
    DownloadJobStore,
    delete_model_cache,
)
from .model_meta import ModelInfo, disk_space_warning, parse_model_info

# message, fraction 0-1 (-1 indeterminate), optional meta dict
ProgressCb = Callable[[str, float, Optional[dict]], None]
StatusCb = Callable[[str], None]


class DownloadPaused(Exception):
    """Raised when the user pauses/cancels an in-flight download."""


def _emit(
    on_progress: Optional[ProgressCb],
    msg: str,
    p: float = -1,
    **meta,
) -> None:
    if not on_progress:
        return
    try:
        on_progress(msg, p, meta or None)
    except TypeError:
        # Back-compat for 2-arg callbacks
        on_progress(msg, p)  # type: ignore[misc]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_disk_gb(path: Optional[Path] = None) -> float:
    p = path or Path.home()
    try:
        usage = shutil.disk_usage(p)
        return usage.free / (1024 ** 3)
    except OSError:
        return 0.0


@dataclass
class LibraryEntry:
    repo_id: str
    status: str = "listed"  # listed | downloading | preparing | paused | ready | error | loaded
    local_path: Optional[str] = None
    shards_path: Optional[str] = None
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    added_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def model_info(self) -> Optional[ModelInfo]:
        if not self.info:
            return None
        return ModelInfo.from_dict(self.info)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LibraryEntry":
        return cls(
            repo_id=data["repo_id"],
            status=data.get("status") or "listed",
            local_path=data.get("local_path"),
            shards_path=data.get("shards_path"),
            info=data.get("info"),
            error=data.get("error"),
            added_at=data.get("added_at") or _now_iso(),
            updated_at=data.get("updated_at") or _now_iso(),
        )


class ModelManager:
    """Manages library registry and download/prepare jobs."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.cfg.ensure_dirs()
        self._entries: Dict[str, LibraryEntry] = {}
        self._lock = threading.RLock()
        self._load()
        # Catalog seed is deferred to idle / first library open (avoid startup hitch)
        self.jobs = DownloadJobStore(self.cfg.data_dir / "download_jobs.json")
        # repo_id -> Event set when user requests pause
        self._pause_flags: Dict[str, threading.Event] = {}
        self._active_threads: Dict[str, threading.Thread] = {}

    def _load(self) -> None:
        path = self.cfg.models_path
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("models") or []:
                e = LibraryEntry.from_dict(item)
                self._entries[e.repo_id] = e
        except (json.JSONDecodeError, OSError, KeyError):
            self._entries = {}

    def _save(self) -> None:
        with self._lock:
            payload = {
                "models": [e.to_dict() for e in self._entries.values()],
                "updated_at": _now_iso(),
            }
            with open(self.cfg.models_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def _seed_examples_if_empty(self) -> None:
        """Seed featured catalog into the local library (listed, not downloaded)."""
        changed = False
        for cat in FEATURED_MODELS:
            if cat.repo_id in self._entries:
                continue
            info = cat.to_model_info()
            self._entries[cat.repo_id] = LibraryEntry(
                repo_id=cat.repo_id,
                status="listed",
                info=info.to_dict(),
            )
            changed = True
        if changed or not self._entries:
            # Always ensure catalog models exist even if library had old seeds
            for cat in FEATURED_MODELS:
                if cat.repo_id not in self._entries:
                    info = cat.to_model_info()
                    self._entries[cat.repo_id] = LibraryEntry(
                        repo_id=cat.repo_id,
                        status="listed",
                        info=info.to_dict(),
                    )
            self._save()

    def ensure_catalog_seeded(self) -> None:
        """Merge missing featured models. Only writes disk when something new appears."""
        with self._lock:
            changed = False
            for cat in FEATURED_MODELS:
                if cat.repo_id in self._entries:
                    continue
                info = cat.to_model_info()
                self._entries[cat.repo_id] = LibraryEntry(
                    repo_id=cat.repo_id,
                    status="listed",
                    info=info.to_dict(),
                )
                changed = True
            if changed:
                self._save()

    def list_ready_models(self) -> List[LibraryEntry]:
        """Models available for the top dropdown (downloaded / ready / loaded)."""
        ready_status = {"ready", "loaded", "preparing", "downloading"}
        with self._lock:
            items = [
                e
                for e in self._entries.values()
                if e.status in ready_status
                or (e.local_path or e.shards_path)
            ]
        # Prefer ready/loaded first
        order = {"loaded": 0, "ready": 1, "preparing": 2, "downloading": 3}
        return sorted(
            items,
            key=lambda e: (order.get(e.status, 9), e.repo_id),
        )

    def list_downloaded_models(self) -> List[LibraryEntry]:
        """Models that are actually on disk and usable in the chat dropdown."""
        with self._lock:
            items = [e for e in self._entries.values() if self._is_downloaded(e)]
        return sorted(items, key=lambda e: e.updated_at, reverse=True)

    @staticmethod
    def _is_downloaded(e: LibraryEntry) -> bool:
        if e.status in ("ready", "loaded"):
            return True
        for raw in (e.shards_path, e.local_path):
            if not raw:
                continue
            p = Path(raw)
            if not p.exists():
                continue
            if p.is_file():
                return True
            try:
                next(p.iterdir())
                return True
            except StopIteration:
                continue
        return False

    def list_models(self) -> List[LibraryEntry]:
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda e: e.updated_at,
                reverse=True,
            )

    def get(self, repo_id: str) -> Optional[LibraryEntry]:
        return self._entries.get(repo_id)

    def upsert_info(self, info: ModelInfo, status: Optional[str] = None) -> LibraryEntry:
        with self._lock:
            e = self._entries.get(info.repo_id) or LibraryEntry(repo_id=info.repo_id)
            e.info = info.to_dict()
            if status:
                e.status = status
            e.updated_at = _now_iso()
            self._entries[info.repo_id] = e
            self._save()
            return e

    def remove(self, repo_id: str) -> None:
        with self._lock:
            self._entries.pop(repo_id, None)
            self._save()

    def fetch_remote_info(
        self, repo_id: str, progress: Optional[ProgressCb] = None
    ) -> ModelInfo:
        """Fetch config.json from Hugging Face without downloading full weights."""
        if progress:
            progress(f"Fetching config for {repo_id}…", 0.1)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required. pip install huggingface_hub"
            ) from exc

        token = self.cfg.hf_token or None
        hub_cache = str(self.cfg.cache_dir / "huggingface" / "hub")
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=token,
            cache_dir=hub_cache,
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        meta = None
        try:
            index_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors.index.json",
                token=token,
                cache_dir=hub_cache,
            )
            with open(index_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None

        info = parse_model_info(repo_id, config, meta)
        self.upsert_info(info, status="listed")
        if progress:
            progress("Config loaded", 1.0)
        return info

    def search_hf(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search Hugging Face model hub."""
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required") from exc

        api = HfApi(token=self.cfg.hf_token or None)
        models = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
        results = []
        for m in models:
            results.append(
                {
                    "repo_id": m.id,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "tags": list(getattr(m, "tags", None) or [])[:8],
                    "pipeline_tag": getattr(m, "pipeline_tag", None),
                }
            )
        return results

    def check_disk(self, repo_id: str) -> tuple[bool, str]:
        e = self.get(repo_id)
        info = e.model_info() if e else ModelInfo(repo_id=repo_id)
        if info is None:
            info = ModelInfo(repo_id=repo_id)
        free = free_disk_gb(self.cfg.data_dir)
        return disk_space_warning(info, free)

    def pause_download(self, repo_id: str) -> None:
        """Request pause; worker checks flag and stops, cache is kept for resume."""
        flag = self._pause_flags.get(repo_id)
        if flag is None:
            flag = threading.Event()
            self._pause_flags[repo_id] = flag
        flag.set()
        self.jobs.update(repo_id, status="paused")
        with self._lock:
            e = self._entries.get(repo_id)
            if e and e.status in ("downloading", "preparing"):
                e.status = "paused"
                e.updated_at = _now_iso()
                self._save()

    def resume_download_async(
        self,
        repo_id: str,
        on_progress: Optional[ProgressCb] = None,
        on_done: Optional[Callable[[LibraryEntry], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_paused: Optional[Callable[[str], None]] = None,
    ) -> Optional[threading.Thread]:
        """Resume a paused/incomplete job (HF cache is reused)."""
        job = self.jobs.get(repo_id)
        if not job:
            return None
        return self.download_and_prepare_async(
            repo_id,
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_paused=on_paused,
            compression=job.compression,
            variant_label=job.variant_label,
            base_repo_id=job.base_repo_id,
        )

    def delete_download(
        self,
        repo_id: str,
        *,
        delete_cache: bool = True,
    ) -> Dict[str, Any]:
        """Stop job if running, remove job record, optionally wipe HF cache + shards."""
        self.pause_download(repo_id)
        self.jobs.remove(repo_id)
        with self._lock:
            e = self._entries.get(repo_id)
            if e:
                e.status = "listed"
                e.local_path = None
                e.shards_path = None
                e.error = None
                e.updated_at = _now_iso()
                self._save()
        summary: Dict[str, Any] = {"repo_id": repo_id, "freed_gb": 0.0}
        if delete_cache:
            summary = delete_model_cache(repo_id, shards_dir=self.cfg.shards_dir)
        self._pause_flags.pop(repo_id, None)
        return summary

    def download_and_prepare_async(
        self,
        repo_id: str,
        on_progress: Optional[ProgressCb] = None,
        on_done: Optional[Callable[[LibraryEntry], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_paused: Optional[Callable[[str], None]] = None,
        compression: Optional[str] = None,
        variant_label: str = "",
        base_repo_id: Optional[str] = None,
    ) -> threading.Thread:
        # Clear any previous pause so resume works
        flag = threading.Event()
        self._pause_flags[repo_id] = flag

        job = self.jobs.get(repo_id) or DownloadJob(repo_id=repo_id)
        job.repo_id = repo_id
        job.status = "downloading"
        job.compression = compression
        job.variant_label = variant_label
        job.base_repo_id = base_repo_id
        job.error = None
        self.jobs.upsert(job)

        def work() -> None:
            try:
                entry = self.download_and_prepare(
                    repo_id,
                    on_progress,
                    compression=compression,
                    variant_label=variant_label,
                    pause_event=flag,
                )
                self.jobs.update(repo_id, status="ready", error=None)
                if on_done:
                    on_done(entry)
            except DownloadPaused:
                self.jobs.update(repo_id, status="paused")
                with self._lock:
                    e = self._entries.get(repo_id) or LibraryEntry(repo_id=repo_id)
                    e.status = "paused"
                    e.updated_at = _now_iso()
                    self._entries[repo_id] = e
                    self._save()
                if on_paused:
                    on_paused(repo_id)
                elif on_progress:
                    _emit(
                        on_progress,
                        f"Paused — resume anytime (cache kept for {repo_id})",
                        -1,
                    )
            except Exception as exc:
                tb = traceback.format_exc()
                self.jobs.update(repo_id, status="error", error=str(exc))
                with self._lock:
                    e = self._entries.get(repo_id) or LibraryEntry(repo_id=repo_id)
                    e.status = "error"
                    e.error = str(exc)
                    e.updated_at = _now_iso()
                    self._entries[repo_id] = e
                    self._save()
                if on_error:
                    on_error(f"{exc}\n{tb}")
            finally:
                self._active_threads.pop(repo_id, None)

        t = threading.Thread(target=work, daemon=True, name=f"prepare-{repo_id}")
        self._active_threads[repo_id] = t
        t.start()
        return t

    def download_and_prepare(
        self,
        repo_id: str,
        on_progress: Optional[ProgressCb] = None,
        compression: Optional[str] = None,
        variant_label: str = "",
        pause_event: Optional[threading.Event] = None,
    ) -> LibraryEntry:
        """Download model and run AirLLM layer split (or demo simulate). Resumable via HF cache."""
        import time

        t0 = time.time()
        pause_event = pause_event or self._pause_flags.get(repo_id) or threading.Event()

        def check_pause() -> None:
            if pause_event.is_set():
                raise DownloadPaused(f"Paused: {repo_id}")

        def prog(msg: str, p: float = -1, **meta) -> None:
            check_pause()
            elapsed = time.time() - t0
            if p is not None and 0 < p < 1 and "eta_seconds" not in meta:
                meta["eta_seconds"] = max(0.0, elapsed * (1.0 / p - 1.0))
            if "elapsed_seconds" not in meta:
                meta["elapsed_seconds"] = elapsed
            if meta.get("downloaded_gb") is not None or meta.get("total_gb") is not None:
                self.jobs.update(
                    repo_id,
                    downloaded_gb=meta.get("downloaded_gb"),
                    total_gb=meta.get("total_gb"),
                    status="downloading" if (p is not None and p < 0.45) else None,
                )
            _emit(on_progress, msg, p, **meta)

        ok, disk_msg = self.check_disk(repo_id)
        prog(disk_msg, 0.02)
        if not ok:
            raise RuntimeError(disk_msg)

        with self._lock:
            e = self._entries.get(repo_id) or LibraryEntry(repo_id=repo_id)
            e.status = "downloading"
            e.error = None
            e.updated_at = _now_iso()
            self._entries[repo_id] = e
            self._save()

        # Refresh metadata
        try:
            info = self.fetch_remote_info(repo_id, progress=None)
        except Exception:
            info = (self.get(repo_id) and self.get(repo_id).model_info()) or ModelInfo(
                repo_id=repo_id
            )

        # Per-download compression overrides global settings when provided
        use_compression = compression if compression is not None else self.cfg.compression
        if use_compression == "":
            use_compression = None

        # IMPORTANT: AirLLM 4bit/8bit still downloads FULL weights first, then compresses.
        # Only pre-quantized HF repos (AWQ/GPTQ/MLX) have a smaller download.
        download_gb = info.estimated_disk_gb or 10.0
        # Try to read real size from the hub (bytes) so the bar doesn't "grow"
        hub_gb = self._hf_repo_size_gb(repo_id)
        if hub_gb and hub_gb > 0:
            download_gb = hub_gb

        final_gb = download_gb
        if use_compression == "4bit":
            final_gb = download_gb * 0.35
        elif use_compression == "8bit":
            final_gb = download_gb * 0.55

        pred_eta = estimate_download_seconds(download_gb, mbps=50.0)
        vnote = f" · {variant_label}" if variant_label else ""
        cnote = f" · {use_compression}" if use_compression else ""
        if use_compression in ("4bit", "8bit"):
            size_msg = (
                f"~{download_gb:.0f} GB download (full weights), "
                f"then AirLLM compresses toward ~{final_gb:.0f} GB"
            )
        else:
            size_msg = f"~{download_gb:.0f} GB download"
        prog(
            f"Starting {repo_id}{vnote}{cnote}: {size_msg}",
            0.03,
            total_gb=download_gb,
            eta_seconds=pred_eta,
            predicted_eta_seconds=pred_eta,
        )

        shards_path = self.cfg.shards_dir / repo_id.replace("/", "__")
        shards_path.mkdir(parents=True, exist_ok=True)

        demo = (
            getattr(self.cfg, "demo_mode", False)
            or not getattr(self.cfg, "airllm_enabled", True)
            or not self._airllm_available()
        )
        if demo:
            return self._demo_prepare(repo_id, info, shards_path, prog, download_gb)

        # One HF pull with accurate progress (cache reused by AirLLM after / on resume)
        try:
            self._hf_snapshot_download(
                repo_id, prog, download_gb, pause_event=pause_event
            )
        except DownloadPaused:
            raise
        except Exception as dl_err:
            check_pause()
            prog(f"HF snapshot note: {dl_err} — AirLLM will download if needed", 0.35)

        check_pause()
        with self._lock:
            e = self._entries[repo_id]
            e.status = "preparing"
            e.updated_at = _now_iso()
            self._save()
        self.jobs.update(repo_id, status="preparing")

        prep_note = ""
        if use_compression in ("4bit", "8bit"):
            prep_note = f" (compressing to {use_compression} while splitting layers)"
        prog(
            f"Preparing AirLLM layer shards{prep_note} — can take a long time…",
            0.45,
            total_gb=download_gb,
        )

        # Ensure MoE routing / MLX patches are active before prepare
        try:
            from airllm_studio.core.mlx_compat import apply_airllm_mlx_patches

            apply_airllm_mlx_patches()
        except Exception:
            pass

        from airllm import AutoModel  # type: ignore

        kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": repo_id,
            "layer_shards_saving_path": str(shards_path),
        }
        if self.cfg.hf_token:
            kwargs["hf_token"] = self.cfg.hf_token
        if use_compression:
            kwargs["compression"] = use_compression
        if self.cfg.delete_original_after_split:
            kwargs["delete_original"] = True

        # MoE on Mac: AirLLMBaseModel + expert streaming needs an explicit device
        try:
            from airllm_studio.core.airllm_moe import default_mac_device, is_moe_config
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(
                repo_id, trust_remote_code=True, token=self.cfg.hf_token or None
            )
            if is_moe_config(cfg):
                from airllm_studio.core.airllm_moe import (
                    ensure_safetensor_shards,
                    force_safetensor_persister,
                )

                kwargs["device"] = default_mac_device()
                kwargs["force_torch_moe"] = True
                force_safetensor_persister()
                # Wipe/convert prior MLX prepares so split writes safetensors
                ensure_safetensor_shards(str(shards_path))
                prog(
                    f"MoE prepare — expert streaming path on {kwargs['device']}",
                    0.46,
                    total_gb=download_gb,
                )
        except Exception:
            pass

        try:
            model = AutoModel.from_pretrained(repo_id, **{
                k: v for k, v in kwargs.items() if k != "pretrained_model_name_or_path"
            })
        except TypeError:
            kwargs.pop("force_torch_moe", None)
            try:
                model = AutoModel.from_pretrained(**kwargs)
            except TypeError:
                kwargs.pop("device", None)
                model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=repo_id,
                    layer_shards_saving_path=str(shards_path),
                    hf_token=self.cfg.hf_token or None,
                    compression=use_compression,
                )

        del model

        with self._lock:
            e = self._entries[repo_id]
            e.status = "ready"
            e.shards_path = str(shards_path)
            e.local_path = str(shards_path)
            e.info = info.to_dict()
            e.updated_at = _now_iso()
            self._save()
            prog(
                f"Model prepared and ready to load · {format_eta(time.time() - t0)} total",
                1.0,
                eta_seconds=0,
                total_gb=download_gb,
            )
            return e

    def _hf_repo_size_gb(self, repo_id: str) -> Optional[float]:
        """Best-effort total size of model files on the Hub (GB)."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.cfg.hf_token or None)
            info = api.model_info(repo_id, files_metadata=True)
            total = 0
            siblings = getattr(info, "siblings", None) or []
            for s in siblings:
                size = getattr(s, "size", None)
                if size:
                    total += int(size)
            if total <= 0:
                return None
            return total / (1024 ** 3)
        except Exception:
            return None

    def _hf_snapshot_download(
        self,
        repo_id: str,
        prog: Callable,
        total_gb: float,
        pause_event: Optional[threading.Event] = None,
    ) -> None:
        """Download full repo with coarse progress callbacks. Resumes from HF cache."""
        import time

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return

        t0 = time.time()
        locked_total_bytes = [int(total_gb * (1024 ** 3)) if total_gb else 0]
        seen_n = [0]
        pause_event = pause_event or threading.Event()

        def _tqdm_class():
            try:
                from tqdm.auto import tqdm as real_tqdm
            except ImportError:
                return None

            class ProgressTqdm(real_tqdm):
                def update(self, n=1):
                    if pause_event.is_set():
                        raise DownloadPaused(f"Paused during download: {repo_id}")
                    r = super().update(n)
                    try:
                        total = int(self.total or 0)
                        if total > locked_total_bytes[0]:
                            locked_total_bytes[0] = total
                        n_bytes = int(self.n or 0)
                        if n_bytes > seen_n[0]:
                            seen_n[0] = n_bytes
                        disp_total = max(locked_total_bytes[0], 1)
                        disp_n = min(seen_n[0], disp_total)
                        frac = min(0.40, 0.05 + 0.35 * (disp_n / disp_total))
                        elapsed = time.time() - t0
                        eta = (
                            elapsed * (1 / max(frac, 0.01) - 1) if frac > 0 else None
                        )
                        speed = (disp_n / (1024**2) / elapsed) if elapsed > 0 else 0
                        tot_gb = disp_total / (1024**3)
                        done_gb = disp_n / (1024**3)
                        prog(
                            f"Downloading {repo_id}: {done_gb:.2f}/{tot_gb:.2f} GB",
                            frac,
                            total_gb=tot_gb,
                            downloaded_gb=done_gb,
                            speed_mbps=speed * 8,
                            eta_seconds=eta,
                        )
                    except DownloadPaused:
                        raise
                    except Exception:
                        pass
                    return r

            return ProgressTqdm

        tqdm_cls = _tqdm_class()
        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "token": self.cfg.hf_token or None,
            # Resume incomplete files after pause / crash / power-off
            "resume_download": True,
            "cache_dir": str(self.cfg.cache_dir / "huggingface" / "hub"),
        }
        if tqdm_cls is not None:
            kwargs["tqdm_class"] = tqdm_cls
        prog(
            f"Downloading full weights from Hugging Face (~{total_gb:.0f} GB)…",
            0.05,
            total_gb=total_gb,
            eta_seconds=estimate_download_seconds(total_gb),
        )
        if pause_event.is_set():
            raise DownloadPaused(f"Paused: {repo_id}")
        try:
            snapshot_download(**kwargs)
        except DownloadPaused:
            raise
        except Exception as e:
            # Some hub versions reject resume_download kwarg
            if "resume_download" in str(e) or isinstance(e, TypeError):
                kwargs.pop("resume_download", None)
                if pause_event.is_set():
                    raise DownloadPaused(f"Paused: {repo_id}")
                snapshot_download(**kwargs)
            else:
                raise
        if pause_event.is_set():
            raise DownloadPaused(f"Paused: {repo_id}")
        prog(
            "Download complete — starting AirLLM layer prepare…",
            0.42,
            total_gb=max(total_gb, locked_total_bytes[0] / (1024**3)),
        )

    def _demo_prepare(
        self,
        repo_id: str,
        info: ModelInfo,
        shards_path: Path,
        prog: Callable,
        total_gb: float = 10.0,
    ) -> LibraryEntry:
        import time

        # Simulate download proportional to size (capped for UI snappiness)
        # ~0.08s per GB, min 2s max 12s for download phase
        dl_seconds = min(12.0, max(2.0, total_gb * 0.06))
        steps = 40
        t0 = time.time()
        for i in range(steps):
            time.sleep(dl_seconds / steps)
            frac = (i + 1) / steps * 0.7  # download = 70% of job
            done_gb = total_gb * frac / 0.7
            elapsed = time.time() - t0
            speed_mbps = (done_gb * 1024 * 8 / elapsed) if elapsed > 0 else 0
            eta = elapsed * (1 / max(frac, 0.01) - 1)
            prog(
                f"Downloading {repo_id}: {done_gb:.1f}/{total_gb:.0f} GB",
                frac,
                total_gb=total_gb,
                downloaded_gb=done_gb,
                speed_mbps=speed_mbps,
                eta_seconds=eta,
            )

        layers = info.num_layers or 32
        for i in range(layers):
            time.sleep(0.02)
            frac = 0.7 + 0.28 * (i + 1) / layers
            prog(
                f"Preparing AirLLM shards · layer {i + 1}/{layers}",
                frac,
                total_gb=total_gb,
                eta_seconds=max(0, (layers - i - 1) * 0.02),
            )

        marker = shards_path / "DEMO_SHARDS.txt"
        marker.write_text(
            f"Demo shards for {repo_id}\nLayers: {layers}\nSize est: {total_gb} GB\n",
            encoding="utf-8",
        )
        with self._lock:
            e = self._entries.get(repo_id) or LibraryEntry(repo_id=repo_id)
            e.status = "ready"
            e.shards_path = str(shards_path)
            e.local_path = str(shards_path)
            e.info = info.to_dict()
            e.updated_at = _now_iso()
            self._entries[repo_id] = e
            self._save()
        prog(
            f"Ready (demo) · install airllm for real inference · took {format_eta(time.time() - t0)}",
            1.0,
            eta_seconds=0,
            total_gb=total_gb,
        )
        return e

    @staticmethod
    def _airllm_available() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("airllm") is not None
        except Exception:
            return False
