"""AirLLM inference engine with progress hooks and demo fallback."""

from __future__ import annotations

import re
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .config import get_config
from .model_meta import ModelInfo


class EngineStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    STREAMING_LAYER = "streaming_layer"
    STREAMING_EXPERT = "streaming_expert"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    use_cache: bool = True


@dataclass
class EngineEvent:
    kind: str  # status | token | progress | done | error | tool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


EventCb = Callable[[EngineEvent], None]


class InferenceEngine:
    """
    Owns the loaded AirLLM model (or demo backend).

    Generation runs on a worker thread; events go to the callback
    (UI should marshal back to main thread).
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        self._model = None
        self._tokenizer = None
        self._repo_id: Optional[str] = None
        self._info: Optional[ModelInfo] = None
        self._status = EngineStatus.UNLOADED
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._demo = False
        self._layer_index = 0
        self._layer_total = 0
        self._probe = None

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def loaded_repo(self) -> Optional[str]:
        return self._repo_id

    @property
    def model_info(self) -> Optional[ModelInfo]:
        return self._info

    @property
    def is_loaded(self) -> bool:
        return self._repo_id is not None and self._status in (
            EngineStatus.READY,
            EngineStatus.GENERATING,
            EngineStatus.STREAMING_LAYER,
            EngineStatus.STREAMING_EXPERT,
        )

    def load_async(
        self,
        repo_id: str,
        info: Optional[ModelInfo] = None,
        shards_path: Optional[str] = None,
        on_event: Optional[EventCb] = None,
    ) -> threading.Thread:
        def work() -> None:
            try:
                self.load(repo_id, info=info, shards_path=shards_path, on_event=on_event)
            except Exception as exc:
                self._status = EngineStatus.ERROR
                if on_event:
                    on_event(
                        EngineEvent(
                            kind="error",
                            message=str(exc),
                            data={"trace": traceback.format_exc()},
                        )
                    )

        t = threading.Thread(target=work, daemon=True, name=f"load-{repo_id}")
        t.start()
        return t

    def load(
        self,
        repo_id: str,
        info: Optional[ModelInfo] = None,
        shards_path: Optional[str] = None,
        on_event: Optional[EventCb] = None,
    ) -> None:
        def emit(kind: str, message: str = "", **data: Any) -> None:
            if on_event:
                on_event(EngineEvent(kind=kind, message=message, data=data))

        with self._lock:
            self.unload()
            self._status = EngineStatus.LOADING
            self._info = info
            emit("status", f"Loading {repo_id}…", status=self._status.value)

        # Fix Qwen/Llama MLX weight mismatches before any AirLLM import path
        try:
            from airllm_studio.core.mlx_compat import apply_airllm_mlx_patches

            apply_airllm_mlx_patches()
        except Exception as patch_exc:
            emit("progress", f"MLX compat note: {patch_exc}")

        demo = self._should_use_demo()
        if demo:
            import time

            self._demo = True
            layers = (info.num_layers if info else None) or 32
            for i in range(min(layers, 8)):
                time.sleep(0.08)
                emit(
                    "progress",
                    f"Demo: warming layer stream {i + 1}/{layers}",
                    layer=i + 1,
                    total=layers,
                    status=EngineStatus.STREAMING_LAYER.value,
                )
            self._repo_id = repo_id
            self._status = EngineStatus.READY
            self._layer_total = layers
            emit("status", f"Demo model ready: {repo_id}", status=self._status.value)
            emit("done", "loaded", repo_id=repo_id, demo=True)
            return

        self._demo = False
        from airllm import AutoModel  # type: ignore

        kwargs: Dict[str, Any] = {}
        if shards_path:
            kwargs["layer_shards_saving_path"] = shards_path
        if self.cfg.hf_token:
            kwargs["hf_token"] = self.cfg.hf_token
        if self.cfg.compression:
            kwargs["compression"] = self.cfg.compression

        # MoE on Mac uses AirLLMBaseModel with device=mps|cpu (patched AutoModel)
        try:
            from airllm_studio.core.airllm_moe import (
                default_mac_device,
                ensure_safetensor_shards,
                force_safetensor_persister,
                is_moe_config,
            )
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=True,
                token=self.cfg.hf_token or None,
            )
            if is_moe_config(cfg):
                kwargs["device"] = default_mac_device()
                kwargs["force_torch_moe"] = True
                force_safetensor_persister()
                # Convert any prior MLX .npz shards so load does not hit
                # nested-dict / is_floating_point crash
                if shards_path:
                    emit(
                        "progress",
                        "MoE: ensuring safetensor layer shards…",
                        status=EngineStatus.LOADING.value,
                    )

                    def _prog(i, total, name):
                        emit(
                            "progress",
                            f"Converting layer shards {i}/{total}: {name}",
                            layer=i,
                            total=total,
                            status=EngineStatus.LOADING.value,
                        )

                    ensure_safetensor_shards(str(shards_path), progress_cb=_prog)
                emit(
                    "progress",
                    f"MoE model — expert streaming on {kwargs['device']}…",
                    status=EngineStatus.LOADING.value,
                )
        except Exception as moe_exc:
            emit("progress", f"MoE setup note: {moe_exc}")

        emit(
            "progress",
            "Initializing AirLLM (layer / expert shards)…",
            status=EngineStatus.LOADING.value,
        )
        try:
            model = AutoModel.from_pretrained(repo_id, **kwargs)
        except TypeError:
            # Older signatures
            kwargs.pop("force_torch_moe", None)
            try:
                model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=repo_id, **kwargs
                )
            except TypeError:
                kwargs.pop("device", None)
                model = AutoModel.from_pretrained(repo_id, **kwargs)

        with self._lock:
            self._model = model
            self._tokenizer = getattr(model, "tokenizer", None)
            self._repo_id = repo_id
            self._status = EngineStatus.READY
            if info and info.num_layers:
                self._layer_total = info.num_layers
            else:
                self._layer_total = len(getattr(model, "layers", []) or [])
            args = getattr(model, "model_args", None)
            if args is not None and getattr(args, "n_layers", None):
                self._layer_total = int(args.n_layers)
            try:
                from airllm_studio.core.activity import ActivityProbe

                self._probe = ActivityProbe()
                self._probe.attach(model)
            except Exception as probe_exc:
                emit("progress", f"Activity probe: {probe_exc}")
                self._probe = None

        emit("status", f"Model loaded: {repo_id}", status=self._status.value)
        emit("done", "loaded", repo_id=repo_id, demo=False)

    def unload(self) -> None:
        with self._lock:
            if self._probe is not None:
                try:
                    self._probe.detach()
                except Exception:
                    pass
                self._probe = None
            self._model = None
            self._tokenizer = None
            self._repo_id = None
            self._status = EngineStatus.UNLOADED
            self._demo = False

    def stop_generation(self) -> None:
        self._stop.set()

    def generate_async(
        self,
        messages: List[Dict[str, str]],
        gen_cfg: Optional[GenerationConfig] = None,
        on_event: Optional[EventCb] = None,
    ) -> threading.Thread:
        self._stop.clear()

        def work() -> None:
            try:
                self.generate(messages, gen_cfg=gen_cfg, on_event=on_event)
            except Exception as exc:
                self._status = EngineStatus.ERROR
                if on_event:
                    on_event(
                        EngineEvent(
                            kind="error",
                            message=str(exc),
                            data={"trace": traceback.format_exc()},
                        )
                    )

        t = threading.Thread(target=work, daemon=True, name="generate")
        t.start()
        return t

    def generate(
        self,
        messages: List[Dict[str, str]],
        gen_cfg: Optional[GenerationConfig] = None,
        on_event: Optional[EventCb] = None,
    ) -> str:
        gen_cfg = gen_cfg or GenerationConfig(
            temperature=self.cfg.default_temperature,
            max_new_tokens=self.cfg.default_max_tokens,
            top_p=self.cfg.default_top_p,
        )

        def emit(kind: str, message: str = "", **data: Any) -> None:
            if on_event:
                on_event(EngineEvent(kind=kind, message=message, data=data))

        if not self._repo_id:
            raise RuntimeError("No model loaded. Load a model from the library first.")

        self._status = EngineStatus.GENERATING
        emit("status", "Generating…", status=self._status.value)

        prompt = self._format_chat(messages)

        if self._demo or self._model is None:
            text = self._demo_generate(prompt, gen_cfg, emit)
            self._status = EngineStatus.READY
            emit("done", text, full_text=text)
            return text

        return self._airllm_generate(prompt, gen_cfg, emit)

    def _airllm_generate(
        self,
        prompt: str,
        gen_cfg: GenerationConfig,
        emit: Callable[..., None],
    ) -> str:
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Model not fully initialized")

        layers = self._layer_total or (
            self._info.num_layers if self._info else 0
        ) or 0

        # Progress feedback while generating (layer streaming is internal to AirLLM)
        if layers:
            emit(
                "progress",
                f"Streaming layers through unified memory (0/{layers})…",
                layer=0,
                total=layers,
                status=EngineStatus.STREAMING_LAYER.value,
            )
        if self._info and self._info.is_moe:
            emit(
                "progress",
                f"MoE: routing tokens to ~{self._info.experts_per_token or '?'} of "
                f"{self._info.num_experts or '?'} experts",
                status=EngineStatus.STREAMING_EXPERT.value,
            )

        max_length = min(len(prompt) + gen_cfg.max_new_tokens * 4, 8192)
        input_tokens = tokenizer(
            [prompt],
            return_tensors="np",
            return_attention_mask=False,
            truncation=True,
            max_length=max(256, max_length // 2),
            padding=False,
        )
        input_ids = input_tokens["input_ids"]

        self._status = EngineStatus.STREAMING_LAYER
        if self._probe is not None:
            self._probe.set_emit(lambda payload: emit("activity", **payload))
        # Stream tokens into the UI as they are produced
        try:
            text = self._run_generate_streaming(
                model,
                tokenizer,
                input_ids,
                gen_cfg,
                emit,
            )
        finally:
            if self._probe is not None:
                self._probe.set_emit(None)

        self._status = EngineStatus.READY
        emit("status", "Ready", status=self._status.value)
        emit("done", text, full_text=text)
        return text

    def _run_generate_streaming(
        self,
        model,
        tokenizer,
        input_ids,
        gen_cfg: GenerationConfig,
        emit: Callable[..., None],
    ) -> str:
        """Generate with true per-token UI updates (not one dump at the end)."""
        temp = float(gen_cfg.temperature) if gen_cfg.temperature > 0.01 else 0.0
        max_new = int(gen_cfg.max_new_tokens)
        is_mlx = type(model).__name__ == "AirLLMLlamaMlx"

        if is_mlx:
            return self._stream_mlx(model, tokenizer, input_ids, temp, max_new, emit)

        return self._stream_torch(
            model, tokenizer, input_ids, gen_cfg, temp, max_new, emit
        )

    def _stream_mlx(
        self,
        model,
        tokenizer,
        input_ids,
        temp: float,
        max_new: int,
        emit: Callable[..., None],
    ) -> str:
        import mlx.core as mx
        import numpy as np

        if hasattr(input_ids, "detach"):
            arr = input_ids.detach().cpu().numpy()
        else:
            arr = np.asarray(input_ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        x = mx.array(arr.astype("int32"))

        token_ids: List[int] = []
        acc = ""

        def push_delta() -> None:
            nonlocal acc
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if text != acc:
                acc = text
                emit("token", acc, full_text=acc)

        # True per-token iterator (AirLLMLlamaMlx.model_generate)
        if hasattr(model, "model_generate"):
            n = 0
            try:
                for token in model.model_generate(x, temperature=temp):
                    if self._stop.is_set():
                        break
                    try:
                        tid = int(token.item()) if hasattr(token, "item") else int(token)
                    except Exception:
                        continue
                    eos = getattr(tokenizer, "eos_token_id", None)
                    if eos is not None and tid == eos:
                        break
                    token_ids.append(tid)
                    push_delta()
                    n += 1
                    if n >= max_new:
                        break
                    if n == 1 or n % 2 == 0:
                        emit(
                            "progress",
                            f"Token {n}/{max_new}",
                            layer=n,
                            total=max_new,
                            status=EngineStatus.STREAMING_LAYER.value,
                        )
            except Exception as mlx_exc:
                emit("progress", f"MLX token stream error: {mlx_exc}")
                if not token_ids:
                    out = model.generate(x, temperature=temp, max_new_tokens=max_new)
                    text = out.strip() if isinstance(out, str) else str(out)
                    self._emit_tokenized(text, tokenizer, emit)
                    return text
            return acc.strip()

        out = model.generate(x, temperature=temp, max_new_tokens=max_new)
        text = out.strip() if isinstance(out, str) else str(out)
        self._emit_tokenized(text, tokenizer, emit)
        return text

    def _stream_torch(
        self,
        model,
        tokenizer,
        input_ids,
        gen_cfg: GenerationConfig,
        temp: float,
        max_new: int,
        emit: Callable[..., None],
    ) -> str:
        """MoE / AirLLMBaseModel (PyTorch) — stream tokens via HF streamer."""
        import threading
        import numpy as np
        import torch

        if not hasattr(input_ids, "to"):
            input_ids = torch.from_numpy(np.asarray(input_ids)).long()
        else:
            input_ids = input_ids.long()

        device = None
        if hasattr(model, "running_device"):
            device = torch.device(model.running_device)
        elif hasattr(model, "device") and isinstance(model.device, torch.device):
            device = model.device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        if device is not None:
            input_ids = input_ids.to(device)

        # Prefer HF TextIteratorStreamer for true per-token (or per-decode-piece) UI
        try:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            do_sample = temp > 0.01
            gen_common: Dict[str, Any] = dict(
                max_new_tokens=max_new,
                do_sample=do_sample,
                streamer=streamer,
            )
            if do_sample:
                gen_common["temperature"] = max(temp, 0.01)
                gen_common["top_p"] = gen_cfg.top_p

            err_box: list = []

            def _worker() -> None:
                try:
                    # transformers 5+ may try to set experts_implementation on generate;
                    # our AirLLM runtime wrapper sometimes rejects that — neutralize it.
                    target = model
                    if hasattr(model, "model") and hasattr(model.model, "generate"):
                        target = model.model
                        # Ensure class accepts experts_implementation no-ops
                        if not hasattr(target, "set_attn_implementation"):
                            pass
                        try:
                            # Prefer eager experts path if API exists
                            if hasattr(target, "set_experts_implementation"):
                                target.set_experts_implementation("eager")
                        except Exception:
                            pass
                        # Also pass attention_mask when missing to silence warnings
                        am = torch.ones_like(input_ids)
                        try:
                            target.generate(
                                input_ids=input_ids,
                                attention_mask=am,
                                **gen_common,
                            )
                        except TypeError:
                            # older / strict signatures
                            try:
                                target.generate(input_ids=input_ids, **gen_common)
                            except Exception as exc2:
                                # Last try via AirLLM wrapper
                                model.generate(input_ids, attention_mask=am, **gen_common)
                    else:
                        model.generate(input_ids, **gen_common)
                except Exception as exc:
                    err_box.append(exc)
                    try:
                        streamer.on_finalized_text("", stream_end=True)
                    except Exception:
                        pass

            th = threading.Thread(target=_worker, daemon=True, name="hf-stream")
            th.start()
            acc = ""
            n = 0
            for piece in streamer:
                if self._stop.is_set():
                    break
                if not piece:
                    continue
                acc += piece
                emit("token", acc, full_text=acc)
                n += 1
                if n == 1 or n % 2 == 0:
                    emit(
                        "progress",
                        f"Token {n}…",
                        status=EngineStatus.STREAMING_LAYER.value,
                    )
            th.join(timeout=600)
            if acc:
                return acc.strip()
            if err_box:
                emit("progress", f"Streamer error: {err_box[0]}")
        except Exception as stream_exc:
            emit("progress", f"Streamer fallback: {stream_exc}")

        # Manual token-by-token loop (slower; real streaming if streamer unavailable)
        try:
            text = self._torch_manual_token_loop(
                model, tokenizer, input_ids, temp, max_new, gen_cfg, emit
            )
            if text:
                return text
        except Exception as manual_exc:
            emit("progress", f"Manual stream failed: {manual_exc}")

        # Non-streaming fallback — still emit per tokenizer piece for UI
        try:
            generation_output = model.generate(
                input_ids,
                max_new_tokens=max_new,
                return_dict_in_generate=True,
                do_sample=temp > 0.01,
                temperature=max(temp, 0.01) if temp > 0.01 else 1.0,
                top_p=gen_cfg.top_p,
            )
        except TypeError:
            generation_output = model.generate(input_ids, max_new_tokens=max_new)

        if isinstance(generation_output, str):
            text = generation_output.strip()
        else:
            try:
                sequences = generation_output.sequences[0]
                prompt_len = input_ids.shape[-1]
                new_tokens = sequences[prompt_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception:
                text = str(generation_output)
        self._emit_tokenized(text, tokenizer, emit)
        return text

    def _torch_manual_token_loop(
        self,
        model,
        tokenizer,
        input_ids,
        temp: float,
        max_new: int,
        gen_cfg: GenerationConfig,
        emit: Callable[..., None],
    ) -> str:
        """Generate one token at a time so the UI updates per token."""
        import torch

        target = model.model if hasattr(model, "model") else model
        cur = input_ids
        acc = ""
        token_ids: List[int] = []
        eos = getattr(tokenizer, "eos_token_id", None)

        for n in range(max_new):
            if self._stop.is_set():
                break
            with torch.no_grad():
                out = target(input_ids=cur)
                logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]
                if temp > 0.01:
                    probs = torch.softmax(logits / max(temp, 0.01), dim=-1)
                    # top-p filter (lightweight)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > gen_cfg.top_p
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_idx.gather(-1, next_sorted)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

            tid = int(next_id.item())
            if eos is not None and tid == eos:
                break
            token_ids.append(tid)
            acc = tokenizer.decode(token_ids, skip_special_tokens=True)
            emit("token", acc, full_text=acc)
            if n == 0 or (n + 1) % 2 == 0:
                emit(
                    "progress",
                    f"Token {n + 1}/{max_new}",
                    layer=n + 1,
                    total=max_new,
                    status=EngineStatus.STREAMING_LAYER.value,
                )
            cur = torch.cat([cur, next_id], dim=-1)
        return acc.strip()

    def _demo_generate(
        self,
        prompt: str,
        gen_cfg: GenerationConfig,
        emit: Callable[..., None],
    ) -> str:
        import time

        layers = self._layer_total or 32
        info = self._info
        moe_note = ""
        if info and info.is_moe:
            moe_note = (
                f" MoE routing: {info.experts_per_token}/{info.num_experts} experts active."
            )

        for i in range(0, min(layers, 12)):
            if self._stop.is_set():
                break
            time.sleep(0.05)
            emit(
                "progress",
                f"Streaming layer {i + 1}/{layers}…{moe_note}",
                layer=i + 1,
                total=layers,
                status=EngineStatus.STREAMING_LAYER.value,
            )
            # Synthetic activity so the left panel animates in demo
            fake = [((i * 7 + k * 13) % 100) / 100.0 for k in range(48)]
            emit(
                "activity",
                layer=i,
                layers_total=layers,
                layers_remaining=max(0, layers - i - 1),
                experts_active=list(range(info.experts_per_token or 0)) if info and info.is_moe else [],
                experts_total=(info.num_experts or 0) if info else 0,
                neurons=fake,
                stage="layer",
            )

        user_tail = prompt.strip().split("User:")[-1].strip()[:200]
        reply = (
            f"[Demo mode — AirLLM not loaded]\n\n"
            f"You said: {user_tail or '(empty)'}\n\n"
            f"In a real session, **{self._repo_id}** would stream "
            f"{'experts and ' if info and info.is_moe else ''}layers through "
            f"unified memory (~"
            f"{(getattr(info, 'estimated_airllm_memory_gb', None) or 2):.1f} GB peak).\n\n"
            f"Install dependencies:\n"
            f"`pip install airllm mlx torch transformers`\n\n"
            f"Then prepare & load the model for true local inference."
        )
        words = reply.split()
        reply = " ".join(words[: gen_cfg.max_new_tokens])
        self._emit_tokenized(reply, None, emit)
        return reply

    def _emit_tokenized(
        self, text: str, tokenizer, emit: Callable[..., None]
    ) -> None:
        """Emit already-produced text as sequential tokens/pieces (fallback only)."""
        import time

        if tokenizer is not None:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                acc = ""
                for i, tid in enumerate(ids):
                    if self._stop.is_set():
                        break
                    acc = tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
                    emit("token", acc, full_text=acc)
                    time.sleep(0.005)
                return
            except Exception:
                pass
        # Character-level last resort
        acc = ""
        for ch in text:
            if self._stop.is_set():
                break
            acc += ch
            emit("token", acc, full_text=acc)
            time.sleep(0.004)

    def _emit_chunked(self, text: str, emit: Callable[..., None], chunk: int = 1) -> None:
        """Legacy progressive UI updates."""
        self._emit_tokenized(text, None, emit)

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Best-effort chat template; falls back to simple roles."""
        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool result: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _should_use_demo(self) -> bool:
        """Demo path when AirLLM toggle is off, forced demo, or package missing."""
        if getattr(self.cfg, "demo_mode", False):
            return True
        if not getattr(self.cfg, "airllm_enabled", True):
            return True
        return not self._airllm_available()

    @staticmethod
    def _airllm_available() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("airllm") is not None
        except Exception:
            return False

    def airllm_active(self) -> bool:
        """True if the next/current load will use real AirLLM (not demo)."""
        return not self._should_use_demo()


# Tool-call extraction helpers for simple agent loop
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\w+)\s*\n?(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# Alternate JSON-ish: call tool_name with query
TOOL_CALL_RE2 = re.compile(
    r"TOOL_CALL:\s*(\w+)\s*\((.*)\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_tool_calls(text: str) -> List[Dict[str, str]]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        calls.append({"name": m.group(1).strip(), "arguments": m.group(2).strip()})
    if not calls:
        for m in TOOL_CALL_RE2.finditer(text):
            calls.append({"name": m.group(1).strip(), "arguments": m.group(2).strip()})
    return calls


SYSTEM_TOOL_PROMPT = """You are a helpful local assistant on Apple Silicon.
Use web_search ONLY for current or external facts. Never search greetings or chit-chat.
If you need it, emit exactly:

<tool_call>
web_search
your search query here
</tool_call>

Otherwise answer normally. Do not invent tool results."""
