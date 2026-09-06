"""
MoE support for AirLLM on Apple Silicon.

On macOS, stock airllm.AutoModel always uses AirLLMLlamaMlx (dense Llama-only).
MoE checkpoints (Mixtral, Qwen2-MoE, DeepSeek-V2/V3-style block_sparse_moe /
mlp.experts) need AirLLMBaseModel's meta-device layer streaming **and**
per-expert hooks when `expert_prefix` is set.

We:
  1. Subclass AirLLMBaseModel with the right layer / expert name maps
  2. Patch AutoModel.from_pretrained to route MoE models here with device=mps|cpu
  3. Force SafetensorModelPersister (not Mlx) and convert any .mlx.npz shards
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple, Type

_PATCHED = False
_MOE_SAFETENSOR_MODE = False


def _detect_expert_prefix(config) -> Optional[str]:
    """Guess HF module path for expert ModuleList inside each decoder layer."""
    archs = [a.lower() for a in (getattr(config, "architectures", None) or [])]
    model_type = str(getattr(config, "model_type", "") or "").lower()
    blob = " ".join(archs) + " " + model_type

    # Mixtral / many sparse MoE: model.layers.i.block_sparse_moe.experts
    if any(x in blob for x in ("mixtral", "block_sparse_moe", "kimik3", "kimi_k3")):
        return "block_sparse_moe.experts"

    # Qwen2-MoE / Qwen1.5-MoE: model.layers.i.mlp.experts
    if any(x in blob for x in ("qwen2_moe", "qwen3_moe", "qwen2moe", "qwen1.5-moe", "qwen_moe")):
        return "mlp.experts"

    # DeepSeek-V2/V3 style often uses mlp.experts
    if any(x in blob for x in ("deepseek", "dbrx", "grok")):
        return "mlp.experts"

    # Generic hints on config
    if getattr(config, "num_local_experts", None) or getattr(config, "n_routed_experts", None):
        if "mixtral" in blob or "mistral" in blob:
            return "block_sparse_moe.experts"
        return "mlp.experts"

    if getattr(config, "num_experts", None):
        if "mixtral" in blob:
            return "block_sparse_moe.experts"
        return "mlp.experts"

    return None


def is_moe_config(config) -> bool:
    if _detect_expert_prefix(config):
        return True
    if getattr(config, "num_local_experts", None):
        return True
    if getattr(config, "n_routed_experts", None):
        return True
    if getattr(config, "num_experts", None):
        return True
    archs = " ".join(getattr(config, "architectures", None) or []).lower()
    return any(x in archs for x in ("moe", "mixtral", "deepseek"))


def _make_moe_class(expert_prefix: str, embed: str, layer_prefix: str, norm: str, lm_head: str):
    from airllm.airllm_base import AirLLMBaseModel

    class AirLLMStudioMoE(AirLLMBaseModel):
        def set_layer_names_dict(self):
            self.layer_names_dict = {
                "embed": embed,
                "layer_prefix": layer_prefix,
                "norm": norm,
                "lm_head": lm_head,
                "expert_prefix": expert_prefix,
            }

        def get_use_better_transformer(self):
            return False

    AirLLMStudioMoE.__name__ = "AirLLMStudioMoE"
    return AirLLMStudioMoE


def pick_moe_class(config) -> Tuple[Type, str]:
    """Return (ModelClass, expert_prefix)."""
    expert = _detect_expert_prefix(config) or "block_sparse_moe.experts"
    archs = " ".join(getattr(config, "architectures", None) or []).lower()

    if "kimik3" in archs or "kimi" in str(getattr(config, "model_type", "")).lower():
        from airllm.airllm_kimi_k3 import AirLLMKimiK3

        return AirLLMKimiK3, "block_sparse_moe.experts"

    cls = _make_moe_class(
        expert_prefix=expert,
        embed="model.embed_tokens",
        layer_prefix="model.layers",
        norm="model.norm",
        lm_head="lm_head",
    )
    return cls, expert


def default_mac_device() -> str:
    """
    Device for MoE (AirLLMBaseModel) on Mac.

    Default is **cpu**. Whole-layer MoE streaming thrashs weights meta↔GPU every
    layer; on Apple MPS that frequently aborts the process with:

        Cannot form weak reference to instance of class MPSGraph
        EXC_CRASH (SIGABRT) on metal gpu stream

    especially on beta macOS. Override with env AIRLLM_MOE_DEVICE=mps if you want
    to try GPU (faster when it works, but can crash the app hard).
    """
    import os

    forced = (os.environ.get("AIRLLM_MOE_DEVICE") or "").strip().lower()
    if forced in ("cpu", "mps"):
        return forced
    if forced in ("gpu", "metal"):
        return "mps"
    # Safe default — process must not SIGABRT under Tk
    return "cpu"


def apply_mps_safety_patches() -> None:
    """
    Harden AirLLM memory free path for MPS.

    Stock clean_memory() only calls cuda.empty_cache(). On MPS, freeing layers to
    meta without synchronize() races the Metal command stream and can SIGABRT.
    """
    try:
        import gc
        import torch
        import airllm.utils as airllm_utils
        from airllm.airllm_base import AirLLMBaseModel
    except Exception as exc:
        print(f"[airllm_moe] mps safety skip: {exc}")
        return

    def _safe_clean_memory() -> None:
        gc.collect()
        try:
            # Linux only; ignore on macOS
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # Drain Metal queue BEFORE releasing MPSGraph-backed tensors
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

    airllm_utils.clean_memory = _safe_clean_memory

    # post_hook frees weights while MPS may still be executing — sync first
    _orig_post = AirLLMBaseModel._post_hook

    def _safe_post_hook(self, module, args, output):
        dev = str(getattr(self, "running_device", "") or "")
        if "mps" in dev:
            try:
                import torch

                torch.mps.synchronize()
            except Exception:
                pass
        return _orig_post(self, module, args, output)

    AirLLMBaseModel._post_hook = _safe_post_hook  # type: ignore

    # Also sync before moving next layer onto MPS
    _orig_pre = AirLLMBaseModel._pre_hook

    def _safe_pre_hook(self, module, args):
        dev = str(getattr(self, "running_device", "") or "")
        if "mps" in dev:
            try:
                import torch

                torch.mps.synchronize()
            except Exception:
                pass
        return _orig_pre(self, module, args)

    AirLLMBaseModel._pre_hook = _safe_pre_hook  # type: ignore
    print("[airllm_moe] MPS safety patches applied (sync before free)")


def force_safetensor_persister() -> None:
    """
    MoE AirLLMBaseModel needs .safetensors layer shards (for seekable expert loads).
    Stock airllm on darwin always installs MlxModelPersister (.mlx.npz) — wrong for MoE.

    We set the singleton *and* patch get_model_persister so later calls cannot
    reinstall the MLX persister while MoE mode is active.
    """
    global _MOE_SAFETENSOR_MODE
    _MOE_SAFETENSOR_MODE = True
    try:
        from airllm.persist import model_persister as mp
        from airllm.persist.safetensor_model_persister import SafetensorModelPersister

        st = SafetensorModelPersister()
        mp.model_persister = st

        # Patch classmethod so get_model_persister never re-creates Mlx on darwin
        _orig_get = mp.ModelPersister.get_model_persister.__func__  # type: ignore

        @classmethod  # type: ignore
        def _get_persister_moe(cls):
            if _MOE_SAFETENSOR_MODE:
                if mp.model_persister is None or type(mp.model_persister).__name__ != "SafetensorModelPersister":
                    mp.model_persister = SafetensorModelPersister()
                return mp.model_persister
            return _orig_get(cls)

        if getattr(mp.ModelPersister.get_model_persister, "__name__", "") != "_get_persister_moe":
            mp.ModelPersister.get_model_persister = _get_persister_moe  # type: ignore

        print("[airllm_moe] ModelPersister → SafetensorModelPersister (MoE)")
    except Exception as exc:
        print(f"[airllm_moe] cannot force safetensor persister: {exc}")


def _iter_shard_dirs(checkpoint_dir: str):
    p = Path(checkpoint_dir)
    for d in (p / "splitted_model", p, p.parent / "splitted_model"):
        if d.is_dir():
            yield d


def convert_mlx_shards_to_safetensors(shard_dir: Path, progress_cb=None) -> int:
    """
    Convert AirLLM MLX layer shards (.mlx.npz) to flat torch .safetensors.

    MLX load returns nested dicts (tree_unflatten) which crash AirLLMBaseModel
    with: 'dict' object has no attribute 'is_floating_point'.
    The raw .mlx.npz files already store float16 numpy tensors with HF key names —
    convert in place so MoE does not need a full re-download.
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file

    mlx_files = sorted(shard_dir.glob("*.mlx.npz"))
    if not mlx_files:
        return 0

    converted = 0
    total = len(mlx_files)
    for i, npz_path in enumerate(mlx_files):
        # model.layers.0.mlx.npz → model.layers.0.safetensors
        stem = npz_path.name[: -len(".mlx.npz")]
        out_path = shard_dir / f"{stem}.safetensors"
        done_path = shard_dir / f"{stem}.safetensors.done"

        if out_path.exists() and out_path.stat().st_size > 0:
            if not done_path.exists():
                done_path.touch()
            converted += 1
            if progress_cb:
                progress_cb(i + 1, total, stem)
            continue

        if progress_cb:
            progress_cb(i + 1, total, stem)

        data = np.load(str(npz_path), allow_pickle=False)
        state: dict = {}
        for key in data.files:
            arr = data[key]
            if not isinstance(arr, np.ndarray):
                continue
            # Contiguous copy required for torch.from_numpy on some arrays
            t = torch.from_numpy(np.ascontiguousarray(arr))
            state[key] = t
        data.close()

        if not state:
            print(f"[airllm_moe] skip empty mlx shard: {npz_path.name}")
            continue

        save_file(state, str(out_path))
        done_path.touch()
        converted += 1
        # Free RAM between layers
        del state

    # Remove MLX shards only after all conversions succeeded
    if converted >= total and total > 0:
        for npz_path in mlx_files:
            try:
                npz_path.unlink(missing_ok=True)
            except Exception:
                pass
            done_mlx = Path(str(npz_path)[: -len(".npz")] + ".done")
            # also model.layers.0.mlx.done
            alt_done = shard_dir / (npz_path.name.replace(".mlx.npz", ".mlx.done"))
            for d in (done_mlx, alt_done):
                try:
                    d.unlink(missing_ok=True)
                except Exception:
                    pass
        print(f"[airllm_moe] converted {converted} MLX shards → safetensors in {shard_dir}")

    return converted


def unpack_packed_expert_shards(shard_dir: Path, progress_cb=None) -> int:
    """
    Split packed 3D expert tensors back into per-expert keys.

    Per-expert keys let safetensors *seek* only the experts a token routes to
    (AirLLM load_layer_subset) — orders of magnitude less I/O than whole-layer load.
    """
    from safetensors.torch import load_file, save_file
    import re

    st_files = sorted(shard_dir.glob("model.layers.*.safetensors"))
    if not st_files:
        return 0

    marker = shard_dir / ".per_expert_shards.done"
    packed_marker = shard_dir / ".packed_experts.done"

    sample = load_file(str(st_files[0]))
    sample_keys = list(sample.keys())
    already_per = any(".experts.0." in k for k in sample_keys)
    is_packed = any(k.endswith(".experts.gate_up_proj") for k in sample_keys)
    del sample

    if already_per and not is_packed:
        if not marker.exists():
            marker.touch()
        if packed_marker.exists():
            try:
                packed_marker.unlink()
            except Exception:
                pass
        return 0
    if not is_packed:
        return 0

    n = 0
    total = len(st_files)
    gu_re = re.compile(r"^(?P<pre>.+\.experts)\.gate_up_proj$")
    dn_re = re.compile(r"^(?P<pre>.+\.experts)\.down_proj$")

    for i, path in enumerate(st_files):
        if progress_cb:
            progress_cb(i + 1, total, path.stem)
        sd = load_file(str(path))
        out = dict(sd)
        # Find packed pairs
        prefixes = set()
        for k in list(out.keys()):
            m = gu_re.match(k)
            if m:
                prefixes.add(m.group("pre"))
        if not prefixes:
            continue
        for pre in prefixes:
            gu_key = f"{pre}.gate_up_proj"
            dn_key = f"{pre}.down_proj"
            if gu_key not in out or dn_key not in out:
                continue
            gate_up = out.pop(gu_key)  # [E, 2I, H]
            down = out.pop(dn_key)  # [E, H, I]
            e = gate_up.shape[0]
            for ei in range(e):
                g, u = gate_up[ei].chunk(2, dim=0)
                out[f"{pre}.{ei}.gate_proj.weight"] = g.contiguous()
                out[f"{pre}.{ei}.up_proj.weight"] = u.contiguous()
                out[f"{pre}.{ei}.down_proj.weight"] = down[ei].contiguous()
        save_file(out, str(path))
        n += 1
        del sd, out

    if n:
        marker.touch()
        try:
            packed_marker.unlink(missing_ok=True)
        except Exception:
            pass
        print(
            f"[airllm_moe] unpacked {n} layer shards → per-expert keys "
            f"(enables seekable expert streaming) in {shard_dir}"
        )
    return n


def ensure_safetensor_shards(checkpoint_dir: str, progress_cb=None) -> None:
    """
    Ensure MoE split dir has .safetensors with **per-expert** keys
    (convert MLX → safetensors, unpack packed 3D if needed).
    """
    import shutil

    for d in _iter_shard_dirs(checkpoint_dir):
        mlx = list(d.glob("*.mlx.npz"))
        st = list(d.glob("*.safetensors"))
        if mlx and not st:
            print(f"[airllm_moe] converting MLX-only shards → safetensors: {d}")
            n = convert_mlx_shards_to_safetensors(d, progress_cb=progress_cb)
            if n == 0:
                print(f"[airllm_moe] convert failed; removing MLX dir for re-split: {d}")
                shutil.rmtree(d, ignore_errors=True)
                continue
            st = list(d.glob("*.safetensors"))
        elif mlx and st:
            convert_mlx_shards_to_safetensors(d, progress_cb=progress_cb)
            st = list(d.glob("*.safetensors"))

        if st:
            try:
                unpack_packed_expert_shards(d, progress_cb=progress_cb)
            except Exception as exc:
                print(f"[airllm_moe] per-expert unpack note: {exc}")


def expand_packed_experts_to_modulelist(model, config) -> int:
    """
    Replace transformers packed Qwen2MoeExperts (3D tensors, no per-expert modules)
    with a ModuleList of small expert MLPs whose forward is called individually.

    AirLLM per-expert hooks only fire when each expert is a real submodule that
    runs in forward — packed tensor indexing never triggers hooks, so the whole
    layer had to load (~all 60 experts) every step.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    try:
        from transformers.activations import ACT2FN
    except Exception:
        ACT2FN = None

    # Locate decoder layers
    layers = None
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__"):
            layers = obj
            break
    if layers is None:
        return 0

    num_experts = int(
        getattr(config, "num_experts", None)
        or getattr(config, "num_local_experts", None)
        or getattr(config, "n_routed_experts", None)
        or 0
    )
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    inter = int(
        getattr(config, "moe_intermediate_size", None)
        or getattr(config, "intermediate_size", None)
        or 0
    )
    if not (num_experts and hidden and inter):
        return 0

    act_name = getattr(config, "hidden_act", "silu")
    if ACT2FN is not None and act_name in ACT2FN:
        act_fn = ACT2FN[act_name]
    else:
        act_fn = nn.SiLU()

    class ExpertMLP(nn.Module):
        """Single MoE expert — AirLLM attaches load/free hooks here."""

        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)
            self.act_fn = act_fn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class ExpertModuleList(nn.ModuleList):
        """Drop-in for Qwen2MoeExperts.forward signature; calls each hit expert module."""

        def __init__(self, modules):
            super().__init__(modules)
            self.num_experts = len(modules)

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            # Match transformers Qwen2MoeExperts routing mask logic
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for row in expert_hit:
                expert_idx = int(row[0].item())
                if expert_idx >= self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                # Submodule call → AirLLM expert pre/post hooks load only this expert
                current_hidden = self[expert_idx](current_state)
                current_hidden = current_hidden * top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden.to(final_hidden_states.dtype)
                )
            return final_hidden_states

    replaced = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            # Mixtral-style
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            experts = getattr(moe, "experts", None)
            if isinstance(experts, nn.ModuleList):
                continue  # already per-expert (Mixtral)
            continue

        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue
        if isinstance(experts, nn.ModuleList):
            # Already list-like; ensure it has our forward if packed was never used
            if type(experts).__name__ == "ExpertModuleList":
                continue
            # Stock ModuleList without custom forward — only works if block iterates experts
            # Leave Mixtral alone
            continue

        # Packed Qwen2MoeExperts (has gate_up_proj / down_proj 3D params, no __len__)
        n = int(getattr(experts, "num_experts", num_experts) or num_experts)
        with torch.device("meta"):
            new_experts = ExpertModuleList([ExpertMLP() for _ in range(n)])
        mlp.experts = new_experts
        replaced += 1

    if replaced:
        print(
            f"[airllm_moe] expanded packed experts → ModuleList "
            f"({num_experts} experts × {replaced} layers) for per-expert streaming"
        )
    return replaced


def _flatten_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Flatten nested dict / list structures into dotted tensor keys."""
    import torch

    out = {}
    if not isinstance(state_dict, dict):
        return out
    for k, v in state_dict.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, torch.Tensor):
            out[key] = v
        elif hasattr(v, "dtype") and hasattr(v, "shape") and not isinstance(v, dict):
            # numpy / mlx array → torch
            try:
                import numpy as np

                if hasattr(v, "tolist") and not isinstance(v, np.ndarray):
                    # mlx array
                    try:
                        import mlx.core as mx

                        v = np.array(v)
                    except Exception:
                        v = np.asarray(v)
                if isinstance(v, np.ndarray):
                    out[key] = torch.from_numpy(np.ascontiguousarray(v))
                    continue
            except Exception:
                pass
        if isinstance(v, dict):
            out.update(_flatten_state_dict(v, key))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten_state_dict(item, f"{key}.{i}"))
                elif isinstance(item, torch.Tensor):
                    out[f"{key}.{i}"] = item
    return out


def patch_should_load_verbatim() -> None:
    """Guard against dict / non-tensor values and packed MoE expert modules."""
    try:
        from airllm.airllm_base import AirLLMBaseModel
        import torch

        _orig = AirLLMBaseModel._should_load_verbatim

        def _safe(self, param_name, value):
            if not isinstance(value, torch.Tensor):
                # Never call .is_floating_point() on nested dicts / metadata
                return True
            return _orig(self, param_name, value)

        AirLLMBaseModel._should_load_verbatim = _safe  # type: ignore

        _orig_move = AirLLMBaseModel.move_layer_to_device

        def _safe_move(self, state_dict):
            if not isinstance(state_dict, dict):
                return []
            # Detect nested (MLX tree_unflatten) or non-tensor payloads
            needs_clean = any(not isinstance(v, torch.Tensor) for v in state_dict.values())
            if needs_clean:
                flat = _flatten_state_dict(state_dict)
                # Prefer only real tensors
                clean = {k: v for k, v in flat.items() if isinstance(v, torch.Tensor)}
                if not clean:
                    # Drop non-tensors but keep flat tensors if any were mixed
                    clean = {
                        k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
                    }
                if not clean:
                    raise TypeError(
                        "Layer shard has no torch tensors (got nested dicts / MLX arrays). "
                        "MoE models need .safetensors shards — re-prepare the model or "
                        "let AirLLM Studio convert .mlx.npz → safetensors."
                    )
                state_dict = clean
            return _orig_move(self, state_dict)

        AirLLMBaseModel.move_layer_to_device = _safe_move  # type: ignore

        # pin_memory path also assumes all values are tensors
        _orig_load = AirLLMBaseModel.load_layer_to_cpu

        def _safe_load_layer(self, layer_name):
            state_dict = _orig_load(self, layer_name)
            if isinstance(state_dict, dict) and any(
                not isinstance(v, torch.Tensor) for v in state_dict.values()
            ):
                flat = _flatten_state_dict(state_dict)
                state_dict = {
                    k: v for k, v in flat.items() if isinstance(v, torch.Tensor)
                }
            # Keep per-expert ModuleList keys as-is (do NOT pack) so load_layer_subset
            # can seek a single expert's tensors.
            return state_dict

        AirLLMBaseModel.load_layer_to_cpu = _safe_load_layer  # type: ignore

        # Expand packed experts → ModuleList *before* AirLLM installs expert hooks
        _orig_install = AirLLMBaseModel._install_streaming_hooks

        def _install_with_expert_expand(self):
            try:
                expand_packed_experts_to_modulelist(self.model, self.config)
            except Exception as exc:
                print(f"[airllm_moe] expert ModuleList expand failed: {exc}")
            return _orig_install(self)

        AirLLMBaseModel._install_streaming_hooks = _install_with_expert_expand  # type: ignore

        # If setup still fails, surface a clear warning (do not silently go whole-layer
        # without saying how to fix). Prefer keeping expert_prefix so we retry expand.
        _orig_expert = AirLLMBaseModel._setup_expert_streaming

        def _safe_expert_setup(self):
            # Second chance expand (in case install path order differed)
            try:
                expand_packed_experts_to_modulelist(self.model, self.config)
            except Exception:
                pass
            try:
                return _orig_expert(self)
            except TypeError as exc:
                err = str(exc).lower()
                if "has no len" in err or "not iterable" in err or "not subscriptable" in err:
                    self._expert_streaming = False
                    self._expert_keys = {}
                    self._non_expert_keys = {}
                    print(
                        "[airllm_moe] WARNING: could not enable per-expert hooks "
                        f"({exc}). Falling back to whole-layer streaming (slow)."
                    )
                    return
                raise
            except Exception as exc:
                self._expert_streaming = False
                self._expert_keys = getattr(self, "_expert_keys", {}) or {}
                self._non_expert_keys = getattr(self, "_non_expert_keys", {}) or {}
                print(f"[airllm_moe] expert streaming disabled ({exc})")

        AirLLMBaseModel._setup_expert_streaming = _safe_expert_setup  # type: ignore

        # transformers 5+ generate() may call set_experts_implementation; AirLLM's
        # dynamic _AirLLMRuntimeModel wrapper rejects it. Add a no-op / eager path.
        _orig_patch_dev = AirLLMBaseModel._patch_device_property

        def _safe_patch_device(self):
            _orig_patch_dev(self)
            cls = type(self.model)
            if getattr(cls, "_airllm_experts_patched", False):
                return

            def set_experts_implementation(this, implementation, **kwargs):
                try:
                    this._experts_implementation = implementation
                except Exception:
                    pass
                return this

            cls.set_experts_implementation = set_experts_implementation  # type: ignore
            cls._supports_experts_implementation = True
            try:
                cls._airllm_experts_patched = True
            except Exception:
                pass
            if not hasattr(self.model, "set_experts_implementation"):
                import types

                self.model.set_experts_implementation = types.MethodType(
                    set_experts_implementation, self.model
                )

        AirLLMBaseModel._patch_device_property = _safe_patch_device  # type: ignore

        print(
            "[airllm_moe] patched load path + ModuleList expert expand (per-expert hooks)"
        )
    except Exception as exc:
        print(f"[airllm_moe] verbatim patch failed: {exc}")


def apply_moe_routing_patch() -> None:
    """
    Patch airllm.AutoModel.from_pretrained so MoE models on macOS use
    AirLLMBaseModel + expert streaming (MPS/CPU) instead of dense-only MLX.
    """
    global _PATCHED
    if _PATCHED:
        return
    if sys.platform != "darwin":
        _PATCHED = True
        return

    try:
        from airllm.auto_model import AutoModel
        from transformers import AutoConfig
    except Exception as exc:
        print(f"[airllm_moe] cannot patch AutoModel: {exc}")
        _PATCHED = True
        return

    patch_should_load_verbatim()
    apply_mps_safety_patches()
    _orig = AutoModel.from_pretrained.__func__  # type: ignore

    @classmethod  # type: ignore
    def from_pretrained_moe(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        force_torch = kwargs.pop("force_torch_moe", False)
        try:
            token = kwargs.get("hf_token")
            if token:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=True, token=token
                )
            else:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=True
                )
        except Exception:
            return _orig(cls, pretrained_model_name_or_path, *inputs, **kwargs)

        if is_moe_config(config) or force_torch:
            # Critical: MoE needs safetensors shards, not MLX nested dicts
            force_safetensor_persister()
            shards = kwargs.get("layer_shards_saving_path")
            paths_to_fix = []
            if shards:
                paths_to_fix.append(str(shards))
            # Also convert under default path next to HF cache if present
            paths_to_fix.append(str(pretrained_model_name_or_path))
            for p in paths_to_fix:
                try:
                    ensure_safetensor_shards(p)
                except Exception as conv_exc:
                    print(f"[airllm_moe] shard ensure note ({p}): {conv_exc}")

            MoEClass, expert = pick_moe_class(config)
            device = kwargs.pop("device", None) or default_mac_device()
            kwargs["device"] = device
            # Prefetching runs a worker thread that races Metal → MPSGraph SIGABRT.
            # Disable on Mac MoE always (CPU is fine; MPS requires this).
            kwargs["prefetching"] = False
            # Prefer float16 on MPS; bf16 + thrashing is flaky
            if device == "mps" and kwargs.get("dtype") is None:
                try:
                    import torch

                    kwargs["dtype"] = torch.float16
                except Exception:
                    pass
            print(
                f"[airllm_moe] MoE route → {MoEClass.__name__} "
                f"expert_prefix={expert!r} device={device} prefetching=False"
            )
            if device == "cpu":
                print(
                    "[airllm_moe] using CPU for MoE (stable). "
                    "Set AIRLLM_MOE_DEVICE=mps to try GPU (may crash)."
                )
            return MoEClass(pretrained_model_name_or_path, *inputs, **kwargs)

        return _orig(cls, pretrained_model_name_or_path, *inputs, **kwargs)

    AutoModel.from_pretrained = from_pretrained_moe  # type: ignore
    _PATCHED = True
    print("[airllm_moe] AutoModel MoE routing patch applied (Mac)")
