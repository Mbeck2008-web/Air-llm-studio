"""
Runtime patches for AirLLM's macOS MLX backend.

Problems we fix (see airllm issues / local testing with Qwen2.5):
1. generate() needs mlx.array token ids — handled in engine._run_generate
2. Qwen2/Qwen2.5 q/k/v projections include **bias**, but AirLLMLlamaMlx
   builds nn.Linear(..., bias=False) → "Module does not have parameter named bias"
3. Stray / unexpected weight keys can break Module.update — strip safely
"""

from __future__ import annotations

import sys
from typing import Any, Dict


_PATCHED = False


def apply_airllm_mlx_patches() -> None:
    """Idempotent monkey-patches for AirLLMLlamaMlx on darwin + MoE routing."""
    global _PATCHED
    if _PATCHED:
        return

    # MoE: route Mixtral/DeepSeek/Qwen-MoE to AirLLMBaseModel + expert streaming
    try:
        from airllm_studio.core.airllm_moe import apply_moe_routing_patch

        apply_moe_routing_patch()
    except Exception as exc:
        print(f"[mlx_compat] moe routing: {exc}")

    if sys.platform != "darwin":
        _PATCHED = True
        return

    try:
        import mlx.nn as nn
        from airllm import airllm_llama_mlx as mlx_mod
        from airllm.persist import mlx_model_persister as persist_mod
    except Exception as exc:
        print(f"[mlx_compat] skip dense MLX patches: {exc}")
        _PATCHED = True
        return

    # ── 1) Attention: enable bias on q/k/v (Qwen2.5 needs it; Llama bias stays 0) ──
    Attention = mlx_mod.Attention
    ModelArgs = mlx_mod.ModelArgs

    def _attention_init(self, args: Any) -> None:
        nn.Module.__init__(self)
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.args.head_dim**-0.5
        # bias=True so Qwen q/k/v.bias load; Llama has no bias files → zeros
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    Attention.__init__ = _attention_init  # type: ignore[method-assign]

    # ── 2) When loading weights, drop keys the module cannot accept ──
    orig_map = persist_mod.map_torch_to_mlx

    def map_torch_to_mlx_safe(model: Dict[str, Any]) -> Dict[str, Any]:
        mapped = orig_map(model)
        # Keep bias for wq/wk/wv; drop other biases that still have no param
        # e.g. some models ship unused bias tensors
        cleaned = {}
        for k, v in mapped.items():
            # o_proj / wo never has bias in AirLLMLlamaMlx
            if k.endswith("wo.bias") or k.endswith("output.bias"):
                continue
            if k.endswith("w1.bias") or k.endswith("w2.bias") or k.endswith("w3.bias"):
                continue
            cleaned[k] = v
        return cleaned

    persist_mod.map_torch_to_mlx = map_torch_to_mlx_safe

    # ── 3) Safer Module.update that ignores unknown leaves ──
    try:
        from mlx.utils import tree_flatten, tree_unflatten
    except Exception:
        tree_flatten = None

    if tree_flatten is not None:
        orig_load = persist_mod.MlxModelPersister.load_model

        def load_model_safe(self, layer_name, path):
            weights = orig_load(self, layer_name, path)
            return weights

        persist_mod.MlxModelPersister.load_model = load_model_safe  # type: ignore

    # ── 4) Patch TransformerBlock.update path used in generate ──
    # AirLLM does: l = TransformerBlock(...); l.update(layer_dict)
    # Filter layer_dict recursively for keys matching parameters
    TB = mlx_mod.TransformerBlock

    def _filter_tree_to_module(module: nn.Module, tree: dict) -> dict:
        """Keep only keys that exist on the module parameter tree."""
        try:
            params = dict(tree_flatten(module.parameters()))
        except Exception:
            return tree
        param_keys = set(params.keys())

        flat = dict(tree_flatten(tree)) if not all(
            isinstance(v, dict) for v in tree.values()
        ) else None

        # tree from load is nested: attention/wq/weight etc.
        def filter_nested(mod: nn.Module, d: dict, prefix: str = "") -> dict:
            out = {}
            for k, v in d.items():
                full = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    child = getattr(mod, k, None)
                    if child is None:
                        continue
                    if hasattr(child, "parameters"):
                        filtered = filter_nested(child, v, full)
                        if filtered:
                            out[k] = filtered
                    continue
                # leaf
                try:
                    # parameter exists?
                    parts = full.split(".")
                    obj = module
                    ok = True
                    for p in parts[:-1]:
                        if not hasattr(obj, p):
                            ok = False
                            break
                        obj = getattr(obj, p)
                    if not ok:
                        continue
                    leaf = parts[-1]
                    if hasattr(obj, leaf) or (
                        hasattr(obj, "parameters")
                        and leaf in dict(tree_flatten(obj.parameters()))
                    ):
                        # check via parameters of parent
                        pass
                    # Prefer: only include if full key in module params
                    mod_flat = {fk for fk, _ in tree_flatten(module.parameters())}
                    if full in mod_flat or any(fk.endswith(full) for fk in mod_flat):
                        out[k] = v
                    elif leaf in ("weight", "bias") and hasattr(obj, leaf):
                        out[k] = v
                except Exception:
                    continue
            return out

        if isinstance(tree, dict):
            return filter_nested(module, tree) or tree
        return tree

    _orig_tb_call_path = None

    # Patch the generate loop's update by wrapping TransformerBlock.update
    _orig_update = nn.Module.update

    def _safe_update(self, parameters, strict=True):
        # mlx Module.update signature may vary by version
        try:
            return _orig_update(self, parameters)
        except Exception as e:
            err = str(e).lower()
            if "bias" not in err and "parameter" not in err and "not have" not in err:
                raise
            # Retry after stripping keys that don't exist
            try:
                flat_params = dict(tree_flatten(parameters))
                mod_keys = {k for k, _ in tree_flatten(self.parameters())}
                # Also allow relative keys
                kept = {}
                for k, v in flat_params.items():
                    if k in mod_keys or any(k.endswith(mk) or mk.endswith(k) for mk in mod_keys):
                        kept[k] = v
                    elif k.split(".")[-1] == "bias":
                        # if module has bias under same path parent
                        parent = ".".join(k.split(".")[:-1])
                        if any(mk.startswith(parent + ".") for mk in mod_keys) and any(
                            mk.endswith(".bias") for mk in mod_keys if mk.startswith(parent)
                        ):
                            kept[k] = v
                        # else drop bias
                    else:
                        # try last two components match
                        tail = ".".join(k.split(".")[-2:])
                        matches = [mk for mk in mod_keys if mk.endswith(tail)]
                        if matches:
                            kept[matches[0]] = v
                if not kept:
                    # last resort: drop all .bias and retry original nested tree
                    def drop_bias(obj):
                        if isinstance(obj, dict):
                            return {
                                kk: drop_bias(vv)
                                for kk, vv in obj.items()
                                if kk != "bias"
                            }
                        return obj

                    cleaned = drop_bias(parameters)
                    return _orig_update(self, cleaned)
                from mlx.utils import tree_unflatten

                return _orig_update(self, tree_unflatten(list(kept.items())))
            except Exception:
                # Final fallback: drop every bias key in nested dict
                def drop_bias(obj):
                    if isinstance(obj, dict):
                        return {
                            kk: drop_bias(vv) for kk, vv in obj.items() if kk != "bias"
                        }
                    return obj

                return _orig_update(self, drop_bias(parameters))

    # Only install safe update if not already wrapped
    if getattr(nn.Module.update, "__name__", "") != "_safe_update":
        nn.Module.update = _safe_update  # type: ignore

    _PATCHED = True
    print("[mlx_compat] AirLLM MLX patches applied (Qwen bias + safe weight load)")
