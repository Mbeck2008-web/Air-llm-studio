"""Live layer / expert / neuron activity for the Studio sidebar.

Neuron field is a 256×256 image: bilinear upsample of |activations|,
averaged across however many experts fired this layer.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Callable, Dict, List, Optional

EmitFn = Callable[[Dict[str, Any]], None]

FRAME_SIZE = 256
_MLX_ORIG_BLOCK_CALL = None


def _unwrap(hidden):
    if isinstance(hidden, (tuple, list)) and hidden:
        return hidden[0]
    return hidden


def _as_abs_tensor(hidden):
    """Torch or MLX hidden → abs float CPU tensor."""
    import torch

    hidden = _unwrap(hidden)
    if hidden is None:
        return None
    if isinstance(hidden, torch.Tensor):
        if hidden.numel() == 0:
            return None
        return hidden.detach().float().abs()
    try:
        import mlx.core as mx
        import numpy as np

        if isinstance(hidden, mx.array):
            mx.eval(hidden)
            return torch.from_numpy(np.ascontiguousarray(hidden)).float().abs()
    except Exception:
        pass
    try:
        import numpy as np

        arr = np.asarray(hidden)
        if arr.dtype == object or arr.size == 0:
            return None
        return torch.from_numpy(np.ascontiguousarray(arr)).float().abs()
    except Exception:
        return None


def _activation_field(hidden, size: int = FRAME_SIZE):
    """Abs activations → size×size float tensor in [0, 1], or None."""
    try:
        import torch.nn.functional as F

        t = _as_abs_tensor(hidden)
        if t is None or t.numel() == 0:
            return None
        t = t.flatten()
        n = int(t.numel())
        side = int(math.ceil(math.sqrt(n)))
        pad = side * side - n
        if pad:
            t = F.pad(t, (0, pad))
        t = t.view(1, 1, side, side)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        t = t.squeeze(0).squeeze(0)
        peak = t.max().clamp(min=1e-8)
        return (t / peak).clamp(0, 1)
    except Exception:
        return None


def _encode_frame(field) -> Optional[str]:
    if field is None:
        return None
    try:
        import torch

        u8 = (field.detach().clamp(0, 1) * 255.0).to(torch.uint8).contiguous().cpu().numpy()
        return base64.b64encode(u8.tobytes()).decode("ascii")
    except Exception:
        return None


class ActivityProbe:
    """Extra forward hooks on an AirLLM model. Safe no-op if attach fails."""

    def __init__(self) -> None:
        self._handles: list = []
        self._emit: Optional[EmitFn] = None
        self.layers_total = 0
        self.experts_total = 0
        self._layer_ids: List[int] = []
        self._active_experts: List[int] = []
        self._expert_heat: List[float] = []
        self._expert_fields: list = []
        self._last_frame: Optional[str] = None
        self._current_layer = 0
        self._attached = False
        self._mlx_i = 0
        self._mlx_patched = False

    def set_emit(self, fn: Optional[EmitFn]) -> None:
        self._emit = fn
        self._mlx_i = 0

    def detach(self) -> None:
        global _MLX_ORIG_BLOCK_CALL
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        if self._mlx_patched:
            try:
                from airllm import airllm_llama_mlx as mlx_mod

                if _MLX_ORIG_BLOCK_CALL is not None:
                    mlx_mod.TransformerBlock.__call__ = _MLX_ORIG_BLOCK_CALL
            except Exception:
                pass
            self._mlx_patched = False
        self._attached = False

    def attach(self, model) -> bool:
        self.detach()
        if model is None:
            return False
        name = type(model).__name__
        if name == "AirLLMLlamaMlx" or not getattr(model, "layers", None):
            if name == "AirLLMLlamaMlx" or hasattr(model, "model_args") or hasattr(model, "model_generate"):
                return self.attach_mlx(model)
        layers = getattr(model, "layers", None)
        names = list(getattr(model, "layer_names", []) or [])
        if not layers:
            return False

        self._layer_ids = list(range(len(layers)))
        self.layers_total = len(layers)
        self.experts_total = 0
        expert_keys = getattr(model, "_expert_keys", {}) or {}
        if expert_keys:
            self.experts_total = max((max(v.keys()) + 1 for v in expert_keys.values() if v), default=0)
        self._expert_heat = [0.0] * max(self.experts_total, 0)
        self._expert_fields = []

        def layer_pre(idx: int):
            def _pre(module, inputs):
                self._current_layer = idx
                self._active_experts = []
                self._expert_heat = [0.0] * max(self.experts_total, 0)
                self._expert_fields = []
                self._push(stage=_stage_name(names, idx))

            return _pre

        def layer_post(idx: int):
            def _post(module, inputs, output):
                if not self._expert_fields:
                    field = _activation_field(output)
                    if field is not None:
                        self._expert_fields = [field]
                self._push(stage=_stage_name(names, idx), send_frame=True)
                return output

            return _post

        for i, mod in enumerate(layers):
            try:
                self._handles.append(mod.register_forward_pre_hook(layer_pre(i)))
                self._handles.append(mod.register_forward_hook(layer_post(i)))
            except Exception:
                continue

        # Individual experts (ModuleList path)
        prefix = (getattr(model, "layer_names_dict", {}) or {}).get("expert_prefix")
        if prefix and expert_keys:
            for layer_idx, per in expert_keys.items():
                try:
                    container = layers[layer_idx]
                    for attr in prefix.split("."):
                        container = getattr(container, attr)
                except Exception:
                    continue
                for eid in per.keys():
                    try:
                        expert = container[int(eid)]
                    except Exception:
                        continue
                    eid_i, lid = int(eid), int(layer_idx)
                    self._handles.append(expert.register_forward_pre_hook(self._expert_pre(eid_i, lid)))
                    self._handles.append(expert.register_forward_hook(self._expert_post(eid_i, lid)))

        self._attached = bool(self._handles)
        return self._attached

    def attach_mlx(self, model) -> bool:
        """Dense AirLLMLlamaMlx: no persistent layers — wrap TransformerBlock.__call__."""
        global _MLX_ORIG_BLOCK_CALL
        n = 0
        args = getattr(model, "model_args", None)
        if args is not None:
            n = int(getattr(args, "n_layers", 0) or 0)
        if not n:
            n = int(getattr(model, "n_layers", 0) or 0)
        cfg = getattr(model, "config", None)
        if not n and cfg is not None:
            n = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        self.layers_total = n
        self.experts_total = 0
        self._expert_heat = []
        self._mlx_i = 0
        try:
            from airllm import airllm_llama_mlx as mlx_mod

            TB = mlx_mod.TransformerBlock
            if _MLX_ORIG_BLOCK_CALL is None:
                _MLX_ORIG_BLOCK_CALL = TB.__call__
            probe = self

            def wrapped(tb_self, x, mask=None, cache=None):
                nlay = max(int(probe.layers_total) or 1, 1)
                idx = int(probe._mlx_i) % nlay
                probe._mlx_i = idx + 1
                probe._current_layer = idx
                probe._expert_fields = []
                probe._push(stage="layer")
                out = _MLX_ORIG_BLOCK_CALL(tb_self, x, mask, cache)
                hidden = out[0] if isinstance(out, tuple) else out
                try:
                    import mlx.core as mx

                    mx.eval(hidden)
                except Exception:
                    pass
                field = _activation_field(hidden)
                if field is not None:
                    probe._expert_fields = [field]
                probe._push(stage="layer", send_frame=True)
                return out

            TB.__call__ = wrapped  # type: ignore[method-assign]
            self._mlx_patched = True
            self._attached = True
            print(f"[activity] MLX dense probe attached · {n} layers")
            return True
        except Exception as exc:
            print(f"[activity] MLX probe failed: {exc}")
            self._attached = False
            return False

    def _hit_expert(self, expert_idx: int, layer_idx: int) -> None:
        if expert_idx not in self._active_experts:
            self._active_experts.append(expert_idx)
        if expert_idx < len(self._expert_heat):
            self._expert_heat[expert_idx] = 1.0
        self._current_layer = layer_idx

    def _expert_pre(self, expert_idx: int, layer_idx: int):
        def _pre(module, inputs):
            self._hit_expert(expert_idx, layer_idx)
            self._push(stage="expert")

        return _pre

    def _expert_post(self, expert_idx: int, layer_idx: int):
        def _post(module, inputs, output):
            self._hit_expert(expert_idx, layer_idx)
            field = _activation_field(output)
            if field is not None:
                self._expert_fields.append(field)
            self._push(stage="expert", send_frame=True)
            return output

        return _post

    def _averaged_field(self):
        if not self._expert_fields:
            return None
        try:
            import torch

            return torch.stack(self._expert_fields, dim=0).mean(dim=0)
        except Exception:
            return self._expert_fields[-1]

    def _push(self, stage: str, send_frame: bool = False) -> None:
        if not self._emit:
            return
        cur = int(self._current_layer)
        remaining = max(0, self.layers_total - cur - 1)
        payload: Dict[str, Any] = {
            "stage": stage,
            "layer": cur,
            "layers_total": self.layers_total,
            "layers_remaining": remaining,
            "experts_active": list(self._active_experts),
            "experts_total": self.experts_total,
            "expert_heat": list(self._expert_heat),
            "frame_n": len(self._expert_fields),
        }
        if send_frame:
            encoded = _encode_frame(self._averaged_field())
            if encoded:
                self._last_frame = encoded
                payload["frame_b64"] = encoded
                payload["frame_w"] = FRAME_SIZE
                payload["frame_h"] = FRAME_SIZE
        try:
            self._emit(payload)
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "attached": self._attached,
            "layer": self._current_layer,
            "layers_total": self.layers_total,
            "layers_remaining": max(0, self.layers_total - self._current_layer - 1)
            if self.layers_total
            else 0,
            "experts_active": list(self._active_experts),
            "experts_total": self.experts_total,
        }


def _stage_name(names: List[str], idx: int) -> str:
    if idx < 0 or idx >= len(names):
        return "layer"
    n = names[idx]
    if "embed" in n:
        return "embed"
    if n.endswith("norm") or n.endswith(".norm"):
        return "norm"
    if "lm_head" in n or n.endswith("head"):
        return "lm_head"
    return "layer"
