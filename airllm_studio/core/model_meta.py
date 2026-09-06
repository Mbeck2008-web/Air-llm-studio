"""Model metadata parsing, MoE detection, and AirLLM memory estimates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelInfo:
    repo_id: str
    display_name: str = ""
    total_params: Optional[float] = None  # billions
    total_params_raw: Optional[int] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    architecture: str = "unknown"
    is_moe: bool = False
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    shared_experts: Optional[int] = None
    dtype: str = "float16"
    estimated_disk_gb: Optional[float] = None
    estimated_airllm_memory_gb: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Model: {self.display_name or self.repo_id}",
            f"Architecture: {self.architecture}",
            f"Type: {'Sparse MoE' if self.is_moe else 'Dense'}",
        ]
        if self.total_params is not None:
            lines.append(f"Parameters: {format_params(self.total_params)}")
        if self.num_layers is not None:
            lines.append(f"Layers: {self.num_layers}")
        if self.is_moe:
            if self.num_experts is not None:
                lines.append(f"Experts: {self.num_experts}")
            if self.experts_per_token is not None:
                lines.append(f"Active experts / token: {self.experts_per_token}")
            if self.shared_experts:
                lines.append(f"Shared experts: {self.shared_experts}")
        if self.estimated_airllm_memory_gb is not None:
            lines.append(
                f"Est. AirLLM peak memory: ~{self.estimated_airllm_memory_gb:.1f} GB"
            )
        if self.estimated_disk_gb is not None:
            lines.append(f"Est. disk (weights): ~{self.estimated_disk_gb:.1f} GB")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def format_params(billions: float) -> str:
    if billions >= 1000:
        return f"{billions / 1000:.2f}T"
    if billions >= 1:
        return f"{billions:.1f}B"
    return f"{billions * 1000:.0f}M"


def _first(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _guess_params_from_name(repo_id: str) -> Optional[float]:
    """Parse common size tags from model names (e.g. 70B, 235B-A22B, 8x7B, 671B)."""
    name = repo_id.split("/")[-1]
    # Mixtral-style: 8x7B ≈ 8 * 7B total sparse params
    m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[Bb]", name, re.I)
    if m:
        return float(m.group(1)) * float(m.group(2))
    # MoE total + active: 235B-A22B → total 235
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\s*-?\s*A\d+(?:\.\d+)?[Bb]", name, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Tt]", name)
    if m:
        return float(m.group(1)) * 1000
    return None


def parse_model_info(
    repo_id: str,
    config: Optional[Dict[str, Any]] = None,
    safetensors_metadata: Optional[Dict[str, Any]] = None,
) -> ModelInfo:
    """Build ModelInfo from HF config.json (and optional weight metadata)."""
    config = config or {}
    info = ModelInfo(
        repo_id=repo_id,
        display_name=repo_id.split("/")[-1],
        raw_config=dict(config),
    )

    archs = config.get("architectures") or []
    if archs:
        info.architecture = archs[0]
    else:
        info.architecture = str(config.get("model_type") or "unknown")

    info.num_layers = _first(
        config.get("num_hidden_layers"),
        config.get("n_layer"),
        config.get("num_layers"),
        config.get("n_layers"),
    )
    info.hidden_size = _first(
        config.get("hidden_size"),
        config.get("d_model"),
        config.get("n_embd"),
    )
    info.intermediate_size = _first(
        config.get("intermediate_size"),
        config.get("ffn_dim"),
        config.get("n_inner"),
    )
    info.num_attention_heads = _first(
        config.get("num_attention_heads"),
        config.get("n_head"),
    )
    info.vocab_size = config.get("vocab_size")

    # MoE detection across common families
    num_experts = _first(
        config.get("num_experts"),
        config.get("num_local_experts"),
        config.get("n_routed_experts"),
        config.get("moe_num_experts"),
        config.get("num_experts_per_tok") and config.get("n_routed_experts"),
    )
    # DeepSeek-style
    if config.get("n_routed_experts") is not None:
        num_experts = config.get("n_routed_experts")
    experts_per_tok = _first(
        config.get("num_experts_per_tok"),
        config.get("num_experts_per_token"),
        config.get("moe_top_k"),
        config.get("top_k"),
    )
    # Some configs use "num_selected_experts"
    if experts_per_tok is None:
        experts_per_tok = config.get("num_selected_experts")

    model_type = str(config.get("model_type") or "").lower()
    arch_l = info.architecture.lower()
    is_moe_name = any(
        x in arch_l or x in model_type or x in repo_id.lower()
        for x in ("moe", "mixtral", "deepseek", "qwen2_moe", "qwen3_moe", "grok")
    )

    if num_experts or (is_moe_name and ("moe" in arch_l or "mixtral" in arch_l or "moe" in model_type)):
        info.is_moe = True
        info.num_experts = int(num_experts) if num_experts else None
        if info.num_experts is None and "mixtral" in arch_l:
            info.num_experts = 8
            info.notes.append("Assumed 8 experts (Mixtral default)")
        info.experts_per_token = int(experts_per_tok) if experts_per_tok else None
        if info.experts_per_token is None and info.is_moe:
            info.experts_per_token = 2
            info.notes.append("Assumed 2 active experts/token (common default)")
        info.shared_experts = _first(
            config.get("n_shared_experts"),
            config.get("num_shared_experts"),
        )
        if info.shared_experts is not None:
            info.shared_experts = int(info.shared_experts)
    else:
        info.is_moe = False

    # Parameter count
    raw = None
    if safetensors_metadata and "total_size" in safetensors_metadata:
        # bytes → rough param count at fp16
        pass
    for key in ("num_parameters", "n_params", "total_params"):
        if key in config:
            raw = int(config[key])
            break

    if raw is None and safetensors_metadata:
        # HF sometimes puts parameter count in safetensors index metadata
        meta = safetensors_metadata.get("metadata") or safetensors_metadata
        if isinstance(meta, dict) and "total_size" in meta:
            # not params — skip
            pass

    if raw is not None:
        info.total_params_raw = raw
        info.total_params = raw / 1e9
    else:
        guessed = _guess_params_from_name(repo_id)
        if guessed is not None:
            info.total_params = guessed
            info.total_params_raw = int(guessed * 1e9)
            info.notes.append("Parameter count inferred from model name")
        elif info.num_layers and info.hidden_size:
            # Rough dense estimate: ~12 * L * H^2
            est = 12 * info.num_layers * (info.hidden_size ** 2)
            if info.vocab_size and info.hidden_size:
                est += info.vocab_size * info.hidden_size * 2
            info.total_params_raw = int(est)
            info.total_params = est / 1e9
            info.notes.append("Parameter count roughly estimated from architecture")

    # Disk estimate (fp16 weights; AirLLM split roughly doubles peak disk during prep)
    if info.total_params is not None:
        info.estimated_disk_gb = info.total_params * 2.0  # fp16
        if info.is_moe:
            # MoE checkpoints still store all experts
            pass

    info.estimated_airllm_memory_gb = estimate_airllm_memory_gb(info)
    return info


def system_memory_gb() -> float:
    """Total unified / system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 16.0  # safe-ish default


def available_memory_gb() -> float:
    """Currently available RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return system_memory_gb() * 0.4


def estimate_full_load_memory_gb(
    info: ModelInfo, bytes_per_param: float = 2.0
) -> float:
    """
    Peak memory if the whole model is loaded at once (AirLLM off).

    fp16 weights ≈ 2 bytes/param; add KV / runtime headroom.
    MoE still needs all experts resident without AirLLM streaming.
    """
    params_b = info.total_params
    if params_b is None and info.total_params_raw:
        params_b = info.total_params_raw / 1e9
    if params_b is None:
        # Last resort from disk estimate
        if info.estimated_disk_gb:
            params_b = info.estimated_disk_gb / 2.0
        else:
            params_b = 7.0

    weights_gb = params_b * bytes_per_param  # billions of params * bytes → GB-ish
    # params_b is in billions; each param is bytes_per_param bytes
    # total bytes = params_b * 1e9 * bytes_per_param → GB = params_b * bytes_per_param
    # yes: 7e9 * 2 / 1e9 = 14 GB for 7B fp16

    # Activations + KV cache + framework overhead (conservative)
    overhead_gb = max(2.0, min(params_b * 0.15, 12.0))
    return round(weights_gb + overhead_gb, 1)


def can_load_without_airllm(
    info: ModelInfo,
    *,
    safety_factor: float = 0.85,
) -> tuple:
    """
    Whether a full (non-AirLLM) load can fit in this machine's RAM.

    Uses total system memory (not momentary free) so a 7B on a 32 GB Mac
    is allowed even if other apps are open; a 70B on 32 GB is blocked.

    Returns (allowed: bool, message: str, need_gb: float, total_gb: float).
    """
    need = estimate_full_load_memory_gb(info)
    total = system_memory_gb()
    avail = available_memory_gb()
    hard_cap = total * safety_factor
    air = info.estimated_airllm_memory_gb or estimate_airllm_memory_gb(info)

    if need <= hard_cap:
        note = ""
        if need > avail:
            note = (
                f" Heads-up: only ~{avail:.0f} GB free right now — "
                f"close other apps if load fails."
            )
        return (
            True,
            (
                f"Full load ~{need:.0f} GB looks OK on this "
                f"{total:.0f} GB machine.{note}"
            ),
            need,
            total,
        )

    return (
        False,
        (
            f"This model needs ~{need:.0f} GB without AirLLM, "
            f"but this Mac only has {total:.0f} GB total RAM "
            f"(safe budget ~{hard_cap:.0f} GB). "
            f"Turn AirLLM On to stream layers/experts (~{air:.1f} GB peak), "
            f"or pick a smaller model."
        ),
        need,
        total,
    )


def estimate_airllm_memory_gb(info: ModelInfo, bytes_per_param: float = 2.0) -> float:
    """
    Rough peak unified-memory estimate for AirLLM streaming.

    AirLLM keeps ~1 layer (or active experts for MoE) in memory, plus KV cache headroom.
    This is intentionally conservative-simple for UI guidance, not a hard guarantee.
    """
    if info.num_layers and info.hidden_size and info.intermediate_size:
        # One transformer block ~ 4*H^2 (attn) + 3*H*I (MLP) parameters
        h, i = info.hidden_size, info.intermediate_size
        layer_params = 4 * h * h + 3 * h * i
        if info.is_moe and info.num_experts and info.experts_per_token:
            # MLP part is experts; only active experts loaded at a time for ultra-sparse path
            # Rough: attn full + (active/total)*mlp of all experts
            mlp_one = 3 * h * i  # if intermediate is per-expert size
            # Many MoE configs set intermediate_size per expert
            active = info.experts_per_token
            total_e = max(info.num_experts, 1)
            # If intermediate_size is per-expert, full layer MLP = total_e * mlp_one
            # AirLLM streams active experts
            layer_params = 4 * h * h + active * mlp_one
            if info.shared_experts:
                layer_params += int(info.shared_experts) * mlp_one
        layer_gb = (layer_params * bytes_per_param) / (1024 ** 3)
        # embeddings + lm_head often resident-ish, plus KV cache buffer
        embed_gb = 0.0
        if info.vocab_size and info.hidden_size:
            embed_gb = (info.vocab_size * info.hidden_size * bytes_per_param) / (1024 ** 3)
        overhead = 0.8  # runtime / activations / KV baseline
        return round(max(layer_gb + min(embed_gb, 2.0) + overhead, 1.0), 2)

    # Fallback from total size tables (AirLLM README ballparks)
    p = info.total_params or 7.0
    if info.is_moe:
        if p >= 600:
            return 12.0
        if p >= 200:
            return 3.0
        if p >= 30:
            return 2.5
        return 2.0
    if p >= 400:
        return 8.0
    if p >= 70:
        return 4.0
    if p >= 30:
        return 3.0
    if p >= 13:
        return 2.5
    return 1.5


def disk_space_warning(
    info: ModelInfo, free_gb: float
) -> Tuple[bool, str]:
    """
    Return (ok_to_proceed, message).
    Preparation can need ~2x model size (original + split shards).
    """
    need = (info.estimated_disk_gb or 10.0) * 2.2
    if free_gb < need:
        return (
            False,
            f"Low disk space: ~{free_gb:.0f} GB free, but preparing "
            f"{info.display_name or info.repo_id} may need ~{need:.0f} GB "
            f"(original weights + AirLLM layer shards). Free more space or "
            f"enable delete-original after split in Settings.",
        )
    if free_gb < need * 1.3:
        return (
            True,
            f"Warning: only ~{free_gb:.0f} GB free. Preparation may use "
            f"~{need:.0f} GB. Consider freeing space first.",
        )
    return True, f"Disk OK (~{free_gb:.0f} GB free; estimate need ~{need:.0f} GB)."
