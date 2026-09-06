"""
Featured one-click model catalog for AirLLM Studio on Apple Silicon.

Dense models → AirLLMLlamaMlx (Metal / MLX), with our Qwen bias patches.
MoE models   → AirLLMBaseModel + per-expert hooks on MPS/CPU (not dense MLX).

See airllm_studio.core.airllm_moe for MoE routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .model_meta import ModelInfo, estimate_airllm_memory_gb


# Size tiers shown in the library UI
SIZE_TIERS: List[Tuple[str, str]] = [
    ("7B", "Fast · laptop-friendly · MLX dense"),
    ("13B", "Strong daily driver · dense or small MoE"),
    ("30B", "Best quality that still feels local"),
    ("70B", "Near frontier · AirLLM streams layers / experts"),
    ("100B+", "Large MoE · per-expert streaming on MPS/CPU"),
    ("400B+", "Frontier open weights · huge disk, streaming peak is small"),
]


@dataclass(frozen=True)
class CatalogModel:
    repo_id: str
    label: str
    params_b: float  # billions
    size_tier: str  # key from SIZE_TIERS
    is_moe: bool = False
    num_layers: int = 32
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    blurb: str = ""
    disk_gb: Optional[float] = None
    gated: bool = False
    roles: Tuple[str, ...] = ("chat",)  # coding | chat | reasoning
    recommended: bool = False
    # Explicit allow-list flag for Mac MLX path
    mlx_supported: bool = True

    def role_label(self) -> str:
        order = ["coding", "chat", "reasoning"]
        tags = [r for r in order if r in self.roles]
        return " · ".join(t.capitalize() for t in tags) if tags else "General"

    def to_model_info(self) -> ModelInfo:
        notes = []
        if self.blurb:
            notes.append(self.blurb)
        notes.append(f"Best for: {self.role_label()}")
        notes.append("AirLLM macOS: Llama-style dense (MLX path)")
        if self.recommended:
            notes.append("Recommended pick in this size class")
        if self.gated:
            notes.append("Gated model — set HF token in Settings")
        info = ModelInfo(
            repo_id=self.repo_id,
            display_name=self.label,
            total_params=self.params_b,
            total_params_raw=int(self.params_b * 1e9),
            num_layers=self.num_layers,
            is_moe=self.is_moe,
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            architecture="dense-llama-mlx",
            estimated_disk_gb=self.disk_gb or (self.params_b * 2.0),
            notes=notes,
        )
        info.estimated_airllm_memory_gb = estimate_airllm_memory_gb(info)
        return info


# ─────────────────────────────────────────────────────────────
# Dense Llama-style models only (AirLLMLlamaMlx on Apple Silicon)
# ─────────────────────────────────────────────────────────────
FEATURED_MODELS: List[CatalogModel] = [
    # ── ~7–9B ──────────────────────────────────────────────
    CatalogModel(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5 Coder 7B",
        7.6,
        "7B",
        num_layers=28,
        blurb="Top small coding model · AirLLM MLX supported",
        disk_gb=15,
        roles=("coding",),
        recommended=True,
    ),
    CatalogModel(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct",
        7.6,
        "7B",
        num_layers=28,
        blurb="Best all-round 7B chat · verified AirLLM MLX path",
        disk_gb=15,
        roles=("chat",),
        recommended=True,
    ),
    CatalogModel(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct v0.3",
        7.0,
        "7B",
        num_layers=32,
        blurb="Classic dense instruct · Llama-like layout · great first test",
        disk_gb=14,
        roles=("chat",),
    ),
    CatalogModel(
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama 3.1 8B Instruct",
        8.0,
        "7B",
        num_layers=32,
        blurb="Meta Llama dense · native MLX layout · gated",
        disk_gb=16,
        roles=("chat", "coding"),
        gated=True,
    ),
    # ── ~13–14B ────────────────────────────────────────────
    CatalogModel(
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen2.5 Coder 14B",
        14.7,
        "13B",
        num_layers=48,
        blurb="Strong local coder · dense · AirLLM MLX",
        disk_gb=29,
        roles=("coding",),
        recommended=True,
    ),
    CatalogModel(
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen2.5 14B Instruct",
        14.7,
        "13B",
        num_layers=48,
        blurb="Strong general chat at mid size",
        disk_gb=29,
        roles=("chat",),
        recommended=True,
    ),
    # ── ~30–32B ────────────────────────────────────────────
    CatalogModel(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen2.5 Coder 32B",
        32.0,
        "30B",
        num_layers=64,
        blurb="Best fully-local coding default · dense MLX path",
        disk_gb=64,
        roles=("coding",),
        recommended=True,
    ),
    CatalogModel(
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen2.5 32B Instruct",
        32.0,
        "30B",
        num_layers=64,
        blurb="Top dense 32B for chat & planning",
        disk_gb=64,
        roles=("chat", "reasoning"),
        recommended=True,
    ),
    CatalogModel(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1 Distill Qwen 32B",
        32.0,
        "30B",
        num_layers=64,
        blurb="Reasoning distill · dense Qwen backbone (not MoE)",
        disk_gb=64,
        roles=("reasoning", "coding"),
    ),
    # ── ~70B ───────────────────────────────────────────────
    CatalogModel(
        "meta-llama/Llama-3.3-70B-Instruct",
        "Llama 3.3 70B Instruct",
        70.0,
        "70B",
        num_layers=80,
        blurb="Latest Meta 70B dense · native MLX · gated",
        disk_gb=140,
        roles=("chat", "coding"),
        recommended=True,
        gated=True,
    ),
    CatalogModel(
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen2.5 72B Instruct",
        72.0,
        "70B",
        num_layers=80,
        blurb="Top open dense ~70B chat · AirLLM streams layers",
        disk_gb=145,
        roles=("chat", "reasoning"),
        recommended=True,
    ),
    CatalogModel(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-R1 Distill Llama 70B",
        70.0,
        "70B",
        num_layers=80,
        blurb="Heavy reasoning · dense Llama backbone",
        disk_gb=140,
        roles=("reasoning", "coding"),
    ),
    CatalogModel(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral 8×7B Instruct (MoE)",
        46.7,
        "70B",
        is_moe=True,
        num_layers=32,
        num_experts=8,
        experts_per_token=2,
        blurb="MoE · AirLLM per-expert streaming on Mac (MPS/CPU) · 2 of 8 experts",
        disk_gb=100,
        roles=("chat",),
        recommended=True,
        mlx_supported=True,  # via MoE torch route, not dense MLX
    ),
    # ── 100B+ MoE ──────────────────────────────────────────
    CatalogModel(
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Mixtral 8×22B Instruct (MoE)",
        141.0,
        "100B+",
        is_moe=True,
        num_layers=56,
        num_experts=8,
        experts_per_token=2,
        blurb="Large Mixtral · expert streaming · big download",
        disk_gb=280,
        roles=("chat",),
        mlx_supported=True,
    ),
    CatalogModel(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "Qwen1.5 MoE A2.7B Chat",
        14.3,
        "13B",
        is_moe=True,
        num_layers=24,
        num_experts=60,
        experts_per_token=4,
        blurb="Small MoE chat · good AirLLM expert-streaming test",
        disk_gb=28,
        roles=("chat",),
        mlx_supported=True,
    ),
    CatalogModel(
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "DeepSeek-V2 Lite Chat (MoE)",
        15.7,
        "30B",
        is_moe=True,
        num_layers=27,
        num_experts=64,
        experts_per_token=6,
        blurb="DeepSeek MoE · mlp.experts streaming",
        disk_gb=32,
        roles=("chat", "coding"),
        mlx_supported=True,
    ),
    # ── 400B+ ──────────────────────────────────────────────
    CatalogModel(
        "meta-llama/Llama-3.1-405B-Instruct",
        "Llama 3.1 405B Instruct",
        405.0,
        "400B+",
        num_layers=126,
        blurb="Largest Meta open dense · AirLLM ~8 GB peak · gated · huge disk",
        disk_gb=810,
        roles=("chat", "coding"),
        recommended=True,
        gated=True,
    ),
    CatalogModel(
        "deepseek-ai/DeepSeek-V2-Chat",
        "DeepSeek-V2 Chat (MoE 236B)",
        236.0,
        "100B+",
        is_moe=True,
        num_layers=60,
        num_experts=160,
        experts_per_token=6,
        blurb="Large DeepSeek MoE · expert streaming · huge download",
        disk_gb=470,
        roles=("chat", "coding"),
        mlx_supported=True,
    ),
]


def featured_by_tier() -> dict:
    out: dict = {}
    for m in FEATURED_MODELS:
        if not m.mlx_supported:
            continue
        out.setdefault(m.size_tier, []).append(m)
    return out


def get_catalog_model(repo_id: str) -> Optional[CatalogModel]:
    for m in FEATURED_MODELS:
        if m.repo_id == repo_id:
            return m
    return None


def is_mac_mlx_supported_repo(repo_id: str) -> bool:
    """True if AirLLM Studio can run this repo on Mac (dense MLX or MoE torch path)."""
    cat = get_catalog_model(repo_id)
    if cat is not None:
        return cat.mlx_supported
    rid = repo_id.lower()
    # Known MoE families → MoE route
    if any(x in rid for x in ("mixtral", "moe", "deepseek-v2", "8x7b", "8x22b", "a2.7b")):
        return True
    allow = ("llama", "mistral", "qwen2.5", "qwen2", "gemma-2")
    return any(a in rid for a in allow)


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h {m}m"


def estimate_download_seconds(disk_gb: float, mbps: float = 50.0) -> float:
    """Rough wall-time estimate at assumed average download speed (Mbps)."""
    if disk_gb <= 0 or mbps <= 0:
        return 0.0
    megabits = disk_gb * 8 * 1024
    return megabits / mbps
