"""
Precision / quantization variants for catalog models.

Each model gets:
  • Base full-precision HF repo (fp16/bf16)
  • AirLLM block compression options (8bit / 4bit) on the same repo
  • Curated Hugging Face quantized forks when known (AWQ, GPTQ, MLX 4-bit, FP8)

If no external quant repo is known, the dropdown still offers AirLLM 4/8-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelVariant:
    """One download/load choice for a model family."""

    key: str  # unique id for UI map
    label: str  # shown in dropdown
    repo_id: str
    precision: str  # "fp16" | "bf16" | "fp8" | "8bit" | "4bit" | "3bit" | "2bit" | "awq" | "gptq" | "mlx4"
    compression: Optional[str] = None  # AirLLM arg: None | "4bit" | "8bit"
    # Relative to full fp16 catalog disk_gb
    download_scale: float = 1.0  # what HF will actually transfer
    final_scale: float = 1.0  # on-disk after prepare / quant
    source: str = "base"  # base | airllm | huggingface
    notes: str = ""

    def download_gb_for(self, base_disk_gb: float) -> float:
        return max(0.5, base_disk_gb * self.download_scale)

    def final_gb_for(self, base_disk_gb: float) -> float:
        return max(0.5, base_disk_gb * self.final_scale)

    # Back-compat for older call sites
    def disk_gb_for(self, base_disk_gb: float) -> float:
        return self.download_gb_for(base_disk_gb)


def _base(repo_id: str) -> ModelVariant:
    return ModelVariant(
        key=f"{repo_id}|fp16",
        label="Full · fp16 / bf16",
        repo_id=repo_id,
        precision="fp16",
        compression=None,
        download_scale=1.0,
        final_scale=1.0,
        source="base",
        notes="Original Hugging Face weights",
    )


def _airllm_bits(repo_id: str, bits: str, final_scale: float) -> ModelVariant:
    """
    AirLLM compression runs *after* the full weights are downloaded.
    Download size ≈ full model; final shards are smaller.
    """
    return ModelVariant(
        key=f"{repo_id}|airllm-{bits}",
        label=f"AirLLM · {bits} (after full download)",
        repo_id=repo_id,
        precision=bits,
        compression=bits,
        download_scale=1.0,  # still pulls full checkpoint from HF
        final_scale=final_scale,
        source="airllm",
        notes=(
            f"Downloads full weights first, then AirLLM compresses to {bits}. "
            "Not a smaller Hugging Face file."
        ),
    )


# Known HF quantized / reduced-precision repos keyed by base catalog repo_id
# (best-effort; missing IDs fall back to AirLLM-only options)
_HF_VARIANTS: Dict[str, List[ModelVariant]] = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-Coder-7B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
            notes="Community AWQ quant on Hugging Face",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-Coder-7B-Instruct|gptq",
            "GPTQ · 4-bit",
            "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
            "gptq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-7B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-7B-Instruct|gptq",
            "GPTQ · 4-bit",
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "gptq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-7B-Instruct|mlx4",
            "MLX · 4-bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx4",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
            notes="Apple MLX community 4-bit",
        ),
    ],
    "Qwen/Qwen2.5-Coder-14B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-Coder-14B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
    ],
    "Qwen/Qwen2.5-14B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-14B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-14B-Instruct|mlx4",
            "MLX · 4-bit",
            "mlx-community/Qwen2.5-14B-Instruct-4bit",
            "mlx4",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
        ),
    ],
    "Qwen/Qwen2.5-Coder-32B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-Coder-32B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
    ],
    "Qwen/Qwen2.5-32B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-32B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-32B-Instruct|mlx4",
            "MLX · 4-bit",
            "mlx-community/Qwen2.5-32B-Instruct-4bit",
            "mlx4",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
        ),
    ],
    "Qwen/Qwen2.5-72B-Instruct": [
        ModelVariant(
            "Qwen/Qwen2.5-72B-Instruct|awq",
            "AWQ · 4-bit",
            "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "Qwen/Qwen2.5-72B-Instruct|gptq",
            "GPTQ · 4-bit",
            "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            "gptq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
    ],
    "mistralai/Mistral-7B-Instruct-v0.3": [
        ModelVariant(
            "mistralai/Mistral-7B-Instruct-v0.3|awq",
            "AWQ · 4-bit",
            "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
        ModelVariant(
            "mistralai/Mistral-7B-Instruct-v0.3|mlx4",
            "MLX · 4-bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx4",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
        ),
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        ModelVariant(
            "meta-llama/Llama-3.1-8B-Instruct|mlx4",
            "MLX · 4-bit",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "mlx4",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
        ),
        ModelVariant(
            "meta-llama/Llama-3.1-8B-Instruct|awq",
            "AWQ · 4-bit",
            "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            "awq",
            download_scale=0.35,
            final_scale=0.35,
            source="huggingface",
        ),
    ],
    "meta-llama/Llama-3.3-70B-Instruct": [
        ModelVariant(
            "meta-llama/Llama-3.3-70B-Instruct|awq",
            "AWQ · 4-bit",
            "casperhansen/llama-3.3-70b-instruct-awq",
            "awq",
            download_scale=0.3,
            final_scale=0.3,
            source="huggingface",
        ),
        ModelVariant(
            "meta-llama/Llama-3.3-70B-Instruct|mlx4",
            "MLX · 4-bit",
            "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "mlx4",
            download_scale=0.28,
            final_scale=0.28,
            source="huggingface",
        ),
    ],
    "meta-llama/Llama-3.1-405B-Instruct": [
        ModelVariant(
            "meta-llama/Llama-3.1-405B-Instruct|fp8",
            "FP8",
            "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8",
            "fp8",
            download_scale=0.55,
            final_scale=0.55,
            source="huggingface",
            notes="FP8 quantized weights on Hugging Face",
        ),
    ],
}


def variants_for(repo_id: str, include_airllm: bool = True) -> List[ModelVariant]:
    """
    Build dropdown options for a base model repo.

    Always includes full precision. Adds AirLLM 8/4-bit. Adds curated HF quants.
    """
    out: List[ModelVariant] = [_base(repo_id)]
    if include_airllm:
        out.append(_airllm_bits(repo_id, "8bit", final_scale=0.55))
        out.append(_airllm_bits(repo_id, "4bit", final_scale=0.35))
    for v in _HF_VARIANTS.get(repo_id, []):
        # Avoid duplicate repo+compression
        if any(
            x.repo_id == v.repo_id
            and x.compression == v.compression
            and x.precision == v.precision
            for x in out
        ):
            continue
        # Legacy disk_scale field → map if present via getattr
        out.append(v)
    return out


def variant_dropdown_labels(repo_id: str) -> Tuple[List[str], Dict[str, ModelVariant]]:
    """Return (labels for CTkOptionMenu, label -> ModelVariant)."""
    vs = variants_for(repo_id)
    labels: List[str] = []
    mapping: Dict[str, ModelVariant] = {}
    for v in vs:
        labels.append(v.label)
        mapping[v.label] = v
    return labels, mapping


def default_variant(repo_id: str) -> ModelVariant:
    return _base(repo_id)
