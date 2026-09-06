"""python -m airllm_studio"""

import os

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
os.environ.setdefault("AIRLLM_MOE_DEVICE", os.environ.get("AIRLLM_MOE_DEVICE", "cpu"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main() -> None:
    try:
        from airllm_studio.runtime_env import apply_sandbox_environment

        apply_sandbox_environment()
    except Exception as exc:
        print(f"[airllm_studio] sandbox env: {exc}")

    try:
        from airllm_studio.core.mlx_compat import apply_airllm_mlx_patches

        apply_airllm_mlx_patches()
    except Exception as exc:
        print(f"[airllm_studio] mlx_compat: {exc}")

    ui = (os.environ.get("AIRLLM_STUDIO_UI") or "web").strip().lower()
    if ui in ("tk", "ctk", "classic"):
        from airllm_studio.ui.app_window import run_app

        run_app()
        return

    from airllm_studio.desktop import run_desktop

    run_desktop()


if __name__ == "__main__":
    main()
