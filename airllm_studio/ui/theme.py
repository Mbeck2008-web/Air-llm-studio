"""Dark theme tokens for AirLLM Studio."""

from __future__ import annotations

import customtkinter as ctk

# Palette — deep slate with teal accent (AirLLM / MLX inspired)
COLORS = {
    "bg": "#0f1117",
    "bg_elevated": "#161b22",
    "bg_panel": "#1a1f2b",
    "bg_input": "#0d1117",
    "border": "#2a3142",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "text_dim": "#6e7681",
    "accent": "#3dcaa3",
    "accent_hover": "#4fd1c5",
    "accent_dim": "#2a9d8f",
    "danger": "#f85149",
    "warning": "#d29922",
    "success": "#3fb950",
    "user_bubble": "#1f3a5f",
    "assistant_bubble": "#1c2333",
    "tool_bubble": "#2d1f3d",
    "moe_active": "#3dcaa3",
    "moe_idle": "#30363d",
    "moe_token": "#58a6ff",
}


def apply_theme() -> None:
    """Apply before creating the root window when possible."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    try:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
    except Exception:
        pass


# Use Tk defaults — SF Pro can fail to resolve in some Python builds
FONT_FAMILY = "Helvetica"
FONT_MONO = "Menlo"


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    try:
        return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)
    except Exception:
        return ctk.CTkFont(size=size, weight=weight)


def mono(size: int = 12) -> ctk.CTkFont:
    try:
        return ctk.CTkFont(family=FONT_MONO, size=size)
    except Exception:
        return ctk.CTkFont(size=size)
