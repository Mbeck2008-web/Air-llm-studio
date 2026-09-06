"""Chat conversation view with generation controls."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import customtkinter as ctk

from airllm_studio.core.chat_store import Message
from airllm_studio.ui.theme import COLORS, font


class ChatView(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_send: Callable[[str], None],
        on_stop: Optional[Callable[[], None]] = None,
        on_model_pick: Optional[Callable[[str], None]] = None,
        on_airllm_toggle: Optional[Callable[[bool], None]] = None,
        airllm_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(master, fg_color=COLORS["bg"], corner_radius=0, **kwargs)
        self.on_send = on_send
        self.on_stop = on_stop
        self.on_model_pick = on_model_pick
        self.on_airllm_toggle = on_airllm_toggle
        self._airllm_enabled = airllm_enabled
        self._streaming = False
        self._model_values: List[str] = ["No downloaded models"]
        self._model_map: dict = {}  # display label -> repo_id
        self._build()

    def _build(self) -> None:
        # Header: chat title | model dropdown + AirLLM toggle | stream status
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_elevated"], height=52, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        self.header_lbl = ctk.CTkLabel(
            header,
            text="Chat",
            font=font(14, "bold"),
            text_color=COLORS["text"],
            width=100,
            anchor="w",
        )
        self.header_lbl.pack(side="left", padx=16, pady=12)

        mid = ctk.CTkFrame(header, fg_color="transparent")
        mid.pack(side="left", expand=True, fill="x", padx=8)

        picker_row = ctk.CTkFrame(mid, fg_color="transparent")
        picker_row.pack(anchor="center")

        ctk.CTkLabel(
            picker_row,
            text="Model",
            font=font(11),
            text_color=COLORS["text_dim"],
        ).pack(side="left", padx=(0, 8))

        self.model_menu = ctk.CTkOptionMenu(
            picker_row,
            values=self._model_values,
            command=self._on_model_menu,
            font=font(12),
            width=280,
            height=32,
            fg_color=COLORS["bg_panel"],
            button_color=COLORS["accent_dim"],
            button_hover_color=COLORS["accent"],
            dropdown_fg_color=COLORS["bg_elevated"],
            dropdown_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            anchor="center",
        )
        self.model_menu.pack(side="left")
        self.model_menu.set("No downloaded models")

        # AirLLM enable toggle — immediately to the right of model selector
        air_wrap = ctk.CTkFrame(picker_row, fg_color="transparent")
        air_wrap.pack(side="left", padx=(14, 0))

        ctk.CTkLabel(
            air_wrap,
            text="AirLLM",
            font=font(11, "bold"),
            text_color=COLORS["accent"],
        ).pack(side="left", padx=(0, 6))

        self.airllm_switch = ctk.CTkSwitch(
            air_wrap,
            text="",
            width=42,
            switch_width=40,
            switch_height=20,
            progress_color=COLORS["accent_dim"],
            button_color=COLORS["text"],
            button_hover_color=COLORS["accent"],
            command=self._on_airllm_switch,
        )
        self.airllm_switch.pack(side="left")
        if self._airllm_enabled:
            self.airllm_switch.select()
        else:
            self.airllm_switch.deselect()

        self.airllm_state_lbl = ctk.CTkLabel(
            air_wrap,
            text="On" if self._airllm_enabled else "Off",
            font=font(10),
            text_color=COLORS["text_muted"],
            width=28,
        )
        self.airllm_state_lbl.pack(side="left", padx=(6, 0))

        self.stream_lbl = ctk.CTkLabel(
            header,
            text="",
            font=font(11),
            text_color=COLORS["accent"],
            width=140,
            anchor="e",
        )
        self.stream_lbl.pack(side="right", padx=16)

        # Messages
        self.msg_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["bg"],
            scrollbar_button_color=COLORS["border"],
        )
        self.msg_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Controls bar
        controls = ctk.CTkFrame(self, fg_color=COLORS["bg_elevated"], corner_radius=0)
        controls.pack(fill="x", padx=0, pady=0)

        inner = ctk.CTkFrame(controls, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=8)

        ctk.CTkLabel(inner, text="Temp", font=font(11), text_color=COLORS["text_dim"]).pack(
            side="left", padx=(0, 4)
        )
        self.temp_slider = ctk.CTkSlider(
            inner, from_=0.0, to=1.5, number_of_steps=30, width=100
        )
        self.temp_slider.set(0.7)
        self.temp_slider.pack(side="left", padx=(0, 12))
        self.temp_val = ctk.CTkLabel(inner, text="0.70", font=font(11), width=36)
        self.temp_val.pack(side="left", padx=(0, 16))
        self.temp_slider.configure(command=self._on_temp)

        ctk.CTkLabel(inner, text="Max tokens", font=font(11), text_color=COLORS["text_dim"]).pack(
            side="left", padx=(0, 4)
        )
        self.max_tokens = ctk.CTkEntry(inner, width=64, font=font(12))
        self.max_tokens.insert(0, "512")
        self.max_tokens.pack(side="left", padx=(0, 16))

        ctk.CTkLabel(inner, text="Top-p", font=font(11), text_color=COLORS["text_dim"]).pack(
            side="left", padx=(0, 4)
        )
        self.top_p = ctk.CTkEntry(inner, width=56, font=font(12))
        self.top_p.insert(0, "0.9")
        self.top_p.pack(side="left")

        # Input
        input_row = ctk.CTkFrame(self, fg_color=COLORS["bg_elevated"], corner_radius=0)
        input_row.pack(fill="x")

        self.input = ctk.CTkTextbox(
            input_row,
            height=80,
            font=font(13),
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            border_width=1,
            wrap="word",
        )
        self.input.pack(side="left", fill="x", expand=True, padx=(12, 8), pady=12)
        self.input.bind("<Command-Return>", self._send_event)
        self.input.bind("<Control-Return>", self._send_event)

        btn_col = ctk.CTkFrame(input_row, fg_color="transparent")
        btn_col.pack(side="right", padx=(0, 12), pady=12)

        self.send_btn = ctk.CTkButton(
            btn_col,
            text="Send",
            font=font(13, "bold"),
            width=88,
            height=36,
            fg_color=COLORS["accent_dim"],
            hover_color=COLORS["accent"],
            text_color=COLORS["bg"],
            command=self._send,
        )
        self.send_btn.pack(pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_col,
            text="Stop",
            font=font(12),
            width=88,
            height=28,
            fg_color=COLORS["bg_panel"],
            hover_color=COLORS["danger"],
            command=self._stop,
            state="disabled",
        )
        self.stop_btn.pack()

    def _on_model_menu(self, choice: str) -> None:
        if not self.on_model_pick:
            return
        if choice in ("No downloaded models", "— Select model —"):
            return
        repo_id = self._model_map.get(choice) or choice
        self.on_model_pick(repo_id)

    def _on_airllm_switch(self) -> None:
        enabled = bool(self.airllm_switch.get())
        self._airllm_enabled = enabled
        self.airllm_state_lbl.configure(text="On" if enabled else "Off")
        if self.on_airllm_toggle:
            self.on_airllm_toggle(enabled)

    def set_airllm_enabled(self, enabled: bool) -> None:
        """Sync switch from app/settings without re-firing callback storms."""
        self._airllm_enabled = enabled
        if enabled:
            self.airllm_switch.select()
        else:
            self.airllm_switch.deselect()
        self.airllm_state_lbl.configure(text="On" if enabled else "Off")

    def get_airllm_enabled(self) -> bool:
        return bool(self.airllm_switch.get())

    def set_model_options(
        self,
        options: Sequence[Tuple[str, str]],
        selected_repo: Optional[str] = None,
    ) -> None:
        """options: list of (repo_id, display_label)"""
        if not options:
            self._model_values = ["No downloaded models"]
            self._model_map = {}
            self.model_menu.configure(values=self._model_values)
            self.model_menu.set(self._model_values[0])
            return

        self._model_map = {label: rid for rid, label in options}
        for rid, label in options:
            self._model_map[rid] = rid
        self._model_values = [label for _, label in options]
        self.model_menu.configure(values=self._model_values)

        if selected_repo:
            for rid, label in options:
                if rid == selected_repo:
                    self.model_menu.set(label)
                    return
        self.model_menu.set(self._model_values[0])

    def _on_temp(self, v: float) -> None:
        self.temp_val.configure(text=f"{float(v):.2f}")

    def _send_event(self, _event=None):
        self._send()
        return "break"

    def _send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text or self._streaming:
            return
        self.input.delete("1.0", "end")
        self.on_send(text)

    def _stop(self) -> None:
        if self.on_stop:
            self.on_stop()

    def get_gen_params(self) -> dict:
        try:
            max_tok = int(self.max_tokens.get().strip() or "512")
        except ValueError:
            max_tok = 512
        try:
            top_p = float(self.top_p.get().strip() or "0.9")
        except ValueError:
            top_p = 0.9
        return {
            "temperature": float(self.temp_slider.get()),
            "max_new_tokens": max_tok,
            "top_p": top_p,
        }

    def set_streaming(self, streaming: bool) -> None:
        self._streaming = streaming
        self.send_btn.configure(state="disabled" if streaming else "normal")
        self.stop_btn.configure(state="normal" if streaming else "disabled")

    def set_stream_status(self, text: str) -> None:
        self.stream_lbl.configure(text=text)

    def set_title(self, title: str) -> None:
        self.header_lbl.configure(text=title)

    def clear_messages(self) -> None:
        for w in self.msg_frame.winfo_children():
            w.destroy()

    def render_messages(self, messages: List[Message]) -> None:
        self.clear_messages()
        for m in messages:
            self.append_message(m.role, m.content, tool_name=m.tool_name)

    def append_message(
        self, role: str, content: str, tool_name: Optional[str] = None
    ) -> ctk.CTkFrame:
        colors = {
            "user": COLORS["user_bubble"],
            "assistant": COLORS["assistant_bubble"],
            "tool": COLORS["tool_bubble"],
            "system": COLORS["bg_panel"],
        }
        labels = {
            "user": "You",
            "assistant": "Assistant",
            "tool": f"Tool · {tool_name or 'result'}",
            "system": "System",
        }
        bubble = ctk.CTkFrame(
            self.msg_frame,
            fg_color=colors.get(role, COLORS["bg_panel"]),
            corner_radius=10,
        )
        bubble.pack(fill="x", pady=6, padx=12, anchor="w")

        ctk.CTkLabel(
            bubble,
            text=labels.get(role, role),
            font=font(11, "bold"),
            text_color=COLORS["accent"] if role == "assistant" else COLORS["text_muted"],
        ).pack(anchor="w", padx=12, pady=(8, 0))

        body = ctk.CTkLabel(
            bubble,
            text=content or " ",
            font=font(13),
            text_color=COLORS["text"],
            wraplength=640,
            justify="left",
            anchor="w",
        )
        body.pack(anchor="w", padx=12, pady=(4, 12), fill="x")
        bubble._body_label = body  # type: ignore[attr-defined]
        try:
            self.msg_frame._parent_canvas.yview_moveto(1.0)  # type: ignore
        except Exception:
            pass
        return bubble

    def update_last_assistant(self, content: str) -> None:
        children = self.msg_frame.winfo_children()
        for w in reversed(children):
            body = getattr(w, "_body_label", None)
            if body is not None:
                kids = w.winfo_children()
                if kids and isinstance(kids[0], ctk.CTkLabel):
                    if kids[0].cget("text") == "Assistant":
                        body.configure(text=content or " ")
                        try:
                            self.msg_frame._parent_canvas.yview_moveto(1.0)  # type: ignore
                        except Exception:
                            pass
                        return
