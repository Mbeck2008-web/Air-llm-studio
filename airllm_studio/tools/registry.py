"""Tool registry — register new tools with one class."""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult
from .web_search import WebSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def run(self, name: str, arguments: str) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {name}. Available: {', '.join(self.names()) or '(none)'}",
            )
        return tool.run(arguments)

    def prompt_section(self) -> str:
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"- {t.schema_hint()}")
        lines.append(
            "To call a tool, output:\n"
            "<tool_call>\nTOOL_NAME\narguments\n</tool_call>"
        )
        return "\n".join(lines)


def get_default_registry(
    web_search: bool = True,
) -> ToolRegistry:
    reg = ToolRegistry()
    if web_search:
        reg.register(WebSearchTool())
    return reg
