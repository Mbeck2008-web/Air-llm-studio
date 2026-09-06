"""Extensible tool base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    success: bool
    output: str
    data: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, arguments: str) -> ToolResult:
        """Execute the tool with free-form argument string from the model."""

    def schema_hint(self) -> str:
        return f"{self.name}: {self.description}"
