"""Built-in web search tool (optional network)."""

from __future__ import annotations

from typing import List

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information. Arguments: search query string."

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results

    def run(self, arguments: str) -> ToolResult:
        query = (arguments or "").strip().strip('"').strip("'")
        if not query:
            return ToolResult(success=False, output="Empty search query.")

        try:
            return self._ddg(query)
        except Exception as primary_err:
            try:
                return self._requests_fallback(query)
            except Exception as fallback_err:
                return ToolResult(
                    success=False,
                    output=(
                        f"Web search failed.\n"
                        f"Primary: {primary_err}\nFallback: {fallback_err}"
                    ),
                )

    def _ddg(self, query: str) -> ToolResult:
        from duckduckgo_search import DDGS

        results: List[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=self.max_results):
                results.append(r)

        if not results:
            return ToolResult(success=True, output=f"No results for: {query}")

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title") or ""
            href = r.get("href") or r.get("link") or ""
            body = r.get("body") or r.get("snippet") or ""
            lines.append(f"{i}. {title}\n   {href}\n   {body}\n")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"results": results, "query": query},
        )

    def _requests_fallback(self, query: str) -> ToolResult:
        """Minimal fallback using DuckDuckGo HTML (no API key)."""
        import re
        import urllib.parse
        import urllib.request

        q = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AirLLMStudio/0.1 (local research tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Rough extract of result snippets
        titles = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>', html, flags=re.DOTALL
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td)', html, flags=re.DOTALL
        )

        def strip_tags(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s).strip()

        lines = [f"Search results for: {query}\n"]
        for i, t in enumerate(titles[: self.max_results], 1):
            sn = strip_tags(snippets[i - 1]) if i - 1 < len(snippets) else ""
            lines.append(f"{i}. {strip_tags(t)}\n   {sn}\n")

        if len(lines) == 1:
            return ToolResult(success=True, output=f"No parseable results for: {query}")
        return ToolResult(success=True, output="\n".join(lines), data={"query": query})
