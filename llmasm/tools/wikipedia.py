"""Wikipedia search tool via the MediaWiki API."""

from __future__ import annotations

import httpx
from typing import Any

from pydantic import BaseModel

from llmasm.schemas import RawText
from llmasm.tools.base import ToolSpec

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 10.0
_HEADERS = {"User-Agent": "llmasm/0.1.0 (https://github.com/codref/llmasm)"}


class WikipediaTool:
    """Search Wikipedia and return article extracts."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="wikipedia.search",
            description="Search Wikipedia for articles matching a query. Returns the article summary.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel, provider: Any = None) -> BaseModel:
        query = getattr(input, "text", str(input)).strip()
        if not query:
            return RawText(text="No query provided.")
        try:
            results = self._search(query)
            if not results:
                return RawText(text=f"No Wikipedia articles found for '{query}'.")
            title = str(results[0]["title"])
            extract = self._extract(title)
            if extract is None:
                return RawText(text=f"No extract available for '{title}'.")
            return RawText(text=f"{title}\n\n{extract}")
        except httpx.HTTPError as exc:
            return RawText(text=f"Wikipedia API error: {exc}")

    def _search(self, query: str) -> list[dict[str, object]]:
        params: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        response = httpx.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return list(data.get("query", {}).get("search", []))

    def _extract(self, title: str) -> str | None:
        params: dict[str, str | int | bool] = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        response = httpx.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract") or None
        return None
