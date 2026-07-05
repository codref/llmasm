"""Tests for the Wikipedia example tool."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from llmasm.schemas import RawText
from llmasm.tools.wikipedia import WikipediaTool, _USER_AGENT


class _MockResponse:
    def __init__(self, json_data: dict[str, Any] | None = None, raise_error: Exception | None = None) -> None:
        self._json = json_data or {}
        self._raise = raise_error

    def raise_for_status(self) -> None:
        if self._raise is not None:
            raise self._raise

    def json(self) -> dict[str, Any]:
        return self._json


def _mock_get(responses: list[_MockResponse]) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []
    index = 0

    def mock_get(url: str, **kwargs: Any) -> _MockResponse:
        nonlocal index
        calls.append({"url": url, "params": kwargs.get("params", {}), "headers": kwargs.get("headers", {})})
        response = responses[index]
        index = min(index + 1, len(responses) - 1)
        return response

    return calls, mock_get


def test_wikipedia_tool_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse({"query": {"search": []}}),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    WikipediaTool().invoke(RawText(text="Ada Lovelace"))

    assert calls[0]["headers"].get("User-Agent") == _USER_AGENT


def test_wikipedia_tool_article_found(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse({"query": {"search": [{"title": "Ada Lovelace"}]}}),
            _MockResponse({"query": {"pages": {"1": {"extract": "First programmer."}}}}),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WikipediaTool().invoke(RawText(text="Ada Lovelace"))

    assert isinstance(result, RawText)
    assert "Ada Lovelace" in result.text
    assert "First programmer." in result.text


def test_wikipedia_tool_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse({"query": {"search": []}}),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WikipediaTool().invoke(RawText(text="xyzxyzxyz"))

    assert isinstance(result, RawText)
    assert "No Wikipedia articles found" in result.text


def test_wikipedia_tool_extract_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse({"query": {"search": [{"title": "Missing Extract"}]}}),
            _MockResponse({"query": {"pages": {"1": {}}}}),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WikipediaTool().invoke(RawText(text="Missing Extract"))

    assert isinstance(result, RawText)
    assert "No extract available" in result.text


def test_wikipedia_tool_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, mock_get = _mock_get(
        [
            _MockResponse(raise_error=httpx.HTTPError("network down")),
        ]
    )
    monkeypatch.setattr(httpx, "get", mock_get)

    result = WikipediaTool().invoke(RawText(text="Anything"))

    assert isinstance(result, RawText)
    assert "Wikipedia API error" in result.text
