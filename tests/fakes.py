"""Shared test doubles for the Anthropic client, used by test_extraction.py
and test_summarization.py so both exercise src.llm.call_structured's retry
loop without hitting the real API (deterministic, free, no network)."""

from __future__ import annotations

import anthropic


class FakeAPIError(anthropic.APIError):
    """A minimal, constructible stand-in — the real APIError requires a live
    httpx request/response we don't want to build just to test retry logic."""

    def __init__(self, message: str = "fake api error"):
        self._message = message

    def __str__(self) -> str:
        return self._message


class FakeResponse:
    def __init__(self, parsed_output=None, stop_reason: str = "end_turn"):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, items):
        self._items = list(items)
        self.calls = 0
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        item = self._items[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, items):
        self.messages = FakeMessages(items)
