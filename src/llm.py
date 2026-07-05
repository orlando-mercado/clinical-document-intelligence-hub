"""Shared helper: a Claude structured-output call with a bounded retry loop.

Structured outputs (`output_format=`) make the API enforce the JSON shape,
but not everything is server-side enforced — numeric bounds like
`confidence` being in [0, 1] are validated client-side by Pydantic (see
extraction.py's module docstring) — and safety refusals / transient API
errors still need a retry regardless. This one loop is reused by the
extraction and summarization pipeline stages so the retry behavior stays
identical rather than reimplemented twice with subtly different bugs.
"""

from __future__ import annotations

import logging
from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredCallError(RuntimeError):
    """Raised when a structured-output call fails after all retry attempts."""


def call_structured(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: str,
    messages: list[dict],
    output_format: Type[ModelT],
    max_tokens: int,
    max_attempts: int,
    label: str,
) -> ModelT:
    """Call `client.messages.parse(...)`, retrying on API errors, safety
    refusals, and validation failures, up to `max_attempts` times.

    `label` is only used for log messages / the final error, so callers can
    describe what was being attempted (e.g. "extraction of lab report from
    'x.pdf'") without this module needing to know about documents at all.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_format=output_format,
            )
        except (anthropic.APIError, ValidationError) as exc:
            last_error = exc
            logger.warning("%s: attempt %d/%d failed: %s", label, attempt, max_attempts, exc)
            continue

        if response.stop_reason == "refusal":
            last_error = StructuredCallError(f"Model declined: {label}.")
            logger.warning("%s: attempt %d/%d refused.", label, attempt, max_attempts)
            continue

        parsed = response.parsed_output
        if parsed is None:
            last_error = StructuredCallError(f"{label}: structured output parsing returned no result.")
            logger.warning("%s: attempt %d/%d — parsed_output is None.", label, attempt, max_attempts)
            continue

        try:
            data = parsed if isinstance(parsed, dict) else parsed.model_dump()
            return output_format.model_validate(data)
        except ValidationError as exc:
            last_error = exc
            logger.warning("%s: attempt %d/%d failed re-validation: %s", label, attempt, max_attempts, exc)
            continue

    raise StructuredCallError(f"{label}: failed after {max_attempts} attempts.") from last_error
