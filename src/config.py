"""Shared configuration: loads .env once and exposes the default model name."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
