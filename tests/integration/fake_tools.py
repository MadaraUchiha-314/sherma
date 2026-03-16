"""Fake tools for integration tests — no HTTP calls."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (fake, for testing)."""
    return f"{city}: temperature=20C, windspeed=5km/h, clear skies"
