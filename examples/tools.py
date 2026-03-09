"""Shared tools for sherma examples."""

from __future__ import annotations

import json

import httpx
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city using Open-Meteo (free, no API key)."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = httpx.get(geo_url, params={"name": city, "count": 1}, timeout=15)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results")
    if not results:
        return f"Could not find location: {city}"

    loc = results[0]
    lat, lon = loc["latitude"], loc["longitude"]
    name = loc.get("name", city)
    country = loc.get("country", "")

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = httpx.get(
        weather_url,
        params={"latitude": lat, "longitude": lon, "current_weather": "true"},
        timeout=15,
    )
    weather_resp.raise_for_status()
    current = weather_resp.json().get("current_weather", {})

    return f"{name}, {country}: {json.dumps(current)}"
