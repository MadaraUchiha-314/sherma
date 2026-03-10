---
name: Weather Lookup
description: Get current weather conditions for any city worldwide.
license: MIT
---
# Weather Lookup Skill

Use the `get_weather` tool to retrieve current weather for a given city.

## Usage

Call `get_weather(city="<city name>")`. The tool returns a JSON string with:
- Location name and country
- Temperature (°C)
- Wind speed (km/h)
- Wind direction (degrees)
- Weather code (WMO standard)

## Notes

- The API is free and requires no authentication.
- City names are resolved via geocoding; use well-known city names for best results.
- Weather data is current (not forecast).
