# Open-Meteo API Reference

## Geocoding Endpoint

`GET https://geocoding-api.open-meteo.com/v1/search`

| Param | Description |
|-------|-------------|
| name  | City name to search |
| count | Max results (use 1) |

Returns `{ "results": [{ "latitude", "longitude", "name", "country" }] }`

## Weather Endpoint

`GET https://api.open-meteo.com/v1/forecast`

| Param | Description |
|-------|-------------|
| latitude | Latitude from geocoding |
| longitude | Longitude from geocoding |
| current_weather | Set to `true` |

Returns `{ "current_weather": { "temperature", "windspeed", "winddirection", "weathercode" } }`
