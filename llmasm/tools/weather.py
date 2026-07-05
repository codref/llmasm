"""Weather lookup tool via the Open-Meteo API (free, no API key)."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from llmasm.schemas import WeatherObservation
from llmasm.tools.base import ToolSpec

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10.0


class WeatherTool:
    """Look up current weather conditions for a location via Open-Meteo."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="weather.lookup",
            description="Get current weather conditions for a location (city name).",
            input_schema="RawText",
            output_schema="WeatherObservation",
        )

    def invoke(self, input: BaseModel) -> BaseModel:
        location = getattr(input, "text", str(input)).strip()
        if not location:
            return WeatherObservation(condition="No location provided.")
        try:
            coords = self._geocode(location)
            if coords is None:
                return WeatherObservation(condition=f"Location '{location}' not found.")
            lat, lon, name = coords
            return self._weather(lat, lon, name)
        except httpx.HTTPError as exc:
            return WeatherObservation(condition=f"Weather API error: {exc}")

    def _geocode(self, location: str) -> tuple[float, float, str] | None:
        response = httpx.get(
            _GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results")
        if not results:
            return None
        item = results[0]
        return (
            float(item["latitude"]),
            float(item["longitude"]),
            str(item.get("name", location)),
        )

    def _weather(self, lat: float, lon: float, name: str) -> WeatherObservation:
        response = httpx.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current_weather", {})
        temp = current.get("temperature", "?")
        speed = current.get("windspeed", "?")
        code = current.get("weathercode", 0)
        desc = _weather_code_description(code)
        condition = f"{name}: {temp}°C, {desc}, wind {speed} km/h"
        source = f"https://open-meteo.com/en/weather/{name.replace(' ', '-').lower()}?latitude={lat}&longitude={lon}"
        return WeatherObservation(condition=condition, source_url=source)


def _weather_code_description(code: int) -> str:
    mapping = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return mapping.get(code, f"weather code {code}")
