weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"},
    },
    "required": ["location", "units", "include_forecast"],
}


@tool(args_schema=weather_schema)
def get_weather(
    location: str, units: str = "celsius", include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"

    if include_forecast:
        result += "\nNext 5-days forecast"

    return result
