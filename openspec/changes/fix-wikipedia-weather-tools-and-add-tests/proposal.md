## Why

The built-in example tools registered in `examples/chat.py` (`WikipediaTool` and `WeatherTool`) do not work reliably. `WikipediaTool` receives HTTP 403 responses from the MediaWiki API because it sends requests without a `User-Agent` header. `WeatherTool` works in isolation but cannot be used from a planner-generated static tool node because its input schema (`WeatherQuery`) does not match the intent node's output schema (`RawText`) and no transform exists. There are also no unit tests for either tool. Fixing these tools and adding tests makes the example path trustworthy and gives users a working reference for authoring their own tools.

## What Changes

- Add a `User-Agent` header to all `WikipediaTool` HTTP requests so the MediaWiki API returns 200 responses.
- Change `WeatherTool` to accept `RawText` input (a city name as plain text) instead of `WeatherQuery`, so it can be wired directly to an intent node in a planner-generated graph.
- Update `WeatherTool.spec()` to reflect the new `RawText` input schema and keep `WeatherObservation` as the output schema.
- Add unit tests for `WikipediaTool` with mocked `httpx` responses, asserting both successful lookup and proper header usage.
- Add unit tests for `WeatherTool` with mocked geocoding and forecast `httpx` responses, asserting a populated `WeatherObservation`.
- Add a test that exercises `WeatherTool` through an `intent → tool → model → final` task graph to verify it fits the static tool-node pipeline.

## Capabilities

### New Capabilities

- `wikipedia-tool`: The built-in Wikipedia search tool must query the MediaWiki API and return article summaries.
- `weather-tool`: The built-in weather lookup tool must resolve a city name to coordinates and return current conditions.

### Modified Capabilities

- None.

## Impact

- `llmasm/tools/wikipedia.py`
- `llmasm/tools/weather.py`
- `tests/unit/test_wikipedia_tool.py` (new)
- `tests/unit/test_weather_tool.py` (new)
- `examples/chat.py` is unaffected; it already registers both tools.
