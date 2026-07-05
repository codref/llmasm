## 1. Fix WikipediaTool API requests

- [x] 1.1 Add a module-level `_USER_AGENT` constant to `llmasm/tools/wikipedia.py`.
- [x] 1.2 Pass `headers={"User-Agent": _USER_AGENT}` to both `httpx.get` calls in `WikipediaTool`.
- [x] 1.3 Run the tool manually against a real query to confirm HTTP 200 and a populated summary.

## 2. Fix WeatherTool input schema

- [x] 2.1 Change `WeatherTool.spec().input_schema` from `"WeatherQuery"` to `"RawText"` in `llmasm/tools/weather.py`.
- [x] 2.2 Update `WeatherTool.invoke()` to read `input.text.strip()` as the location instead of expecting `WeatherQuery`.
- [x] 2.3 Verify the tool still works in isolation with a real city name.

## 3. Add unit tests for WikipediaTool

- [x] 3.1 Create `tests/unit/test_wikipedia_tool.py`.
- [x] 3.2 Add a `_MockResponse` helper and monkeypatch `httpx.get` to return search + extract responses.
- [x] 3.3 Write a test asserting that `WikipediaTool.invoke()` sends a `User-Agent` header.
- [x] 3.4 Write tests for: article found, no results, extract unavailable, and API error.

## 4. Add unit tests for WeatherTool

- [x] 4.1 Create `tests/unit/test_weather_tool.py`.
- [x] 4.2 Add a `_MockResponse` helper and monkeypatch `httpx.get` to return geocoding + forecast responses.
- [x] 4.3 Write tests for: location found, location not found, empty input, and API error.
- [x] 4.4 Write an integration test that builds an `intent → weather.lookup → model → final` graph and runs it through the executor with `FakeProvider`, asserting the run succeeds.

## 5. Verify the change

- [x] 5.1 Run `make test` (or `python -m pytest tests/unit/test_wikipedia_tool.py tests/unit/test_weather_tool.py -v`) and confirm all new tests pass.
- [x] 5.2 Run `make lint` and `make typecheck` and fix any issues.
