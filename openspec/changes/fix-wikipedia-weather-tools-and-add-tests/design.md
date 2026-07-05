## Context

`examples/chat.py` registers four built-in tools: `wikipedia.search`, `weather.lookup`, `calculator.eval`, and `file_reader.read`. The calculator and file reader already accept `RawText` input, which lets the planner wire them directly to an `intent` node. `WikipediaTool` also accepts `RawText`, but its HTTP requests are rejected by the MediaWiki API with HTTP 403 because they lack a `User-Agent` header. `WeatherTool` uses a typed `WeatherQuery` input, which cannot be produced by the intent node; the default transform registry has no `RawText → WeatherQuery` transform, so a planner-generated static tool graph fails validation.

There are no unit tests covering these tools, so the failures were only visible when running the live example.

## Goals / Non-Goals

**Goals:**
- Make `WikipediaTool` return article summaries reliably against the live MediaWiki API.
- Make `WeatherTool` usable from planner-generated static tool-node graphs.
- Add deterministic unit tests for both tools that do not depend on external network availability.
- Keep the changes minimal and focused on the built-in example tools.

**Non-Goals:**
- Adding new tools or changing the generic tool-use runtime loop.
- Supporting non-English Wikipedia endpoints or advanced WeatherQuery parameters (date, units).
- Refactoring the tool base protocol or registry.
- Adding integration tests against live APIs.

## Decisions

1. **Wikipedia: include a `User-Agent` header**
   - Add a module-level `_USER_AGENT` constant (e.g. `"llmasm/0.1 (llmasm-tool)"`) and pass it as `headers={"User-Agent": _USER_AGENT}` to every `httpx.get` call.
   - Rationale: MediaWiki's API policy requires a descriptive user agent; without it the API returns 403.
   - Alternative considered: use a third-party Wikipedia wrapper library. Rejected to avoid a new dependency for a single example tool.

2. **Weather: accept `RawText` input instead of `WeatherQuery`**
   - Change `WeatherTool.spec().input_schema` to `"RawText"` and parse the location from `input.text.strip()`.
   - Rationale: All other chat-oriented built-in tools use `RawText` input, allowing the planner to connect the intent node directly. This removes the need for a `RawText → WeatherQuery` transform or an intermediate model node.
   - Alternative considered: register a new transform. Rejected because it adds registry complexity for one example tool and is harder for tool authors to discover.

3. **Tests: mock `httpx.get`**
   - Use `pytest.monkeypatch` to replace `httpx.get` with a function that returns a `_MockResponse` object.
   - Rationale: Keeps the test suite fast, deterministic, and CI-friendly.
   - For `WeatherTool`, mock both the geocoding call and the forecast call in sequence.
   - For `WikipediaTool`, mock the search call and the extract call, and assert the `User-Agent` header is present.

4. **Integration test for `WeatherTool` static node**
   - Build an `intent → weather.lookup → model → final` graph and run it through the executor using `FakeProvider`.
   - Rationale: Verifies that the schema change actually fixes the planner/executor path, not just the tool in isolation.

## Risks / Trade-offs

- **[Risk]** MediaWiki may still rate-limit or block the generic user agent in production.
  **Mitigation:** The example is illustrative; users deploying this tool can override the constant or switch to a library with rate-limit handling.

- **[Risk]** Changing `WeatherTool` input from `WeatherQuery` to `RawText` removes the explicit `date` field.
  **Mitigation:** The tool only ever supported current weather; the field was unused. This is an example tool, not a public API surface.

- **[Risk]** Tests that mock `httpx.get` at the module level may not catch regressions in URL construction.
  **Mitigation:** Assert on the mocked `url` and `params` in the test to pin the request shape.

## Migration Plan

No migration needed. `examples/chat.py` already registers both tools and will continue to work after the schema change.

## Open Questions

- Should the `User-Agent` string include a project URL or contact email to satisfy MediaWiki’s policy more completely?
