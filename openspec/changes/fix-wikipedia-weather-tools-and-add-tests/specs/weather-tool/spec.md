## ADDED Requirements

### Requirement: WeatherTool accepts a plain-text city name
`WeatherTool` SHALL accept `RawText` input and use the trimmed text as the location to look up.

#### Scenario: City name provided
- **WHEN** the input text is a non-empty city name
- **THEN** the tool geocodes the city and requests current weather

#### Scenario: Empty input
- **WHEN** the input text is empty or whitespace only
- **THEN** the tool returns a `WeatherObservation` with a condition indicating no location was provided

### Requirement: WeatherTool returns current conditions
`WeatherTool` SHALL resolve the city name to latitude and longitude via the Open-Meteo geocoding API, then request current weather and return a `WeatherObservation` with a human-readable condition string.

#### Scenario: Location found
- **WHEN** the geocoding API returns results
- **THEN** the tool returns a `WeatherObservation` containing the resolved name, temperature, weather description, wind speed, and a source URL

#### Scenario: Location not found
- **WHEN** the geocoding API returns no results
- **THEN** the tool returns a `WeatherObservation` with a condition stating the location was not found

#### Scenario: API error
- **WHEN** either the geocoding or forecast API returns an HTTP error
- **THEN** the tool returns a `WeatherObservation` describing the error without raising
