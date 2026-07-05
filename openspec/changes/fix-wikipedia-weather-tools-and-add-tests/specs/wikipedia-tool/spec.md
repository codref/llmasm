## ADDED Requirements

### Requirement: WikipediaTool queries the MediaWiki API with a User-Agent header
`WikipediaTool` SHALL include a descriptive `User-Agent` header in every request to the MediaWiki API.

#### Scenario: Successful search
- **WHEN** `WikipediaTool` searches for a query
- **THEN** the request includes a `User-Agent` header

### Requirement: WikipediaTool returns an article summary
`WikipediaTool` SHALL search Wikipedia for the query, select the first result, fetch its extract, and return the title and extract as `RawText`.

#### Scenario: Article found
- **WHEN** the search returns at least one result and the article has an extract
- **THEN** the tool returns `RawText` containing the article title and extract

#### Scenario: No search results
- **WHEN** the search returns no results
- **THEN** the tool returns `RawText` stating that no articles were found

#### Scenario: Extract unavailable
- **WHEN** the search returns a result but the article has no extract
- **THEN** the tool returns `RawText` stating that no extract is available

#### Scenario: API error
- **WHEN** the MediaWiki API returns an HTTP error
- **THEN** the tool returns `RawText` describing the error without raising
