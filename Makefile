.PHONY: test lint typecheck e2e-stack-up e2e-stack-down

test:
	python -m pytest

lint:
	ruff check .

typecheck:
	mypy llmasm

e2e-stack-up:
	docker compose up -d --build postgres graph-viewer --remove-orphans

e2e-stack-down:
	docker compose down
