.PHONY: install test lint run-api

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests

run-api:
	uvicorn advanced_rag.api.main:app --reload
