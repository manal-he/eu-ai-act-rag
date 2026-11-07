# Common tasks. Override the interpreter with `make PY=python3.11 ...`.
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif

.PHONY: help install install-all ingest fixture dev ui test lint format typecheck eval eval-full \
        eval-gen docker-up docker-down clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python  ## core + dev dependencies (offline mode)
	$(BIN)/pip install -e ".[dev]"

install-all: $(BIN)/python  ## + local models, pgvector and cloud providers
	$(BIN)/pip install -e ".[dev,local,pg,vertex,anthropic,openai]"

ingest:  ## download FR/EN texts from EUR-Lex and parse -> data/processed/corpus.jsonl
	$(BIN)/euai ingest

fixture: ingest  ## regenerate the committed offline fixture
	$(BIN)/euai make-fixture

dev:  ## run the API with reload on :8000
	$(BIN)/uvicorn eu_ai_act_rag.api.main:app_factory --factory --reload --port 8000

ui:  ## run the UI dev server on :5173 (proxies /api to :8000)
	cd ui && npm install && npm run dev

test:  ## offline test-suite (mock LLM, no keys)
	$(BIN)/pytest

lint:  ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check src tests evals

format:
	$(BIN)/ruff format src tests evals
	$(BIN)/ruff check --fix .

typecheck:
	$(BIN)/mypy

eval:  ## offline retrieval ablation on the fixture (BM25 + hashing), writes evals/results/offline
	$(BIN)/python evals/run_ablation.py --profile offline --generation

eval-full: ingest  ## full ablation: downloaded corpus, multilingual-e5 + cross-encoder (needs [local])
	$(BIN)/python evals/run_ablation.py --profile full

eval-gen:  ## end-to-end eval with a real LLM + LLM judge (needs keys in .env)
	$(BIN)/python evals/run_ablation.py --profile full --generation --llm gemini --judge llm --judge-llm anthropic

docker-up:  ## api + pgvector + ui
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache ui/dist data/cache
