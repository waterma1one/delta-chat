# Makefile — one-command entrypoints. Critical path (demo/run/chat/eval/markup/test) is zero-infra:
# no Docker, no Phoenix/Prometheus/Grafana, no LLM key required for ingest/delta/eval-gate. Grounded
# chat answers still need an LLM key (ANTHROPIC_API_KEY/OPENAI_API_KEY in .env) since generation is
# inherently model-backed — everything upstream of that (ingest, delta, retrieval, the eval gate)
# is pure code and runs with zero external services.
#
# Override PID_A / PID_B on the command line; defaults point at the synthesized golden sample pair.
#
# PYTHONPATH=src on every non-pytest target below: this repo's `uv run` editable install is
# order-dependent flaky in some environments (the fast .pth-based editable link `uv run` relinks on
# every invocation doesn't always get picked up by the interpreter's `site` module, even though it
# points at a real, existing directory) — `uv run pytest` never hits this because pytest's own
# `pythonpath = ["src"]` config (pyproject.toml) bypasses package-install resolution entirely.
# Setting PYTHONPATH explicitly here does the same for the CLI/eval entrypoints, deterministically.

PID_A ?= data/samples/pair_01_native/rev_a.pdf
PID_B ?= data/samples/pair_01_native/rev_b.pdf
RUN   := PYTHONPATH=src uv run

.PHONY: install samples demo run chat eval markup test up up-obs down

install:            ## sync deps (uv, Python 3.11+) + seed .env + synthesize sample pairs
	uv sync
	test -f .env || cp .env.example .env
	$(MAKE) samples

samples:             ## regenerate data/samples/ + eval gold (fast, deterministic; gitignored, not committed)
	$(RUN) python scripts/synth_samples.py

demo: samples        ## ACCEPTANCE COMMAND: ingest -> delta report -> grounded chat, one invocation
	$(RUN) delta-chat demo --pid-a "$(PID_A)" --pid-b "$(PID_B)"

run: samples         ## ingest both -> delta report (JSON+MD) + markup (best-effort), writes a trace file
	$(RUN) delta-chat run --pid-a "$(PID_A)" --pid-b "$(PID_B)"

chat: samples        ## interactive grounded chat in the terminal over an ingested pair
	$(RUN) delta-chat chat --pid-a "$(PID_A)" --pid-b "$(PID_B)"

eval: samples        ## scorecard.json + budget.md; diffs eval/baseline_scorecard.json; non-zero on regression
	$(RUN) python -m eval.run_eval --offline --baseline eval/baseline_scorecard.json

markup: samples      ## annotated PDF overlay (bonus)
	$(RUN) delta-chat markup --pid-a "$(PID_A)" --pid-b "$(PID_B)" --out markup.pdf

test: samples        ## deterministic + mocked-LLM suite (excludes @pytest.mark.llm; none defined yet)
	uv run pytest -m "not llm"

up:                  ## app only, no obs stack (file-based traces still write to ./traces)
	docker compose up --build

up-obs:              ## app + Phoenix (6006) + Prometheus (9090) + Grafana (3000), opt-in, never required
	docker compose --profile obs up --build

down:                ## stop and remove all compose services
	docker compose down
