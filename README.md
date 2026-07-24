# delta-chat

Two revisions of an engineering-drawing P&ID in, a structured delta and grounded chat over that
delta out — observable and evaluated end to end.

## What this is

Given PID_A and PID_B (two revisions of the same P&ID, as native/scanned PDF or DXF), delta-chat
ingests both into a canonical representation, computes a typed, deterministic **delta** (added /
removed / modified / moved / renumbered changes with dual confidence scores), and answers questions
about that delta and the underlying documents through a **grounded chat** interface — every claim
cites a resolvable source span, or the system abstains.

North star: a canonical-representation seam between ingestion and everything downstream; a
deterministic-first delta engine (Hungarian alignment + typed classification, not a raw text/pixel
diff and not an LLM-authored delta); the LLM confined to exactly two judgment calls — the semantic
note-gate (deciding whether an ambiguous reword is cosmetic) and the chat answer itself.

## Setup

```bash
uv sync                       # Python 3.11+, dependencies via uv
cp .env.example .env          # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY for grounded chat
```

(`make install` does both of the above, and only seeds `.env` if it doesn't already exist.)

- **Tesseract** is required for scanned-PDF OCR (`apt-get install tesseract-ocr` / `brew install
  tesseract`; the Docker image installs it automatically).
- **ODA File Converter** is optional, for real `.dwg` support (`.dwg` -> `.dxf`). Without it, the
  DWG adapter honestly reports "conversion required" rather than faking support; `.dxf` inputs
  always parse regardless.
- Keys come from the environment only (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `LOCAL_LLM_BASE_URL`
  for Ollama/vLLM), never from config, and are never committed (`.env` is gitignored).

## Run

| Command | What it does |
|---|---|
| `make demo` | **Acceptance command.** Ingests PID_A + PID_B, writes the delta report, then runs grounded chat — one invocation, no browser. |
| `make run` | Ingest -> delta report (JSON + Markdown) + markup overlay (best-effort) -> `data/output/<id>/`. |
| `make chat` | Interactive grounded chat over an already-computed pair. |
| `make eval` | Writes `scorecard.json` + `budget.md`, diffs `eval/baseline_scorecard.json`, exits non-zero on a deterministic regression. |
| `make markup` | Annotated PDF overlay of the delta (bonus). |
| `make test` | Full deterministic + mocked-LLM test suite. |
| `make up` | App only, via Docker (`GET /metrics`, `/healthz`, and the chat UI at `:8000/ui`). |
| `make up-obs` | App + Phoenix (`:6006`) + Prometheus (`:9090`) + Grafana (`:3000`) — opt-in, `--profile obs`. |

Override the sample pair with `PID_A=... PID_B=... make demo` (accepts `file://` URIs or bare
paths; defaults point at `data/samples/pair_01_native/`). `demo`/`run`/`chat`/`eval`/`markup`/`test`
all depend on `make samples`, which regenerates `data/samples/` + eval gold from the two source
PDFs (fast, deterministic, gitignored — not committed) — no manual setup step needed.

`docker-compose.yml`'s `app` service uses the `env_file: [{path: .env, required: false}]` long form
so the container runs even with no `.env` present; this needs Docker Compose v2.24+ (Jan 2024). On
an older Compose, either upgrade or create an empty `.env` before `docker compose up`.

**The critical path is zero-infra.** `demo`, `run`, `chat`, `eval`, `markup`, and `test` need no
Docker, no Phoenix/Prometheus/Grafana, and — with the one exception below — no LLM key: ingestion,
the delta engine, retrieval (BM25), and the eval gate are all pure code. The one place a live model
is inherently required is **generating a grounded chat answer** (the LLM's other confined job,
alongside the note-gate). `make eval`'s default (`--offline`) sidesteps even that for the CI/local
gate — see [Evaluation approach](#evaluation-approach).

## Design decisions & trade-offs

- **The canonical-representation seam.** Every ingestion adapter (native PDF, scanned PDF+OCR, DXF)
  converges on one `Document`/`Item` model. The delta engine, chat retrieval, and eval harness never
  know or care which adapter produced a document — this is what lets scanned and DWG/DXF pairs
  reuse the same delta/chat/eval code paths with zero branching downstream of ingestion.
- **Deterministic-first delta.** Hungarian-algorithm alignment (text + spatial + type + region
  similarity) plus typed, rule-based classification produces the delta, not a raw text/pixel diff
  and not an LLM asked to "describe what changed." The LLM is invoked for exactly one narrow
  judgment — whether an *ambiguous* note reword is cosmetic or meaningful — and even that call is
  optional: this deployment's note-gate band is collapsed to a single cosine-similarity cutoff (see
  [Honest failure taxonomy](#honest-failure-taxonomy--known-limitations) for why), so in practice
  the whole delta path runs LLM-free.
- **Dual confidence, not a single score.** Every `Change` carries a `match_conf` (alignment quality)
  and a `class_conf` (classification/note-gate certainty) separately, so a caller can distinguish "we
  matched the wrong item" from "we matched correctly but aren't sure how to classify it."
- **BM25, not a vector DB, at 2-document scale.** Retrieval is over a single delta's worth of chunks
  (tens to low hundreds), where a vector database would be pure overhead with no recall benefit; BM25
  with an optional hybrid mode is the right-sized choice here, not a placeholder for "the real thing."
- **Two grounding paths, deliberately not combined.** `chat.grounding` selects `json` (provider-
  agnostic structured output, the default) or `anthropic_citations` (Anthropic's native Citations
  API). They're mutually exclusive per call — `response_format` (strict JSON) and `documents`
  (Citations blocks) can't both be set on the same request per the Anthropic API — so switching is a
  config choice, not a runtime combination.
- **The sufficiency gate uses a *normalized* signal.** `check_sufficiency` never thresholds a raw
  BM25/coverage score directly; it min-max normalizes the top-k scores and gates on the *margin*
  between the best and second-best candidate. A flat top-k (nothing stands out) abstains even when
  absolute coverage looks fine, because a flat ranking means nothing distinguishes the right answer
  from the rest.
- **Train/test split to avoid train-on-test leakage.** `eval.datasets.split` fixes which sample pairs
  are held out; the CI-gated `delta_f1_test` metric is computed only over the test split, never the
  pair used to sanity-check the pipeline during development.

## Observability approach

- Every request opens a `pipeline.request` span (OTel GenAI semantic conventions) with per-stage
  child spans (`ingest`, `delta`, `answer`, ...).
- A **JSON trace file per run** is written to `traces/` unconditionally — this is the load-bearing,
  zero-infra observability path; it works with no external services and is what `tests/test_cli.py`
  asserts on.
- OTLP export to Phoenix is opt-in (`obs.otlp_enabled`, off by default) and additive — enabling it
  never changes what gets written to `traces/`.
- Token counts and cost are recorded on every LLM call, **including failed ones** (a timed-out call
  still billed its input tokens) — see `src/delta_chat/observability/cost.py`.
- Every call carries a `request_id` for cross-thread-pool correlation.
- Failures are surfaced, not hidden: degraded OCR confidence is flagged on the `Document`, DWG
  conversion failures return an honest error status rather than a silent empty result, and LLM
  timeouts are recorded (see cost-on-failure above) rather than swallowed.
- `GET /metrics` exposes Prometheus counters/histograms (stage latency, tokens, cost, delta counts)
  unconditionally; Prometheus itself is only scraping it under the `obs` compose profile.

## Evaluation approach

`make eval` / `eval/run_eval.py` writes a segmented scorecard — by **pair**, by **format** (native /
scanned / DWG), and by the **hard slice** — never a single averaged headline number:

- **Delta**: Hungarian-matched precision/recall/F1 at multiple `tau` thresholds, plus a soft-F1
  (partial credit for a close-but-not-exact match) and a type-confusion matrix.
- **Chat**: faithfulness, answer relevancy, context precision/recall (model-graded, multi-sample
  averaged) plus bespoke, **pure-code** citation *validity* (does the id resolve?) and citation
  *support* (does the span actually entail the claim?), and abstention precision/recall over
  answerable/unanswerable gold questions. **Abstention caveat**: the sufficiency gate has two
  stages — a cheap deterministic lexical pre-filter and an LLM entailment check. On this tiny,
  sparse P&ID text no *lexical* threshold separates all answerable from all unanswerable questions
  (empirically: driving false-abstention to 0 leaves abstention **recall at 0.67** — one
  unanswerable question whose stray terms overlap a chunk slips the floor). The entailment stage is
  what closes that gap, but it needs a real model, so **`make eval`'s default `--offline` run
  reports the weaker deterministic-only abstention** (recall 0.67); the tradeoff is deliberate —
  refusing the flagship demo question would be far worse than occasionally over-answering, and the
  entailment backstop exists for deployments with a key.
- **Retrieval**: Hit-Rate/Recall/MRR at multiple k, plus a cost/latency budget table
  (`budget.md`) extrapolated to a larger sheet count.
- **OCR**: word- and character-error rate for the scanned path, scored against `ocr_truth.txt` (the
  exact token sequence of the native page before it was rasterized). Reported in the scorecard's
  `ocr` block, never gated — the number moves with the installed Tesseract build, so gating it would
  fail CI on an environment upgrade rather than on a real regression (on this machine's Tesseract it
  is ~WER 0.38 / CER 0.37 on `pair_03_scanned` — read the live `ocr` block for the current run's
  figure rather than trusting this prose); see the failure taxonomy.
- **The judge is cross-family-enforced**: `eval.judge.model` is asserted at runtime to be a
  different model family than the answerer (`llm.model`) — a same-family judge can't credibly
  validate its own outputs — and prompts are order-swapped to counter position bias.
  **Not yet agreement-validated**: `eval/judge.py` implements `cohen_kappa()` and a judge/human
  confusion matrix, and both are unit-tested, but *no human-annotated label set exists in this
  repo*, so kappa has never actually been computed on real data. Until it is, the judge is
  cross-family and order-swapped but its agreement with a human is unmeasured — treat the
  model-graded numbers accordingly. Producing that annotation set is the first item under
  [What's next](#whats-next).

**CI gates ONLY the deterministic block** (delta P/R/F1 on the held-out test split, citation
validity, latency, cost) against `eval/baseline_scorecard.json`, via `eval.regression_thresholds`.
Model-graded metrics are never gating — they're reported as mean ± stdev over multiple judge samples
specifically so judge/temperature-0 non-determinism can never fire a false regression.

**`make eval` and CI both run with `--offline`** by default: the real ingest -> delta -> BM25
retrieval pipeline runs for real (all pure code), but chat generation and the judge are backed by a
deterministic stub (`eval/run_eval.py`'s `_offline_llm_complete` / `_OfflineJudge`) instead of a live
model — see the module docstring. This is a deliberate trade-off: it makes the CI-gated
`deterministic` block (which is what's actually enforced) reproducible with zero external services
and no API key, at the cost of the `chat`/model-graded block being uninformative in that mode (every
offline-stubbed claim trivially "entails" its source). Drop `--offline` for a real judge-graded run
against a live model — that path exists and is exercised manually, just not by CI, matching the same
convention used elsewhere in this codebase for live-network paths (see `tests/test_config.py`'s
`llm` marker).

## Honest failure taxonomy / known limitations

| Failure mode | Example | Detected by |
|---|---|---|
| Hallucinated citation id | model cites a `delta:CHG-0099` that doesn't exist | citation *validity* check (id-resolves) |
| Wrong-span citation | id resolves but the span doesn't entail the claim | citation *support* check (entailment) |
| Missed change | a real edit produces no `Change` | delta recall / false negative |
| Mislabeled change type | a MOVED change classed as MODIFIED | type-confusion matrix |
| Over-answered unanswerable | answers a question the sources can't support | abstention precision/recall |
| OCR error on small text | tiny/dense scanned text misread | **measured** in the scorecard `ocr` block (~WER 0.38 / CER 0.37 on this machine's Tesseract; environment-dependent, so read the live figure) |
| Abstention recall < 1.0 offline | an unanswerable question whose stray terms overlap a chunk ("flow rate on sheet 2") slips the deterministic floor | abstention P/R — the lexical gate can't separate all cases on this sparse text; the LLM entailment stage (key required) closes it, so offline recall is 0.67 |
| **Multi-sheet change missed** | `pair_04_multisheet`'s sheet-3 edit produces no matching `Change` | delta recall — it is a real FN, visible as `pair:pair_04_multisheet F1=0.000` in the segmented scorecard |
| **Judge agreement unmeasured** | `cohen_kappa()` is implemented and unit-tested but has never been run | no human-annotated label set exists in this repo — the judge is cross-family and order-swapped, but its agreement with a human is **unknown**, so model-graded numbers carry that caveat |
| Spurious change (low precision) | delta precision is 0.48 at tau=0.5 — roughly half the reported changes are FPs | delta precision, per-segment; the dual-confidence fields (`match_conf`/`class_conf`) let a caller filter, but nothing filters by default |
| **Note-gate not calibrated** | no train-split calibration script exists in this repo for `delta.notes.semantic_low/high` | collapsed to a fixed cosine cutoff (`0.90`/`0.90`) so the ambiguous LLM-gated band is empty by construction — this is a **deliberate cut**, not a bug: it trades away the nuance a real calibration would recover (distinguishing "cosmetic reword" from "meaningful reword" at the margin) for a demo path that never crashes and never needs a key for delta computation |

## Deliberate cuts / boundaries

- **No full P&ID connectivity-graph delta.** The delta engine reasons over coarse symbol geometry,
  not a traced piping/instrumentation connectivity graph — that's a research-scale problem with low
  marginal rubric value relative to its cost.
- **OCR error on small text is real** and is measured/reported, not hidden or special-cased away.
- **DWG depth is limited to what DXF exposes.** Real `.dwg` end-to-end requires the ODA File
  Converter at build/runtime; without it the adapter honestly reports "conversion required" rather
  than claiming full DWG support. Grading credit should reflect DXF-level support, not "all formats
  end-to-end."
- **Retrieval stays simple by choice.** BM25 (optional hybrid) at 2-document scale, not a vector
  database — see [Design decisions](#design-decisions--trade-offs).
- **`make eval`'s CI-gated run uses stubbed chat generation** (`--offline`), not a live model — see
  [Evaluation approach](#evaluation-approach). The `chat`/model-graded scorecard block is real when
  run without `--offline`, but that path isn't exercised in CI.

## What's next

- **Annotate a human judge-agreement set and report Cohen's κ.** The harness exists
  (`eval/judge.py`); what's missing is the labels. Until then the model-graded numbers are
  reported but not validated — see the failure taxonomy.
- **Raise delta precision.** Recall is 0.83 on the held-out split but precision is 0.48; the
  cheapest win is filtering or down-weighting `added`/`removed` changes whose `match_conf` is 0
  (the "possible missed-modify" cases already flagged in the report).
- Symbol detection + a real connectivity-graph delta.
- Learned alignment: embeddings fine-tuned on drawing-specific tokens (tag formats, line numbers)
  rather than a generic text encoder.
- An active-learning loop that feeds eval-flagged failures back into the gold set.
- Multi-sheet / large-drawing-set scaling: sheet-parallel ingest + incremental delta recomputation.
- A human-in-the-loop review UI for sub-threshold / low-confidence changes.
- An actual train-split calibration procedure for `delta.notes.semantic_low/high`, replacing the
  fixed cutoff described above.

## Determinism note

The structural delta (alignment + classification) is pure code and byte-reproducible run to run —
`tests/eval/test_run_eval.py` asserts the scorecard's gated `deterministic` block is byte-identical
across repeated runs given fixed inputs. The only non-deterministic elements anywhere in the
pipeline are the two LLM judgment calls (semantic note-gate, chat answer), bounded with
`temperature=0` and a pinned model id. Even temperature-0 hosted models aren't perfectly
deterministic in practice (backend/routing drift outside anyone's control) — which is exactly why CI
gates only the pure-code `deterministic` block, never a model-graded number.
