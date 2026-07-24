# DEMO

A 2–4 minute walkthrough: one delta, one grounded chat exchange (answerable + abstention), and the
eval scorecard. All commands are copy-paste runnable from the repo root after `make install`.

## 0. Setup (one time)

```bash
make install                  # uv sync + seeds .env from .env.example if missing
# edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY for the chat steps below
```

## 1. One delta

```bash
make run
```

Ingests the golden sample pair (`data/samples/pair_01_native/`), computes the delta, and writes a
report — no LLM key needed for this step.

Expected shape of `data/output/<id>/report.json`:

```json
{
  "pid_a": "data/samples/pair_01_native/rev_a.pdf",
  "pid_b": "data/samples/pair_01_native/rev_b.pdf",
  "changes": [
    {
      "id": "CHG-0001",
      "type": "modified",
      "item_type": "note",
      "description": "Note reworded",
      "old": "...",
      "new": "...",
      "match_conf": 1.0,
      "class_conf": 0.9
    }
  ],
  "counts": {"added": 1, "removed": 0, "modified": 6, "moved": 0}
}
```

`report.md` renders the same changes as a human-readable table, and (best-effort)
`data/output/<id>/markup.pdf` overlays the changes on the drawing itself.

## 2. One grounded chat exchange — answerable + abstention

Requires a real LLM key (`.env`), since answer generation is one of the two places this system
calls a model.

```bash
uv run delta-chat chat --pid-a data/samples/pair_01_native/rev_a.pdf \
                        --pid-b data/samples/pair_01_native/rev_b.pdf
```

Then at the `>` prompt, try an answerable question first:

```
> what changed about the setpoint?
```

Expected shape:

```
[Grounded] The setpoint for FV-9040 changed from 150 psi to 175 psi.
  cite: delta:CHG-0003 (delta) page=0 sheet=1
```

Then an out-of-scope question, to see honest abstention:

```
> what is the weather in Houston?
```

Expected shape:

```
[Rejected] 
  refusal_reason: retrieved evidence has no clear best match (normalized score margin 0.00 < 0.15) — top candidates are too similar in relevance to trust
```

(Or, if retrieval finds *some* superficially related-but-wrong evidence, the sufficiency gate's
entailment check rejects it instead — either way, the system declines to answer rather than
fabricating one.)

`exit` or a blank line quits.

## 3. The scorecard

```bash
make eval
```

Zero-infra, no LLM key required (see README's [Evaluation
approach](README.md#evaluation-approach) for why). Writes `scorecard.json` (gitignored — it embeds
document text) and `budget.md`, and prints nothing if there's no regression against
`eval/baseline_scorecard.json`, or `REGRESSION: ...` lines if there is.

Expected shape of `scorecard.json`'s gated section:

```json
{
  "deterministic": {
    "delta_f1_test": 0.62,
    "citation_validity": 1.0,
    "latency_p95_ms": 850.0,
    "cost_usd_per_request": 0.0
  },
  "delta": {"test": {"by_segment": {"format:pdf_native": {...}, "format:pdf_scanned": {...}, "format:dwg": {...}}}},
  "chat": {"faithfulness": {"mean": 1.0, "stdev": 0.0}, "abstention": {"precision": 0.3, "recall": 1.0, "f1": 0.46}},
  "retrieval": {"...": "Hit-Rate/Recall/MRR at k=1,3,5,10"}
}
```

`budget.md` is a per-stage latency/token/cost table, extrapolated to a larger (500-sheet) run.
