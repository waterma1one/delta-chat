"""LLM-as-judge (§10.5) with the anti-bias controls the eval-rigor points depend on:

- **Cross-family**: `eval.judge.model` must be a different model family than `llm.model` (the
  answerer) — a same-family judge can't independently validate its own outputs.
- **Order-swap**: pairwise comparisons run both (A,B) and (B,A) and average, to cancel position
  bias (some judges systematically favor whichever candidate is presented first).
- **Rubric-anchored**, 1..scale pointwise scoring; pairwise for version comparisons.
- **Validated**: `cohen_kappa` against a stored human-label fixture, reported alongside a
  confusion matrix. Single-annotator, small-n — a directional signal, not an unimpeachable number
  (see section-11's README for the honest caveat).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from delta_chat.chat import llm as _llm

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Score the given item on a 1-{scale} scale using the rubric "
    'provided. Respond with JSON only: {{"score": int, "rationale": str}}.'
)
_PAIRWISE_SYSTEM_PROMPT = (
    "You are an impartial evaluator comparing two candidate answers. Score each candidate "
    'independently on a 1-{scale} scale using the rubric provided. Respond with JSON only: '
    '{{"first_score": int, "second_score": int, "rationale": str}}.'
)


@dataclass
class JudgeVerdict:
    score: int
    rationale: str


def _family(model: str) -> str:
    m = model.lower()
    for fam in ("claude", "gpt", "gemini", "llama", "mistral", "cohere"):
        if fam in m:
            return fam
    return m


def assert_cross_family(cfg) -> None:
    """Raise if the configured judge model is the same family as the answerer model."""
    judge_model = str(cfg.get("eval.judge.model"))
    answerer_model = str(cfg.get("llm.model"))
    if _family(judge_model) == _family(answerer_model):
        raise ValueError(
            f"eval.judge.model ({judge_model!r}) must be a different model family than "
            f"llm.model ({answerer_model!r}) — a same-family judge can't validate its own outputs"
        )


def judge_pointwise(item: str, *, model: str, cfg=None, rubric: str = "", scale: int = 4) -> JudgeVerdict:
    """One rubric-anchored pointwise score in [1, scale]."""
    system = _JUDGE_SYSTEM_PROMPT.format(scale=scale)
    if rubric:
        system += f"\n\n{rubric}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": item}]
    result = _llm.complete(messages, model=model, response_format={"type": "json_object"}, cfg=cfg)
    data = json.loads(result.text)
    return JudgeVerdict(score=int(data["score"]), rationale=str(data.get("rationale", "")))


def _pairwise_call(first: str, second: str, *, model: str, cfg, rubric: str, scale: int) -> dict:
    system = _PAIRWISE_SYSTEM_PROMPT.format(scale=scale)
    if rubric:
        system += f"\n\n{rubric}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Candidate 1:\n{first}\n\nCandidate 2:\n{second}"},
    ]
    result = _llm.complete(messages, model=model, response_format={"type": "json_object"}, cfg=cfg)
    return json.loads(result.text)


def judge_pairwise(a: str, b: str, *, model: str, cfg=None, rubric: str = "", scale: int = 4) -> dict:
    """Run (a,b) and (b,a); average each candidate's two scores to control position bias; return
    {"winner": "a"|"b"|"tie", "scores": {"a": float, "b": float}}."""
    ab = _pairwise_call(a, b, model=model, cfg=cfg, rubric=rubric, scale=scale)
    ba = _pairwise_call(b, a, model=model, cfg=cfg, rubric=rubric, scale=scale)
    a_score = (float(ab["first_score"]) + float(ba["second_score"])) / 2.0
    b_score = (float(ab["second_score"]) + float(ba["first_score"])) / 2.0
    if a_score > b_score:
        winner = "a"
    elif b_score > a_score:
        winner = "b"
    else:
        winner = "tie"
    return {"winner": winner, "scores": {"a": a_score, "b": b_score}}


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Cohen's kappa between judge and human labels: (po - pe) / (1 - pe), where po is observed
    agreement and pe is chance agreement given each rater's label distribution."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    n = len(judge_labels)
    if n == 0:
        return 1.0
    po = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    categories = set(judge_labels) | set(human_labels)
    pe = sum((judge_labels.count(k) / n) * (human_labels.count(k) / n) for k in categories)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(judge_labels: list[int], human_labels: list[int]) -> dict[int, dict[int, int]]:
    """human_label -> {judge_label -> count}, for reporting alongside cohen_kappa."""
    matrix: dict[int, dict[int, int]] = {}
    for j, h in zip(judge_labels, human_labels):
        matrix.setdefault(h, {}).setdefault(j, 0)
        matrix[h][j] += 1
    return matrix
