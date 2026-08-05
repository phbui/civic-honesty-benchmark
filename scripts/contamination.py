"""contamination.py -- no-tool ablation for the civic-honesty benchmark.

Contamination is a real but tractable threat: this module's job is to
"run a no-tool ablation on every answerable and
unreliable item." NYC Open Data has no licensing blocker on reuse (Local Law 11
of 2012, NYC Admin Code Sec 23-502(d)), and LiveBench (arXiv:2406.19314) already
uses Socrata as a contamination-limiting live source -- the structural
precedent this design follows.

WHAT THIS MEASURES: for every ANSWERABLE and UNRELIABLE question, query the
model WITHOUT tool access and check whether it independently reproduces the
ground-truth answer from parametric memory alone. High no-tool accuracy on an
item means an eval that claims to test "retrieval honesty" on that item is
actually testing memorization -- the item's honesty signal is void and should
be flagged for downstream filtering.

WHY UNANSWERABLE IS EXCLUDED (deliberate, not an oversight): there is no
correct VALUE for an unanswerable question by construction, so "the no-tool
model produced the correct answer" is not a coherent event for that class --
the only correct no-tool response would be an abstention/refusal driven purely
by an unreliable, out-of-scope kind of self-knowledge ("I don't have access to
that field") rather than by the retrieval-contamination mechanism this ablation
is built to detect. Running it there would silently redefine "contamination"
mid-benchmark; skip it explicitly instead. See `metrics.py`'s own
`_UNANSWERABLE_CALIBRATION_NOTE` for the parallel reasoning on the calibration
side.

INTERFACE ONLY FOR THE MODEL CALL. By design, this module
makes NO paid LLM API calls anywhere. `NoToolQueryFn` is the seam a caller
wires a real model into later (outside this module, outside this repo's
test/CI path); the default is `NeverCallQueryFn`, which raises if actually
invoked, so a caller cannot accidentally trigger a live call just by forgetting
to pass `query_fn`. Everything downstream of that seam -- scoring, the
per-item contamination flag, per-class contamination rate, and the
downstream-analysis filter -- is implemented and tested now against a
synthetic/fake query function.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

QUESTION_CLASSES_FOR_CONTAMINATION: tuple[str, ...] = ("answerable", "unreliable")
# "unanswerable" is deliberately excluded -- see module docstring.

_MISSING = object()


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Flexible attribute/dict-key accessor. Deliberately duplicated from
    metrics.py's identical helper rather than imported, so this module has
    zero import-time dependency on metrics.py or on any other sibling
    file (socrata.py / questions.py / groundtruth.py / agent_env.py /
    confidence.py / rewards.py) -- it can run standalone against a bare list
    of question dicts."""
    for name in names:
        v = obj.get(name, _MISSING) if isinstance(obj, Mapping) else getattr(obj, name, _MISSING)
        if v is not _MISSING:
            return v
    return default


class NoToolQueryFn(Protocol):
    """Seam for the real model call. A real implementation queries the model
    with tool access explicitly disabled/withheld and returns its raw text
    answer. MUST NOT be a paid API call from within this repo's test/CI path."""

    # Positional-only: a callback seam must not constrain the caller's own
    # parameter names (a bare `lambda q: ...` is a legitimate implementation).
    def __call__(self, question: Any, /) -> str: ...


class NoToolScoreFn(Protocol):
    def __call__(self, no_tool_answer: str | None, gold_answer: Any, question: Any, /) -> bool: ...


class NeverCallQueryFn:
    """Default `NoToolQueryFn` -- raises if actually invoked. Ensures
    `run_no_tool_ablation` cannot silently make a live model/API call just
    because a caller forgot to pass `query_fn`."""

    def __call__(self, question: Any) -> str:
        raise NotImplementedError(
            "no NoToolQueryFn was wired in -- this module defines the contamination "
            "ablation interface and implements its scoring/reporting half, but "
            "deliberately makes no model or API calls itself. Pass a real `query_fn` "
            "to run_no_tool_ablation(); any paid-API implementation of it belongs "
            "outside this repo's test/CI path."
        )


def default_no_tool_score(
    no_tool_answer: str | None, gold_answer: Any, question: Any = None
) -> bool:
    """Default execution-match-style scorer: numeric tolerance match, else
    case-insensitive substring match of the stringified gold answer inside the
    model's raw text. Deliberately permissive -- a contamination check should
    err toward FLAGGING an item as contaminated (false positive: an
    uncontaminated item is needlessly dropped from later analysis) rather than
    missing one (false negative: a contaminated item's void honesty signal
    silently pollutes the main results). `question` is accepted but unused by
    the default scorer; it exists so a real scorer can consult item-specific
    tolerance/units if needed."""
    if no_tool_answer is None or gold_answer is None:
        return False
    text = str(no_tool_answer).strip()
    try:
        return math.isclose(float(text), float(gold_answer), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        pass
    return str(gold_answer).strip().lower() in text.lower()


@dataclass
class ContaminationItemResult:
    question_id: Any
    question_class: str
    no_tool_answer: str | None
    no_tool_correct: bool
    contaminated: bool  # == no_tool_correct; kept as a separate field for readability


@dataclass
class ContaminationClassReport:
    question_class: str
    n_items: int
    n_contaminated: int
    contamination_rate: float


@dataclass
class ContaminationReport:
    per_item: list[ContaminationItemResult]
    per_class: dict[str, ContaminationClassReport]
    contaminated_ids: set[Any]

    def is_contaminated(self, question_id: Any) -> bool:
        return question_id in self.contaminated_ids


def run_no_tool_ablation(
    questions: Sequence[Any],
    *,
    query_fn: NoToolQueryFn | None = None,
    score_fn: NoToolScoreFn = default_no_tool_score,
) -> list[ContaminationItemResult]:
    """Runs the no-tool ablation over every ANSWERABLE/UNRELIABLE item in
    `questions` (skips unanswerable and any unrecognized class, warning rather
    than raising -- graceful degradation against questions.py's actual
    schema). Each question record needs an id, a class field, and a gold
    answer (see `_get`'s alias lists) -- identical accessor contract to
    `metrics.py::normalize_item`, duplicated rather than imported (see module
    docstring).

    `query_fn` defaults to `NeverCallQueryFn()` (raises on use) so this cannot
    accidentally make a live call."""
    if query_fn is None:
        query_fn = NeverCallQueryFn()
    results: list[ContaminationItemResult] = []
    for q in questions:
        # Aliases include both the documented contract names and the concrete
        # names used by questions.py::Question / groundtruth.py::GroundTruthRow
        # (`id`/`question_id`, `cls`, `gold_value`) -- see metrics.py::
        # normalize_item's matching comment.
        qid = _get(q, "question_id", "id", "qid", "item_id")
        qclass = _get(q, "question_class", "cls", "class_", "qclass", "category")
        gold = _get(q, "gold_answer", "gold_value", "gold", "ground_truth", "answer_gold")
        if qclass not in QUESTION_CLASSES_FOR_CONTAMINATION:
            continue
        if gold is None:
            warnings.warn(
                f"question {qid!r}: no gold answer available, skipping contamination check",
                stacklevel=2,
            )
            continue
        answer = query_fn(q)
        correct = bool(score_fn(answer, gold, q))
        results.append(ContaminationItemResult(qid, qclass, answer, correct, correct))
    return results


def summarize_contamination(results: Sequence[ContaminationItemResult]) -> ContaminationReport:
    """Per-class contamination rate + the set of flagged item ids, ready for
    `filter_non_contaminated` to consume."""
    per_class: dict[str, ContaminationClassReport] = {}
    for cls in QUESTION_CLASSES_FOR_CONTAMINATION:
        cls_items = [r for r in results if r.question_class == cls]
        n = len(cls_items)
        n_c = sum(1 for r in cls_items if r.contaminated)
        per_class[cls] = ContaminationClassReport(
            question_class=cls,
            n_items=n,
            n_contaminated=n_c,
            contamination_rate=(n_c / n if n else math.nan),
        )
    contaminated_ids = {r.question_id for r in results if r.contaminated}
    return ContaminationReport(
        per_item=list(results), per_class=per_class, contaminated_ids=contaminated_ids
    )


def flag_contaminated(results: Sequence[ContaminationItemResult]) -> set[Any]:
    """Per-item contamination flag as a bare set of question_ids, for a caller
    that wants the flag without the full report object."""
    return {r.question_id for r in results if r.contaminated}


def filter_non_contaminated(
    items: Sequence[Any],
    report: ContaminationReport,
    *,
    id_field_names: tuple[str, ...] = ("question_id", "id", "qid", "item_id"),
) -> list[Any]:
    """Downstream-analysis filter: drops any item (e.g. the tool-using
    trajectory/eval records `metrics.py` analyzes) whose question_id is
    flagged contaminated by the no-tool ablation. `items` here are NOT the
    ablation's own results -- they are whatever records a later analysis
    stage is filtering."""
    out = []
    for it in items:
        qid = _get(it, *id_field_names)
        if qid is not None and report.is_contaminated(qid):
            continue
        out.append(it)
    return out
