"""GRPO-family RL training facade over `agent_env.py`'s episode machinery.

This module provides a Gymnasium-style facade so the episode machinery in
`agent_env.py` can be driven by a step-based RL trainer -- a GRPO-family
GROUP ROLLOUT loop (sample K prompts, G completions each, score, compute
group-relative advantage) -- instead of only by a frozen/prompted policy.
`agent_env.py`'s `Policy.act` currently returns one already-decided
`PolicyAction` per call, which is right for a frozen/prompted policy but
too coarse for a trainer; this facade closes that gap for the coarse
(one-decided-action-per-call) case, i.e. it makes `agent_env.py`'s existing
turn granularity steppable by an external trainer loop: one full
`PolicyAction` per model generation, not a token-level facade. A true
token-level facade would additionally need a tokenizer bridge to
`confidence_schema()`, which remains unbuilt and is NOT attempted here (see
the design-scope note at the bottom of this docstring). The prompted
evaluation reported in the paper does not use any of this; it exists so a
future RL-trained arm could be added without a second implementation of the
scoring/grading logic.

REUSE -- deliberately NOT a second copy of the scoring loop:
  - `RLEnv.step()` below calls `agent_env._run_tool_call`
    and `agent_env._run_report` DIRECTLY, unmodified,
    on the SAME `EpisodeLog` instance those functions already know how to
    mutate (`agent_env.py`'s own `run_episode` does exactly this). Every byte of
    grading (`Grader.grade`), parsing
    (`confidence.parse_confidence_report`), and reward computation
    (`rewards.score_all_arms`) that `run_episode` would have run for a given
    turn is IDENTICAL here, because it is the identical function call, not a
    reimplementation of it.
  - `agent_env.EpisodeLog`, `agent_env.Turn`, `agent_env.ToolCallEvent`,
    `agent_env.ReportEvent`, `agent_env.AbstainEvent`, `agent_env.Grader`,
    `agent_env.SocrataQueryClient`, `agent_env.PolicyAction`,
    `agent_env.ToolCallRequest`, `agent_env.QuestionRecord` are imported and
    used as-is -- no shadow dataclasses.
  - `confidence.confidence_schema()` is reused (not re-declared) for
    `action_space_description()`'s `report` action's structured-output shape.

WHAT IS NOT REUSED, AND WHY (the one honest gap in the reuse story): the
outer per-turn dispatch loop -- "if action.kind == 'quit': break; if
action.kind == 'tool_call': ...; else: ..." -- is
REIMPLEMENTED here as an inverted, step-driven version of the same branch
structure, because `run_episode` is not step-able as written: it owns its
`for turn_index in range(max_turns)` loop internally and only returns once
the WHOLE episode ends, calling `policy.act()` itself rather than accepting
one externally-supplied action at a time. Making it literally step-able
without touching this fact requires one of:
  (1) editing `agent_env.py` to add a generator/coroutine-based variant of
      `run_episode` (out of scope for this module -- noted here as a
      possible follow-up, not implemented); or
  (2) driving the existing `run_episode` in a background thread with a
      `Policy.act()` that blocks on a queue for the next externally-supplied
      action (considered; rejected for this facade -- the added deadlock/
      timeout surface is not justified by the size of the ~30-line loop body
      being mirrored, which contains no scoring logic of its own).
This module takes neither path and instead reimplements ONLY the dispatch
(no grading, no parsing, no reward math lives in this file) --
`test_rl_env.py::test_rl_env_step_loop_matches_run_episode_exactly` is a
PARITY TEST that runs the identical scripted actions through both
`run_episode` and this facade and asserts field-for-field identical
`EpisodeLog.to_dict()` output, guarding against the two implementations
ever silently diverging.

DESIGN SCOPE: wrappable as a coarse (one-`PolicyAction`-per-step) facade
with NO changes to `agent_env.py`; a true token-level facade for
structured-output-constrained decoding remains unbuilt, as noted above.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # sibling-import convention, see test_pipeline.py:31

import agent_env as ae  # noqa: E402
from confidence import confidence_schema  # noqa: E402
from rewards import DEFAULT_EPS, RewardArm  # noqa: E402

# ---------------------------------------------------------------------------
# Reward-arm selection -- construction parameter (the design requirement that
# the three arms be identical except the scoring formula). Every `RLEnv.step()` call
# computes ALL three arms every turn (`_run_report` calls
# `rewards.score_all_arms` internally, unmodified) -- `reward_arm` only
# selects which single key of that already-identical computation becomes the
# canonical `StepResult.reward` a trainer optimizes against. This is what
# makes "LEVEL/INCREMENT/CONTROL runs differ ONLY in that parameter"
# structurally guaranteed rather than merely intended: the two runs share
# 100% of the reward *computation* and differ only in which dict key is read.
# ---------------------------------------------------------------------------
_VALID_ARMS: tuple[str, ...] = tuple(a.value for a in RewardArm)


@dataclass(frozen=True, slots=True)
class Observation:
    """The observation a policy/trainer actually sees, matching
    `agent_env.Policy.act(question, history)`'s signature exactly
    (agent_env.py:257) -- this facade does not invent a different
    observation shape, it exposes the one the existing `Policy` protocol
    already defines. `history` is a SNAPSHOT (`list(...)` copy) taken at
    the moment this `Observation` is constructed, not a live reference into
    `RLEnv`'s internal `EpisodeLog.turns` -- so a trainer that stashes an
    `Observation` across steps does not see it silently grow."""

    question: ae.QuestionRecord
    history: list[ae.Turn]


@dataclass(frozen=True, slots=True)
class StepResult:
    """One `RLEnv.step()` outcome.

    `reward` is the scalar for the CONSTRUCTION-TIME `reward_arm`. `all_arm_rewards`
    carries the full `{"level", "increment", "control"}` dict computed for
    this turn regardless of which arm is canonical -- this is what lets a
    single rollout be re-scored under every arm after the fact (the design
    requirement that a single rollout be scored both ways for analysis,
    generalized here to three arms via `rewards.score_all_arms`), exactly as
    `agent_env.EpisodeLog` already does for a non-stepped rollout.

    `terminated` is True only for an agent-chosen `quit` (Gymnasium
    convention: the episode reached an actual terminal state).
    `truncated` is True only when `max_turns` was exhausted (a time-limit
    cutoff, not an agent decision) -- `run_episode`'s `final_action ==
    "max_turns"` path (agent_env.py:542-543). Exactly one of the two is True
    at episode end; both are False mid-episode.
    """

    observation: Observation
    reward: float
    all_arm_rewards: dict[str, float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RLEnv:
    """Reset/step facade over one `agent_env.QuestionRecord` episode.

    Construction parameters are deliberately split into "identical across
    every arm" (socrata client, grader, max_turns, eps -- everything
    `run_episode` itself takes) and the ONE parameter that varies the
    experiment: `reward_arm`. LEVEL/INCREMENT/CONTROL `RLEnv` instances built
    from the same other arguments are byte-identical in every code path
    except which key of `_run_report`'s already-computed `score_all_arms`
    dict is surfaced as `StepResult.reward` (see module docstring).
    """

    def __init__(
        self,
        socrata: ae.SocrataQueryClient,
        grader: ae.Grader,
        *,
        reward_arm: str = "level",
        max_turns: int = 12,
        eps: float = DEFAULT_EPS,
        clock: Any = time.time,
        episode_id_prefix: str = "rl",
    ) -> None:
        if reward_arm not in _VALID_ARMS:
            raise ValueError(f"reward_arm must be one of {_VALID_ARMS}, got {reward_arm!r}")
        self._socrata = socrata
        self._grader = grader
        self._reward_arm = reward_arm
        self._max_turns = max_turns
        self._eps = eps
        self._clock = clock
        self._episode_id_prefix = episode_id_prefix

        self._question: ae.QuestionRecord | None = None
        self._log: ae.EpisodeLog | None = None
        self._prev_report: Any = None  # rewards.TurnReport | None
        self._turn_index = 0
        self._finished = False

    @property
    def reward_arm(self) -> str:
        return self._reward_arm

    @property
    def episode_log(self) -> ae.EpisodeLog:
        """The live (possibly mid-episode) `EpisodeLog` -- the SAME object
        `_run_tool_call`/`_run_report` mutate, so `.to_dict()` /
        `.total_reward()` are always consistent with every `StepResult`
        already returned. Raises if `reset()` was never called."""
        if self._log is None:
            raise RuntimeError("RLEnv.episode_log: reset() has not been called yet")
        return self._log

    def reset(self, question: ae.QuestionRecord, *, episode_id: str | None = None) -> Observation:
        """Starts a fresh episode over `question`. Mirrors `run_episode`'s
        own `EpisodeLog` construction (agent_env.py:502-507) field-for-field."""
        self._question = question
        self._episode_id = (
            episode_id or f"{self._episode_id_prefix}-{question.id}-{uuid.uuid4().hex[:8]}"
        )
        self._log = ae.EpisodeLog(
            episode_id=self._episode_id,
            question_id=question.id,
            question_class=question.cls,
            started_at=self._clock(),
        )
        self._prev_report = None
        self._turn_index = 0
        self._finished = False
        return Observation(question=question, history=list(self._log.turns))

    def action_space_description(self) -> dict[str, Any]:
        """Static description of what a policy may return each turn.
        Deliberately NOT a `gymnasium.spaces.*` object (unlike a
        fixed-cardinality grid-action multi-agent RL environment, where
        `Discrete`/`Box` space builders are the natural fit): this
        env's action is either a structured tool call or a JSON report, not
        a fixed-dimension vector, so a Discrete/Box encoding would
        misrepresent it rather than describe it. The `report` sub-schema is
        `confidence.confidence_schema()` verbatim, reused not re-declared."""
        return {
            "kinds": list(get_args(ae.PolicyActionKind)),  # ("tool_call", "report", "quit")
            "tool_call": {
                "kinds": ["query", "describe_dataset"],
                "fields": {
                    "dataset_id": "str, required",
                    "soql": "str, required iff kind=='query', else null",
                },
            },
            "report": {"schema": confidence_schema()},
            "quit": {"fields": {}},
        }

    def step(self, action: ae.PolicyAction) -> StepResult:
        """One turn. Reuses `agent_env._run_tool_call`/`_run_report`
        verbatim (see module docstring) -- this method's own logic is
        limited to: which of those two functions to call (or neither, for
        `quit`), turn-index bookkeeping, and packaging the result. No
        parsing, grading, or reward arithmetic happens in this file.
        """
        if self._log is None or self._question is None:
            raise RuntimeError("RLEnv.step: call reset() before step()")
        if self._finished:
            raise RuntimeError("RLEnv.step: episode already finished; call reset() again")

        ts = self._clock()
        log = self._log

        if action.kind == "quit":
            # Mirrors agent_env.py:514-516 exactly: quit never calls
            # _run_report, so no reward is EVER computed for a quit turn --
            # rewards.score_all_arms is not invoked. The zero dict below is
            # not a fabricated score; it documents "this turn was never
            # scored by any arm," matching run_episode's own behavior.
            log.final_action = "quit"
            log.ended_at = self._clock()
            self._finished = True
            zero = {a: 0.0 for a in _VALID_ARMS}
            return StepResult(
                observation=Observation(question=self._question, history=list(log.turns)),
                reward=zero[self._reward_arm],
                all_arm_rewards=zero,
                terminated=True,
                truncated=False,
                info={"turn": None, "final_action": "quit"},
            )

        if action.kind == "tool_call":
            event = ae._run_tool_call(
                self._socrata, action.tool_call, turn_index=self._turn_index, ts=ts
            )
            log.tool_calls.append(event)
            turn = ae.Turn(
                turn_index=self._turn_index, timestamp=ts, kind="tool_call", tool_call=event
            )
            log.turns.append(turn)
            # rewards.py never scores a tool call (score_all_arms is only
            # invoked from _run_report) -- zero here for the same reason as
            # the quit path: an accurate "not scored," not a guess.
            all_rewards = {a: 0.0 for a in _VALID_ARMS}
        else:  # action.kind == "report"
            report_event, report = ae._run_report(
                self._question,
                action.confidence_raw,
                self._prev_report,
                log,
                self._grader,
                turn_index=self._turn_index,
                ts=ts,
                eps=self._eps,
            )
            log.reports.append(report_event)
            turn = ae.Turn(
                turn_index=self._turn_index, timestamp=ts, kind="report", report=report_event
            )
            log.turns.append(turn)
            self._prev_report = report
            all_rewards = dict(report_event.rewards)

        self._turn_index += 1
        truncated = self._turn_index >= self._max_turns
        if truncated:
            log.final_action = "max_turns"
            log.ended_at = self._clock()
            self._finished = True

        return StepResult(
            observation=Observation(question=self._question, history=list(log.turns)),
            reward=all_rewards[self._reward_arm],
            all_arm_rewards=all_rewards,
            terminated=False,
            truncated=truncated,
            info={"turn": turn},
        )


def run_full_episode(
    env: RLEnv,
    question: ae.QuestionRecord,
    policy_fn: Any,
    *,
    episode_id: str | None = None,
) -> list[StepResult]:
    """Convenience driver: `reset()` then `step()` until terminated/truncated,
    calling `policy_fn(question, history) -> PolicyAction` each turn -- the
    identical signature as `agent_env.Policy.act`, so any existing `Policy`
    (including `agent_env.ScriptedPolicy`) can be used unmodified via
    `policy_fn=some_policy.act`. Returns every `StepResult` in order. Used by
    `test_rl_env.py`'s parity test and mock-trainer rollout collector, kept
    here (not duplicated in the test file) so both call the same driver."""
    obs = env.reset(question, episode_id=episode_id)
    results: list[StepResult] = []
    while True:
        action = policy_fn(obs.question, obs.history)
        result = env.step(action)
        results.append(result)
        obs = result.observation
        if result.terminated or result.truncated:
            return results


# ---------------------------------------------------------------------------
# Class-balanced, seeded sampler with a held-out split.
# Seeded randomness only: a per-instance
# `np.random.default_rng(seed)`, never a global RNG (`random.seed` /
# `np.random.seed` are never called here).
# ---------------------------------------------------------------------------


@dataclass
class QuestionSampler:
    """Splits `questions` into a held-out set (per-class, seeded) and
    samples class-balanced batches from either split.

    The held-out split is computed ONCE at construction, per class, via a
    seeded `rng.permutation` -- so `held_out` is a fixed set for the
    lifetime of one `QuestionSampler` instance (never resampled), which is
    what makes it usable as an actual held-out evaluation set rather than a
    moving target. `sample_batch` draws WITH replacement (seeded,
    `rng.integers`) round-robin across classes present in the requested
    split, so `n` may exceed the split's size (needed for GRPO's G-way
    group repeats of the same small pilot question set) while staying
    class-balanced to within one item.
    """

    questions: list[ae.QuestionRecord]
    seed: int
    held_out_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("QuestionSampler requires at least one question")
        if not (0.0 <= self.held_out_fraction < 1.0):
            raise ValueError(f"held_out_fraction must be in [0, 1), got {self.held_out_fraction}")
        self._rng = np.random.default_rng(self.seed)

        by_cls: dict[str, list[ae.QuestionRecord]] = {}
        for q in self.questions:
            by_cls.setdefault(q.cls, []).append(q)
        self._classes = sorted(by_cls)

        train: list[ae.QuestionRecord] = []
        held_out: list[ae.QuestionRecord] = []
        for cls in self._classes:
            items = by_cls[cls]
            perm = self._rng.permutation(len(items))
            n_held = int(round(len(items) * self.held_out_fraction))
            held_idx = set(perm[:n_held].tolist())
            for i, item in enumerate(items):
                (held_out if i in held_idx else train).append(item)

        self._train = train
        self._held_out = held_out
        self._pool: dict[str, dict[str, list[ae.QuestionRecord]]] = {
            "train": {c: [q for q in train if q.cls == c] for c in self._classes},
            "held_out": {c: [q for q in held_out if q.cls == c] for c in self._classes},
        }

    @property
    def train(self) -> list[ae.QuestionRecord]:
        return list(self._train)

    @property
    def held_out(self) -> list[ae.QuestionRecord]:
        return list(self._held_out)

    def sample_batch(
        self, n: int, *, split: Literal["train", "held_out"] = "train"
    ) -> list[ae.QuestionRecord]:
        """Class-balanced (round-robin over non-empty classes present in
        `split`), seeded WITH-replacement sample of size `n`."""
        pool = self._pool[split]
        classes = [c for c in self._classes if pool[c]]
        if not classes:
            raise ValueError(f"QuestionSampler: no questions available in split={split!r}")
        out: list[ae.QuestionRecord] = []
        for i in range(n):
            cls = classes[i % len(classes)]
            items = pool[cls]
            idx = int(self._rng.integers(0, len(items)))
            out.append(items[idx])
        return out


# ---------------------------------------------------------------------------
# Unit tests (pytest-collectible; plain-assert, no `import pytest` needed at
# module scope) -- same embedded-test convention as confidence.py/rewards.py.
# `test_rl_env.py` also imports this module directly, so these are collected
# either way; kept here (a few structural smoke tests only) because they
# check THIS module's own facade contract, distinct from test_rl_env.py's
# GRPO mock-trainer proof, which is a separate concern.
#   uv run pytest scripts/rl_env.py -v
# ---------------------------------------------------------------------------


def _make_stub_env(*, reward_arm: str = "level") -> tuple[RLEnv, ae.QuestionRecord]:
    dataset_id = "6yyb-pb25"
    soql = "select count(*) as result where boroughname='Bronx'"
    q = ae.QuestionRecord(
        id="rl-env-test-q1",
        cls="answerable",
        question="How many segments in the Bronx?",
        dataset_id=dataset_id,
        soql=soql,
        gold="42",
    )
    socrata = ae.StubSocrataClient(rows_by_soql={(dataset_id, soql): [{"result": "42"}]})
    env = RLEnv(socrata, ae.StubGrader(), reward_arm=reward_arm)
    return env, q


def test_reset_returns_empty_history_observation():
    env, q = _make_stub_env()
    obs = env.reset(q)
    assert obs.question is q
    assert obs.history == []


def test_quit_is_terminal_and_unscored():
    env, q = _make_stub_env()
    env.reset(q)
    result = env.step(ae.PolicyAction(kind="quit"))
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.0
    assert all(v == 0.0 for v in result.all_arm_rewards.values())


def test_max_turns_sets_truncated_not_terminated():
    env, q = _make_stub_env()
    env.reset(q)
    env._max_turns = 1  # force truncation on the first report turn
    raw = ae._report("answer", "42", 0.6)
    result = env.step(ae.PolicyAction(kind="report", confidence_raw=raw))
    assert result.truncated is True
    assert result.terminated is False
    assert env.episode_log.final_action == "max_turns"


def test_reward_arm_selects_from_all_arm_rewards():
    env, q = _make_stub_env(reward_arm="increment")
    env.reset(q)
    raw = ae._report("answer", "42", 0.6)
    result = env.step(ae.PolicyAction(kind="report", confidence_raw=raw))
    assert result.reward == result.all_arm_rewards["increment"]
    assert set(result.all_arm_rewards) == {"level", "increment", "control"}


def test_step_before_reset_raises():
    env = RLEnv(ae.StubSocrataClient(), ae.StubGrader())
    try:
        env.step(ae.PolicyAction(kind="quit"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_step_after_terminated_raises():
    env, q = _make_stub_env()
    env.reset(q)
    env.step(ae.PolicyAction(kind="quit"))
    try:
        env.step(ae.PolicyAction(kind="quit"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_question_sampler_is_class_balanced_and_seeded():
    questions = (
        [ae.QuestionRecord(id=f"a{i}", cls="answerable", question="q") for i in range(6)]
        + [ae.QuestionRecord(id=f"u{i}", cls="unreliable", question="q") for i in range(6)]
        + [ae.QuestionRecord(id=f"n{i}", cls="unanswerable", question="q") for i in range(6)]
    )
    s1 = QuestionSampler(questions, seed=7, held_out_fraction=0.2)
    s2 = QuestionSampler(questions, seed=7, held_out_fraction=0.2)
    b1 = [q.id for q in s1.sample_batch(9, split="train")]
    b2 = [q.id for q in s2.sample_batch(9, split="train")]
    assert b1 == b2  # seeded determinism
    counts = {"answerable": 0, "unreliable": 0, "unanswerable": 0}
    for qid in b1:
        for cls, prefix in (("answerable", "a"), ("unreliable", "u"), ("unanswerable", "n")):
            if qid.startswith(prefix):
                counts[cls] += 1
    assert max(counts.values()) - min(counts.values()) <= 1  # class-balanced to within 1
    held_out_ids = {q.id for q in s1.held_out}
    train_ids = {q.id for q in s1.train}
    assert held_out_ids.isdisjoint(train_ids)  # held-out split is a genuine partition
    assert held_out_ids  # non-empty at held_out_fraction=0.2 over 6 items/class


if __name__ == "__main__":
    import inspect

    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
                failures.append(name)
                print(f"FAIL {name}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} failing: {failures}")
    print(f"{sum(1 for n in globals() if n.startswith('test_'))} tests passed")
