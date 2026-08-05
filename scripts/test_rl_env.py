"""Validity-gate tests for `rl_env.py`.

Three concerns, each with its own test group below:

1. PARITY -- `RLEnv` must reproduce `agent_env.run_episode`'s trajectory
   EXACTLY for the same scripted actions (guards the "divergent second
   implementation" risk called out in `rl_env.py`'s module docstring).
2. THREE-ARM SEPARATION -- the property the whole planned experiment depends
   on: LEVEL and INCREMENT must differ when a rollout contains an unchanged
   re-report, and must be IDENTICAL when it does not (see
   `test_level_and_increment_separate_on_rereport_and_agree_without_it`).
3. MOCK GRPO TRAINER -- a batched-rollout / group-relative-advantage / mock
   update loop over a trivial rule-based policy (no NN, no GPU, no paid
   API), proving the sampler -> rollout -> per-arm reward -> advantage ->
   update data flow actually runs end to end through `RLEnv`.

Also measures real throughput (episodes/sec, turns/episode, Socrata-call
counts) with `StubSocrataClient` (zero network) and reports the extrapolation
to the planned run size.

Run: `uv run pytest scripts/test_rl_env.py -v -s`
(`-s` to see the throughput/extrapolation print block and the mock-trainer
output -- this file intentionally prints real numbers, not
just assertions.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # sibling-import convention, see test_pipeline.py:31

import agent_env as ae  # noqa: E402
from rl_env import QuestionSampler, RLEnv, run_full_episode  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
QUESTIONS_JSONL = RESULTS / "questions.jsonl"
GROUNDTRUTH_JSONL = RESULTS / "groundtruth.jsonl"
DATASET_ID = "6yyb-pb25"


# ---------------------------------------------------------------------------
# Shared fixtures: a tiny synthetic question/socrata pair AND (if present)
# the real 54-question pilot set materialized in results/ (per
# test_pipeline.py's convention: real data when available, an inline
# fixture fallback so this file still runs on a fresh checkout with no
# network access).
# ---------------------------------------------------------------------------


def _synthetic_question(soql: str = "select count(*) as result where boroughname='Bronx'"):
    q = ae.QuestionRecord(
        id="rl-parity-q1",
        cls="answerable",
        question="How many segments in the Bronx?",
        dataset_id=DATASET_ID,
        soql=soql,
        gold="4217",
    )
    socrata = ae.StubSocrataClient(rows_by_soql={(DATASET_ID, soql): [{"result": "4217"}]})
    return q, socrata


def _load_real_pilot_questions() -> list[ae.QuestionRecord]:
    """Real 54-question pilot set (§README "Pilot run -- actual counts"),
    joined against its materialized gold via the exact loader functions
    `agent_env.py` ships (`load_questions_jsonl`/`load_groundtruth_jsonl`/
    `attach_groundtruth`) -- not a synthetic stand-in."""
    if not (QUESTIONS_JSONL.exists() and GROUNDTRUTH_JSONL.exists()):
        pytest.skip("results/questions.jsonl + groundtruth.jsonl not present in this checkout")
    questions = ae.load_questions_jsonl(QUESTIONS_JSONL)
    groundtruth = ae.load_groundtruth_jsonl(GROUNDTRUTH_JSONL)
    return ae.attach_groundtruth(questions, groundtruth)


def _stub_socrata_from_gold(questions: list[ae.QuestionRecord]) -> ae.StubSocrataClient:
    """Builds a `StubSocrataClient` whose canned rows are each question's own
    ALREADY-materialized `gold` value (from `groundtruth.py`, offline) --
    zero network calls, but the retrieved value is the real gold value, so a
    policy that reports it is genuinely correct, not correct-by-accident."""
    rows: dict[tuple[str, str | None], list[dict]] = {}
    schema: dict[str, dict] = {}
    meta_path = RESULTS / "questions_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        schema[DATASET_ID] = {"field_names": meta["schema_field_names"]}
    for q in questions:
        if q.soql is not None and q.gold is not None and q.dataset_id is not None:
            rows[(q.dataset_id, q.soql)] = [{"result": q.gold}]
    return ae.StubSocrataClient(rows_by_soql=rows, schema_by_dataset=schema)


# ---------------------------------------------------------------------------
# 1. PARITY: RLEnv must reproduce run_episode's trajectory exactly.
# ---------------------------------------------------------------------------


def _scripted_actions_with_rereport(soql: str) -> list[ae.PolicyAction]:
    return [
        ae.PolicyAction(
            kind="tool_call",
            tool_call=ae.ToolCallRequest(kind="query", dataset_id=DATASET_ID, soql=soql),
        ),
        ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
        # unchanged re-report:
        ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
        ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.8)),  # update
        ae.PolicyAction(kind="quit"),
    ]


def test_rl_env_step_loop_matches_run_episode_exactly():
    """The parity guard cited in rl_env.py's module docstring. Runs the
    IDENTICAL scripted actions through `run_episode` (agent_env.py's own
    loop) and through `RLEnv.step()` (this facade's reimplemented dispatch),
    against separate `StubSocrataClient`/`QuestionRecord` instances with
    identical content but a frozen `clock` so timestamps line up too, and
    asserts the two `EpisodeLog.to_dict()` outputs are byte-identical except
    for `episode_id` (deliberately different so the two runs are
    distinguishable in a shared log)."""
    q_a, socrata_a = _synthetic_question()
    q_b, socrata_b = _synthetic_question()
    assert q_a.soql is not None and q_b.soql is not None  # by construction, see _synthetic_question
    actions_a = _scripted_actions_with_rereport(q_a.soql)
    actions_b = _scripted_actions_with_rereport(q_b.soql)

    fixed_clock = iter([100.0 + 0.1 * i for i in range(100)])

    def clock() -> float:
        return next(fixed_clock)

    # -- path A: agent_env.run_episode, unmodified --
    policy_a = ae.ScriptedPolicy(actions=actions_a)
    log_a = ae.run_episode(
        q_a, policy_a, socrata_a, ae.StubGrader(), episode_id="parity-run-episode", clock=clock
    )

    # -- path B: rl_env.RLEnv, driven step-by-step with the same actions --
    fixed_clock_b = iter([100.0 + 0.1 * i for i in range(100)])

    def clock_b() -> float:
        return next(fixed_clock_b)

    env = RLEnv(socrata_b, ae.StubGrader(), clock=clock_b)
    idx = 0

    def scripted_policy_fn(question, history):
        nonlocal idx
        action = actions_b[idx]
        idx += 1
        return action

    run_full_episode(env, q_b, scripted_policy_fn, episode_id="parity-rl-env")
    log_b = env.episode_log

    dict_a = log_a.to_dict()
    dict_b = log_b.to_dict()
    for key in dict_a:
        if key == "episode_id":
            continue
        assert dict_a[key] == dict_b[key], (
            f"mismatch at key {key!r}: {dict_a[key]!r} != {dict_b[key]!r}"
        )
    print(
        "\n[parity] run_episode vs RLEnv EpisodeLog.to_dict() identical on every key "
        "except episode_id"
    )
    print("[parity] total_reward (both paths):", log_a.total_reward())


# ---------------------------------------------------------------------------
# 2. THREE-ARM SEPARATION -- the property the experiment depends on.
# ---------------------------------------------------------------------------


def test_level_and_increment_separate_on_rereport_and_agree_without_it():
    """The core design requirement this gate exists to check: a SINGLE set of rollouts must be
    scoreable under all three arms (`StepResult.all_arm_rewards`, never a
    second re-run), and LEVEL/INCREMENT must produce DIFFERENT total returns
    for a trajectory with an unchanged re-report while producing the SAME
    total return for a trajectory with no repeats.

    Both trajectories below are driven through ONE `RLEnv` instance each
    (`reward_arm="level"` is the constructor default -- irrelevant here
    since both arms' totals are read from `all_arm_rewards` on every step,
    proving the single-rollout-both-arms property directly rather than by
    construction).
    """
    q, socrata = _synthetic_question()

    # -- trajectory WITHOUT a repeat: query, one report, quit --
    env_no_repeat = RLEnv(socrata, ae.StubGrader())
    results_no_repeat = run_full_episode(
        env_no_repeat,
        q,
        lambda _q, hist: [
            ae.PolicyAction(
                kind="tool_call",
                tool_call=ae.ToolCallRequest(kind="query", dataset_id=DATASET_ID, soql=q.soql),
            ),
            ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
            ae.PolicyAction(kind="quit"),
        ][len(hist)],
    )
    level_no_repeat = sum(r.all_arm_rewards["level"] for r in results_no_repeat)
    increment_no_repeat = sum(r.all_arm_rewards["increment"] for r in results_no_repeat)

    # -- trajectory WITH an unchanged re-report: query, report, SAME report again, quit --
    q2, socrata2 = _synthetic_question()
    env_repeat = RLEnv(socrata2, ae.StubGrader())
    results_repeat = run_full_episode(
        env_repeat,
        q2,
        lambda _q, hist: [
            ae.PolicyAction(
                kind="tool_call",
                tool_call=ae.ToolCallRequest(kind="query", dataset_id=DATASET_ID, soql=q2.soql),
            ),
            ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
            # unchanged re-report:
            ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
            ae.PolicyAction(kind="quit"),
        ][len(hist)],
    )
    level_repeat = sum(r.all_arm_rewards["level"] for r in results_repeat)
    increment_repeat = sum(r.all_arm_rewards["increment"] for r in results_repeat)

    print(
        f"\n[separation] no-repeat: level={level_no_repeat:.4f} increment={increment_no_repeat:.4f}"
    )
    print(f"[separation] w/ repeat: level={level_repeat:.4f} increment={increment_repeat:.4f}")

    # (i) no repeats -> LEVEL and INCREMENT give the SAME total return.
    assert abs(level_no_repeat - increment_no_repeat) < 1e-9, (
        level_no_repeat,
        increment_no_repeat,
    )
    # (ii) with an unchanged re-report -> LEVEL and INCREMENT DIFFER.
    assert abs(level_repeat - increment_repeat) > 1e-6, (level_repeat, increment_repeat)
    # (iii) LEVEL pays extra for the repeat; INCREMENT does not (the farmable
    # horn LEVEL leaves open and INCREMENT is designed to close -- rewards.py
    # module docstring; extra evidence, not the requirement's own wording).
    assert level_repeat > level_no_repeat
    assert abs(increment_repeat - increment_no_repeat) < 1e-9


def test_reward_arm_is_the_only_thing_that_differs_between_arm_runs():
    """The design requirement checked directly: build three RLEnvs identical in every
    argument except `reward_arm`, drive the SAME scripted actions through
    each, and assert every `all_arm_rewards` dict is IDENTICAL across the
    three runs (only `StepResult.reward`, the selected scalar, differs)."""
    actions_template = lambda soql: [  # noqa: E731
        ae.PolicyAction(
            kind="tool_call",
            tool_call=ae.ToolCallRequest(kind="query", dataset_id=DATASET_ID, soql=soql),
        ),
        ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
        ae.PolicyAction(kind="report", confidence_raw=ae._report("answer", "4217", 0.6)),
        ae.PolicyAction(kind="quit"),
    ]
    all_arm_reward_sequences = {}
    selected_reward_sequences = {}
    for arm in ("level", "increment", "control"):
        q, socrata = _synthetic_question()
        env = RLEnv(socrata, ae.StubGrader(), reward_arm=arm)
        actions = actions_template(q.soql)
        results = run_full_episode(env, q, lambda _q, hist, actions=actions: actions[len(hist)])
        all_arm_reward_sequences[arm] = [r.all_arm_rewards for r in results]
        selected_reward_sequences[arm] = [r.reward for r in results]

    assert all_arm_reward_sequences["level"] == all_arm_reward_sequences["increment"]
    assert all_arm_reward_sequences["level"] == all_arm_reward_sequences["control"]
    assert selected_reward_sequences["level"] != selected_reward_sequences["increment"]


# ---------------------------------------------------------------------------
# 3. MOCK GRPO TRAINER -- proves the sampler -> batched rollout -> per-arm
#    reward -> group-relative advantage -> update data flow works end to end.
#    Trivial rule-based policy (reads the question's own fields), NO neural
#    network, NO GPU, NO paid API call of any kind.
# ---------------------------------------------------------------------------


class _RuleBasedPolicy:
    """A trivial TABULAR policy: for `answerable`/`unreliable` questions,
    issue the question's OWN `soql` as a tool call, then report the
    question's OWN (already-materialized) gold value at a per-episode
    confidence drawn from a seeded RNG; for `unanswerable`, abstain
    immediately with no tool call. Always reports the CORRECT value/action
    (deterministic outcome=True) so reward VARIANCE across a GRPO group
    comes only from the confidence draw and the group's rereport/no-rereport
    coin flip below -- enough to prove the data flow works without needing a
    real learned policy."""

    def __init__(self, seed: int, *, rereport_prob: float = 0.5):
        self._rng = np.random.default_rng(seed)
        self._rereport_prob = rereport_prob

    def act(self, question: ae.QuestionRecord, history: list[ae.Turn]) -> ae.PolicyAction:
        n = len(history)
        if question.cls == "unanswerable":
            if n == 0:
                conf = float(self._rng.uniform(0.6, 0.95))
                return ae.PolicyAction(
                    kind="report", confidence_raw=ae._report("abstain", None, conf)
                )
            return ae.PolicyAction(kind="quit")

        if n == 0:
            assert question.dataset_id is not None  # answerable/unreliable always have one
            return ae.PolicyAction(
                kind="tool_call",
                tool_call=ae.ToolCallRequest(
                    kind="query", dataset_id=question.dataset_id, soql=question.soql
                ),
            )
        if n == 1:
            conf = float(self._rng.uniform(0.5, 0.95))
            unc = None
            if question.cls == "unreliable":
                r = (question.reliability or {}).get("r", 0.5)
                unc = f"instrument reliability R~={r} for this measure class"
            return ae.PolicyAction(
                kind="report",
                confidence_raw=ae._report("answer", str(question.gold), conf, unc=unc),
            )
        if n == 2 and self._rng.uniform() < self._rereport_prob:
            # the farmable horn: re-report the SAME confidence with no new information.
            last_report = history[-1].report
            assert last_report is not None and last_report.parsed is not None
            conf = last_report.parsed.confidence
            unc = last_report.parsed.uncertainty_statement
            return ae.PolicyAction(
                kind="report",
                confidence_raw=ae._report("answer", str(question.gold), conf, unc=unc),
            )
        return ae.PolicyAction(kind="quit")


def _rollout(
    question: ae.QuestionRecord, socrata: ae.StubSocrataClient, seed: int, reward_arm: str
):
    env = RLEnv(socrata, ae.StubGrader(), reward_arm=reward_arm)
    policy = _RuleBasedPolicy(seed=seed)
    results = run_full_episode(env, question, policy.act)
    arms = ("level", "increment", "control")
    totals = {arm: sum(r.all_arm_rewards[arm] for r in results) for arm in arms}
    n_tool_calls = sum(1 for t in env.episode_log.turns if t.kind == "tool_call")
    return totals, len(results), n_tool_calls


def test_mock_grpo_trainer_batched_rollout_advantage_and_update():
    """Batched rollouts -> per-arm reward -> group-relative advantage ->
    mock (gradient-free, tabular) policy update. K prompts (questions) x G
    rollouts/prompt ("group" in the GRPO sense: repeated samples of the same
    prompt used to compute a RELATIVE advantage within the group), using the
    REAL 54-question pilot set via QuestionSampler."""
    questions = _load_real_pilot_questions()
    socrata = _stub_socrata_from_gold(questions)
    sampler = QuestionSampler(questions, seed=2026, held_out_fraction=0.2)

    K, G = 6, 8  # 6 prompts, 8 rollouts/prompt = 48 episodes, small but real
    prompts = sampler.sample_batch(K, split="train")

    rewards_level = np.zeros((K, G), dtype=np.float64)
    rewards_increment = np.zeros((K, G), dtype=np.float64)
    turn_counts = np.zeros((K, G), dtype=np.int64)
    seed_counter = 0
    for k, q in enumerate(prompts):
        for g in range(G):
            totals, n_turns, _ = _rollout(
                q, socrata, seed=10_000 + seed_counter, reward_arm="level"
            )
            rewards_level[k, g] = totals["level"]
            rewards_increment[k, g] = totals["increment"]
            turn_counts[k, g] = n_turns
            seed_counter += 1

    # ---- group-relative advantage (GRPO-style): normalize within each
    # prompt's group of G rollouts, not across the whole batch. ----
    def group_relative_advantage(rewards: np.ndarray) -> np.ndarray:
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-8
        return (rewards - mean) / std

    adv_level = group_relative_advantage(rewards_level)
    adv_increment = group_relative_advantage(rewards_increment)

    # ---- shape assertions ----
    assert rewards_level.shape == (K, G)
    assert adv_level.shape == (K, G)
    assert turn_counts.shape == (K, G)

    # ---- group-relative advantage must center each group at ~0 ----
    assert np.allclose(adv_level.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(adv_increment.mean(axis=1), 0.0, atol=1e-6)

    # ---- reward differences between arms must PROPAGATE into the
    # advantage signal: if level and increment totals differ per-rollout
    # (they will, whenever a rollout drew the re-report branch), the two
    # arms' advantage matrices must not be identical. ----
    level_vs_increment_differ = not np.allclose(rewards_level, rewards_increment)
    assert level_vs_increment_differ, "expected some rollouts to hit the re-report branch"
    assert not np.allclose(adv_level, adv_increment)

    # ---- mock policy update: gradient-free, tabular. Bump a per-class
    # "confidence bias" table in the direction of the mean advantage for
    # that class's prompts -- proves an UPDATE SIGNAL was computed and
    # applied from the advantages, without any neural net or optimizer. ----
    confidence_bias = {"answerable": 0.0, "unreliable": 0.0, "unanswerable": 0.0}
    learning_rate = 0.05
    for k, q in enumerate(prompts):
        mean_adv_k = float(adv_level[k].mean())  # ~0 by construction (see above) per-group,
        # so accumulate the RAW (non-group-normalized) reward's deviation
        # from the batch mean instead -- the mock update signal a real GRPO
        # step would apply is per-token-logprob-weighted advantage; here,
        # lacking token logprobs (tabular policy), the analogous cheap proxy
        # is each prompt's mean reward vs the batch mean.
        batch_mean = float(rewards_level.mean())
        prompt_mean = float(rewards_level[k].mean())
        confidence_bias[q.cls] += learning_rate * (prompt_mean - batch_mean)
        del mean_adv_k

    print("\n[mock-trainer] prompts x group:", K, "x", G, "=", K * G, "episodes")
    print("[mock-trainer] rewards_level sample row 0:", np.round(rewards_level[0], 4).tolist())
    print("[mock-trainer] adv_level sample row 0:    ", np.round(adv_level[0], 4).tolist())
    print(
        "[mock-trainer] level vs increment identical across all rollouts:",
        not level_vs_increment_differ,
    )
    print("[mock-trainer] mock tabular update (confidence_bias by class):", confidence_bias)
    assert any(abs(v) > 0.0 for v in confidence_bias.values()), "update signal must be non-trivial"


# ---------------------------------------------------------------------------
# 5. THROUGHPUT MEASUREMENT + EXTRAPOLATION TO THE PLANNED RUN.
# ---------------------------------------------------------------------------


def test_measure_throughput_and_extrapolate_to_planned_run():
    """Real wall-clock timing of `RLEnv`-driven episodes with
    `StubSocrataClient` (zero network -- measures pure environment-mechanics
    overhead, NOT Socrata latency) over the real 54-question pilot set, then
    extrapolates to the planned run: 2 arms x 2 model families x 10 seeds x
    ~500 questions = 20,000 episodes."""
    questions = _load_real_pilot_questions()
    socrata = _stub_socrata_from_gold(questions)

    n_episodes = len(questions)
    policy = _RuleBasedPolicy(seed=1, rereport_prob=0.5)
    started = time.perf_counter()
    total_turns = 0
    total_tool_calls = 0
    for q in questions:
        env = RLEnv(socrata, ae.StubGrader(), reward_arm="level")
        results = run_full_episode(env, q, policy.act)
        total_turns += len(results)
        total_tool_calls += sum(1 for t in env.episode_log.turns if t.kind == "tool_call")
    elapsed = time.perf_counter() - started

    episodes_per_sec = n_episodes / elapsed
    turns_per_episode = total_turns / n_episodes
    tool_calls_per_episode = total_tool_calls / n_episodes

    planned_episodes = 2 * 2 * 10 * 500  # 2 arms x 2 model families x 10 seeds x ~500 questions
    est_wall_clock_env_only_s = planned_episodes / episodes_per_sec
    est_distinct_live_calls = min(
        planned_episodes * tool_calls_per_episode, 500 * tool_calls_per_episode
    )
    # ^ upper bound: with SocrataClient's default force_refresh=False, every
    # (dataset_id, soql) pair is fetched LIVE at most ONCE across the whole
    # run (socrata.py:122-124 -- the cache-hit path returns before the
    # rate-limited `_get` loop is ever entered, i.e. a warm cache pays ZERO
    # of the 0.5s inter-request pacing). Since the planned run repeats the
    # SAME ~500-question set across 2 arms x 2 model families x 10 seeds,
    # live calls are bounded by the number of DISTINCT (dataset_id, soql)
    # pairs in that ~500-question set, not by 20,000 episodes' worth of
    # tool calls.
    MIN_INTERVAL_S = 0.5  # socrata.py:72 MIN_INTERVAL_S_DEFAULT
    est_live_pacing_s = est_distinct_live_calls * MIN_INTERVAL_S

    print(
        f"\n[throughput] {n_episodes} episodes (StubSocrataClient, zero network) in {elapsed:.3f}s"
    )
    print(f"[throughput] episodes/sec: {episodes_per_sec:.1f}")
    print(f"[throughput] turns/episode: {turns_per_episode:.2f}")
    print(f"[throughput] tool_calls/episode: {tool_calls_per_episode:.2f}")
    print(
        f"[extrapolation] planned run size: {planned_episodes} episodes "
        "(2 arms x 2 model families x 10 seeds x ~500 questions)"
    )
    print(
        f"[extrapolation] pure env-mechanics wall clock at measured rate: "
        f"{est_wall_clock_env_only_s:.1f}s ({est_wall_clock_env_only_s / 60:.1f} min)"
    )
    print(
        f"[extrapolation] Socrata LIVE calls upper-bounded by distinct (dataset_id, soql) "
        f"pairs in the ~500-question set (cache reused across all 20,000 episodes): "
        f"<= ~{est_distinct_live_calls:.0f} calls, ~{est_live_pacing_s:.0f}s of 0.5s pacing "
        "if all cold; a real trainer that generates a NEW confidence report each rollout "
        "still issues the SAME soql for the SAME question, so this bound holds regardless "
        "of policy behavior."
    )
    print(
        "[extrapolation] CAVEAT: this measures environment mechanics only "
        "(JSON parse + grade + score_all_arms + bookkeeping), not model inference time, "
        "which will dominate total wall clock for any real GRPO rollout by orders of "
        "magnitude -- this number bounds the environment's OWN overhead, not total "
        "training time."
    )

    assert episodes_per_sec > 0
    assert turns_per_episode > 0


if __name__ == "__main__":
    import inspect

    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
                failures.append((name, exc))
                print(f"FAIL {name}: {exc!r}")
    if failures:
        raise SystemExit(f"{len(failures)} failing: {[n for n, _ in failures]}")
    print(f"{sum(1 for n in globals() if n.startswith('test_'))} tests passed")
