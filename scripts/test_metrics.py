"""Unit tests for metrics.py / contamination.py, validated against SYNTHETIC data
where the correct answer is known analytically -- this is
the only way to know the implementations are right. Not wired into any
project-wide default test task
(pyproject `testpaths = ["tests"]` scopes the default pytest run to
any parent package; these scripts are standalone
analysis code, not CI-gated pytest). Run explicitly:

    uv run pytest scripts/test_metrics.py -v
"""

from __future__ import annotations

import math
import warnings

import contamination as ct
import metrics as m
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# ECE
# --------------------------------------------------------------------------- #


def test_ece_perfectly_calibrated_predictor_near_zero():
    rng = np.random.default_rng(0)
    n = 20_000
    conf = rng.uniform(0.02, 0.98, size=n)
    correct = (rng.random(n) < conf).astype(float)
    for strategy in ("equal_mass", "equal_width"):
        res = m.expected_calibration_error(conf, correct, n_bins=15, strategy=strategy)
        assert res.ece < 0.03, (
            f"{strategy}: ECE={res.ece} should be near 0 for a calibrated predictor"
        )


def test_ece_badly_miscalibrated_predictor_is_large():
    rng = np.random.default_rng(1)
    n = 5000
    conf = np.full(n, 0.95)  # always very confident
    correct = (rng.random(n) < 0.3).astype(float)  # but usually wrong
    res = m.expected_calibration_error(conf, correct, n_bins=15, strategy="equal_mass")
    assert res.ece > 0.5


def test_equal_mass_binning_collapses_on_heavy_ties():
    conf = np.array([0.1] * 40 + [0.5] * 40 + [0.9] * 40)
    correct = np.array([0.0] * 20 + [1.0] * 20 + [0.0] * 20 + [1.0] * 20 + [0.0] * 20 + [1.0] * 20)
    res = m.expected_calibration_error(conf, correct, n_bins=15, strategy="equal_mass")
    # exactly 3 distinct confidence values -> exactly 3 bins, none split, none empty
    assert res.n_bins_used == 3
    assert res.n_bins_used < res.n_bins_requested
    assert all(b.n == 40 for b in res.bins)


# --------------------------------------------------------------------------- #
# Murphy decomposition
# --------------------------------------------------------------------------- #


def test_murphy_decomposition_sums_to_binned_brier():
    rng = np.random.default_rng(2)
    for n in (37, 500, 4000):
        conf = rng.uniform(0.0, 1.0, size=n)
        correct = (rng.random(n) < conf**0.5).astype(float)  # some miscalibration by construction
        for strategy in ("equal_mass", "equal_width"):
            dec = m.brier_murphy_decomposition(conf, correct, n_bins=10, strategy=strategy)
            reconstructed = dec.reliability - dec.resolution + dec.uncertainty
            assert math.isclose(reconstructed, dec.brier_binned, abs_tol=1e-9), (
                f"n={n} strategy={strategy}: reliability-resolution+uncertainty="
                f"{reconstructed} != brier_binned={dec.brier_binned}"
            )


def test_murphy_decomposition_perfect_forecaster_zero_reliability():
    # forecast IS the bin-mean outcome exactly (already bin-constant) -> raw == binned,
    # and a forecaster whose confidence always equals the bin's true frequency has
    # zero reliability (no miscalibration) by definition.
    conf = np.array([0.0] * 50 + [1.0] * 50)
    correct = np.array([0.0] * 50 + [1.0] * 50)
    dec = m.brier_murphy_decomposition(conf, correct, n_bins=2, strategy="equal_width")
    assert math.isclose(dec.reliability, 0.0, abs_tol=1e-9)
    assert math.isclose(dec.brier_raw, dec.brier_binned, abs_tol=1e-9)
    assert math.isclose(dec.brier_binned, 0.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# AUROC
# --------------------------------------------------------------------------- #


def test_auroc_perfectly_discriminating_is_one():
    n = 200
    conf = np.concatenate([np.full(n, 0.4), np.full(n, 0.9)])  # incorrect group, correct group
    correct = np.concatenate([np.zeros(n), np.ones(n)])
    res = m.auroc_confidence_correctness(conf, correct)
    assert res.auroc == pytest.approx(1.0)
    assert res.undefined_reason is None


def test_auroc_constant_confidence_is_half():
    rng = np.random.default_rng(3)
    n = 400
    conf = np.full(n, 0.7)
    correct = (rng.random(n) < 0.5).astype(bool)
    assert correct.sum() > 0 and (~correct).sum() > 0
    res = m.auroc_confidence_correctness(conf, correct)
    assert res.auroc == pytest.approx(0.5)


def test_auroc_undefined_on_constant_correctness():
    conf = np.array([0.1, 0.5, 0.9, 0.3])
    correct_all_true = np.array([True, True, True, True])
    res = m.auroc_confidence_correctness(conf, correct_all_true)
    assert math.isnan(res.auroc)
    assert res.undefined_reason is not None and "constant" in res.undefined_reason


# --------------------------------------------------------------------------- #
# Risk-coverage / AURC
# --------------------------------------------------------------------------- #


def test_aurc_perfect_ranking_beats_random_beats_worst():
    rng = np.random.default_rng(4)
    n = 1000
    err_rate = 0.3
    correct = np.ones(n)
    n_err = int(n * err_rate)
    correct[:n_err] = 0.0
    rng.shuffle(correct)

    # perfect ranker: confidence strictly higher for correct items
    conf_perfect = np.where(correct == 1.0, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    # random ranker: confidence independent of correctness
    conf_random = rng.uniform(0.0, 1.0, n)
    # worst ranker: confidence strictly higher for INCORRECT items
    conf_worst = np.where(correct == 1.0, rng.uniform(0.0, 0.4, n), rng.uniform(0.6, 1.0, n))

    rc_perfect = m.risk_coverage_curve(conf_perfect, correct)
    rc_random = m.risk_coverage_curve(conf_random, correct)
    rc_worst = m.risk_coverage_curve(conf_worst, correct)

    assert rc_perfect.aurc < rc_random.aurc < rc_worst.aurc
    # random ranker's risk should hover near the base error rate at every coverage
    assert rc_random.aurc == pytest.approx(err_rate, abs=0.05)
    # perfect ranker should have near-zero risk until coverage exceeds the accuracy
    assert rc_perfect.risk[: int(n * (1 - err_rate)) - 5].max() < 0.02


def test_aurc_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    conf = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    correct = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    rc = m.risk_coverage_curve(conf, correct)
    out = tmp_path / "rc_smoke"
    m.plot_risk_coverage_curve(rc, title="smoke", out_path=out)
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()


# --------------------------------------------------------------------------- #
# Abstention / verification split
# --------------------------------------------------------------------------- #


def test_abstention_report_verified_split():
    items = [
        m.EvalItem(
            "q1", "unanswerable", None, None, None, [], True, 3, None
        ),  # verified-then-abstained
        m.EvalItem(
            "q2", "unanswerable", None, None, None, [], True, 0, None
        ),  # abstained-without-verification
        m.EvalItem("q3", "unanswerable", None, None, None, [], True, None, None),  # unknown
        m.EvalItem(
            "q4", "unanswerable", None, "some answer", 0.9, [], False, 2, False
        ),  # not abstained
    ]
    rep = m.abstention_report(items)
    assert rep.n_items == 4
    assert rep.n_abstained == 3
    assert rep.n_verified_then_abstained == 1
    assert rep.n_abstained_without_verification == 1
    assert rep.n_unknown_verification == 1
    assert rep.abstention_rate == pytest.approx(0.75)
    assert rep.verified_then_abstained_rate == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Fabrication rate
# --------------------------------------------------------------------------- #


def test_fabrication_rate_execution_match():
    items = [
        m.EvalItem("q1", "answerable", 42, 42, 0.9, [], False, 1, True),
        m.EvalItem("q2", "answerable", 42, 7, 0.9, [], False, 1, False),
        m.EvalItem("q3", "answerable", 42, None, 0.9, [], False, 1, None),  # unscored
        m.EvalItem("q4", "answerable", 42, None, None, [], True, 1, None),  # abstained, excluded
    ]
    rate, n_scored, n_unscored = m.fabrication_rate(items)
    assert n_scored == 2
    assert n_unscored == 1
    assert rate == pytest.approx(0.5)


def test_judge_fabrication_rate_only_hits_unscored_subset():
    items = [
        m.EvalItem(
            "q1", "answerable", 42, 42, 0.9, [], False, 1, True
        ),  # scored already, judge not called
        m.EvalItem("q2", "answerable", "no data available", None, 0.2, [], False, 1, None),
    ]
    calls = []

    def fake_judge(final_answer, gold_answer, context):
        calls.append(context["question_id"])
        return True  # "honest" (not fabricated)

    rate, n = m.judge_fabrication_rate(items, fake_judge)
    assert calls == ["q2"]
    assert n == 1
    assert rate == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# normalize_item / stratify / compute_metric_suite end-to-end
# --------------------------------------------------------------------------- #


def _make_raw(qid, cls, gold, final, conf, abstained, tool_calls):
    return {
        "id": qid,
        "question_class": cls,
        "gold_answer": gold,
        "final_answer": final,
        "confidence": conf,
        "abstained": abstained,
        "tool_call_count": tool_calls,
    }


def test_compute_metric_suite_unanswerable_skips_calibration():
    rng = np.random.default_rng(5)
    raw = []
    for i in range(20):
        abstained = i % 4 != 0  # 75% correctly abstain, 25% fabricate an answer
        raw.append(
            _make_raw(
                f"u{i}",
                "unanswerable",
                None,
                None if abstained else "fabricated value",
                None if abstained else float(rng.uniform(0.5, 1.0)),
                abstained,
                int(rng.integers(0, 3)),
            )
        )
    suite = m.compute_metric_suite(raw)
    rep = suite["unanswerable"]
    assert (
        rep.ece is None and rep.murphy is None and rep.auroc is None and rep.risk_coverage is None
    )
    assert rep.abstention.abstention_rate == pytest.approx(0.75)
    assert any("SKIPPED" in note for note in rep.notes)


def test_compute_metric_suite_answerable_computes_full_suite():
    rng = np.random.default_rng(6)
    raw = []
    for i in range(200):
        gold = rng.integers(0, 100)
        conf = float(rng.uniform(0.0, 1.0))
        answered_correctly = rng.random() < conf
        final = int(gold) if answered_correctly else int(gold) + 1
        raw.append(
            _make_raw(f"a{i}", "answerable", gold, final, conf, False, int(rng.integers(0, 4)))
        )
    suite = m.compute_metric_suite(raw)
    rep = suite["answerable"]
    assert (
        rep.ece is not None
        and rep.murphy is not None
        and rep.auroc is not None
        and rep.risk_coverage is not None
    )
    assert rep.n_scored == 200
    assert 0.0 <= rep.ece.ece <= 1.0
    assert not math.isnan(rep.auroc.auroc)


def test_never_pools_across_classes():
    raw = [
        _make_raw("a1", "answerable", 1, 1, 0.9, False, 1),
        _make_raw("r1", "unreliable", 2, 2, 0.9, False, 1),
        _make_raw("n1", "unanswerable", None, None, None, True, 1),
    ]
    suite = m.compute_metric_suite(raw)
    assert set(suite.keys()) == {"answerable", "unanswerable", "unreliable"}
    assert suite["answerable"].n_items == 1
    assert suite["unreliable"].n_items == 1
    assert suite["unanswerable"].n_items == 1


# --------------------------------------------------------------------------- #
# Sample-size floors (Colas et al. arXiv:1904.06979)
# --------------------------------------------------------------------------- #


def test_permutation_test_refuses_below_n10():
    a = np.array([1.0] * 5)
    b = np.array([0.0] * 5)
    with pytest.raises(m.SampleSizeError):
        m.guarded_paired_permutation_test(a, b)


def test_permutation_test_warns_when_told_to():
    a = np.array([1.0] * 5)
    b = np.array([0.0] * 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = m.guarded_paired_permutation_test(a, b, on_violation="warn")
    assert result is None
    assert any("N>=10" in str(x.message) for x in w)


def test_permutation_test_runs_at_n10():
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 1, 10)
    b = a - 0.5  # a clear paired shift
    p = m.guarded_paired_permutation_test(a, b)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_bootstrap_ci_refuses_below_n50():
    values = np.arange(20, dtype=float)
    with pytest.raises(m.SampleSizeError):
        m.bootstrap_ci(values)


def test_bootstrap_ci_warns_when_told_to():
    values = np.arange(20, dtype=float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lo, hi = m.bootstrap_ci(values, on_violation="warn")
    assert math.isnan(lo) and math.isnan(hi)
    assert any("N>=50" in str(x.message) for x in w)


def test_bootstrap_ci_runs_at_n50():
    rng = np.random.default_rng(8)
    values = rng.normal(0.5, 0.05, size=60)
    lo, hi = m.bootstrap_ci(values, seed=0)
    assert lo < np.mean(values) < hi


def test_stratified_family_correction_reuses_holm():
    pvals = {"answerable": 0.01, "unreliable": 0.04, "unanswerable": 0.20}
    corrected = m.stratified_family_correction(pvals)
    assert set(corrected) == set(pvals)
    assert all(corrected[k] >= pvals[k] for k in pvals)
    # Holm-Bonferroni monotonicity: sorted-ascending corrected p's are non-decreasing
    ordered = sorted(pvals, key=lambda k: pvals[k])
    vals = [corrected[k] for k in ordered]
    assert vals == sorted(vals)


# --------------------------------------------------------------------------- #
# contamination.py
# --------------------------------------------------------------------------- #


def test_never_call_query_fn_raises():
    with pytest.raises(NotImplementedError):
        ct.NeverCallQueryFn()({"id": "q1"})


def test_run_no_tool_ablation_excludes_unanswerable_and_flags_correctly():
    questions = [
        {"id": "a1", "question_class": "answerable", "gold_answer": 42},
        {"id": "a2", "question_class": "answerable", "gold_answer": 99},
        {"id": "r1", "question_class": "unreliable", "gold_answer": 7},
        {"id": "u1", "question_class": "unanswerable", "gold_answer": None},
    ]
    # fake model: "knows" a1 and r1 from parametric memory, guesses wrong on a2
    fake_answers = {"a1": "42", "a2": "0", "r1": "7"}

    def fake_query(q):
        return fake_answers[ct._get(q, "id")]

    results = ct.run_no_tool_ablation(questions, query_fn=fake_query)
    ids_seen = {r.question_id for r in results}
    assert ids_seen == {"a1", "a2", "r1"}  # u1 excluded (unanswerable)

    report = ct.summarize_contamination(results)
    assert report.per_class["answerable"].n_items == 2
    assert report.per_class["answerable"].n_contaminated == 1
    assert report.per_class["answerable"].contamination_rate == pytest.approx(0.5)
    assert report.per_class["unreliable"].contamination_rate == pytest.approx(1.0)
    assert report.is_contaminated("a1") is True
    assert report.is_contaminated("a2") is False


def test_filter_non_contaminated_drops_flagged_items():
    questions = [
        {"id": "a1", "question_class": "answerable", "gold_answer": 42},
        {"id": "a2", "question_class": "answerable", "gold_answer": 99},
    ]
    results = ct.run_no_tool_ablation(
        questions, query_fn=lambda q: {"a1": "42", "a2": "0"}[ct._get(q, "id")]
    )
    report = ct.summarize_contamination(results)
    downstream_items = [{"question_id": "a1"}, {"question_id": "a2"}]
    filtered = ct.filter_non_contaminated(downstream_items, report)
    assert filtered == [{"question_id": "a2"}]


def test_default_no_tool_score_numeric_and_substring():
    assert ct.default_no_tool_score("42.0000001", 42, None) is True
    assert ct.default_no_tool_score("the answer is 42 units", "42", None) is True
    assert ct.default_no_tool_score("nope", "42", None) is False
    assert ct.default_no_tool_score(None, 42, None) is False
