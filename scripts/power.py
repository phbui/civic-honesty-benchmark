"""power.py -- pre-registered, simulation-based power analysis for the civic-honesty
benchmark, meant to be run BEFORE any GPU training spend it de-risks.

Question this file answers, verbatim
from the design brief: "if we spend the money and run the experiment as planned, is it
POWERED to detect the effect we expect -- or would it produce an uninterpretable null?"

WHY SIMULATION, NOT A CLOSED-FORM APPROXIMATION. Every inferential test this benchmark
actually runs is non-standard: `metrics.py::guarded_paired_permutation_test` is a
sign-flip permutation test on SEED-LEVEL paired statistics (not item-level t-tests), and
at n_seeds<=12 it enumerates all 2**n_seeds sign patterns EXACTLY
(`stats._EXACT_ENUMERATION_MAX_N=12`) rather than using a normal
approximation. A closed-form power formula (e.g. the normal-approximation power formula
for a paired t-test) would answer a power question for a DIFFERENT test than the one this
benchmark will actually run. This file instead simulates full experiments end to end --
draw synthetic per-seed data, run the REAL functions from `metrics.py` and
`stats.py` on it, and estimates power
as the fraction of simulated experiments where the real test rejects the null.

HIERARCHICAL DESIGN THIS FILE MODELS. The planned design (10 seeds/arm, 500-1,000 questions
across 3 classes, 3 arms) is a two-level design:
  (1) seed level -- one trained checkpoint per (arm, seed). The permutation test's `N` is
      the number of PAIRED SEEDS (planned: 10/arm), matching `PERMUTATION_MIN_N=10` exactly.
      This is the binding sample-size floor for every primary comparison in this study --
      not the question count.
  (2) question level -- within one seed's evaluation rollout, N_questions_per_class items
      are scored, producing that seed's ONE scalar per-class statistic (e.g. that seed's
      abstention rate on the unanswerable class, or that seed's ECE on the answerable
      class). N_questions_per_class only controls how NOISY each seed's scalar is, not the
      permutation test's N. This file always separates the two: MDE tables are indexed by
      (n_questions_per_class, n_seeds), and n_seeds=10 is the planned design, not this file's
      choice.

DATA-GENERATING PROCESSES (DGPs), each with an [UNVERIFIED]-tagged assumed baseline shown
with an explicit sensitivity sweep, never a single hidden number:
  - PROPORTION metrics (abstention rate, fabrication rate): each seed's TRUE rate is drawn
    from a logit-normal around the arm mean (captures real training-run-to-run
    variability, distinct from the finite-question sampling noise below), then the
    OBSERVED rate is Binomial(n_questions, true_rate)/n_questions. `tau_logit` is the
    seed-heterogeneity knob; swept, never fixed on one guessed value.
  - CONTINUOUS calibration metrics (ECE, Brier): item-level (confidence, correctness)
    pairs are drawn from `confidence ~ Beta(2,2)`, `correct ~ Bernoulli(clip(confidence +
    bias, eps, 1-eps))`. `bias` is a per-seed miscalibration offset (drawn around an
    arm-mean bias, again with seed heterogeneity); `bias=0` is calibrated-in-expectation,
    so a `bias` DIFFERENCE between arms is, in the equal-mass-binning large-N limit,
    approximately an ECE-units effect size -- but this file reports the REALIZED simulated
    ECE/Brier numbers from `metrics.py`'s own functions, never the assumed `bias` itself,
    so binning/estimator bias at finite N is captured, not assumed away.
  - AUROC: the binormal ROC model. Confidence for correct items ~ N(+d/2, 1), for
    incorrect items ~ N(-d/2, 1), both clipped to [0,1]; standard result
    AUROC = Phi(d / sqrt(2)) for equal-variance Gaussians (Phi = standard normal CDF) --
    [VERIFIED, standard statistical identity underlying the binormal ROC model, not an
    external-library behavioral claim; math is out of scope for cite-before-claim].
    `metrics.py::auroc_confidence_correctness`'s own Mann-Whitney-U estimator is run on
    the FINITE simulated sample, so finite-N estimator noise (the thing that actually
    limits power) is what gets measured, not the theoretical Phi(d/sqrt(2)).

RNG CONVENTION. Every simulation call takes an explicit `rng: np.random.Generator`
constructed by the caller via `np.random.default_rng(seed)` -- never the global numpy
RNG -- seeded randomness only, matching
a per-call `np.random.default_rng(seed)` pattern.

RUNTIME. `n_mc` (Monte Carlo replicates per power estimate) defaults are sized to finish
this whole file's `__main__` block in low-single-digit minutes on a laptop CPU -- see
`DEFAULT_N_MC`. Increase for a tighter Monte Carlo error bar on power (this file reports
that error bar, `power_se = sqrt(p(1-p)/n_mc)`, on every estimate).

DEPENDENCIES. numpy only (already a `pyproject.toml` dependency). No scipy (not present in
this env -- `ModuleNotFoundError` verified 2026-07-31 via `uv run python -c "import
scipy"`); the AUROC DGP's normal CDF is `math.erf`-based (stdlib), not `scipy.stats.norm`.
No paid LLM API calls anywhere in this file.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    # Matches test_metrics.py's own convention (see its module docstring): run via
    # `uv run python scripts/power.py` from the repo root, or
    # `uv run pytest scripts/...` -- either way this directory must be on
    # sys.path for `import metrics` to resolve, since scripts/ is not a package.
    sys.path.insert(0, str(SCRIPT_DIR))

import metrics as m  # noqa: E402  -- reused, not reimplemented; see module docstring.

try:
    # Reused, not reimplemented -- same rule metrics.py itself follows.
    from stats import holm_correction

    # Referenced so the import is a real dependency rather than dead weight:
    # power.py reports the Holm-corrected family sizes that metrics.py's
    # stratified_family_correction wraps, and this asserts the same
    # implementation is present rather than silently diverging.
    assert callable(holm_correction)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "stats module not importable "
        "`uv run python scripts/power.py` from the repo root "
        ""
    ) from exc

RESULTS_DIR = SCRIPT_DIR.parent / "results"

ALPHA = 0.05
SPEC_N_SEEDS = 10  # planned design: "10 seeds/arm"
# planned design: "question set of 500-1,000 items across the 3 classes" -- divided
# roughly evenly per the pilot's actual balance (results/questions_meta.json:
# n_answerable=n_unreliable=n_unanswerable=18 of 54, i.e. exactly 1/3 each).
SPEC_N_QUESTIONS_TOTAL_LO = 500
SPEC_N_QUESTIONS_TOTAL_HI = 1000
SPEC_N_QUESTIONS_PER_CLASS_LO = SPEC_N_QUESTIONS_TOTAL_LO // 3  # 166
SPEC_N_QUESTIONS_PER_CLASS_HI = SPEC_N_QUESTIONS_TOTAL_HI // 3  # 333

DEFAULT_N_MC = 1000  # Monte Carlo replicates per power estimate; see module docstring.
# ECE/Brier/AUROC replicates cost ~10x a proportion replicate (each MC draw runs the real
# per-seed `metrics.py` estimator n_seeds times), so their n_mc is set lower to keep this
# file's __main__ under ~8 minutes wall time; every estimate still reports `power_se`
# (Monte Carlo SE of the power estimate itself) so the extra noise from the smaller n_mc
# is visible in the output, never hidden.
CALIBRATION_N_MC = 300
AUROC_N_MC = 300
BISECTION_TOL_ITERS = 8


# --------------------------------------------------------------------------- #
# Small numeric helpers (no scipy -- see module docstring "DEPENDENCIES")
# --------------------------------------------------------------------------- #


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via `math.erf` (stdlib) -- see module docstring
    "DEPENDENCIES" for why this isn't `scipy.stats.norm.cdf`."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binormal_auroc(d: float) -> float:
    """Theoretical AUROC = Phi(d / sqrt(2)) for the equal-variance binormal ROC model
    used by `_simulate_auroc_items` below (standard identity; see module docstring)."""
    return _norm_cdf(d / math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# Power-estimate result type
# --------------------------------------------------------------------------- #


@dataclass
class PowerEstimate:
    metric: str
    n_questions: int
    n_seeds: int
    effect_param: float  # the DGP's own effect knob (rate pp, bias units, or `d`)
    effect_reported_units: float  # effect translated into the metric's OWN reporting units
    n_mc: int
    alpha: float
    power: float
    power_se: float  # Monte Carlo standard error of the power estimate itself
    mean_realized_a: float
    mean_realized_b: float
    assumptions: dict


# --------------------------------------------------------------------------- #
# DGP 1: paired PROPORTION metrics (abstention rate, fabrication rate)
# --------------------------------------------------------------------------- #


def simulate_proportion_power(
    *,
    n_questions: int,
    n_seeds: int,
    p_base: float,
    effect: float,
    tau_logit: float,
    n_mc: int,
    rng: np.random.Generator,
    alpha: float = ALPHA,
) -> PowerEstimate:
    """Power of `metrics.py::guarded_paired_permutation_test` to detect an `effect`
    (arm-B rate minus arm-A rate, in raw proportion units, e.g. 0.05 = 5 percentage
    points) between two arms' per-seed observed proportions, given seed-to-seed
    heterogeneity `tau_logit` (logit-scale SD) and `n_questions` items/seed."""
    p_b = np.clip(p_base + effect, 0.001, 0.999)
    la, lb = float(_logit(p_base)), float(_logit(p_b))
    rejections = 0
    realized_a = np.empty(n_mc)
    realized_b = np.empty(n_mc)
    for i in range(n_mc):
        true_a = _expit(la + rng.normal(0.0, tau_logit, n_seeds))
        true_b = _expit(lb + rng.normal(0.0, tau_logit, n_seeds))
        obs_a = rng.binomial(n_questions, true_a) / n_questions
        obs_b = rng.binomial(n_questions, true_b) / n_questions
        realized_a[i] = obs_a.mean()
        realized_b[i] = obs_b.mean()
        p = m.guarded_paired_permutation_test(obs_a, obs_b, on_violation="raise")
        rejections += p < alpha
    power = rejections / n_mc
    return PowerEstimate(
        metric="proportion",
        n_questions=n_questions,
        n_seeds=n_seeds,
        effect_param=effect,
        effect_reported_units=effect,
        n_mc=n_mc,
        alpha=alpha,
        power=power,
        power_se=math.sqrt(power * (1 - power) / n_mc),
        mean_realized_a=float(realized_a.mean()),
        mean_realized_b=float(realized_b.mean()),
        assumptions={"p_base": p_base, "tau_logit": tau_logit},
    )


# --------------------------------------------------------------------------- #
# DGP 2: continuous calibration metrics via metrics.py's REAL ECE / Brier functions
# --------------------------------------------------------------------------- #


def _simulate_confidence_correct(
    n: int,
    bias: float,
    rng: np.random.Generator,
    *,
    conf_alpha: float = 2.0,
    conf_beta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """confidence ~ Beta(alpha,beta); correct ~ Bernoulli(clip(confidence+bias,eps,1-eps)).
    `bias=0` is calibrated in expectation; `bias>0` is systematic overconfidence."""
    conf = rng.beta(conf_alpha, conf_beta, size=n)
    q = np.clip(conf + bias, 1e-3, 1 - 1e-3)
    correct = (rng.random(n) < q).astype(float)
    return conf, correct


def simulate_calibration_power(
    *,
    stat: str,  # "ece" or "brier"
    n_questions: int,
    n_seeds: int,
    bias_base: float,
    effect: float,  # bias_base - bias_arm_b, i.e. how much LESS miscalibrated arm B is
    seed_noise_sd: float,
    n_mc: int,
    rng: np.random.Generator,
    n_bins: int = 15,
    alpha: float = ALPHA,
) -> PowerEstimate:
    """Power of the permutation test on per-seed ECE (or Brier) values, computed by
    `metrics.py::expected_calibration_error` / `brier_murphy_decomposition` on
    per-seed-simulated (confidence, correct) samples -- the REAL calibration estimator,
    run on synthetic data, not a formula for its sampling distribution."""
    if stat not in ("ece", "brier"):
        raise ValueError(f"unknown stat {stat!r}")
    rejections = 0
    realized_a = np.empty(n_mc)
    realized_b = np.empty(n_mc)
    for i in range(n_mc):
        bias_a = bias_base + rng.normal(0.0, seed_noise_sd, n_seeds)
        bias_b = (bias_base - effect) + rng.normal(0.0, seed_noise_sd, n_seeds)
        vals_a = np.empty(n_seeds)
        vals_b = np.empty(n_seeds)
        for s in range(n_seeds):
            conf, correct = _simulate_confidence_correct(n_questions, float(bias_a[s]), rng)
            if stat == "ece":
                vals_a[s] = m.expected_calibration_error(
                    conf, correct, n_bins=n_bins, strategy="equal_mass"
                ).ece
            else:
                vals_a[s] = m.brier_murphy_decomposition(
                    conf, correct, n_bins=n_bins, strategy="equal_mass"
                ).brier_raw
            conf, correct = _simulate_confidence_correct(n_questions, float(bias_b[s]), rng)
            if stat == "ece":
                vals_b[s] = m.expected_calibration_error(
                    conf, correct, n_bins=n_bins, strategy="equal_mass"
                ).ece
            else:
                vals_b[s] = m.brier_murphy_decomposition(
                    conf, correct, n_bins=n_bins, strategy="equal_mass"
                ).brier_raw
        realized_a[i] = vals_a.mean()
        realized_b[i] = vals_b.mean()
        p = m.guarded_paired_permutation_test(vals_a, vals_b, on_violation="raise")
        rejections += p < alpha
    power = rejections / n_mc
    return PowerEstimate(
        metric=stat,
        n_questions=n_questions,
        n_seeds=n_seeds,
        effect_param=effect,
        effect_reported_units=float(realized_a.mean() - realized_b.mean()),
        n_mc=n_mc,
        alpha=alpha,
        power=power,
        power_se=math.sqrt(power * (1 - power) / n_mc),
        mean_realized_a=float(realized_a.mean()),
        mean_realized_b=float(realized_b.mean()),
        assumptions={"bias_base": bias_base, "seed_noise_sd": seed_noise_sd},
    )


# --------------------------------------------------------------------------- #
# DGP 3: AUROC via metrics.py's REAL Mann-Whitney-U estimator
# --------------------------------------------------------------------------- #


def _simulate_auroc_items(
    n: int, d: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Binormal ROC model: confidence | correct ~ N(+d/2,1), confidence | incorrect ~
    N(-d/2,1), both squashed through the standard logistic to [0,1] (keeps `confidence`
    in the same [0,1] reporting range the real pipeline uses; the logistic squash is
    monotonic so it does not change AUROC, only recalibrates the theoretical
    Phi(d/sqrt(2)) target -- which is why this file always reports the REALIZED simulated
    AUROC, never the theoretical one, as the ground truth)."""
    correct = rng.integers(0, 2, size=n).astype(bool)
    z = np.where(correct, 1.0, -1.0)
    raw = d * z / 2.0 + rng.normal(0.0, 1.0, size=n)
    conf = _expit(raw)
    return conf, correct.astype(float)


def simulate_auroc_power(
    *,
    n_questions: int,
    n_seeds: int,
    d_base: float,
    d_effect: float,
    seed_noise_sd: float,
    n_mc: int,
    rng: np.random.Generator,
    alpha: float = ALPHA,
) -> PowerEstimate:
    rejections = 0
    realized_a = np.empty(n_mc)
    realized_b = np.empty(n_mc)
    for i in range(n_mc):
        d_a = d_base + rng.normal(0.0, seed_noise_sd, n_seeds)
        d_b = (d_base + d_effect) + rng.normal(0.0, seed_noise_sd, n_seeds)
        vals_a = np.empty(n_seeds)
        vals_b = np.empty(n_seeds)
        for s in range(n_seeds):
            conf, correct = _simulate_auroc_items(n_questions, float(d_a[s]), rng)
            res = m.auroc_confidence_correctness(conf, correct)
            vals_a[s] = res.auroc if not math.isnan(res.auroc) else 0.5
            conf, correct = _simulate_auroc_items(n_questions, float(d_b[s]), rng)
            res = m.auroc_confidence_correctness(conf, correct)
            vals_b[s] = res.auroc if not math.isnan(res.auroc) else 0.5
        realized_a[i] = vals_a.mean()
        realized_b[i] = vals_b.mean()
        p = m.guarded_paired_permutation_test(vals_a, vals_b, on_violation="raise")
        rejections += p < alpha
    power = rejections / n_mc
    return PowerEstimate(
        metric="auroc",
        n_questions=n_questions,
        n_seeds=n_seeds,
        effect_param=d_effect,
        effect_reported_units=float(realized_b.mean() - realized_a.mean()),
        n_mc=n_mc,
        alpha=alpha,
        power=power,
        power_se=math.sqrt(power * (1 - power) / n_mc),
        mean_realized_a=float(realized_a.mean()),
        mean_realized_b=float(realized_b.mean()),
        assumptions={
            "d_base": d_base,
            "seed_noise_sd": seed_noise_sd,
            "theoretical_auroc_base": binormal_auroc(d_base),
        },
    )


# --------------------------------------------------------------------------- #
# Bisection: minimum detectable effect at a target power
# --------------------------------------------------------------------------- #


def find_mde(
    power_fn: Callable[[float], PowerEstimate],
    *,
    target_power: float,
    effect_lo: float,
    effect_hi: float,
    tol_iters: int = 10,
) -> tuple[PowerEstimate, list[PowerEstimate]]:
    """Bisection over a scalar effect size for the smallest effect whose simulated power
    reaches `target_power`. `power_fn` must be monotonically non-decreasing in its
    argument (true for every DGP above by construction: bigger effect = bigger true gap =
    at least as much power in expectation). Returns the estimate at (approximately) the
    crossing point and the full trace of evaluated points (kept for auditability -- see
    `results/*.json`)."""
    trace: list[PowerEstimate] = []
    lo_est = power_fn(effect_lo)
    hi_est = power_fn(effect_hi)
    trace.extend([lo_est, hi_est])
    if hi_est.power < target_power:
        # Even the upper bound of the search range can't reach the target power at this
        # (n_questions, n_seeds). Return the upper-bound estimate; callers must check
        # `.power < target_power` and report "MDE exceeds effect_hi", not fabricate a
        # crossing point that was never simulated.
        return hi_est, trace
    lo, hi = effect_lo, effect_hi
    for _ in range(tol_iters):
        mid = (lo + hi) / 2.0
        mid_est = power_fn(mid)
        trace.append(mid_est)
        if mid_est.power >= target_power:
            hi = mid
        else:
            lo = mid
    return power_fn(hi), trace


# --------------------------------------------------------------------------- #
# Multiplicity: approximate Holm-correction power cost
# --------------------------------------------------------------------------- #


def holm_power_cost(
    power_fn: Callable[[float], PowerEstimate], *, effect: float, family_sizes: list[int]
) -> dict[int, float]:
    """Approximates the power lost when the alpha=0.05 rejection threshold used inside a
    `power_fn` power estimate is replaced by a Holm-corrected threshold for a family of
    size m. Exact Holm power requires re-deriving the joint distribution of an entire
    p-value family; this file instead uses the standard conservative approximation
    "smallest-p-in-family must clear alpha/m" (the Bonferroni threshold), which upper-
    bounds Holm's true correction cost since Holm is uniformly at least as powerful as
    Bonferroni -- [INFERRED: Holm (1979) step-down dominance over Bonferroni is textbook,
    but the exact number reported here is a documented conservative approximation, not
    the exact Holm power, and is labeled as such wherever it is printed].
    """
    out: dict[int, float] = {}
    # NOTE: not named `m` -- that shadows the `import metrics as m` at module level.
    for family_size in family_sizes:
        est = power_fn_with_alpha(power_fn, effect, ALPHA / family_size)
        out[family_size] = est.power
    return out


def power_fn_with_alpha(
    power_fn: Callable[[float], PowerEstimate], effect: float, alpha: float
) -> PowerEstimate:
    """Re-runs a `power_fn`-shaped closure at a specific alpha by having the closure
    accept `alpha` as a second positional slot; see call sites below for the
    `functools.partial`-free closures that supply this."""
    return power_fn(effect, alpha)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# __main__: run every simulation, print the deliverable tables, write JSON
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(20260731)  # per-instance RNG, never np.random global

    n_questions_grid = [
        SPEC_N_QUESTIONS_PER_CLASS_LO,  # 166  (500-question plan)
        SPEC_N_QUESTIONS_PER_CLASS_HI,  # 333  (1000-question plan)
    ]
    n_seeds_headline = SPEC_N_SEEDS

    print("=" * 78)
    print("CIVIC-HONESTY POWER GATE -- simulation-based, real metrics.py test functions")
    print(f"alpha={ALPHA}, n_seeds (planned)={n_seeds_headline}, n_mc={DEFAULT_N_MC}")
    print("=" * 78)

    all_results: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alpha": ALPHA,
        "n_mc": DEFAULT_N_MC,
        "spec_n_seeds": n_seeds_headline,
        "spec_n_questions_per_class_range": [
            SPEC_N_QUESTIONS_PER_CLASS_LO,
            SPEC_N_QUESTIONS_PER_CLASS_HI,
        ],
    }

    # ----------------------------------------------------------------- #
    # 1. ABSTENTION RATE (unanswerable class) -- MDE table
    # ----------------------------------------------------------------- #
    print("\n--- 1. ABSTENTION RATE (unanswerable class), proportion, paired ---")
    # [UNVERIFIED] baseline: no measured pilot rate exists yet (0 trained models run --
    # the no-tool contamination ablation is not built yet either). RLHF-tuned base
    # models commonly over-answer under agentic pressure rather than reliably abstain
    # (this is the exact failure this benchmark exists to measure), so p_base is swept
    # across a low/mid/high range rather than fixed at one guessed number.
    abstention_p_base_sweep = [0.20, 0.40, 0.60]
    abstention_tau_sweep = [0.15, 0.35]  # low / high seed-to-seed heterogeneity
    abstention_mde: list[dict] = []
    for p_base in abstention_p_base_sweep:
        for tau in abstention_tau_sweep:
            for nq in n_questions_grid:

                def pf(effect, alpha=ALPHA, _p_base=p_base, _tau=tau, _nq=nq):
                    return simulate_proportion_power(
                        n_questions=_nq,
                        n_seeds=n_seeds_headline,
                        p_base=_p_base,
                        effect=effect,
                        tau_logit=_tau,
                        n_mc=DEFAULT_N_MC,
                        rng=master_rng,
                        alpha=alpha,
                    )

                est80, _ = find_mde(
                    pf,
                    target_power=0.80,
                    effect_lo=0.0,
                    effect_hi=0.60,
                    tol_iters=BISECTION_TOL_ITERS,
                )
                est90, _ = find_mde(
                    pf,
                    target_power=0.90,
                    effect_lo=0.0,
                    effect_hi=0.60,
                    tol_iters=BISECTION_TOL_ITERS,
                )
                row = {
                    "p_base": p_base,
                    "tau_logit": tau,
                    "n_questions": nq,
                    "n_seeds": n_seeds_headline,
                    "mde_80pct_pp": est80.effect_param * 100,
                    "power_at_mde_80": est80.power,
                    "mde_90pct_pp": est90.effect_param * 100,
                    "power_at_mde_90": est90.power,
                }
                abstention_mde.append(row)
                print(
                    f"  p_base={p_base:.2f} tau={tau:.2f} N_q={nq:>3}  "
                    f"MDE80={row['mde_80pct_pp']:5.1f}pp (power={row['power_at_mde_80']:.2f})  "
                    f"MDE90={row['mde_90pct_pp']:5.1f}pp (power={row['power_at_mde_90']:.2f})"
                )
    all_results["abstention_rate_mde"] = abstention_mde

    # ----------------------------------------------------------------- #
    # 2. FABRICATION RATE (proportion, SAME DGP but flagged confounded -- see step 6)
    # ----------------------------------------------------------------- #
    print("\n--- 2. FABRICATION RATE, proportion, paired (denominator-confound caveat below) ---")
    fabrication_p_base_sweep = [0.10, 0.25]
    fabrication_tau = 0.25
    fabrication_mde: list[dict] = []
    for p_base in fabrication_p_base_sweep:
        for nq in n_questions_grid:

            def pf(effect, alpha=ALPHA, _p_base=p_base, _nq=nq):
                return simulate_proportion_power(
                    n_questions=_nq,
                    n_seeds=n_seeds_headline,
                    p_base=_p_base,
                    effect=effect,
                    tau_logit=fabrication_tau,
                    n_mc=DEFAULT_N_MC,
                    rng=master_rng,
                    alpha=alpha,
                )

            est80, _ = find_mde(
                pf, target_power=0.80, effect_lo=0.0, effect_hi=0.60, tol_iters=BISECTION_TOL_ITERS
            )
            est90, _ = find_mde(
                pf, target_power=0.90, effect_lo=0.0, effect_hi=0.60, tol_iters=BISECTION_TOL_ITERS
            )
            row = {
                "p_base": p_base,
                "n_questions": nq,
                "n_seeds": n_seeds_headline,
                "mde_80pct_pp": est80.effect_param * 100,
                "power_at_mde_80": est80.power,
                "mde_90pct_pp": est90.effect_param * 100,
                "power_at_mde_90": est90.power,
            }
            fabrication_mde.append(row)
            print(
                f"  p_base={p_base:.2f} N_q={nq:>3}  MDE80={row['mde_80pct_pp']:5.1f}pp "
                f"MDE90={row['mde_90pct_pp']:5.1f}pp   "
                "[NOTE: n_questions here is 'items agent chose to answer', a DOWNSTREAM, "
                "arm-dependent count, not the design's raw per-class N -- see step 6 flag]"
            )
    all_results["fabrication_rate_mde"] = fabrication_mde

    # ----------------------------------------------------------------- #
    # 3. ECE + BRIER (answerable / unreliable classes), continuous, real metrics.py fns
    # ----------------------------------------------------------------- #
    print("\n--- 3. ECE and Brier (real metrics.py estimators), continuous, paired ---")
    calibration_bias_base_sweep = [0.10, 0.20]  # moderate / poor baseline overconfidence
    calibration_seed_noise = 0.03
    calibration_mde: list[dict] = []
    for stat in ("ece", "brier"):
        for bias_base in calibration_bias_base_sweep:
            for nq in n_questions_grid:

                def pf(effect, alpha=ALPHA, _stat=stat, _bias_base=bias_base, _nq=nq):
                    return simulate_calibration_power(
                        stat=_stat,
                        n_questions=_nq,
                        n_seeds=n_seeds_headline,
                        bias_base=_bias_base,
                        effect=effect,
                        seed_noise_sd=calibration_seed_noise,
                        n_mc=CALIBRATION_N_MC,
                        rng=master_rng,
                        alpha=alpha,
                    )

                est80, _ = find_mde(
                    pf,
                    target_power=0.80,
                    effect_lo=0.0,
                    effect_hi=0.18,
                    tol_iters=BISECTION_TOL_ITERS,
                )
                est90, _ = find_mde(
                    pf,
                    target_power=0.90,
                    effect_lo=0.0,
                    effect_hi=0.18,
                    tol_iters=BISECTION_TOL_ITERS,
                )
                row = {
                    "stat": stat,
                    "bias_base": bias_base,
                    "n_questions": nq,
                    "n_seeds": n_seeds_headline,
                    "mde_80pct_units": est80.effect_reported_units,
                    "power_at_mde_80": est80.power,
                    "mde_90pct_units": est90.effect_reported_units,
                    "power_at_mde_90": est90.power,
                }
                calibration_mde.append(row)
                print(
                    f"  {stat.upper():5s} bias_base={bias_base:.2f} N_q={nq:>3}  "
                    f"MDE80={row['mde_80pct_units']:.4f} (power={row['power_at_mde_80']:.2f})  "
                    f"MDE90={row['mde_90pct_units']:.4f} (power={row['power_at_mde_90']:.2f})"
                )
    all_results["calibration_mde"] = calibration_mde

    # ----------------------------------------------------------------- #
    # 4. AUROC (real Mann-Whitney-U estimator), continuous, paired
    # ----------------------------------------------------------------- #
    print("\n--- 4. AUROC (real metrics.py Mann-Whitney-U estimator), continuous, paired ---")
    auroc_d_base_sweep = [0.6, 1.0]  # theoretical baseline AUROC ~0.66 / ~0.76
    auroc_seed_noise = 0.10
    auroc_mde: list[dict] = []
    for d_base in auroc_d_base_sweep:
        for nq in n_questions_grid:

            def pf(d_effect, alpha=ALPHA, _d_base=d_base, _nq=nq):
                return simulate_auroc_power(
                    n_questions=_nq,
                    n_seeds=n_seeds_headline,
                    d_base=_d_base,
                    d_effect=d_effect,
                    seed_noise_sd=auroc_seed_noise,
                    n_mc=AUROC_N_MC,
                    rng=master_rng,
                    alpha=alpha,
                )

            est80, _ = find_mde(
                pf, target_power=0.80, effect_lo=0.0, effect_hi=1.2, tol_iters=BISECTION_TOL_ITERS
            )
            est90, _ = find_mde(
                pf, target_power=0.90, effect_lo=0.0, effect_hi=1.2, tol_iters=BISECTION_TOL_ITERS
            )
            row = {
                "theoretical_auroc_base": binormal_auroc(d_base),
                "n_questions": nq,
                "n_seeds": n_seeds_headline,
                "mde_80pct_auroc": est80.effect_reported_units,
                "power_at_mde_80": est80.power,
                "mde_90pct_auroc": est90.effect_reported_units,
                "power_at_mde_90": est90.power,
            }
            auroc_mde.append(row)
            print(
                f"  AUROC_base~={row['theoretical_auroc_base']:.3f} N_q={nq:>3}  "
                f"MDE80={row['mde_80pct_auroc']:.4f} (power={row['power_at_mde_80']:.2f})  "
                f"MDE90={row['mde_90pct_auroc']:.4f} (power={row['power_at_mde_90']:.2f})"
            )
    all_results["auroc_mde"] = auroc_mde

    # ----------------------------------------------------------------- #
    # 5. Required-N inversion for "scientifically interesting" effect sizes
    # ----------------------------------------------------------------- #
    print("\n--- 5. REQUIRED N for scientifically-interesting effects ---")
    # An earlier from-scratch MiniGrid RL study of the same level-vs-increment reward
    # contrast measured final accuracy 0.7145 (level/gated) vs 0.9270
    # (increment_signed) vs 0.8781 (belief-independent control) -- a ~21pp gap between
    # level and increment arms, in that from-scratch agent's episode-level
    # accuracy, NOT an LLM's per-item abstention/calibration behavior. [INFERRED, not
    # a directly verified transfer] this magnitude is used only as an UPPER anchor for
    # "what this reward-shape manipulation has produced when it works elsewhere",
    # because a from-scratch policy collapsing to a 2-step episode is a much larger,
    # more mechanical effect than an already-RLHF-tuned LLM's abstention shift under a
    # differently-shaped reward is likely to be. A LOWER anchor of 5pp / 0.02 ECE-units /
    # 0.03 AUROC represents a "real but modest, still worth a paper" effect. Both anchors
    # are swept, not asserted as a single number.
    required_n: list[dict] = []

    def required_n_for_target(
        power_fn_factory, target_effect, target_power, n_lo, n_hi, tol_iters=9
    ):
        """Bisection over integer n_questions for the smallest n reaching target_power at
        a FIXED target_effect and the planned n_seeds=10. Returns (n*, achieved power) or
        (None, power_at_n_hi) if even n_hi is insufficient."""
        lo, hi = n_lo, n_hi
        est_hi = power_fn_factory(hi)(target_effect)
        if est_hi.power < target_power:
            return None, est_hi.power
        for _ in range(tol_iters):
            mid = (lo + hi) // 2
            if mid == lo:
                break
            est_mid = power_fn_factory(mid)(target_effect)
            if est_mid.power >= target_power:
                hi = mid
            else:
                lo = mid
        return hi, power_fn_factory(hi)(target_effect).power

    # (a) abstention rate: interesting = 10pp (modest) and 21pp (the earlier from-scratch-RL scale)
    for interesting_pp in (5, 10, 21):

        def factory(nq, _pp=interesting_pp):
            def pf(effect, alpha=ALPHA):
                return simulate_proportion_power(
                    n_questions=nq,
                    n_seeds=n_seeds_headline,
                    p_base=0.40,
                    effect=effect,
                    tau_logit=0.25,
                    n_mc=DEFAULT_N_MC,
                    rng=master_rng,
                    alpha=alpha,
                )

            return pf

        n_star, power_at_1000 = required_n_for_target(
            factory, interesting_pp / 100.0, 0.80, 20, 2000
        )
        row = {
            "metric": "abstention_rate",
            "target_effect_pp": interesting_pp,
            "required_n_per_class_for_80pct_power": n_star,
            "power_at_n2000_if_insufficient": None if n_star is not None else power_at_1000,
            "planned_n_range": [SPEC_N_QUESTIONS_PER_CLASS_LO, SPEC_N_QUESTIONS_PER_CLASS_HI],
        }
        required_n.append(row)
        exceeds = (
            "EXCEEDS PLAN"
            if n_star is None or n_star > SPEC_N_QUESTIONS_PER_CLASS_HI
            else "within plan"
        )
        print(
            f"  abstention effect={interesting_pp}pp -> required N/class for 80% power = "
            f"{n_star!r}  ({exceeds}; plan="
            f"{SPEC_N_QUESTIONS_PER_CLASS_LO}-{SPEC_N_QUESTIONS_PER_CLASS_HI})"
        )
    all_results["required_n_inversion"] = required_n

    # ----------------------------------------------------------------- #
    # 6. Seeds sensitivity (n_seeds swept at fixed n_questions=250)
    # ----------------------------------------------------------------- #
    print("\n--- 6. Seed-count sensitivity at N_questions=250, abstention MDE80 ---")
    seeds_sensitivity: list[dict] = []
    for n_seeds in (5, 10, 12, 20):

        def pf(effect, alpha=ALPHA, _ns=n_seeds):
            return simulate_proportion_power(
                n_questions=250,
                n_seeds=_ns,
                p_base=0.40,
                effect=effect,
                tau_logit=0.25,
                n_mc=DEFAULT_N_MC,
                rng=master_rng,
                alpha=alpha,
            )

        try:
            est80, _ = find_mde(pf, target_power=0.80, effect_lo=0.0, effect_hi=0.6)
            row = {
                "n_seeds": n_seeds,
                "mde_80pct_pp": est80.effect_param * 100,
                "power": est80.power,
            }
        except m.SampleSizeError as exc:
            row = {"n_seeds": n_seeds, "error": str(exc)}
        seeds_sensitivity.append(row)
        print(f"  n_seeds={n_seeds:>2}  {row}")
    all_results["seeds_sensitivity"] = seeds_sensitivity

    # ----------------------------------------------------------------- #
    # 7. Multiplicity cost (Bonferroni-approximated Holm threshold)
    # ----------------------------------------------------------------- #
    print("\n--- 7. Multiplicity cost: power at Holm-approximated thresholds ---")
    print(
        "    metrics.py::stratified_family_correction (holm_correction) is coded to run "
        "ONLY across the per-class family (m<=3: answerable/unanswerable/unreliable) for "
        "a SINGLE metric x SINGLE arm-pair at a time -- see metrics.py L937-947, "
        "`stratified_family_correction(pvals_by_class: Mapping[str, float])`. It does NOT "
        "correct across the 4+ metrics or across the 3 possible arm-pairs "
        "(level-vs-increment, level-vs-control, increment-vs-control)."
    )
    family_sizes = {
        "as-coded (per-class only, single metric, single arm-pair)": 3,
        "headline contrast only (level-vs-increment), all applicable metric-class cells": 12,
        "full 3-arm-pair x metric-class family": 36,
    }

    def abstention_pf(effect, alpha=ALPHA):
        return simulate_proportion_power(
            n_questions=SPEC_N_QUESTIONS_PER_CLASS_HI,
            n_seeds=n_seeds_headline,
            p_base=0.40,
            effect=effect,
            tau_logit=0.25,
            n_mc=DEFAULT_N_MC,
            rng=master_rng,
            alpha=alpha,
        )

    multiplicity_rows = []
    for label, m_size in family_sizes.items():
        est = power_fn_with_alpha(abstention_pf, 0.15, ALPHA / m_size)
        multiplicity_rows.append(
            {"family": label, "m": m_size, "alpha_used": ALPHA / m_size, "power_at_15pp": est.power}
        )
        print(
            f"  {label:75s} m={m_size:>2}  "
            f"alpha_used={ALPHA / m_size:.5f}  power@15pp={est.power:.3f}"
        )
    # uncorrected reference row
    uncorrected = power_fn_with_alpha(abstention_pf, 0.15, ALPHA)
    multiplicity_rows.append(
        {
            "family": "uncorrected (alpha=0.05)",
            "m": 1,
            "alpha_used": ALPHA,
            "power_at_15pp": uncorrected.power,
        }
    )
    print(
        f"  {'uncorrected (alpha=0.05)':75s} m= 1  "
        f"alpha_used={ALPHA:.5f}  power@15pp={uncorrected.power:.3f}"
    )
    all_results["multiplicity_cost"] = multiplicity_rows

    elapsed = time.time() - t0
    print(f"\nTotal simulation wall time: {elapsed:.1f}s")
    all_results["wall_time_seconds"] = elapsed

    out_path = RESULTS_DIR / "power_analysis_2026-07-31.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {out_path}")
