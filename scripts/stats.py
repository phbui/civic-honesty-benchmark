"""Standalone statistics helpers for the benchmark harness.

Holm-Bonferroni step-down correction and a paired sign-flip permutation
test. Both are standard methods, implemented here so the harness has no
external dependencies beyond numpy.
"""

from __future__ import annotations

import itertools

import numpy as np

# Below this sample count, all 2**n sign patterns are enumerated exactly;
# above it, a Monte-Carlo approximation with n_perm draws is used.
_EXACT_ENUMERATION_MAX_N = 12


def _sign_patterns(n: int, *, n_perm: int, seed: int) -> tuple[np.ndarray, int]:
    """Returns (signs, total): an (total, n) array of +/-1 rows covering every
    one of the 2**n sign-flip patterns when n <= _EXACT_ENUMERATION_MAX_N,
    else n_perm Monte-Carlo draws."""
    if n <= _EXACT_ENUMERATION_MAX_N:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        return signs, signs.shape[0]
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    return signs, n_perm


def paired_permutation_test(
    a: np.ndarray, b: np.ndarray, *, n_perm: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for the matched difference between two paired score
    arrays `a` and `b` (same items, same order). Sign-flip permutation null
    (exchangeability under H0: differences symmetric about 0), add-one
    smoothed. For n <= 12 paired samples every one of the 2**n sign-flip
    patterns is enumerated exactly (the p-value is then independent of
    `seed`); larger n uses an n_perm-draw Monte-Carlo estimate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test requires equal-length, matched score arrays")
    diff = a - b
    observed = abs(float(np.mean(diff)))
    n = diff.shape[0]
    signs, total = _sign_patterns(n, n_perm=n_perm, seed=seed)
    at_least = int(np.count_nonzero(np.abs(signs @ diff / n) >= observed))
    return (at_least + 1) / (total + 1)


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction across a family of m hypothesis
    tests, controlling the family-wise error rate less conservatively than
    plain Bonferroni. Sort p-values ascending; the i-th smallest (0-indexed)
    is scaled by (m - i); each adjusted p is floored at the running max of
    prior adjusted p's (monotonicity), and every result is clipped at 1.0.
    Returns a dict keyed identically to `pvals`, in its original order."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    running_max = 0.0
    corrected: dict[str, float] = {}
    for i, key in enumerate(order):
        val = min((m - i) * pvals[key], 1.0)
        running_max = max(running_max, val)
        corrected[key] = running_max
    return {key: corrected[key] for key in pvals}
