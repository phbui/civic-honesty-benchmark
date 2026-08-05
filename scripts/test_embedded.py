"""Collection shim for the tests embedded in `rewards.py` and `confidence.py`.

Those two modules keep their tests inline (they are self-contained scripts, so
the properness proofs sit next to the rule they constrain). pytest's default
discovery pattern is `test_*.py`/`*_test.py`, so it never collected them:
`uv run pytest scripts/` reported 27 passing tests while 24 more
existed and were silently skipped -- exactly the kind of quiet gap that lets a
broken scoring rule ship green.

Star-importing here re-exports those `test_*` functions into a module pytest
DOES collect, without moving them away from the code they constrain and
without widening `python_files` repo-wide.

If you add a test to either module, it is picked up here automatically. The
counts below are asserted so that deleting or renaming a test is a visible
failure rather than a silent reduction in coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import confidence as _confidence  # noqa: E402
import rewards as _rewards  # noqa: E402
from confidence import *  # noqa: E402, F403
from rewards import *  # noqa: E402, F403

EXPECTED_REWARD_TESTS = 13
EXPECTED_CONFIDENCE_TESTS = 11


def _count_tests(module: object) -> int:
    return sum(1 for name in dir(module) if name.startswith("test_"))


def test_all_embedded_reward_tests_are_collected() -> None:
    assert _count_tests(_rewards) == EXPECTED_REWARD_TESTS, (
        "rewards.py's embedded test count changed; update EXPECTED_REWARD_TESTS "
        "deliberately rather than letting collection drift silently"
    )


def test_all_embedded_confidence_tests_are_collected() -> None:
    assert _count_tests(_confidence) == EXPECTED_CONFIDENCE_TESTS, (
        "confidence.py's embedded test count changed; update "
        "EXPECTED_CONFIDENCE_TESTS deliberately rather than letting collection "
        "drift silently"
    )
