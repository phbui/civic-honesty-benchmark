"""run_contamination.py -- runnable CLI wiring a real model into
`contamination.run_no_tool_ablation` over the real civic-honesty question set.

Contamination is a real but tractable threat: this script's job is to
"run a no-tool ablation on
every answerable and unreliable item." `contamination.py` (a separate module,
NOT edited here) implements the scoring/reporting half against a synthetic query
function; this script wires the actual model-call seam (`NoToolQueryFn`) it
deliberately leaves unwired.

CREDENTIAL POLICY -- READ BEFORE RUNNING:
  This script NEVER silently falls back to a stub that would produce fake
  "uncontaminated" results. `--provider bedrock` (the default) requires:
    1. `boto3` importable (NOT a pyproject.toml dependency -- run via
       `uv run --with boto3 python scripts/run_contamination.py ...`,
       the same `uv run --with <pkg>` convention `cost_model.py` and this
       directory's README already use for `matplotlib`).
    2. AWS credentials that actually resolve for `--profile` (default
       `admin-dev`) / `--region` (default `us-east-1`) -- checked with a FREE
       `sts:GetCallerIdentity` call before any paid Bedrock call is attempted.
    3. An explicit `--i-know-this-costs-money` flag. This is a real-money
       spend; the flag is a deliberate extra confirmation step, not a
       formality -- omitting it fails loudly with the exact reason, it never
       proceeds "for convenience."
  Any failure in 1-3 raises `SystemExit` with an actionable message. It never
  degrades to a scripted/fake answer under `--provider bedrock`.

  `--provider fake` exists ONLY to prove the wiring end-to-end with zero cost
  and zero network calls (a small deterministic scripted responder). Every
  output row and the top-level report are stamped `"provider": "fake"` in
  that mode so a downstream consumer can never mistake a wiring-proof run for
  a real contamination measurement.

MODEL / REGION DEFAULTS: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` via
Bedrock's inference-profile id form (the bare `anthropic.*` model id form is
reported to 403/ResourceNotFound on this account -- an inference-profile
prefix is required), profile `admin-dev`, region `us-east-1`. These are
configuration defaults only; supplying them does not itself invoke anything.

CACHING / RATE LIMITING: every (model_id, question_id, prompt) triple is
cached to disk under `.cache/contamination/` (sha256-keyed
JSON files, same convention as `socrata.py`'s `.cache/socrata/` --
`.cache/` is already gitignored, see README). A cache hit
skips both the API call and the rate-limit sleep, so a rerun (e.g. after a
crash, or to add more items) is free. Real calls are paced by
`--rate-limit-seconds` (default 1.0s, conservative given no published
numeric Bedrock throttling ceiling was assumed here either -- same posture
`socrata.py`'s module docstring documents for the Socrata API).

Run (wiring proof, zero cost):
  uv run python scripts/run_contamination.py --provider fake

Run (real, a paid call -- one command away):
  uv run --with boto3 python scripts/run_contamination.py \\
      --provider bedrock --i-know-this-costs-money
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
CIVIC_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = CIVIC_DIR / "results"
DEFAULT_CACHE_DIR = CIVIC_DIR / ".cache" / "contamination"
sys.path.insert(0, str(SCRIPTS_DIR))

from agent_env import (  # noqa: E402 -- real sibling modules, imported not edited
    attach_groundtruth,
    load_groundtruth_jsonl,
    load_questions_jsonl,
)
from contamination import (  # noqa: E402
    ContaminationReport,
    default_no_tool_score,
    run_no_tool_ablation,
    summarize_contamination,
)

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_PROFILE = "admin-dev"
DEFAULT_REGION = "us-east-1"

NO_TOOL_SYSTEM_PROMPT = (
    "Answer the following question about NYC Open Data using only what you already "
    "know. You do NOT have access to any tool, database, or web search right now -- "
    "answer directly from memory. If you are not confident, give your best guess "
    "anyway; do not refuse to answer and do not say you lack access. Respond with "
    "just the answer value, no explanation."
)


# ---------------------------------------------------------------------------
# Disk cache -- content-addressed by (model_id, question_id, prompt).
# ---------------------------------------------------------------------------


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key_path(self, model_id: str, question_id: str, prompt: str) -> Path:
        h = hashlib.sha256(f"{model_id}:{question_id}:{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, model_id: str, question_id: str, prompt: str) -> str | None:
        path = self._key_path(model_id, question_id, prompt)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["response_text"]
        self.misses += 1
        return None

    def put(self, model_id: str, question_id: str, prompt: str, response_text: str) -> None:
        path = self._key_path(model_id, question_id, prompt)
        path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "question_id": question_id,
                    "prompt": prompt,
                    "response_text": response_text,
                    "cached_at": time.time(),
                }
            )
        )


# ---------------------------------------------------------------------------
# Credential check -- FREE (sts:GetCallerIdentity), run before any paid call.
# ---------------------------------------------------------------------------


def _check_bedrock_credentials(profile: str, region: str) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(
            "boto3 is not installed in this environment. It is deliberately NOT a "
            "pyproject.toml dependency (this directory's policy: no dependency beyond "
            "stdlib unless a script genuinely needs one, stated inline). Run this "
            "script via:\n"
            "  uv run --with boto3 python scripts/run_contamination.py ...\n"
            f"(original ImportError: {exc})"
        ) from exc

    try:
        session = boto3.Session(profile_name=profile, region_name=region)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
    except Exception as exc:  # noqa: BLE001 -- surface any credential failure loudly
        raise SystemExit(
            f"AWS credentials for profile '{profile}' in region '{region}' did not "
            f"resolve (checked with the FREE sts:GetCallerIdentity call, no Bedrock "
            f"call was attempted). This script REFUSES to fall back to a stub -- fix "
            f"credentials and rerun. Likely fixes:\n"
            f"  aws sso login --profile {profile}\n"
            f"  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    print(
        f"[credentials] resolved OK for profile '{profile}': "
        f"{identity.get('Arn', identity)} (free STS check, no Bedrock call made yet)"
    )


# ---------------------------------------------------------------------------
# NoToolQueryFn implementations.
# ---------------------------------------------------------------------------


class BedrockQueryFn:
    """Real `contamination.NoToolQueryFn` implementation over AWS Bedrock's
    `invoke_model`. Constructing an instance does NOT make any network call;
    only `__call__` does, and only on a cache miss."""

    def __init__(
        self,
        *,
        model_id: str,
        profile: str,
        region: str,
        max_tokens: int,
        rate_limit_seconds: float,
        cache: DiskCache,
    ):
        import boto3  # already verified importable by _check_bedrock_credentials

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.rate_limit_seconds = rate_limit_seconds
        self.cache = cache
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("bedrock-runtime")

    def _prompt_for(self, question: Any) -> str:
        text = getattr(question, "question", None) or (question.raw or {}).get("question", "")
        return f"{NO_TOOL_SYSTEM_PROMPT}\n\nQuestion: {text}"

    def __call__(self, question: Any, /) -> str:
        qid = getattr(question, "id", None) or str(question)
        prompt = self._prompt_for(question)
        cached = self.cache.get(self.model_id, qid, prompt)
        if cached is not None:
            print(f"[bedrock] cache HIT for {qid}")
            return cached

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        payload = json.loads(response["body"].read())
        response_text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        self.cache.put(self.model_id, qid, prompt, response_text)
        time.sleep(self.rate_limit_seconds)
        print(f"[bedrock] LIVE call for {qid} -> {response_text[:80]!r}")
        return response_text


class FakeQueryFn:
    """Scripted, deterministic, zero-network `NoToolQueryFn` -- proves the
    CLI's wiring (loading, joining, ablation, scoring, report-writing) end to
    end with NO paid call and NO real model. Every result this produces is
    stamped provider='fake' downstream; never treat its contamination rate as
    a measurement of anything. Deterministic rule (so the demo is
    reproducible): items whose id hashes even 'know' the gold value (a
    stand-in for 'this item happens to be contaminated'), odd-hashing items
    answer 'I don't know' (a stand-in for 'this item is clean') -- chosen to
    exercise BOTH branches of `default_no_tool_score` in one run, not to
    claim anything about real memorization."""

    def __call__(self, question: Any, /) -> str:
        qid = getattr(question, "id", None) or str(question)
        if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 == 0:
            return str(getattr(question, "gold", None))
        return "I don't have that information."


# ---------------------------------------------------------------------------
# Report serialization.
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, set):
        return sorted(_jsonable(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    return obj


def write_report(
    report: ContaminationReport, *, out_path: Path, provider: str, cache: DiskCache
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "per_class": _jsonable(report.per_class),
        "per_item": _jsonable(report.per_item),
        "contaminated_ids": _jsonable(report.contaminated_ids),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=RESULTS_DIR / "questions.jsonl")
    parser.add_argument("--groundtruth", type=Path, default=RESULTS_DIR / "groundtruth.jsonl")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "contamination_report.json")
    parser.add_argument("--provider", choices=("bedrock", "fake"), default="bedrock")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-items", type=int, default=None, help="limit items processed (smoke test)"
    )
    parser.add_argument(
        "--i-know-this-costs-money",
        action="store_true",
        help="required to actually run --provider bedrock; this is a real-money spend",
    )
    args = parser.parse_args()

    if args.provider == "bedrock" and not args.i_know_this_costs_money:
        raise SystemExit(
            "--provider bedrock makes real, paid AWS Bedrock invoke_model calls "
            f"(model_id={args.model_id!r}). Pass --i-know-this-costs-money to confirm "
            "you intend to spend real money right now. This script will not proceed "
            "without it -- there is no implicit default consent for a paid run."
        )

    questions = load_questions_jsonl(args.questions)
    groundtruth = load_groundtruth_jsonl(args.groundtruth) if args.groundtruth.exists() else {}
    joined = attach_groundtruth(questions, groundtruth)
    if args.max_items is not None:
        joined = joined[: args.max_items]
    print(
        f"Loaded {len(joined)} questions from {args.questions} "
        f"(groundtruth: {len(groundtruth)} rows)"
    )

    cache = DiskCache(args.cache_dir)

    if args.provider == "fake":
        print("=" * 78)
        print("PROVIDER = fake -- THIS IS A WIRING PROOF, NOT A CONTAMINATION MEASUREMENT.")
        print("No network call, no model, no cost. Results are stamped provider='fake'.")
        print("=" * 78)
        query_fn = FakeQueryFn()
    else:
        _check_bedrock_credentials(args.profile, args.region)
        query_fn = BedrockQueryFn(
            model_id=args.model_id,
            profile=args.profile,
            region=args.region,
            max_tokens=args.max_tokens,
            rate_limit_seconds=args.rate_limit_seconds,
            cache=cache,
        )

    results = run_no_tool_ablation(joined, query_fn=query_fn, score_fn=default_no_tool_score)
    report = summarize_contamination(results)

    print()
    print("=" * 78)
    print(f"CONTAMINATION REPORT (provider={args.provider})")
    print("=" * 78)
    for cls, cls_report in report.per_class.items():
        print(
            f"  {cls:14s}: {cls_report.n_contaminated}/{cls_report.n_items} flagged "
            f"({cls_report.contamination_rate:.1%})"
        )
    print(f"  cache: {cache.hits} hits, {cache.misses} misses")

    write_report(report, out_path=args.out, provider=args.provider, cache=cache)


if __name__ == "__main__":
    main()
