"""socrata.py — thin, dependency-light Socrata SODA client for the civic-honesty
benchmark's data layer.

WHY THIS EXISTS. The benchmark's ground-truth requirement is that every
answerable/unreliable question's gold
answer be "the exact answer computed directly against the API" — so the data layer
needs a client that (a) actually hits the live SODA API, (b) caches responses on disk
keyed by a hash of the exact query so repeated pipeline runs are free and reproducible,
(c) is polite about rate limits, (d) retries transient 5xx failures with backoff, and
(e) supports an app-token hook via env var without requiring one (NYC Open Data
allows unauthenticated access — see RATE LIMITS below).

STDLIB ONLY. `pyproject.toml` at repo root does not list `requests` (or `httpx`) among
its dependencies, and the parent project's dependencies are all RL/ML packages. Rather
than add an HTTP dependency for this pilot, this client uses `urllib.request` — the
SODA API is plain JSON-over-HTTPS and needs nothing more. If a downstream agent later
wants connection pooling / async, swapping to `httpx` is a contained change confined to
this file. No dependency was added for this deliverable.

RATE LIMITS — VERIFIED 2026-07-31 against https://dev.socrata.com/docs/app-tokens.html
(WebFetch quote, live page):
  - WITH an app token: "Currently we do not throttle API requests that are using an
    application token, unless those requests are determined to be abusive or
    malicious." No published numeric ceiling for the token case either.
  - WITHOUT an app token: "IP addresses that make too many requests during a given
    period may be subject to throttling" — tracked by source IP, shared pool. The page
    does NOT publish an exact requests-per-hour number for the unauthenticated case
    (a WebSearch summary claimed "1,000 requests per rolling hour" for the app-token
    case, but that exact figure could not be found on the current live page and is
    NOT relied on here — treat it as [UNVERIFIED] and do not cite it downstream).
  Because no exact unauthenticated ceiling is published, this client defaults to a
  conservative fixed inter-request interval (MIN_INTERVAL_S_DEFAULT, ~2 req/s)
  regardless of whether a token is configured, rather than assuming a specific quota.

APP TOKEN HOOK. Set the `SOCRATA_APP_TOKEN` environment variable; if unset, requests
are sent unauthenticated (still works against data.cityofnewyork.us, just subject to
the unauthenticated IP-throttling above). No token is required to run this pipeline.

CACHE. Every GET is cached at `<cache_dir>/<sha256(url)>.json` as
`{"url", "fetched_at", "body"}`. Cache is keyed on the full resolved URL (domain +
path + query string, including the SoQL `$query` param), so two different SoQL
queries never collide and the same query never re-hits the network. Pass
`force_refresh=True` to bypass the cache for a single call (used by groundtruth.py's
drift detection, which must see live data, not a stale cache entry).

Run standalone for a live smoke test against 6yyb-pb25 (NYC Street Pavement Ratings):
  uv run python scripts/socrata.py --describe
  uv run python scripts/socrata.py --query "select count(*) as result"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DOMAIN = "data.cityofnewyork.us"
HERE = Path(__file__).parent
DEFAULT_CACHE_DIR = HERE.parent / ".cache" / "socrata"

APP_TOKEN_ENV_VAR = "SOCRATA_APP_TOKEN"

# See module docstring "RATE LIMITS" for the citation behind these defaults.
MIN_INTERVAL_S_DEFAULT = 0.5  # polite fixed pacing; no published exact quota to target
MAX_RETRIES_DEFAULT = 4
BACKOFF_BASE_S = 1.5
TIMEOUT_S_DEFAULT = 30.0


class SocrataError(RuntimeError):
    """Raised for any non-recoverable SODA API failure (HTTP error, malformed
    response, exhausted retries). Never swallowed silently — the whole point of this
    benchmark is that fabricated/guessed data is worse than an explicit failure."""


@dataclass
class SocrataClient:
    domain: str = DEFAULT_DOMAIN
    app_token: str | None = None
    cache_dir: Path = DEFAULT_CACHE_DIR
    min_interval_s: float = MIN_INTERVAL_S_DEFAULT
    max_retries: int = MAX_RETRIES_DEFAULT
    timeout_s: float = TIMEOUT_S_DEFAULT
    _last_request_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.app_token is None:
            self.app_token = os.environ.get(APP_TOKEN_ENV_VAR) or None
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # low-level GET with cache + rate limit + retry/backoff
    # ------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def _get(
        self, path: str, params: dict[str, Any] | None = None, *, force_refresh: bool = False
    ) -> tuple[Any, bool]:
        """Returns (parsed_json_body, cache_hit)."""
        params = dict(params or {})
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        url = f"https://{self.domain}{path}" + (f"?{query}" if query else "")
        cache_path = self._cache_path(url)

        if not force_refresh and cache_path.exists():
            payload = json.loads(cache_path.read_text())
            return payload["body"], True

        headers = {"Accept": "application/json"}
        if self.app_token:
            headers["X-App-Token"] = self.app_token

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            req = urllib.request.Request(url, headers=headers)
            self._last_request_ts = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                body = json.loads(raw)
                cache_path.write_text(
                    json.dumps({"url": url, "fetched_at": time.time(), "body": body})
                )
                return body, False
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code >= 500 and attempt < self.max_retries:
                    sleep_s = BACKOFF_BASE_S * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                detail = e.read().decode("utf-8", errors="replace")[:500]
                raise SocrataError(f"GET {url} failed: HTTP {e.code} {e.reason} — {detail}") from e
            except urllib.error.URLError as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(BACKOFF_BASE_S * (2**attempt))
                    continue
                raise SocrataError(f"GET {url} failed: {e.reason}") from e
        raise SocrataError(f"GET {url} exhausted {self.max_retries} retries") from last_exc

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def query(
        self, dataset_id: str, soql: str, *, force_refresh: bool = False
    ) -> tuple[list[dict[str, Any]], bool]:
        """Execute a SoQL query against `dataset_id` via the `$query` parameter.
        Returns (rows, cache_hit). Raises SocrataError on any non-2xx response or a
        response body that isn't a JSON list (the shape every SODA row-query
        endpoint returns)."""
        body, cache_hit = self._get(
            f"/resource/{dataset_id}.json", {"$query": soql}, force_refresh=force_refresh
        )
        if not isinstance(body, list):
            raise SocrataError(f"unexpected SODA response shape for query {soql!r}: {type(body)}")
        return body, cache_hit

    def describe_dataset(self, dataset_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch the dataset's column schema via the Views metadata endpoint. This is
        the source of truth the unanswerable-question class is built against: a field
        name must be absent from `field_names` here before questions.py is allowed to
        ask about it."""
        body, cache_hit = self._get(f"/api/views/{dataset_id}.json", force_refresh=force_refresh)
        columns = [
            {
                "field_name": c["fieldName"],
                "display_name": c.get("name"),
                "data_type": c.get("dataTypeName"),
            }
            for c in body.get("columns", [])
        ]
        return {
            "dataset_id": dataset_id,
            "domain": self.domain,
            "name": body.get("name"),
            "columns": columns,
            "field_names": sorted({c["field_name"] for c in columns}),
            "cache_hit": cache_hit,
        }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--dataset", default="6yyb-pb25", help="Socrata 4x4 dataset id")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--describe", action="store_true", help="print describe_dataset() schema")
    ap.add_argument("--query", help="run a SoQL query and print the rows")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    client = SocrataClient(domain=args.domain)
    token_state = "set" if client.app_token else "UNSET (unauthenticated)"
    print(f"SOCRATA_APP_TOKEN: {token_state}")

    if args.describe:
        schema = client.describe_dataset(args.dataset, force_refresh=args.force_refresh)
        print(json.dumps(schema, indent=2))
    if args.query:
        rows, cache_hit = client.query(args.dataset, args.query, force_refresh=args.force_refresh)
        print(f"cache_hit={cache_hit} rows={len(rows)}")
        print(json.dumps(rows[:10], indent=2))
    if not args.describe and not args.query:
        ap.print_help()


if __name__ == "__main__":
    _cli()
