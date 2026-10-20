"""llm_providers.py -- provider adapters for run_prompted_eval.py.

WHY THIS FILE EXISTS. The published Haiku 4.5 run drove `PromptedPolicy` and
`run_notool` through the Anthropic Messages `.create(**kwargs)` surface and
consumed Anthropic-shaped responses (`.content` blocks of type text/tool_use,
`.stop_reason`, `.usage.input_tokens/.output_tokens`). The paper's stated
limitation is that it is a single-model-family study. Rather than fork the
episode driver per vendor (which would make the second family's episodes
non-comparable by construction), every other vendor is reached through an
adapter that presents the SAME `.create(**kwargs)` surface and returns the
SAME response shape. `PromptedPolicy` and `run_notool` are therefore byte-for-
byte unchanged between families; only the transport differs.

ONE ADAPTER, MANY VENDORS. `OpenAICompatMessages` speaks the OpenAI chat-
completions dialect, which OpenAI, Groq, DeepInfra, Together, OpenRouter,
Gemini (its `/openai/` compatibility endpoint), vLLM and Ollama all serve.
`KNOWN_BASE_URLS` maps a provider name to its endpoint; `--base-url` overrides.

TRANSLATION RULES (Anthropic request -> OpenAI request):
  system=<str>                      -> leading {"role": "system"} message
  user content str                  -> {"role": "user", "content": str}
  user content [tool_result ...]    -> one {"role": "tool", "tool_call_id", "content"} per block
  assistant [text | tool_use ...]   -> {"role": "assistant", "content": text-or-None,
                                        "tool_calls": [{id, type: function,
                                        function: {name, arguments: <JSON string>}}]}
  tools [{name, description, input_schema}]
                                    -> [{"type": "function", "function": {name,
                                        description, parameters}}]
  tool_choice {"type": "auto", "disable_parallel_tool_use": True}
                                    -> tool_choice="auto", parallel_tool_calls=False
                                       (omitted for dialects that reject the flag)

TRANSLATION RULES (OpenAI response -> Anthropic response):
  message.content                   -> {"type": "text", "text": ...} block (if non-empty)
  message.tool_calls[0]             -> {"type": "tool_use", "id", "name", "input": <parsed JSON>}
  message.tool_calls[1:]            -> DROPPED and counted (`stats.dropped_parallel_calls`).
                                       agent_env consumes exactly one action per turn; a
                                       second call would sit in the transcript with no
                                       tool_result after it. Servers that ignore
                                       parallel_tool_calls=False land here.
  unparseable function.arguments    -> input={"__malformed_arguments__": <raw string>}
                                       and counted. For `report` this flows into
                                       parse_confidence_report as an unknown key, i.e. a
                                       recorded ParseFailure -- never a crash, never a
                                       silently defaulted confidence (confidence.py).
  finish_reason stop / length / tool_calls
                                    -> stop_reason end_turn / max_tokens / tool_use
  usage.prompt_tokens / completion_tokens
                                    -> usage.input_tokens / output_tokens
  tool_call.id missing (Gemini compat sometimes omits ids)
                                    -> synthesized `call_<n>` so the next request's
                                       tool message can reference it.

SCHEMA DIALECTS. `confidence_schema()` uses JSON-Schema type unions
(`["string", "null"]`) and `additionalProperties: false`. Gemini's function-
declaration schema rejects both; `schema_dialect="gemini"` rewrites unions to
the non-null type plus `nullable: true` and drops `additionalProperties`. The
OpenAI dialect passes the schema through untouched.

REASONING MODELS. Some models spend the per-turn token budget on hidden
reasoning or reject `temperature`. The adapter exposes `omit_temperature` and an
`extra` dict merged into every request (e.g. {"reasoning_effort": "minimal"}).
Both are stamped into the run summary by run_prompted_eval.py so any
comparability caveat against the Haiku run (700 tokens, temperature 0) is on
the record, not in someone's memory.

FAKES. `FakeOpenAIClient` mimics `openai.OpenAI().chat.completions.create` with
a deterministic script, in two modes: "clean" mirrors run_prompted_eval's
`FakeMessages` move for move (so a `--provider fake-openai` run must produce
the same flat rows as `--provider fake`), and "nasty" emits two parallel tool
calls on the first turn and malformed `report` arguments on the second, so the
drop-and-count and ParseFailure branches are exercised at zero cost before any
paid call. Dry-run discipline: the exact command that will spend money is run
first against one of these.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

KNOWN_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

OPENAI_COMPAT_PROVIDERS = tuple(KNOWN_BASE_URLS) + ("openai-compat",)
FAKE_PROVIDERS = ("fake", "fake-openai", "fake-openai-nasty")

# Vendors whose compat layer is known or suspected to reject `parallel_tool_calls`.
# The adapter still truncates to one call on the response side, so correctness
# does not depend on the flag being honoured.
_NO_PARALLEL_FLAG_DIALECTS = frozenset({"gemini"})


# ---------------------------------------------------------------------------
# Anthropic-shaped response objects
# ---------------------------------------------------------------------------


class Block:
    """Attribute bag standing in for an Anthropic content block."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __repr__(self) -> str:
        return f"Block({self.__dict__!r})"


class Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = int(input_tokens or 0)
        self.output_tokens = int(output_tokens or 0)


class Response:
    def __init__(self, content: list[Block], stop_reason: str, usage: Usage) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def model_slug(model: str, provider: str) -> str:
    """Filesystem-safe model label for output filenames.

    Anthropic keeps the legacy rule (`split("-2")[0]`, which turns
    `claude-haiku-4-5-20251001` into `claude-haiku-4-5`) so the published
    results/prompted_eval/ filenames and their resume keys are unchanged.
    Every other provider keeps the whole model id with path-hostile characters
    replaced, so `meta-llama/Llama-3.3-70B` cannot create a subdirectory and
    `gpt-4o-2024-08-06` is not truncated to `gpt-4o`.
    """
    if provider in ("anthropic", "fake"):
        return model.split("-2")[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def host_slug(base_url: str) -> str:
    host = re.sub(r"^https?://", "", base_url).split("/")[0]
    return re.sub(r"[^A-Za-z0-9.-]+", "-", host)


# ---------------------------------------------------------------------------
# Schema dialects
# ---------------------------------------------------------------------------


def sanitize_schema(schema: dict, dialect: str) -> dict:
    """Return a copy of an Anthropic `input_schema` acceptable to `dialect`."""
    if dialect == "openai":
        return copy.deepcopy(schema)
    if dialect == "gemini":
        return _gemini_schema(copy.deepcopy(schema))
    raise ValueError(f"unknown schema dialect {dialect!r}")


def _gemini_schema(node: Any) -> Any:  # noqa: ANN401
    if isinstance(node, list):
        return [_gemini_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    node.pop("additionalProperties", None)
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1:
            node["type"] = non_null[0]
        elif non_null:
            # Gemini has no union type; keep the first and record the loss in the description.
            node["type"] = non_null[0]
            accepted = "|".join(non_null)
            node["description"] = f"{node.get('description', '')} (accepts {accepted})".strip()
        else:
            node.pop("type")
        if "null" in t:
            node["nullable"] = True
    return {k: _gemini_schema(v) for k, v in node.items()}


def anthropic_tools_to_openai(tools: list[dict], dialect: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": sanitize_schema(t.get("input_schema", {"type": "object"}), dialect),
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def anthropic_messages_to_openai(
    system: str | None,
    messages: list[dict],
    call_extras: Mapping[str, dict] | None = None,
) -> list[dict]:
    """`call_extras` maps a tool_call id to opaque fields the server attached to
    that call and expects echoed back verbatim on the assistant message (Gemini
    3.x's `extra_content.google.thought_signature`; the API returns 400 without
    it). PromptedPolicy stores only {id, name, input} per tool_use block, so the
    adapter keeps these on the side and re-attaches them here by id."""
    call_extras = call_extras or {}
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "user":
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content", "")
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": c if isinstance(c, str) else json.dumps(c),
                        }
                    )
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b.get("text", "")})
            continue
        # assistant
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [
            {
                "id": b["id"],
                "type": "function",
                "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})},
                **call_extras.get(b["id"], {}),
            }
            for b in content
            if b.get("type") == "tool_use"
        ]
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = calls
        out.append(msg)
    return out


_FINISH_TO_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


@dataclass
class AdapterStats:
    """Counted translation events. Reported in the run summary so a weak tool
    caller's format failures are visible next to its honesty numbers, never
    folded into them."""

    requests: int = 0
    retries: int = 0
    dropped_parallel_calls: int = 0
    malformed_arguments: int = 0
    unknown_tool_names: int = 0
    synthesized_ids: int = 0
    call_extras_carried: int = 0
    rate_limit_waits: int = 0
    recovered_calls: int = 0
    rate_limit_wait_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def to_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class OpenAICompatMessages:
    """Anthropic-Messages-shaped facade over an OpenAI chat-completions client.

    `client` is anything exposing `.chat.completions.create(**kwargs)` and
    returning an object with `.choices[0].message.{content,tool_calls}`,
    `.choices[0].finish_reason`, and `.usage.{prompt_tokens,completion_tokens}`
    -- the real `openai.OpenAI()` or `FakeOpenAIClient` below.
    """

    client: Any
    schema_dialect: str = "openai"
    omit_temperature: bool = False
    use_max_completion_tokens: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    known_tool_names: tuple[str, ...] = ()
    max_retries: int = 4
    backoff_s: float = 2.0
    min_interval_s: float = 0.0
    patient_rate_limits: bool = False
    patient_max_waits: int = 400
    recover_reasoning_calls: bool = False
    _gate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _next_ok_at: float = 0.0
    stats: AdapterStats = field(default_factory=AdapterStats)
    _id_counter: int = 0
    _id_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # tool_call id -> opaque server fields to echo back (see anthropic_messages_to_openai).
    # Ids are unique per server response, so one shared map across worker threads is safe.
    _call_extras: dict[str, dict] = field(default_factory=dict)

    def _next_id(self) -> str:
        with self._id_lock:
            self._id_counter += 1
            return f"call_synth_{self._id_counter}"

    def _gate(self) -> None:
        """Serialize request starts min_interval_s apart across worker threads."""
        if self.min_interval_s <= 0:
            return
        with self._gate_lock:
            now = time.monotonic()
            start = max(now, self._next_ok_at)
            self._next_ok_at = start + self.min_interval_s
        if start > now:
            time.sleep(start - now)

    def _request(self, **kwargs: Any) -> Any:  # noqa: ANN401
        attempt = 0
        waits = 0
        while True:
            self._gate()
            try:
                self.stats.bump("requests")
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classify, then re-raise or retry
                if self.patient_rate_limits and _is_rate_limit(exc) and waits < self.patient_max_waits:
                    wait = _stated_wait_seconds(exc)
                    if wait is not None:
                        waits += 1
                        self.stats.bump("rate_limit_waits")
                        self.stats.bump("rate_limit_wait_seconds", wait)
                        time.sleep(wait)
                        continue
                if attempt >= self.max_retries or not _retryable(exc):
                    raise
                attempt += 1
                self.stats.bump("retries")
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> Response:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages_to_openai(system, messages, self._call_extras),
        }
        budget_key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
        kwargs[budget_key] = max_tokens
        if not self.omit_temperature:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = anthropic_tools_to_openai(tools, self.schema_dialect)
            kwargs["tool_choice"] = "auto"
            if self.schema_dialect not in _NO_PARALLEL_FLAG_DIALECTS:
                kwargs["parallel_tool_calls"] = False
        kwargs.update(self.extra)

        resp = self._request(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        blocks: list[Block] = []
        text = getattr(msg, "content", None)
        if text:
            blocks.append(Block(type="text", text=text))
        calls = list(getattr(msg, "tool_calls", None) or [])
        recovered = None
        if self.recover_reasoning_calls and not calls and not text:
            recovered = self._recover_call(getattr(msg, "reasoning_content", None))
            if recovered is not None:
                blocks.append(recovered)
        if len(calls) > 1:
            self.stats.bump("dropped_parallel_calls", len(calls) - 1)
        blocks.extend(self._tool_use_block(tc) for tc in calls[:1])
        finish = getattr(choice, "finish_reason", None) or "stop"
        stop_reason = "tool_use" if (calls or recovered is not None) else _FINISH_TO_STOP.get(finish, finish)
        u = getattr(resp, "usage", None)
        usage = Usage(
            getattr(u, "prompt_tokens", 0) if u else 0,
            getattr(u, "completion_tokens", 0) if u else 0,
        )
        return Response(blocks, stop_reason, usage)

    def _recover_call(self, reasoning: Any) -> Block | None:  # noqa: ANN401
        """Rebuild a tool call a server filed under reasoning_content (see --recover-reasoning-calls)."""
        if not isinstance(reasoning, str) or not reasoning.strip().startswith("{"):
            return None
        try:
            obj = json.loads(reasoning)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        keys = set(obj)
        report_keys = {"action", "answer", "confidence", "uncertainty_statement"}
        if "soql" in keys and isinstance(obj.get("soql"), str):
            name, args = "query", {"soql": obj["soql"]}
        elif "action" in keys and keys <= report_keys:
            name, args = "report", obj
        elif keys <= {"dataset_id"}:
            name, args = "describe_dataset", obj
        else:
            return None
        if self.known_tool_names and name not in self.known_tool_names:
            return None
        self.stats.bump("recovered_calls")
        self.stats.bump("synthesized_ids")
        return Block(type="tool_use", id=self._next_id(), name=name, input=args)

    def _parse_arguments(self, raw_args: Any) -> dict:  # noqa: ANN401
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):  # some servers already decode
            return raw_args
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            self.stats.bump("malformed_arguments")
            return {"__malformed_arguments__": raw_args}
        return args

    def _tool_use_block(self, tc: Any) -> Block:  # noqa: ANN401
        fn = tc.function
        args = self._parse_arguments(fn.arguments)
        if self.known_tool_names and fn.name not in self.known_tool_names:
            self.stats.bump("unknown_tool_names")
        call_id = getattr(tc, "id", None)
        if not call_id:
            self.stats.bump("synthesized_ids")
            call_id = self._next_id()
        extra = _call_extra_fields(tc)
        if extra:
            self.stats.bump("call_extras_carried")
            with self._id_lock:
                self._call_extras[call_id] = extra
        return Block(type="tool_use", id=call_id, name=fn.name, input=args)


_ECHOED_CALL_FIELDS = ("extra_content",)


def _call_extra_fields(tc: Any) -> dict:  # noqa: ANN401
    """Opaque per-call fields a server wants echoed back. The openai SDK keeps
    unknown response fields in `model_extra`; fakes set them as attributes."""
    out: dict = {}
    model_extra = getattr(tc, "model_extra", None) or {}
    for name in _ECHOED_CALL_FIELDS:
        val = getattr(tc, name, None)
        if val is None:
            val = model_extra.get(name)
        if val is not None:
            out[name] = val if isinstance(val, dict | str | list) else _to_plain(val)
    return out


def _to_plain(obj: Any) -> Any:  # noqa: ANN401
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(obj, SimpleNamespace):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return obj


class PatchKwargs:
    """Pass-through over a Messages-shaped API that removes named keyword
    arguments and merges fixed ones before `.create`. Used for the Anthropic SDK
    >= 1.x, whose `Messages.create` no longer accepts `temperature` (the published
    run passed temperature=0 through anthropic 0.x) and whose 5.x models think by
    default (`thinking={"type": "disabled"}` restores the Haiku 4.5 regime). Both
    the dropped names and the added fields are stamped into the run summary by
    run_prompted_eval.py as comparability caveats."""

    def __init__(
        self,
        inner: Any,  # noqa: ANN401
        drop: tuple[str, ...] = (),
        add: Mapping[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self.drop = drop
        self.add = dict(add or {})

    def create(self, **kwargs: Any) -> Any:  # noqa: ANN401
        for k in self.drop:
            kwargs.pop(k, None)
        kwargs.update(self.add)
        return self._inner.create(**kwargs)


DropKwargs = PatchKwargs  # backward-compatible name


def _is_rate_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "RateLimitError" or getattr(exc, "status_code", None) == 429


_WAIT_RE = re.compile(r"try again in\s+(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?", re.I)


def _stated_wait_seconds(exc: Exception, cap_s: float = 26 * 3600.0) -> float | None:
    """Seconds a 429 asks us to wait ('Please try again in 1.6s' or '9h53m16.8s'), plus one second of
    slack, capped at 26 hours. None when the message states no wait (then the ordinary retry applies)."""
    m = _WAIT_RE.search(str(exc))
    if not m or not any(m.groups()):
        return None
    h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return min(cap_s, h * 3600 + mi * 60 + sec + 1.0)


def _retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504}


def make_openai_compat(
    provider: str,
    *,
    api_key: str | None,
    base_url: str | None,
    schema_dialect: str | None,
    omit_temperature: bool,
    use_max_completion_tokens: bool,
    extra: dict[str, Any] | None,
    known_tool_names: tuple[str, ...],
    timeout_s: float = 120.0,
    min_interval_s: float = 0.0,
    patient_rate_limits: bool = False,
    recover_reasoning_calls: bool = False,
) -> tuple[OpenAICompatMessages, str]:
    """Build the real adapter. Returns (adapter, resolved_base_url)."""
    import openai  # deferred so fake providers need no SDK

    url = base_url or KNOWN_BASE_URLS.get(provider)
    if not url:
        raise SystemExit(f"--provider {provider} requires --base-url")
    dialect = schema_dialect or ("gemini" if provider == "gemini" else "openai")
    client = openai.OpenAI(
        api_key=api_key or "none", base_url=url, timeout=timeout_s, max_retries=0
    )
    adapter = OpenAICompatMessages(
        client=client,
        schema_dialect=dialect,
        omit_temperature=omit_temperature,
        use_max_completion_tokens=use_max_completion_tokens,
        extra=dict(extra or {}),
        known_tool_names=known_tool_names,
        min_interval_s=min_interval_s,
        patient_rate_limits=patient_rate_limits,
        recover_reasoning_calls=recover_reasoning_calls,
    )
    return adapter, url


# ---------------------------------------------------------------------------
# Fakes (zero-cost dry runs of the translation layer)
# ---------------------------------------------------------------------------


class FakeOpenAIClient:
    """Scripted stand-in for `openai.OpenAI()`; see module docstring, FAKES.

    Per-thread step counter, like run_prompted_eval.FakeMessages, so a
    multi-worker dry run stays deterministic per episode.
    """

    def __init__(self, mode: str = "clean") -> None:
        if mode not in ("clean", "nasty"):
            raise ValueError(mode)
        self.mode = mode
        self._n_by_thread: dict[int, int] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.requests: list[dict] = []
        self._req_lock = threading.Lock()

    @staticmethod
    def _resp(content: str | None, tool_calls: list | None, finish: str) -> Any:  # noqa: ANN401
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=tool_calls),
                    finish_reason=finish,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

    @staticmethod
    def _call(cid: str | None, name: str, arguments: str) -> Any:  # noqa: ANN401
        fn = SimpleNamespace(name=name, arguments=arguments)
        return SimpleNamespace(id=cid, type="function", function=fn)

    def _create(self, **kwargs: Any) -> Any:  # noqa: ANN401
        with self._req_lock:
            self.requests.append(kwargs)
        tid = threading.get_ident()
        n = self._n_by_thread.get(tid, 0)
        self._n_by_thread[tid] = n + 1
        has_tools = bool(kwargs.get("tools"))
        report = {
            "action": "abstain",
            "answer": None,
            "confidence": 0.5,
            "uncertainty_statement": None,
        }
        if not has_tools:
            return self._resp(json.dumps(report), None, "stop")
        step = n % 3
        if step == 0:
            soql = json.dumps({"soql": "select count(*) as result"})
            calls = [self._call("fake-1", "query", soql)]
            if self.mode == "nasty":
                # An opaque per-call field the server expects echoed back (Gemini 3.x
                # thought_signature shape), plus a second, parallel call emitted
                # despite parallel_tool_calls=False.
                calls[0].extra_content = {"google": {"thought_signature": "sig-fake-1"}}
                calls.append(self._call("fake-1b", "describe_dataset", "{}"))
            return self._resp(None, calls, "tool_calls")
        if step == 1:
            if self.mode == "nasty":
                # Truncated JSON and no id: the malformed-arguments and synthesized-id branches.
                truncated = '{"action": "abstain", "confidence": 0.5'
                return self._resp(None, [self._call(None, "report", truncated)], "tool_calls")
            return self._resp(
                None, [self._call("fake-2", "report", json.dumps(report))], "tool_calls"
            )
        return self._resp("DONE", None, "stop")
