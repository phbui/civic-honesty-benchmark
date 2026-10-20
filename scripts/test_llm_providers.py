"""Tests for llm_providers.py -- the Anthropic-shaped facade over OpenAI-
compatible endpoints that lets run_prompted_eval.py run other model families
through the unchanged PromptedPolicy / run_notool code.

Discrimination: `test_model_slug_keeps_full_id_for_non_anthropic` fails against
the legacy `args.model.split("-2")[0]` rule it replaces (which turns
`gpt-4o-2024-08-06` into `gpt-4o` and lets `meta-llama/...` create a
subdirectory); the translation and nasty-branch tests fail if the adapter
returns the raw OpenAI shape or crashes on malformed arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # sibling-import convention, see test_pipeline.py:31

from confidence import ParseFailure, confidence_schema, parse_confidence_report  # noqa: E402
from llm_providers import (  # noqa: E402
    FakeOpenAIClient,
    OpenAICompatMessages,
    PatchKwargs,
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    host_slug,
    model_slug,
    sanitize_schema,
)

TOOLS = [
    {
        "name": "query",
        "description": "q",
        "input_schema": {
            "type": "object",
            "properties": {"soql": {"type": "string"}},
            "required": ["soql"],
        },
    },
    {
        "name": "describe_dataset",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    },
    {"name": "report", "description": "r", "input_schema": confidence_schema()},
]
TOOL_NAMES = ("query", "describe_dataset", "report")


# --- naming -----------------------------------------------------------------


def test_model_slug_keeps_legacy_rule_for_anthropic() -> None:
    # The published results/prompted_eval/ filenames depend on this exact truncation.
    assert model_slug("claude-haiku-4-5-20251001", "anthropic") == "claude-haiku-4-5"
    assert model_slug("claude-haiku-4-5-20251001", "fake") == "claude-haiku-4-5"


def test_model_slug_keeps_full_id_for_non_anthropic() -> None:
    assert model_slug("gpt-4o-2024-08-06", "openai") == "gpt-4o-2024-08-06"
    assert (
        model_slug("meta-llama/Llama-3.3-70B-Instruct", "deepinfra")
        == "meta-llama-Llama-3.3-70B-Instruct"
    )
    assert "/" not in model_slug("a/b:c d", "openai-compat")


def test_host_slug() -> None:
    assert host_slug("https://api.groq.com/openai/v1") == "api.groq.com"
    assert host_slug("http://localhost:8000/v1") == "localhost-8000"


# --- schema dialects ----------------------------------------------------------


def test_openai_dialect_is_passthrough() -> None:
    s = confidence_schema()
    assert sanitize_schema(s, "openai") == s
    assert sanitize_schema(s, "openai") is not s  # a copy, never the shared dict


def test_gemini_dialect_rewrites_unions_and_drops_additional_properties() -> None:
    g = sanitize_schema(confidence_schema(), "gemini")
    assert "additionalProperties" not in g
    for key in ("answer", "uncertainty_statement", "rationale"):
        prop = g["properties"][key]
        assert prop["type"] == "string"
        assert prop["nullable"] is True
    assert g["properties"]["confidence"]["type"] == "number"
    assert g["properties"]["action"]["enum"] == ["answer", "abstain"]
    assert g["required"] == confidence_schema()["required"]


def test_tools_translate_to_openai_function_shape() -> None:
    out = anthropic_tools_to_openai(TOOLS, "openai")
    assert [t["type"] for t in out] == ["function"] * 3
    assert [t["function"]["name"] for t in out] == list(TOOL_NAMES)
    assert out[2]["function"]["parameters"] == confidence_schema()


# --- message translation -----------------------------------------------------


def test_messages_translate_system_tool_use_and_tool_result() -> None:
    history = [
        {"role": "user", "content": "How many rows?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "query",
                    "input": {"soql": "select count(*)"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "[{...}]"}],
        },
    ]
    out = anthropic_messages_to_openai("SYS", history)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "How many rows?"}
    assert out[2]["role"] == "assistant"
    assert out[2]["content"] == "Let me check."
    call = out[2]["tool_calls"][0]
    assert call["id"] == "c1" and call["type"] == "function"
    assert call["function"]["name"] == "query"
    assert json.loads(call["function"]["arguments"]) == {"soql": "select count(*)"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "[{...}]"}


def test_call_extras_are_echoed_on_the_assistant_tool_call() -> None:
    history = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c9", "name": "query", "input": {}}],
        }
    ]
    extras = {"c9": {"extra_content": {"google": {"thought_signature": "abc"}}}}
    out = anthropic_messages_to_openai(None, history, extras)
    assert out[0]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "abc"}}
    # Unknown ids get nothing extra.
    assert (
        "extra_content" not in anthropic_messages_to_openai(None, history, {})[0]["tool_calls"][0]
    )


def test_nasty_fake_signature_round_trips_into_the_next_request() -> None:
    ad, client = _adapter("nasty")
    msgs: list[dict] = [{"role": "user", "content": "q"}]
    r1 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    b = r1.content[0]
    # Mimic PromptedPolicy: it stores only {type,id,name,input} and never sees extra_content.
    msgs.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}],
        }
    )
    msgs.append(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id, "content": "ok"}]}
    )
    ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    sent = client.requests[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "sig-fake-1"}
    }
    assert ad.stats.call_extras_carried == 1


def test_patch_kwargs_drops_temperature_and_adds_fixed_fields() -> None:
    seen: dict = {}

    class Inner:
        def create(self, **kw: object) -> str:
            seen.update(kw)
            return "resp"

    shim = PatchKwargs(Inner(), drop=("temperature",), add={"thinking": {"type": "disabled"}})
    assert shim.create(model="m", temperature=0.0, max_tokens=5) == "resp"
    assert seen == {"model": "m", "max_tokens": 5, "thinking": {"type": "disabled"}}


def test_assistant_tool_use_only_has_null_content() -> None:
    out = anthropic_messages_to_openai(
        None,
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "report", "input": {}}],
            }
        ],
    )
    assert out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"


# --- adapter: request side -----------------------------------------------------


def _adapter(mode: str = "clean", **kw: object) -> tuple[OpenAICompatMessages, FakeOpenAIClient]:
    client = FakeOpenAIClient(mode)
    return OpenAICompatMessages(client=client, known_tool_names=TOOL_NAMES, **kw), client


def test_request_carries_tools_tool_choice_and_no_parallel_flag() -> None:
    ad, client = _adapter()
    ad.create(
        model="m",
        max_tokens=700,
        temperature=0.0,
        system="S",
        tools=TOOLS,
        messages=[{"role": "user", "content": "q"}],
    )
    req = client.requests[0]
    assert req["model"] == "m" and req["max_tokens"] == 700 and req["temperature"] == 0.0
    assert req["messages"][0] == {"role": "system", "content": "S"}
    assert req["tool_choice"] == "auto" and req["parallel_tool_calls"] is False
    assert [t["function"]["name"] for t in req["tools"]] == list(TOOL_NAMES)


def test_gemini_dialect_omits_parallel_flag_and_sanitizes_schema() -> None:
    ad, client = _adapter(schema_dialect="gemini")
    ad.create(
        model="m",
        max_tokens=1,
        temperature=0.0,
        tools=TOOLS,
        messages=[{"role": "user", "content": "q"}],
    )
    req = client.requests[0]
    assert "parallel_tool_calls" not in req
    assert "additionalProperties" not in req["tools"][2]["function"]["parameters"]


def test_omit_temperature_max_completion_tokens_and_extra() -> None:
    ad, client = _adapter(
        omit_temperature=True, use_max_completion_tokens=True, extra={"reasoning_effort": "minimal"}
    )
    ad.create(model="m", max_tokens=5, temperature=0.0, messages=[{"role": "user", "content": "q"}])
    req = client.requests[0]
    assert "temperature" not in req and "max_tokens" not in req
    assert req["max_completion_tokens"] == 5 and req["reasoning_effort"] == "minimal"


# --- adapter: response side ------------------------------------------------------


def test_clean_script_returns_anthropic_shape() -> None:
    ad, _ = _adapter()
    msgs = [{"role": "user", "content": "q"}]
    r1 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    assert r1.stop_reason == "tool_use"
    (b,) = r1.content
    assert (b.type, b.id, b.name, b.input) == (
        "tool_use",
        "fake-1",
        "query",
        {"soql": "select count(*) as result"},
    )
    r2 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    assert r2.content[0].name == "report"
    assert (
        parse_confidence_report(json.dumps(r2.content[0].input), question_class="answerable")
        is not None
    )
    r3 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    assert (
        r3.stop_reason == "end_turn"
        and r3.content[0].type == "text"
        and r3.content[0].text == "DONE"
    )
    assert r3.usage.input_tokens == 0 and r3.usage.output_tokens == 0
    assert ad.stats.to_dict() == {
        "requests": 3,
        "retries": 0,
        "dropped_parallel_calls": 0,
        "malformed_arguments": 0,
        "unknown_tool_names": 0,
        "synthesized_ids": 0,
        "call_extras_carried": 0,
        "recovered_calls": 0,
        "rate_limit_waits": 0,
        "rate_limit_wait_seconds": 0.0,
    }


def test_nasty_script_drops_parallel_counts_malformed_and_synthesizes_ids() -> None:
    ad, _ = _adapter("nasty")
    msgs = [{"role": "user", "content": "q"}]
    r1 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    assert len(r1.content) == 1 and r1.content[0].name == "query"  # second call dropped
    r2 = ad.create(model="m", max_tokens=1, temperature=0.0, tools=TOOLS, messages=msgs)
    (b,) = r2.content
    assert b.name == "report" and b.id.startswith("call_synth_")
    assert "__malformed_arguments__" in b.input
    # The malformed payload must surface as a recorded ParseFailure, never a crash or a default.
    parsed = parse_confidence_report(json.dumps(b.input), question_class="answerable")
    assert isinstance(parsed, ParseFailure)
    s = ad.stats.to_dict()
    assert (s["dropped_parallel_calls"], s["malformed_arguments"], s["synthesized_ids"]) == (
        1,
        1,
        1,
    )


def test_notool_path_returns_text_block() -> None:
    ad, _ = _adapter()
    r = ad.create(
        model="m",
        max_tokens=1,
        temperature=0.0,
        system="S",
        messages=[{"role": "user", "content": "q"}],
    )
    assert r.stop_reason == "end_turn"
    assert json.loads(r.content[0].text)["action"] == "abstain"


def test_unknown_tool_name_is_counted_not_raised() -> None:
    ad, client = _adapter()
    client.chat.completions.create = lambda **kw: FakeOpenAIClient._resp(
        None, [FakeOpenAIClient._call("z", "hallucinated_tool", "{}")], "tool_calls"
    )
    r = ad.create(
        model="m",
        max_tokens=1,
        temperature=0.0,
        tools=TOOLS,
        messages=[{"role": "user", "content": "q"}],
    )
    assert r.content[0].name == "hallucinated_tool"
    assert ad.stats.unknown_tool_names == 1


@pytest.mark.parametrize(
    "finish,expected",
    [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "content_filter")],
)
def test_finish_reason_mapping(finish: str, expected: str) -> None:
    ad, client = _adapter()
    client.chat.completions.create = lambda **kw: FakeOpenAIClient._resp("t", None, finish)
    r = ad.create(
        model="m", max_tokens=1, temperature=0.0, messages=[{"role": "user", "content": "q"}]
    )
    assert r.stop_reason == expected


# --- exploratory schema variant (2026-10-17) -----------------------------------


def test_null_count_schema_client_puts_counts_first_and_within_summary_cap() -> None:
    from agent_env import _summarize
    from run_prompted_eval import NullCountSchemaClient

    class Inner:
        def describe_dataset(self, dataset_id: str, **kw: object) -> dict:
            return {"dataset_id": dataset_id, "columns": [{"field_name": "x"}] * 50}

        def query(self, dataset_id: str, soql: str, **kw: object) -> tuple:
            return ([], False)

    rates = {
        "fields": {
            "road_type": {"n_total": 100, "n_nonnull": 9, "null_rate": 0.91},
            "boroughname": {"n_total": 100, "n_nonnull": 99, "null_rate": 0.01},
        }
    }
    c = NullCountSchemaClient(Inner(), rates)
    out = c.describe_dataset("6yyb-pb25")
    assert list(out)[0] == "null_counts_for_sparse_fields"
    assert out["null_counts_for_sparse_fields"] == {
        "road_type": {"null_rate": 0.91, "n_null": 91, "n_total": 100}
    }
    # The 300-character summary the model actually sees still carries the number.
    assert '"road_type": {"null_rate": 0.91, "n_null": 91' in _summarize(out)
    assert c.query("d", "select 1") == ([], False)



def test_result_max_chars_override_rebinds_summarize(monkeypatch):
    """--result-max-chars rebinds agent_env.RESULT_MAX_CHARS, and _summarize honours it (2026-10-19)."""
    import agent_env
    monkeypatch.setattr(agent_env, "RESULT_MAX_CHARS", 50)
    big = {"rows": [{"k": "x" * 200}]}
    out = agent_env._summarize(big)
    assert out.endswith("...<truncated>") and len(out) == 50 + len("...<truncated>")
    monkeypatch.setattr(agent_env, "RESULT_MAX_CHARS", 4000)
    assert agent_env._summarize(big) == __import__("json").dumps(big, default=str)



class _RateLimitError(Exception):
    status_code = 429


_RateLimitError.__name__ = "RateLimitError"


def test_stated_wait_parses_seconds_and_hours():
    from llm_providers import _stated_wait_seconds
    assert abs(_stated_wait_seconds(_RateLimitError("Please try again in 1.605s.")) - 2.605) < 1e-6
    assert abs(_stated_wait_seconds(_RateLimitError("Please try again in 9h53m16.799s.")) - (9 * 3600 + 53 * 60 + 16.799 + 1)) < 1e-6
    assert abs(_stated_wait_seconds(_RateLimitError("try again in 7m12s")) - (7 * 60 + 12 + 1)) < 1e-6
    assert _stated_wait_seconds(_RateLimitError("slow down")) is None
    assert _stated_wait_seconds(_RateLimitError("try again in 400h")) == 26 * 3600.0


def test_patient_mode_sleeps_the_stated_wait_then_succeeds(monkeypatch):
    import llm_providers as L

    calls = {"n": 0}

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise _RateLimitError("Rate limit. Please try again in 2s.")
                    return "OK"

    slept = []
    monkeypatch.setattr(L.time, "sleep", lambda x: slept.append(x))
    a = L.OpenAICompatMessages(client=_C, patient_rate_limits=True)
    assert a._request(model="m") == "OK"
    assert slept == [3.0, 3.0] and a.stats.rate_limit_waits == 2 and a.stats.retries == 0
    # default mode still raises after max_retries and uses the old backoff
    calls["n"] = -10
    b = L.OpenAICompatMessages(client=_C, patient_rate_limits=False, max_retries=1)
    slept.clear()
    try:
        b._request(model="m")
        raise AssertionError("should have raised")
    except _RateLimitError:
        pass
    assert slept == [2.0]


def test_min_interval_gates_request_starts(monkeypatch):
    import llm_providers as L

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return "OK"

    t = {"now": 100.0}
    slept = []
    monkeypatch.setattr(L.time, "monotonic", lambda: t["now"])
    monkeypatch.setattr(L.time, "sleep", lambda x: (slept.append(round(x, 3)), t.__setitem__("now", t["now"] + x)))
    a = L.OpenAICompatMessages(client=_C, min_interval_s=10.0)
    a._request(model="m"); a._request(model="m"); a._request(model="m")
    assert slept == [10.0, 10.0]



def test_describe_dataset_arg_is_opt_in(monkeypatch):
    """The default tool spec keeps describe_dataset argument-free; the opt-in adds an optional dataset_id."""
    import run_prompted_eval as R

    def props():
        return next(t for t in R._tools_spec() if t["name"] == "describe_dataset")["input_schema"]["properties"]

    monkeypatch.setattr(R, "DESCRIBE_DATASET_ARG", False)
    assert props() == {}
    monkeypatch.setattr(R, "DESCRIBE_DATASET_ARG", True)
    assert list(props()) == ["dataset_id"]



def test_describe_dataset_arg_reaches_the_policy_through_main(tmp_path, monkeypatch):
    """The override must rebind the module global that _tools_spec reads. A local assignment in main()
    would stamp true in the summary and leave the tool spec unchanged (found 2026-10-19)."""
    import run_prompted_eval as R

    if not (R.RESULTS_DIR / "questions_canonical.jsonl").exists():
        pytest.skip("needs the released snapshot; see the README quickstart")
    seen = {}
    orig = R._tools_spec

    def spy():
        spec = orig()
        seen["props"] = next(t for t in spec if t["name"] == "describe_dataset")["input_schema"]["properties"]
        return spec

    monkeypatch.setattr(R, "_tools_spec", spy)
    for flag in (True, False):
        seen.clear()
        argv = ["x", "--provider", "fake-openai", "--conditions", "neutral", "--limit", "1", "--workers", "1",
                "--out-dir", str(tmp_path / f"o{flag}")] + (["--describe-dataset-arg"] if flag else [])
        monkeypatch.setattr(R.sys, "argv", argv)
        try:
            R.main()
        except SystemExit:
            pass
        assert (list(seen["props"]) == ["dataset_id"]) is flag
    monkeypatch.setattr(R, "DESCRIBE_DATASET_ARG", False)



def _msg(content, reasoning, tool_calls=()):
    from types import SimpleNamespace as NS
    m = NS(content=content, reasoning_content=reasoning, tool_calls=list(tool_calls))
    resp = NS(choices=[NS(message=m, finish_reason="stop")], usage=NS(prompt_tokens=1, completion_tokens=1))

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return resp

    return _C


def test_recover_reasoning_calls_is_conservative_and_counted():
    import llm_providers as L

    names = ("query", "describe_dataset", "report")

    def run(content, reasoning, flag=True):
        a = L.OpenAICompatMessages(client=_msg(content, reasoning), recover_reasoning_calls=flag, known_tool_names=names)
        r = a.create(model="m", max_tokens=1, temperature=0.0, messages=[{"role": "user", "content": "q"}])
        return a, r

    a, r = run("", '{}')
    assert r.stop_reason == "tool_use" and r.content[0].name == "describe_dataset" and a.stats.recovered_calls == 1
    a, r = run("", '{"dataset_id":"6yyb-pb25"}')
    assert r.content[0].name == "describe_dataset"
    a, r = run("", '{"soql":"select 1"}')
    assert r.content[0].name == "query" and r.content[0].input == {"soql": "select 1"}
    a, r = run("", '{"action":"answer","answer":"5","confidence":0.9,"uncertainty_statement":null}')
    assert r.content[0].name == "report"
    # ambiguous or not a call: left alone
    for reasoning in ('We need to describe the dataset.', '{"foo":1}', '[1]', '{"soql": 3}', None):
        a, r = run("", reasoning)
        assert not r.content and r.stop_reason == "end_turn" and a.stats.recovered_calls == 0, reasoning
    # visible text wins; flag off recovers nothing
    a, r = run("DONE", "{}")
    assert r.content[0].type == "text" and a.stats.recovered_calls == 0
    a, r = run("", "{}", flag=False)
    assert not r.content and a.stats.recovered_calls == 0



def test_replay_block_carries_thinking_blocks_unchanged():
    from types import SimpleNamespace as NS
    import run_prompted_eval as R
    th = NS(type="thinking", thinking="", signature="sig==")
    assert R._replay_block(th) == {"type": "thinking", "thinking": "", "signature": "sig=="}
    red = NS(type="redacted_thinking", data="blob")
    assert R._replay_block(red) == {"type": "redacted_thinking", "data": "blob"}
    assert R._replay_block(NS(type="text", text="DONE")) == {"type": "text", "text": "DONE"}
    tu = NS(type="tool_use", id="t1", name="query", input={"soql": "select 1"})
    assert R._replay_block(tu) == {"type": "tool_use", "id": "t1", "name": "query", "input": {"soql": "select 1"}}
