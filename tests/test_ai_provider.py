"""The OpenRouter and Ollama calls, against httpx.MockTransport: no network, no key."""
import base64
import json
from typing import List

import httpx
import pytest
from pydantic import BaseModel, ValidationError

import ai_provider as ap


class Answer(BaseModel):
    windows: List[str]
    score: int


JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-bytes"
PNG = b"\x89PNG\r\n\x1a\n" + b"png-bytes"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("AI_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "AI_TIMEOUT", "BILLING_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    ap._model_cache.update(at=0.0, value=None)


def _transport(monkeypatch, handler):
    seen = []

    def recording(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(ap, "_client", lambda timeout: httpx.Client(
        transport=httpx.MockTransport(recording), timeout=timeout))
    return seen


def _chat_answer(content, cost=0.0123, prompt_tokens=1000, completion_tokens=40, finish="stop"):
    return {
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "cost": cost, "completion_tokens_details": {"reasoning_tokens": 0}},
    }


# --- selection and settings -------------------------------------------------------- #

def test_nothing_is_chosen_by_default():
    with pytest.raises(ap.AIUnavailableError, match="Choose a provider"):
        ap.resolve()


def test_a_server_may_name_its_provider_and_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model")
    s = ap.resolve(api_key="sk")
    assert (s.provider, s.model, s.api_key) == ("openrouter", "vendor/model", "sk")
    assert s.ready


def test_a_missing_model_is_never_guessed(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    s = ap.resolve(api_key="sk")
    assert s.model == "" and s.ready is False
    with pytest.raises(ap.AIUnavailableError, match="Choose an OpenRouter model"):
        ap.generate_text("hi", config=s)


def test_the_request_choice_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")
    s = ap.resolve(provider="ollama")
    assert s.provider == "ollama" and s.model == "llava:7b"
    assert s.base_url == ap.OLLAMA_DEFAULT_BASE_URL
    assert s.num_ctx == ap.DEFAULT_NUM_CTX


def test_the_request_model_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "env/model")
    s = ap.resolve(provider="openrouter", api_key="sk", model="chosen/model")
    assert s.model == "chosen/model"


def test_a_model_name_is_checked(monkeypatch):
    with pytest.raises(ap.AIUnavailableError, match="not valid"):
        ap.resolve(provider="openrouter", api_key="sk", model="bad model; rm -rf")


def test_unknown_provider_is_refused():
    with pytest.raises(ap.AIUnavailableError, match="Unknown AI provider"):
        ap.resolve(provider="gemini")


def test_cloud_mode_runs_on_openrouter_with_the_operators_model(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "1")
    monkeypatch.setenv("OPENROUTER_MODEL", "operator/model")
    with pytest.raises(ap.AIUnavailableError, match="OpenRouter only"):
        ap.provider_name("ollama")
    s = ap.resolve(provider="openrouter", api_key="managed", model="user/picked")
    assert s.model == "operator/model"  # cloud mode ignores the user's model choice


def test_cloud_mode_refuses_ollama_from_the_environment(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    with pytest.raises(ap.AIUnavailableError):
        ap.provider_name()


def test_describe_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
    info = ap.describe()
    assert info == {"keyOnServer": True}
    assert "sk-or-secret-value" not in json.dumps(info)


def test_configured_needs_a_model_and_an_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "m/x")
    assert ap.configured("openrouter") is False
    assert ap.configured("openrouter", api_key="sk-request") is True
    assert ap.configured("ollama") is False  # a model is still needed
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")
    assert ap.configured("ollama") is True


# --- OpenRouter JSON ----------------------------------------------------------------- #

def test_openrouter_json_request_carries_schema_images_and_cost(monkeypatch):
    seen = _transport(monkeypatch, lambda request: httpx.Response(
        200, json=_chat_answer(json.dumps({"windows": ["a"], "score": 81}), cost=0.00421)))

    value, cost = ap.generate_json("pick", Answer, images=[JPEG], config=ap.Settings(
        provider="openrouter", model="m/x", api_key="sk-test"))

    assert value == {"windows": ["a"], "score": 81}
    assert cost["total_cost"] == pytest.approx(0.00421)
    assert cost["input_tokens"] == 1000 and cost["output_tokens"] == 40
    assert cost["provider"] == "openrouter" and cost["local"] is False

    request = seen[0]
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "m/x"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Answer"
    assert body["provider"] == {"require_parameters": True}
    content = body["messages"][-1]["content"]
    assert content[0]["type"] == "text" and content[0]["text"] == "pick"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()


def test_openrouter_falls_back_from_json_schema_to_json_object(monkeypatch):
    answers = iter([
        httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}}),
        httpx.Response(200, json=_chat_answer('{"windows": [], "score": 3}')),
    ])
    seen = _transport(monkeypatch, lambda request: next(answers))

    value, _ = ap.generate_json("pick", Answer, config=ap.Settings(
        provider="openrouter", model="m/x", api_key="sk"))

    assert value["score"] == 3
    types = [json.loads(r.content)["response_format"]["type"] for r in seen]
    assert types == ["json_schema", "json_object"]


def test_openrouter_drops_the_format_when_nothing_is_accepted(monkeypatch):
    answers = iter([
        httpx.Response(400, json={"error": {"message": "unsupported response_format"}}),
        httpx.Response(400, json={"error": {"message": "unsupported response_format"}}),
        httpx.Response(200, json=_chat_answer('```json\n{"windows": ["w"], "score": 5}\n```')),
    ])
    seen = _transport(monkeypatch, lambda request: next(answers))

    value, _ = ap.generate_json("pick", Answer, config=ap.Settings(
        provider="openrouter", model="m/x", api_key="sk"))

    assert value == {"windows": ["w"], "score": 5}
    assert "response_format" not in json.loads(seen[-1].content)


def test_openrouter_401_is_unavailable_not_retryable(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(
        401, json={"error": {"message": "No auth credentials found"}}))
    with pytest.raises(ap.AIUnavailableError, match="rejected the key"):
        ap.generate_json("pick", Answer, config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


def test_openrouter_402_says_there_are_no_credits(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(
        402, json={"error": {"message": "Insufficient credits"}}))
    with pytest.raises(ap.AIUnavailableError, match="no credits"):
        ap.generate_text("hi", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


def test_openrouter_429_is_a_retryable_error(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(ap.AIError) as info:
        ap.generate_text("hi", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))
    assert not isinstance(info.value, ap.AIUnavailableError)


def test_openrouter_content_filter_is_blocked(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(
        200, json=_chat_answer("", finish="content_filter")))
    with pytest.raises(ap.AIBlockedError):
        ap.generate_json("pick", Answer, config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


def test_error_inside_a_200_answer_is_raised(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(
        200, json={"error": {"code": 502, "message": "upstream failed"}}))
    with pytest.raises(ap.AIError, match="upstream failed"):
        ap.generate_text("hi", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


def test_schema_violation_is_a_validation_error(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(
        200, json=_chat_answer('{"windows": "not a list", "score": 1}')))
    with pytest.raises(ValidationError):
        ap.generate_json("pick", Answer, config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


# --- Ollama -------------------------------------------------------------------------- #

def test_ollama_request_is_native_chat_with_bare_base64(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "message": {"role": "assistant", "content": '{"windows": [], "score": 9}'},
        "done_reason": "stop", "prompt_eval_count": 500, "eval_count": 40}))

    value, cost = ap.generate_json("pick", Answer, images=[JPEG])

    assert value["score"] == 9
    request = seen[0]
    assert request.url.path == "/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "llava:7b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["format"] == "json"
    assert body["options"]["num_ctx"] == 8192
    message = body["messages"][-1]
    assert message["content"] == "pick"
    assert message["images"] == [base64.b64encode(JPEG).decode("ascii")]
    assert not message["images"][0].startswith("data:")
    assert cost["input_tokens"] == 500 and cost["output_tokens"] == 40
    assert cost["total_cost"] == 0.0 and cost["local"] is True and cost["provider"] == "ollama"


def test_ollama_unreachable_says_how_to_start_it(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")

    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)

    _transport(monkeypatch, refused)
    with pytest.raises(ap.AIUnavailableError, match="ollama serve"):
        ap.generate_text("hi")


def test_ollama_running_out_of_context_is_an_error(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")
    _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "message": {"role": "assistant", "content": "{"}, "done_reason": "length"}))
    with pytest.raises(ap.AIError, match="OLLAMA_NUM_CTX"):
        ap.generate_json("pick", Answer)


def test_ollama_lists_what_is_downloaded_with_vision_where_it_says_so(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llava:7b", "size": 4_700_000_000,
                 "details": {"parameter_size": "7B", "quantization_level": "Q4_0", "families": ["llama", "clip"]}},
                {"name": "qwen3:8b", "size": 5_200_000_000, "details": {"parameter_size": "8B", "families": ["qwen3"]}},
                {"name": "mystery:1b", "size": 1, "details": {}},
            ]})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            caps = {"qwen3:8b": ["completion"], "llava:7b": ["completion", "vision"]}
            return httpx.Response(200, json={"capabilities": caps[name]} if name in caps else {})
        return httpx.Response(404)

    _transport(monkeypatch, handler)
    out = ap.ollama_models()

    assert out["available"] is True
    by_name = {m["name"]: m for m in out["models"]}
    assert by_name["llava:7b"]["vision"] is True
    assert by_name["qwen3:8b"]["vision"] is False
    assert by_name["mystery:1b"]["vision"] is None
    assert by_name["llava:7b"]["parameterSize"] == "7B"


def test_ollama_not_running_is_reported_not_raised(monkeypatch):
    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    _transport(monkeypatch, refused)
    out = ap.ollama_models()
    assert out["available"] is False and out["models"] == []
    assert "ollama serve" in out["reason"]


# --- model lists ------------------------------------------------------------------------ #

def test_openrouter_models_lists_the_text_answers(monkeypatch):
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json={"data": [
        {"id": "vendor/text", "name": "Text", "context_length": 128000,
         "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
         "supported_parameters": ["response_format"]},
        {"id": "vendor/draw", "name": "Draw", "context_length": 8000,
         "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["image", "text"]},
         "supported_parameters": []},
        {"id": "bad id!", "name": "skipped", "architecture": {"output_modalities": ["text"]}},
    ]}))

    out = ap.openrouter_models()

    assert [m["id"] for m in out["text"]] == ["vendor/draw", "vendor/text"]
    assert "image" not in out
    text = {m["id"]: m for m in out["text"]}
    assert text["vendor/text"]["vision"] is True and text["vendor/text"]["json"] is True
    assert text["vendor/draw"]["json"] is False
    assert seen[0].url.path == "/api/v1/models"
    assert "authorization" not in {k.lower() for k in seen[0].headers}  # the list is public: no key sent
    ap.openrouter_models()
    assert len(seen) == 1  # cached


# --- free text and pictures ------------------------------------------------------------ #

def test_generate_text_returns_text_and_cost(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_chat_answer("A title", cost=0.001)))
    text, cost = ap.generate_text("title?", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))
    assert text == "A title"
    assert cost["total_cost"] == pytest.approx(0.001)


def test_generate_image_asks_the_chat_route_for_a_picture(monkeypatch):
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "",
                                 "images": [{"image_url": {"url": data_url}}]}}],
        "usage": {"cost": 0.02}}))

    raw, cost = ap.generate_image("a thumbnail", model="img/m", config=ap.Settings(
        provider="openrouter", model="m", api_key="sk"))

    assert raw == PNG
    assert seen[0].url.path == "/api/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == "img/m" and body["modalities"] == ["image", "text"]
    assert body["messages"][-1]["content"] == "a thumbnail"  # no references: a plain prompt
    assert cost["total_cost"] == pytest.approx(0.02)


def test_generate_image_with_references_sends_the_pictures(monkeypatch):
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "",
                                 "images": [{"image_url": {"url": data_url}}]}}],
        "usage": {"cost": 0.0}}))

    raw, _ = ap.generate_image("draw me", reference_images=[JPEG], model="img/m", config=ap.Settings(
        provider="openrouter", model="m", api_key="sk"))

    assert raw == PNG
    body = json.loads(seen[0].content)
    assert body["modalities"] == ["image", "text"]
    assert body["messages"][-1]["content"][1]["type"] == "image_url"


def test_generate_image_is_off_without_an_image_model(monkeypatch):
    with pytest.raises(ap.AIUnavailableError, match="Thumbnails are off"):
        ap.generate_image("draw", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))


def test_generate_image_never_fetches_a_link(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "", "images": [{"image_url": {"url": "https://example.com/x.png"}}]}}]}))
    with pytest.raises(ap.AIError, match="does not fetch"):
        ap.generate_image("draw", reference_images=[JPEG], model="img", config=ap.Settings(
            provider="openrouter", model="m", api_key="sk"))


def test_generate_image_is_refused_on_ollama(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    with pytest.raises(ap.AIUnavailableError, match="Ollama writes text only"):
        ap.generate_image("draw")


# --- cost ------------------------------------------------------------------------------ #

def test_every_answer_is_recorded_in_the_ledger_and_the_tracked_list(monkeypatch, tmp_path):
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_chat_answer("A title", cost=0.005)))
    with ap.track_costs() as costs:
        ap.generate_text("title?", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))
        ap.generate_text("again", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))

    assert len(costs) == 2
    summary = ap.summarize(costs)
    assert summary["total_cost"] == pytest.approx(0.01)
    assert summary["calls"] == 2 and summary["local"] is False
    lines = (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cost_usd"] == pytest.approx(0.005)


def test_answers_outside_a_tracked_block_still_reach_the_ledger(monkeypatch, tmp_path):
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_chat_answer("t", cost=0.002)))
    ap.generate_text("alone", config=ap.Settings(provider="openrouter", model="m", api_key="sk"))
    assert len((tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_local_answer_costs_nothing_but_counts(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llava:7b")
    _transport(monkeypatch, lambda request: httpx.Response(200, json={
        "message": {"content": "hi"}, "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 3}))
    with ap.track_costs() as costs:
        ap.generate_text("hello")
    summary = ap.summarize(costs)
    assert summary["total_cost"] == 0.0 and summary["local"] is True
    assert summary["input_tokens"] == 7 and summary["output_tokens"] == 3


# --- the JSON parser ----------------------------------------------------------------------- #

def test_parser_takes_the_object_out_of_fences_and_prose():
    text = 'Sure, here it is:\n```json\n{"windows": ["x"], "score": 2}\n```\nThanks.'
    assert ap.parse_json_response_text(text) == {"windows": ["x"], "score": 2}


def test_parser_repairs_invalid_unicode_escapes():
    text = '{"windows": ["bad \\uZZZZ escape"], "score": 1}'
    value = ap.parse_json_response_text(text)
    assert value["score"] == 1
    assert "\\uZZZZ" in value["windows"][0]


def test_parser_rejects_empty_and_non_json():
    with pytest.raises(ValueError):
        ap.parse_json_response_text("")
    with pytest.raises(ValueError):
        ap.parse_json_response_text("no object here")
