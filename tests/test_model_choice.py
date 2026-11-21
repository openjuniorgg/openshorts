"""The model choice a request names, the lists Settings shows, the lifetime total, and the
environment a pipeline child gets. No network: the provider calls are stubbed."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import ai_provider
import ai_usage
import app as app_module


def _request(headers=None):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "query_string": b"", "headers": raw})


@pytest.fixture()
def server_env(monkeypatch, tmp_path):
    for name in ("AI_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OLLAMA_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    return tmp_path


def test_the_request_names_the_provider_and_the_model(server_env):
    ai = asyncio.run(app_module.resolve_ai(_request({
        "X-AI-Provider": "openrouter", "X-AI-Model": "vendor/text",
        "X-OpenRouter-Key": "sk-request"})))
    assert (ai.provider, ai.model, ai.api_key) == ("openrouter", "vendor/text", "sk-request")


def test_a_request_with_no_choice_is_refused_with_what_to_do(server_env):
    with pytest.raises(app_module.HTTPException) as info:
        asyncio.run(app_module.resolve_ai(_request({})))
    assert info.value.status_code == 400
    assert "Choose a provider" in info.value.detail


def test_openrouter_without_a_key_or_a_model_says_which_is_missing(server_env):
    with pytest.raises(app_module.HTTPException) as info:
        asyncio.run(app_module.resolve_ai(_request({"X-AI-Provider": "openrouter", "X-AI-Model": "m"})))
    assert "OpenRouter key" in info.value.detail
    with pytest.raises(app_module.HTTPException) as info:
        asyncio.run(app_module.resolve_ai(_request({"X-AI-Provider": "ollama"})))
    assert "Choose a model" in info.value.detail


def test_ollama_needs_no_key(server_env):
    ai = asyncio.run(app_module.resolve_ai(_request({"X-AI-Provider": "ollama", "X-AI-Model": "llava:7b"})))
    assert ai.provider == "ollama" and ai.model == "llava:7b" and ai.api_key == ""


def test_cloud_mode_ignores_the_request_and_uses_the_operators_model(monkeypatch, server_env):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setenv("OPENROUTER_MODEL", "operator/model")

    async def user(request):
        return object()

    monkeypatch.setattr(app_module, "_user_from_request", user)
    # The cloud package is loaded only in cloud mode: stand in for the managed-key module here.
    monkeypatch.setattr(app_module, "managed_keys", SimpleNamespace(
        has_active_entitlement=lambda u: True, openrouter_key=lambda: "managed-key"))
    ai = asyncio.run(app_module.resolve_ai(_request({
        "X-AI-Provider": "ollama", "X-AI-Model": "x", "X-OpenRouter-Key": "mine"})))
    assert (ai.provider, ai.model, ai.api_key) == ("openrouter", "operator/model", "managed-key")


def test_a_pipeline_child_gets_the_chosen_model_and_no_stale_choice(server_env):
    env = {"OPENROUTER_API_KEY": "stale", "OLLAMA_MODEL": "stale", "AI_PROVIDER": "ollama"}
    ai = ai_provider.Settings(provider="openrouter", model="vendor/text", api_key="sk-x")
    out = app_module.ai_child_env(env, ai)
    assert out["AI_PROVIDER"] == "openrouter"
    assert out["OPENROUTER_API_KEY"] == "sk-x"
    assert out["OPENROUTER_MODEL"] == "vendor/text"
    assert "OPENROUTER_IMAGE_MODEL" not in out
    assert "OLLAMA_MODEL" not in out
    assert out["AI_USAGE_LOG"] == app_module.AI_USAGE_LOG

    local = app_module.ai_child_env({"OPENROUTER_API_KEY": "stale"},
                                    ai_provider.Settings(provider="ollama", model="llava:7b"))
    assert local["OLLAMA_MODEL"] == "llava:7b"
    assert "OPENROUTER_API_KEY" not in local


def test_the_lifetime_total_reads_only_whole_new_lines(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text(json.dumps({"ts": 1.0, "cost_usd": 0.5, "local": False}) + "\n", encoding="utf-8")
    assert ai_usage.summary(str(path))["total_usd"] == 0.5
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 2.0, "cost_usd": 0.25, "local": True}) + "\n")
        f.write('{"ts": 3.0, "cost_usd": 9')  # a line still being written waits for its newline
    second = ai_usage.summary(str(path))
    assert second["total_usd"] == 0.75 and second["calls"] == 2 and second["local_calls"] == 1
    assert ai_usage.summary(str(tmp_path / "missing.jsonl"))["total_usd"] == 0.0


def test_the_header_total_comes_from_the_ledger(server_env):
    (server_env / "usage.jsonl").write_text(
        json.dumps({"ts": 1.0, "cost_usd": 0.5, "local": False, "input_tokens": 3, "output_tokens": 1}) + "\n",
        encoding="utf-8")
    body = TestClient(app_module.app).get("/api/ai/usage").json()
    assert body["enabled"] is True and body["total_usd"] == 0.5 and body["calls"] == 1


def test_cloud_mode_shows_no_model_cost(monkeypatch, server_env):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    assert TestClient(app_module.app).get("/api/ai/usage").json() == {"enabled": False}


def test_settings_lists_the_ollama_models_it_is_given(monkeypatch, server_env):
    monkeypatch.setattr(ai_provider, "ollama_models", lambda timeout=5.0: {
        "available": True, "reason": None,
        "models": [{"name": "llava:7b", "vision": True, "size": 1, "parameterSize": "7B", "quantization": "Q4_0"}]})
    body = TestClient(app_module.app).get("/api/ai/ollama/models").json()
    assert body["available"] is True and body["models"][0]["name"] == "llava:7b"


def test_settings_lists_the_openrouter_models_it_is_given(monkeypatch, server_env):
    monkeypatch.setattr(ai_provider, "openrouter_models", lambda: {
        "text": [{"id": "a/b", "name": "B", "context": 1, "vision": False, "json": True}]})
    body = TestClient(app_module.app).get("/api/ai/openrouter/models").json()
    assert body["available"] is True and body["text"][0]["id"] == "a/b"
