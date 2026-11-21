"""Retry behaviour of the AI scoring/detail stage (``main._run_ai_stage``).

A model that answers 200 with an empty body raises while parsing, not while
calling, so it must still be retried; a content block, a missing key and a
spent credit fail at once, because the same call fails the same way again.
"""
import pytest

# main pulls in cv2/torch/mediapipe at import time; the minimal CI env lacks
# them, so skip there. Runs fully in the container/local where deps exist.
main = pytest.importorskip("main")
import ai_provider  # noqa: E402


class _Stub:
    """Stands in for ai_provider.generate_json: empty answers first, then a good one."""

    def __init__(self, blips, payload=None, error=None):
        self.blips = blips
        self.calls = 0
        self.payload = payload if payload is not None else {"windows": [{"id": "w0", "score": 90}]}
        self.error = error

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.calls <= self.blips:
            raise ValueError("The model returned an empty response body.")
        return self.payload, {"input_tokens": 1}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


def _patch(monkeypatch, stub):
    monkeypatch.setattr(ai_provider, "generate_json", stub)


def test_recovers_from_a_single_empty_body(monkeypatch):
    stub = _Stub(blips=1)
    _patch(monkeypatch, stub)
    parsed, _cost = main._run_ai_stage("prompt", object)
    assert stub.calls == 2
    assert parsed["windows"][0]["score"] == 90


def test_recovers_from_two_consecutive_blips(monkeypatch):
    stub = _Stub(blips=2)
    _patch(monkeypatch, stub)
    parsed, _cost = main._run_ai_stage("prompt", object)
    assert stub.calls == 3
    assert parsed["windows"]


def test_gives_up_after_three_attempts(monkeypatch):
    stub = _Stub(blips=99)
    _patch(monkeypatch, stub)
    with pytest.raises(ValueError) as exc:
        main._run_ai_stage("prompt", object)
    assert stub.calls == 3
    assert "empty response body" in str(exc.value)


def test_non_transient_errors_are_not_retried(monkeypatch):
    stub = _Stub(blips=0, error=TypeError("bad argument"))
    _patch(monkeypatch, stub)
    with pytest.raises(TypeError):
        main._run_ai_stage("prompt", object)
    assert stub.calls == 1


def test_a_rate_limit_is_retried(monkeypatch):
    calls = []

    def flaky(prompt, schema, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ai_provider.AIError("OpenRouter is rate limiting this key (429)")
        return {"windows": []}, None

    _patch(monkeypatch, flaky)
    assert main._run_ai_stage("prompt", object)[0] == {"windows": []}
    assert len(calls) == 2


def test_succeeds_without_retrying_when_the_first_call_is_fine(monkeypatch):
    stub = _Stub(blips=0)
    _patch(monkeypatch, stub)
    main._run_ai_stage("prompt", object)
    assert stub.calls == 1


def test_content_block_fails_fast_without_retrying(monkeypatch):
    # A content block is deterministic: three retries would only burn quota.
    stub = _Stub(blips=0, error=ai_provider.AIBlockedError("content_filter: blocked this answer"))
    _patch(monkeypatch, stub)
    with pytest.raises(ai_provider.AIBlockedError) as exc:
        main._run_ai_stage("prompt", object)
    assert stub.calls == 1
    assert "content_filter" in str(exc.value)


def test_missing_key_or_credit_fails_fast_without_retrying(monkeypatch):
    stub = _Stub(blips=0, error=ai_provider.AIUnavailableError("No OpenRouter key."))
    _patch(monkeypatch, stub)
    with pytest.raises(ai_provider.AIUnavailableError):
        main._run_ai_stage("prompt", object)
    assert stub.calls == 1
