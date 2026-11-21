"""The one place that talks to a model: OpenRouter or Ollama.

Every stage of the pipeline that needs a model calls this module. Nothing here
imports a vendor SDK, so the two providers are the whole list.

The user picks the provider and the models in Settings. Nothing is preselected and
no model is built in. A request names its choice (``X-AI-Provider``, ``X-AI-Model``; see
app.resolve_ai). A server may name its own in the environment (``AI_PROVIDER``,
``OPENROUTER_MODEL``, ``OLLAMA_MODEL``) for callers that send no choice. A call with no
choice at all is refused, never guessed. No image model is offered: thumbnail pictures
are refused (see generate_image).

OpenRouter speaks OpenAI-style chat completions at https://openrouter.ai/api/v1. The
key comes from the request (``X-OpenRouter-Key``, self-host), from
``OPENROUTER_API_KEY`` on the server, or from the managed key in cloud mode. Pictures
go in as ``image_url`` parts with data URLs. JSON comes back through
``response_format``: ``json_schema``, then ``json_object``, then no format at all,
whichever the routed provider accepts. Every answer carries ``usage.cost``.

Ollama runs on the user's own machine through its native ``POST {OLLAMA_BASE_URL}/api/chat``,
with ``/api/tags`` and ``/api/show`` for the models it has. Pictures go in as bare
base64 strings in each message's ``images``. JSON comes back through ``format:
"json"``; the prompts spell out the exact shape. ``think`` is off, because the JSON
is the whole answer, and ``options.num_ctx`` is set on every call so a long prompt is
not cut short by the server's default context. Ollama is refused in cloud mode: it
would need the operator's own GPU.

Whatever comes back is validated with the pydantic model the caller asked for, so the
pipeline sees one shape whichever provider answered. Retries live in the callers.

Every answer's cost is recorded twice: on the caller's tracked list (``track_costs``,
which gives one clip its own total) and in the lifetime ledger (``usage_log_path``,
one JSON line per answer).
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import httpx
from pydantic import BaseModel

PROVIDERS = ("openrouter", "ollama")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
# No model is a default. The user picks the provider and the models in Settings; a
# server may name its own in the environment. A call with no choice is refused.
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
MAX_OLLAMA_MODELS = 60
MODEL_LIST_TTL_SECONDS = 3600.0
USAGE_LOG_NAME = ".ai_usage.jsonl"
DEFAULT_NUM_CTX = 32768  # 12 frames at 1024px plus a transcript need well over 8k
DEFAULT_TIMEOUT = 600.0  # a local model on a modest GPU can take minutes per batch


class AIError(RuntimeError):
    """A model call failed in a way a retry may fix: network, rate limit, 5xx, bad JSON."""


class AIUnavailableError(AIError):
    """The provider cannot be used here: no key, no credits, not reachable, not allowed,
    no choice made. Retrying does not help, so callers fail with the message instead."""


class AIBlockedError(ValueError):
    """The provider refused this material on policy grounds. The same input is refused
    every time, so callers must fail fast instead of retrying."""


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: float = DEFAULT_TIMEOUT
    num_ctx: int = DEFAULT_NUM_CTX

    @property
    def ready(self) -> bool:
        """A call can be made: a model is chosen, and OpenRouter has a key (Ollama needs none)."""
        if not self.model:
            return False
        return self.provider == "ollama" or bool(self.api_key)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def billing_enabled() -> bool:
    return _env("BILLING_ENABLED").lower() in ("1", "true", "yes", "on")


def provider_name(requested: Optional[str] = None) -> str:
    """The provider a call uses: the one the request names (the user's choice in Settings),
    else the server's AI_PROVIDER if the operator set it. There is no built-in default: a
    call that names neither is refused. Cloud mode is OpenRouter only."""
    if billing_enabled():
        if requested and requested.strip().lower() not in ("", "openrouter"):
            raise AIUnavailableError("Cloud mode runs on OpenRouter only.")
        if _env("AI_PROVIDER", "openrouter").lower() != "openrouter":
            raise AIUnavailableError("Cloud mode runs on OpenRouter only.")
        return "openrouter"
    name = (requested or _env("AI_PROVIDER")).strip().lower()
    if not name:
        raise AIUnavailableError("Choose a provider in Settings: OpenRouter or Ollama.")
    if name not in PROVIDERS:
        raise AIUnavailableError(f"Unknown AI provider {name!r}. Use 'openrouter' or 'ollama'.")
    return name


def _checked_model(value: str, what: str) -> str:
    value = (value or "").strip()
    if value and not MODEL_NAME_RE.match(value):
        raise AIUnavailableError(f"That {what} name is not valid.")
    return value


def resolve(provider: Optional[str] = None, api_key: Optional[str] = None,
            model: Optional[str] = None) -> Settings:
    """Settings for one call. ``provider`` and ``model`` are the user's choice (self-host);
    ``api_key`` is the request's own OpenRouter key, if any. A value not named falls back to
    the server's environment, never to a built-in model. Cloud mode ignores the model choice:
    the operator's OPENROUTER_MODEL decides there."""
    name = provider_name(provider)
    if name == "ollama":
        return Settings(
            provider="ollama",
            model=_checked_model(model or "", "model") or _checked_model(_env("OLLAMA_MODEL"), "model"),
            base_url=_env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL).rstrip("/"),
            timeout=_float_env("AI_TIMEOUT", DEFAULT_TIMEOUT),
            num_ctx=_int_env("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
        )
    chosen = "" if billing_enabled() else _checked_model(model or "", "model")
    return Settings(
        provider="openrouter",
        model=chosen or _checked_model(_env("OPENROUTER_MODEL"), "model"),
        api_key=(api_key or "").strip() or _env("OPENROUTER_API_KEY"),
        timeout=_float_env("AI_TIMEOUT", DEFAULT_TIMEOUT),
    )


def configured(provider: Optional[str] = None, api_key: Optional[str] = None,
               model: Optional[str] = None) -> bool:
    """True when a call could be made with these settings (see Settings.ready)."""
    try:
        return resolve(provider, api_key, model).ready
    except AIUnavailableError:
        return False


def describe() -> Dict[str, Any]:
    """What /api/config tells the dashboard. Never a key and never a choice: the user picks
    the provider and the models. ``keyOnServer`` says whether OpenRouter can run without a
    key typed into the browser."""
    return {"keyOnServer": bool(_env("OPENROUTER_API_KEY"))}


# --- JSON answers ----------------------------------------------------------- #

def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _escape_invalid_unicode_escapes(text: str) -> str:
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "u":
            hex_digits = text[i + 2:i + 6]
            if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                chars.append("\\\\u")
                i += 2
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def parse_json_response_text(text: str) -> dict:
    """The JSON object in a model's answer, tolerant of code fences and prose around it."""
    if not text:
        raise ValueError("The model returned an empty response body.")
    candidate = _extract_json_candidate(text).replace("\x00", "").strip()
    if not candidate:
        raise ValueError("The model's response did not contain a JSON object.")
    attempts = [candidate]
    sanitized = _escape_invalid_unicode_escapes(candidate)
    if sanitized != candidate:
        attempts.append(sanitized)
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as e:
            last_error = e
    raise ValueError(f"Failed to parse the model's JSON response: {last_error}")


# --- pictures ---------------------------------------------------------------- #

def _mime_of(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def _data_url(data: bytes) -> str:
    return f"data:{_mime_of(data)};base64,{base64.b64encode(data).decode('ascii')}"


def _decode_data_url(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.partition(",")[2])
    # A link is never fetched: the model could point the server at any address.
    raise AIError("The image came back as a link, which this app does not fetch.")


# --- HTTP ---------------------------------------------------------------------- #

def _client(timeout: float) -> httpx.Client:
    """Factory so tests can swap in httpx.MockTransport."""
    return httpx.Client(timeout=timeout)


def _error_text(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(err, str):
            return err[:300]
    except Exception:
        pass
    return (resp.text or "")[:300]


def _http_error(provider: str, resp: httpx.Response) -> AIError:
    code = resp.status_code
    detail = _error_text(resp)
    if provider == "openrouter":
        if code == 401:
            return AIUnavailableError(f"OpenRouter rejected the key (401): {detail}")
        if code == 402:
            return AIUnavailableError(f"This OpenRouter key has no credits left (402): {detail}")
        if code == 429:
            return AIError(f"OpenRouter is rate limiting this key (429): {detail}")
    return AIError(f"{provider} answered {code}: {detail}")


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 404, 422):
        return False
    body = (resp.text or "").lower()
    return any(word in body for word in ("response_format", "json_schema", "json_object",
                                         "structured", "parameter", "format"))


# --- OpenRouter ---------------------------------------------------------------- #

def _openrouter_headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "X-OpenRouter-Title": "OpenShorts"}


def _openrouter_messages(system: Optional[str], prompt: str, images: Sequence[bytes]) -> List[dict]:
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        parts: List[dict] = [{"type": "text", "text": prompt}]
        parts += [{"type": "image_url", "image_url": {"url": _data_url(img), "detail": "auto"}}
                  for img in images]
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _json_formats(schema: Optional[Type[BaseModel]]) -> List[Optional[dict]]:
    if schema is None:
        return [None]
    name = re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64] or "response"
    return [
        {"type": "json_schema",
         "json_schema": {"name": name, "strict": False, "schema": schema.model_json_schema()}},
        {"type": "json_object"},
        None,
    ]


def _openrouter_chat(s: Settings, *, system: Optional[str], prompt: str, images: Sequence[bytes],
                     model: Optional[str], schema: Optional[Type[BaseModel]], temperature: float,
                     web: bool = False, modalities: Optional[List[str]] = None,
                     json_mode: bool = False) -> dict:
    if not s.api_key:
        raise AIUnavailableError(
            "No OpenRouter key. Add one in Settings, or set OPENROUTER_API_KEY on the server.")
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    use_model = model or s.model
    if not use_model:
        raise AIUnavailableError("Choose an OpenRouter model in Settings.")
    last_rejection = ""
    if schema is None and json_mode:
        formats: List[Optional[dict]] = [{"type": "json_object"}, None]
    else:
        formats = _json_formats(schema)
    with _client(s.timeout) as client:
        for fmt in formats:
            body: Dict[str, Any] = {
                "model": use_model,
                "messages": _openrouter_messages(system, prompt, images),
                "temperature": temperature,
            }
            if fmt is not None:
                body["response_format"] = fmt
                # Route only to providers that honour the JSON request; never a silent plain answer.
                body["provider"] = {"require_parameters": True}
            if web:
                body["plugins"] = [{"id": "web"}]
            if modalities:
                body["modalities"] = modalities
            try:
                resp = client.post(url, json=body, headers=_openrouter_headers(s.api_key))
            except httpx.HTTPError as e:
                raise AIError(f"Could not reach OpenRouter: {e}") from e
            if fmt is not None and _is_format_rejection(resp):
                last_rejection = _error_text(resp)
                continue
            if resp.status_code >= 400:
                raise _http_error("openrouter", resp)
            return resp.json()
    raise AIError(f"OpenRouter refused every JSON format for {use_model}: {last_rejection}")


def _openrouter_text(data: dict, model: str) -> str:
    if data.get("error"):
        err = data["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise AIError(f"OpenRouter returned an error inside a 200 answer: {message}")
    choices = data.get("choices") or []
    if not choices:
        raise AIError(f"OpenRouter returned no choices for {model}.")
    choice = choices[0] or {}
    if choice.get("finish_reason") == "content_filter":
        raise AIBlockedError(
            "OpenRouter blocked this answer (content_filter). The provider's usage policies "
            "reject this material, so it can't be analyzed.")
    content = ((choice.get("message") or {}).get("content"))
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _openrouter_cost(data: dict, model: str) -> dict:
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": int(details.get("reasoning_tokens") or 0),
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": float(usage.get("cost") or 0.0),
        "model": model,
        "price_estimated": False,
        "provider": "openrouter",
        "local": False,
    }


def _generated_image_url(data: dict) -> str:
    choices = data.get("choices") or []
    message = (choices[0] or {}).get("message") or {} if choices else {}
    for item in message.get("images") or []:
        url = ((item or {}).get("image_url") or {}).get("url")
        if url:
            return url
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    return url
    raise AIError("The model returned no image.")


# --- Ollama ---------------------------------------------------------------------- #

def _ollama_chat(s: Settings, *, system: Optional[str], prompt: str, images: Sequence[bytes],
                 model: Optional[str], json_mode: bool, temperature: float) -> Tuple[str, dict]:
    url = f"{s.base_url}/api/chat"
    use_model = model or s.model
    if not use_model:
        raise AIUnavailableError("Choose an Ollama model in Settings: one already downloaded here.")
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = [base64.b64encode(img).decode("ascii") for img in images]
    messages = ([{"role": "system", "content": system}] if system else []) + [message]
    body: Dict[str, Any] = {
        "model": use_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": s.num_ctx},
    }
    if json_mode:
        body["format"] = "json"
    try:
        with _client(s.timeout) as client:
            resp = client.post(url, json=body)
    except httpx.ConnectError as e:
        raise AIUnavailableError(
            f"Ollama is not reachable at {s.base_url}. Start it with `ollama serve`, "
            "or fix OLLAMA_BASE_URL.") from e
    except httpx.HTTPError as e:
        raise AIError(f"Ollama call to {url} failed: {e}") from e
    if resp.status_code >= 400:
        raise _http_error("ollama", resp)
    data = resp.json()
    if data.get("error"):
        raise AIError(f"Ollama error: {data['error']}")
    if data.get("done_reason") == "length":
        raise AIError(
            f"Ollama ran out of context ({s.num_ctx} tokens) before it finished. "
            "Raise OLLAMA_NUM_CTX.")
    text = (data.get("message") or {}).get("content") or ""
    cost = {
        "input_tokens": int(data.get("prompt_eval_count") or 0),
        "output_tokens": int(data.get("eval_count") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": use_model,
        "price_estimated": False,
        "provider": "ollama",
        "local": True,
    }
    return text, cost


def ollama_models(timeout: float = 5.0) -> Dict[str, Any]:
    """The models this machine's Ollama has downloaded, with vision where Ollama says so.
    Never raises: the Settings screen shows the reason instead. The address is the server's
    own (OLLAMA_BASE_URL), never a request's."""
    base = _env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    models: List[Dict[str, Any]] = []
    try:
        with _client(timeout) as client:
            tags = client.get(f"{base}/api/tags")
            if tags.status_code >= 400:
                return {"available": False, "reason": f"Ollama answered {tags.status_code}.",
                        "models": []}
            for row in (tags.json().get("models") or [])[:MAX_OLLAMA_MODELS]:
                row = row or {}
                name = str(row.get("name") or row.get("model") or "")
                if not MODEL_NAME_RE.match(name):
                    continue
                details = row.get("details") or {}
                # An older Ollama marks a vision projector with a "clip" family; a
                # capabilities list from /api/show, when present, decides.
                vision: Optional[bool] = True if "clip" in (details.get("families") or []) else None
                try:
                    show = client.post(f"{base}/api/show", json={"model": name})
                    capabilities = show.json().get("capabilities") if show.status_code < 400 else None
                except (httpx.HTTPError, ValueError):
                    capabilities = None
                if isinstance(capabilities, list):
                    vision = "vision" in capabilities
                models.append({
                    "name": name,
                    "size": int(row.get("size") or 0),
                    "parameterSize": str(details.get("parameter_size") or ""),
                    "quantization": str(details.get("quantization_level") or ""),
                    "vision": vision,
                })
    except httpx.HTTPError:
        return {"available": False,
                "reason": "Ollama is not running on this machine. Start it with `ollama serve`.",
                "models": []}
    except ValueError:
        return {"available": False,
                "reason": "Ollama answered with something that is not a model list.", "models": []}
    return {"available": True, "reason": None, "models": models}


# --- model lists ----------------------------------------------------------------- #

_model_cache: Dict[str, Any] = {"at": 0.0, "value": None}
_model_cache_lock = threading.Lock()


def openrouter_models() -> Dict[str, Any]:
    """The models OpenRouter serves that answer in text. The list is public, so no key is
    sent. Cached for an hour; a failed fetch is not cached."""
    with _model_cache_lock:
        cached = _model_cache["value"]
        if cached is not None and time.time() - _model_cache["at"] < MODEL_LIST_TTL_SECONDS:
            return cached
    try:
        with _client(20.0) as client:
            resp = client.get(f"{OPENROUTER_BASE_URL}/models")
    except httpx.HTTPError as e:
        raise AIError(f"Could not reach OpenRouter: {e}") from e
    if resp.status_code >= 400:
        raise _http_error("openrouter", resp)
    text_models: List[Dict[str, Any]] = []
    for row in resp.json().get("data") or []:
        row = row or {}
        mid = str(row.get("id") or "")
        if not MODEL_NAME_RE.match(mid):
            continue
        arch = row.get("architecture") or {}
        outputs = arch.get("output_modalities") or []
        item = {
            "id": mid,
            "name": str(row.get("name") or mid),
            "context": int(row.get("context_length") or 0),
            "vision": "image" in (arch.get("input_modalities") or []),
            "json": "response_format" in (row.get("supported_parameters") or []),
        }
        if "text" in outputs:
            text_models.append(item)
    value = {"text": sorted(text_models, key=lambda m: m["id"])}
    with _model_cache_lock:
        _model_cache["at"] = time.time()
        _model_cache["value"] = value
    return value


# --- cost ------------------------------------------------------------------------ #

_tracking = threading.local()
_ledger_lock = threading.Lock()


def usage_log_path() -> str:
    """Where the lifetime ledger lives: one JSON line per answer. The server names it
    (AI_USAGE_LOG), under output/ as a dot entry that the job sweeps never touch."""
    return _env("AI_USAGE_LOG") or os.path.join("output", USAGE_LOG_NAME)


@contextmanager
def track_costs() -> Iterator[List[dict]]:
    """Collect the cost of every answer this thread gets inside the block: one clip's own total."""
    bucket: List[dict] = []
    previous = getattr(_tracking, "bucket", None)
    _tracking.bucket = bucket
    try:
        yield bucket
    finally:
        _tracking.bucket = previous


def record_usage(cost: Optional[dict]) -> None:
    """Note one answer's cost on this thread's tracked list and in the lifetime ledger.
    A ledger that cannot be written never fails the call."""
    if not cost:
        return
    bucket = getattr(_tracking, "bucket", None)
    if bucket is not None:
        bucket.append(cost)
    line = json.dumps({
        "ts": round(time.time(), 3),
        "provider": cost.get("provider"),
        "model": cost.get("model"),
        "input_tokens": int(cost.get("input_tokens") or 0),
        "output_tokens": int(cost.get("output_tokens") or 0),
        "cost_usd": float(cost.get("total_cost") or 0.0),
        "local": bool(cost.get("local")),
    })
    path = usage_log_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _ledger_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def summarize(costs: Sequence[dict]) -> Dict[str, Any]:
    """One total for a list of answers: what a clip, or a job, used."""
    return {
        "total_cost": round(sum(float(c.get("total_cost") or 0.0) for c in costs), 6),
        "input_tokens": sum(int(c.get("input_tokens") or 0) for c in costs),
        "output_tokens": sum(int(c.get("output_tokens") or 0) for c in costs),
        "calls": len(costs),
        "provider": costs[-1].get("provider") if costs else None,
        "model": costs[-1].get("model") if costs else None,
        "local": bool(costs) and all(bool(c.get("local")) for c in costs),
    }


# --- public calls ------------------------------------------------------------------- #

def generate_json(prompt: str, schema: Type[BaseModel], *, images: Sequence[bytes] = (),
                  system: Optional[str] = None, model: Optional[str] = None,
                  config: Optional[Settings] = None, temperature: float = 0.2,
                  web: bool = False) -> Tuple[dict, dict]:
    """One answer that must be JSON matching ``schema``. Returns ``(validated, cost)``.

    Raises AIError (retryable), AIUnavailableError or AIBlockedError. A schema violation
    raises a ValueError from pydantic, which the caller retries like any bad answer."""
    s = config or resolve()
    if s.provider == "ollama":
        text, cost = _ollama_chat(s, system=system, prompt=prompt, images=images, model=model,
                                  json_mode=True, temperature=temperature)
    else:
        data = _openrouter_chat(s, system=system, prompt=prompt, images=images, model=model,
                                schema=schema, temperature=temperature, web=web)
        use_model = model or s.model
        text = _openrouter_text(data, use_model)
        cost = _openrouter_cost(data, use_model)
    record_usage(cost)
    parsed = parse_json_response_text(text)
    return schema.model_validate(parsed).model_dump(), cost


def generate_text(prompt: str, *, images: Sequence[bytes] = (), system: Optional[str] = None,
                  model: Optional[str] = None, config: Optional[Settings] = None,
                  temperature: float = 0.5, web: bool = False,
                  json_mode: bool = False) -> Tuple[str, dict]:
    """One answer as text. ``json_mode`` asks for a JSON object without a schema (the
    caller parses it). Returns ``(text, cost)``."""
    s = config or resolve()
    if s.provider == "ollama":
        text, cost = _ollama_chat(s, system=system, prompt=prompt, images=images, model=model,
                                  json_mode=json_mode, temperature=temperature)
        record_usage(cost)
        return text, cost
    use_model = model or s.model
    data = _openrouter_chat(s, system=system, prompt=prompt, images=images, model=model,
                            schema=None, temperature=temperature, web=web, json_mode=json_mode)
    cost = _openrouter_cost(data, use_model)
    record_usage(cost)
    return _openrouter_text(data, use_model), cost


def generate_image(prompt: str, *, reference_images: Sequence[bytes] = (),
                   model: Optional[str] = None, config: Optional[Settings] = None) -> Tuple[bytes, dict]:
    """One picture, through the chat route with image output. This build offers no image model,
    so a call that names none is refused: thumbnails are off. OpenRouter only."""
    s = config or resolve()
    if s.provider != "openrouter":
        raise AIUnavailableError("Thumbnail images need OpenRouter: Ollama writes text only.")
    if not model:
        raise AIUnavailableError("Thumbnails are off in this build: no image model is offered.")
    data = _openrouter_chat(s, system=None, prompt=prompt, images=reference_images,
                            model=model, schema=None, temperature=0.7,
                            modalities=["image", "text"])
    cost = _openrouter_cost(data, model)
    record_usage(cost)
    return _decode_data_url(_generated_image_url(data)), cost
