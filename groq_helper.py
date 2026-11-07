"""Groq chat-completions client with validated, bounded-retry structured output.

Uses the OpenAI-compatible REST endpoint through ``requests`` (no vendor SDK needed).
The API key is never logged and never included in exception messages.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

import requests

from config import Settings, get_settings, provider_config, resolve_api_key
from schemas import SchemaError
from utils import redact

log = logging.getLogger("ventureiq.llm")


class LLMError(Exception):
    """Base class: live AI generation failed."""


class LLMUnavailable(LLMError):
    """No API key configured."""


class LLMAuthError(LLMError):
    """Key rejected by the provider."""


class LLMQuotaError(LLMError):
    """Rate limit or quota exhausted."""


class LLMTimeout(LLMError):
    """Request timed out."""


class LLMNetworkError(LLMError):
    """Connection or provider-side failure."""


class LLMResponseError(LLMError):
    """The model never produced output that satisfied the schema."""


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class GroqClient:
    """OpenAI-compatible chat client. Despite the legacy name it serves OpenRouter and Groq."""

    def __init__(self, api_key: str | None, model: str, base_url: str, timeout: float = 60.0,
                 max_attempts: int = 2, network_retries: int = 1, max_tokens: int = 3500,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 provider: str = "groq"):
        self.provider = provider
        self._key = api_key
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.max_attempts, self.network_retries, self.max_tokens = max_attempts, network_retries, max_tokens
        self._session = session or requests.Session()
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings | None = None, api_key_override: str | None = None,
                      session: Any = None) -> "GroqClient | None":
        s = settings or get_settings()
        key = resolve_api_key(api_key_override)
        if not key:
            return None
        pc = provider_config(api_key_override)
        return cls(key, pc["model"], pc["base_url"], s.llm_timeout_s, s.llm_max_attempts,
                   s.llm_network_retries, s.llm_max_tokens, session, provider=pc["provider"])

    # ---- transport
    def _headers(self) -> dict[str, str]:
        if not self._key:
            raise LLMUnavailable("No API key is configured (set OPENROUTER_API_KEY or GROQ_API_KEY in .env).")
        h = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":   # optional attribution headers recommended by OpenRouter
            h["HTTP-Referer"] = "http://localhost:3000"
            h["X-Title"] = "VentureIQ"
        return h

    def _post(self, messages: list[dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.2,
                "max_tokens": self.max_tokens}
        if self.provider == "groq":   # router/free models on OpenRouter may reject response_format
            body["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(self.network_retries + 1):
            try:
                resp = self._session.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                          json=body, timeout=self.timeout)
            except requests.Timeout as exc:
                last = LLMTimeout(f"The AI provider did not respond within {self.timeout:.0f}s.")
                log.warning("LLM timeout (attempt %d)", attempt + 1)
            except requests.RequestException as exc:
                last = LLMNetworkError(f"Could not reach the AI provider ({type(exc).__name__}).")
                log.warning("LLM network error (attempt %d): %s", attempt + 1, type(exc).__name__)
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        content = resp.json()["choices"][0]["message"]["content"]
                        if not isinstance(content, str) or not content.strip():
                            raise TypeError("empty content")
                        return content
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise LLMResponseError("The provider returned an unexpected response format.") from exc
                if status in (401, 403):
                    raise LLMAuthError(f"The API key was rejected by {self.provider}. Check the key in your .env file.")
                if status == 402:
                    raise LLMQuotaError(f"{self.provider} reports insufficient credits for this model.")
                if status == 429:
                    retry = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
                    hint = f" Retry after about {retry}s." if retry else ""
                    raise LLMQuotaError("Rate limit or quota reached." + hint)
                if status == 400 or status == 404:
                    raise LLMError(f"The provider rejected the request (HTTP {status}); check the model name '{self.model}'.")
                last = LLMNetworkError(f"The AI provider returned HTTP {status}.")
                log.warning("LLM HTTP %s (attempt %d)", status, attempt + 1)
            if attempt < self.network_retries:
                self._sleep(min(2.0 * (attempt + 1), 5.0))
        assert last is not None
        raise last

    # ---- public API
    def chat_json(self, system: str, user: str, validator: Callable[[Any], dict]) -> dict:
        """Return validator(parsed_json). Retries (bounded) with error feedback on invalid output."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        problem = ""
        for attempt in range(1, self.max_attempts + 1):
            content = self._post(messages)
            try:
                return validator(_extract_json(content))
            except (ValueError, SchemaError) as exc:   # SchemaError and JSONDecodeError are ValueErrors
                problem = redact(str(exc))[:300]
                log.warning("LLM output invalid (attempt %d/%d): %s", attempt, self.max_attempts, problem)
                messages += [{"role": "assistant", "content": content[:4000]},
                             {"role": "user", "content": "Your previous reply was invalid: "
                              f"{problem}\nReturn ONLY a corrected JSON object that follows the schema exactly."}]
        raise LLMResponseError(f"The model did not return a valid structured report after {self.max_attempts} attempts ({problem}).")

    def ping(self) -> tuple[bool, str]:
        """Lightweight connectivity/credential check (lists models; no tokens consumed)."""
        try:
            path = "/key" if self.provider == "openrouter" else "/models"
            resp = self._session.get(f"{self.base_url}{path}", headers=self._headers(), timeout=min(self.timeout, 15))
        except LLMUnavailable as exc:
            return False, str(exc)
        except requests.Timeout:
            return False, "Timed out reaching the provider."
        except requests.RequestException as exc:
            return False, f"Could not reach the provider ({type(exc).__name__})."
        if resp.status_code == 200:
            return True, "Provider reachable and key accepted."
        if resp.status_code in (401, 403):
            return False, "Key rejected by the provider (Authentication failed)."
        return False, f"Provider returned HTTP {resp.status_code}."


def test_llm_key(api_key: str, provider: str = "auto", model: str | None = None) -> dict:
    """Test an API key against the provider with latency measurement and clear diagnostics."""
    from config import PROVIDERS, active_provider
    clean_key = (api_key or "").strip()
    if not clean_key:
        return {"ok": False, "latency_ms": 0, "message": "No API key provided.", "provider": "unknown"}

    chosen_prov = provider.lower() if provider and provider != "auto" else active_provider(clean_key)
    p_info = PROVIDERS.get(chosen_prov, PROVIDERS["groq"])
    chosen_model = model or p_info["model"]

    client = GroqClient(
        api_key=clean_key,
        model=chosen_model,
        base_url=p_info["url"],
        timeout=15.0,
        provider=chosen_prov,
    )
    t0 = time.perf_counter()
    ok, msg = client.ping()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "ok": ok,
        "latency_ms": elapsed_ms,
        "message": msg,
        "provider": chosen_prov,
        "model": chosen_model,
        "get_key_url": p_info["get_key"],
    }
