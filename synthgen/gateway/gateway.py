"""LLM gateway for locally run models (Ollama).

Safeguards:
- Connects only to a model server on localhost. Any other address is refused.
- Every call uses a pinned model name and can require a JSON schema for the response.
- Responses are cached by request, so repeated experiments don't re-run the model.
- Calls are logged (model, duration, cache hit), never the prompt's example values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class GatewayError(RuntimeError):
    pass


@dataclass
class LLMRequest:
    model: str
    system: str
    prompt: str
    json_schema: dict[str, Any] | None = None
    temperature: float = 0.1
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


class LLMGateway:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 600):
        host = urlparse(base_url).hostname
        if host not in LOCAL_HOSTS:
            raise GatewayError(
                f"Refusing non-local model server '{base_url}'. All LLMs must run locally."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    def available_models(self) -> set[str]:
        """Return the model tags the local server has pulled, or an empty set if it is down."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            resp.raise_for_status()
        except requests.RequestException:
            return set()
        return {m["name"] for m in resp.json().get("models", [])}

    def complete(self, request: LLMRequest) -> str:
        """Send a chat request to the local model and return the response text."""
        key = request.cache_key()
        if key in self._cache:
            log.info("llm call model=%s cache=hit", request.model)
            return self._cache[key]

        body: dict[str, Any] = {
            "model": request.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "options": {"temperature": request.temperature, **request.options},
        }
        if request.seed is not None:
            body["options"]["seed"] = request.seed
        if request.json_schema is not None:
            body["format"] = request.json_schema

        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise GatewayError(f"Local model server call failed: {exc}") from exc
        text = resp.json()["message"]["content"]
        log.info(
            "llm call model=%s cache=miss seconds=%.1f",
            request.model,
            time.perf_counter() - start,
        )
        self._cache[key] = text
        return text

    def complete_json(self, request: LLMRequest, retries: int = 1) -> dict[str, Any]:
        """Return a parsed JSON response, retrying once if the model returns invalid JSON."""
        last_error: Exception | None = None
        for _ in range(retries + 1):
            text = self.complete(request)
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                self._cache.pop(request.cache_key(), None)
        raise GatewayError(f"Model returned invalid JSON: {last_error}")
