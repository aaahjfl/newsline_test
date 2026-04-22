"""Small Ollama client used by local pipeline stages."""

from __future__ import annotations

import json
from socket import timeout as SocketTimeout
from typing import Any
from urllib import error, parse, request


class OllamaRequestError(RuntimeError):
    """Raised when the local Ollama API cannot complete a generation request."""


def normalize_local_ollama_url(url: str) -> str:
    """Prefer explicit IPv4 loopback to avoid localhost/proxy/VPN edge cases."""
    parsed = parse.urlparse(url)
    if parsed.hostname != "localhost":
        return url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return parse.urlunparse(parsed._replace(netloc=netloc))


def check_ollama_available(url: str, *, timeout_seconds: int = 5) -> None:
    """Fail fast if the local Ollama server is not reachable."""
    normalized_url = normalize_local_ollama_url(url)
    parsed = parse.urlparse(normalized_url)
    tags_url = parse.urlunparse(parsed._replace(path="/api/tags", query="", params="", fragment=""))
    opener = request.build_opener(request.ProxyHandler({}))
    http_request = request.Request(tags_url, method="GET")
    try:
        with opener.open(http_request, timeout=timeout_seconds) as response:
            response.read()
    except (error.URLError, TimeoutError, SocketTimeout) as exc:
        raise OllamaRequestError(
            f"Cannot reach local Ollama at {tags_url} within {timeout_seconds}s. "
            "If a proxy or virtual network adapter is enabled, make sure 127.0.0.1:11434 bypasses it."
        ) from exc


def generate_with_ollama(
    prompt: str,
    *,
    model: str,
    url: str,
    timeout_seconds: int,
    keep_alive: str = "0s",
    think: bool | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Call Ollama's generate API and return the completed response text."""
    url = normalize_local_ollama_url(url)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if think is not None:
        payload["think"] = think
    if options:
        payload["options"] = dict(options)

    opener = request.build_opener(request.ProxyHandler({}))

    def post(current_payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(current_payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        data = post(payload)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if "think" in payload and exc.code in {400, 422}:
            # Older Ollama builds may reject the `think` field. Retry once without
            # it; downstream parsers still strip visible <think> blocks if needed.
            fallback_payload = dict(payload)
            fallback_payload.pop("think", None)
            try:
                data = post(fallback_payload)
            except Exception as retry_exc:  # pragma: no cover - depends on local Ollama.
                raise OllamaRequestError(
                    f"Ollama request failed after retrying without think field: {retry_exc}"
                ) from retry_exc
        else:
            raise OllamaRequestError(f"Ollama HTTP {exc.code}: {body}") from exc
    except (error.URLError, TimeoutError, SocketTimeout) as exc:
        raise OllamaRequestError(
            f"Ollama request failed or timed out after {timeout_seconds}s. "
            "Check that Ollama is running and try a smaller --llm-batch-size or larger --llm-timeout-seconds."
        ) from exc

    result = str(data.get("response") or "").strip()
    if not result:
        raise RuntimeError("Ollama returned an empty response.")
    return result
