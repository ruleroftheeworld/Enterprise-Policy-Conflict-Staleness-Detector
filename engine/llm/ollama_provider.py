from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


@dataclass(frozen=True)
class OllamaProvider:
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_URL
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        endpoint = f"{self.base_url.rstrip('/')}/api/generate"

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                },
            }
        ).encode("utf-8")

        http_request = request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")

        except error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}"
            ) from exc

        except error.URLError as exc:
            raise RuntimeError(
                f"Unable to connect to Ollama: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                "Ollama request timed out"
            ) from exc

        try:
            response_payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON"
            ) from exc

        if not isinstance(response_payload, dict):
            raise RuntimeError(
                "Ollama response must be a JSON object"
            )

        generated = response_payload.get("response")

        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError(
                "Ollama response does not contain generated text"
            )

        return generated.strip()


__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_URL",
    "OllamaProvider",
]