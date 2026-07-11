import json
from io import BytesIO
from urllib import error

import pytest

from engine.llm import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaProvider,
)


class FakeHTTPResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_default_configuration():
    provider = OllamaProvider()

    assert provider.model == DEFAULT_OLLAMA_MODEL
    assert provider.base_url == DEFAULT_OLLAMA_URL
    assert provider.timeout_seconds == 60.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": ""}, "model must not be empty"),
        ({"base_url": ""}, "base_url must not be empty"),
        (
            {"timeout_seconds": 0},
            "timeout_seconds must be greater than zero",
        ),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OllamaProvider(**kwargs)


def test_generate_sends_expected_request(monkeypatch):
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["request"] = http_request
        captured["timeout"] = timeout

        return FakeHTTPResponse(
            {
                "response": json.dumps(
                    {
                        "verified": True,
                        "confidence": 0.91,
                        "explanation": "Confirmed.",
                    }
                )
            }
        )

    monkeypatch.setattr(
        "engine.llm.ollama_provider.request.urlopen",
        fake_urlopen,
    )

    provider = OllamaProvider(
        model="test-model",
        base_url="http://localhost:11434/",
        timeout_seconds=12.0,
    )

    result = provider.generate("verify this finding")

    assert captured["request"].full_url == (
        "http://localhost:11434/api/generate"
    )

    assert captured["timeout"] == 12.0

    payload = json.loads(
        captured["request"].data.decode("utf-8")
    )

    assert payload["model"] == "test-model"
    assert payload["prompt"] == "verify this finding"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0

    parsed = json.loads(result)

    assert parsed["verified"] is True
    assert parsed["confidence"] == 0.91


def test_http_error_is_wrapped(monkeypatch):
    def fake_urlopen(http_request, timeout):
        raise error.HTTPError(
            url=http_request.full_url,
            code=500,
            msg="server error",
            hdrs=None,
            fp=BytesIO(),
        )

    monkeypatch.setattr(
        "engine.llm.ollama_provider.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        OllamaProvider().generate("verify")


def test_connection_error_is_wrapped(monkeypatch):
    def fake_urlopen(http_request, timeout):
        raise error.URLError("connection refused")

    monkeypatch.setattr(
        "engine.llm.ollama_provider.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to connect to Ollama",
    ):
        OllamaProvider().generate("verify")


def test_invalid_json_response_is_rejected(monkeypatch):
    class InvalidJSONResponse:
        def read(self):
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        "engine.llm.ollama_provider.request.urlopen",
        lambda http_request, timeout: InvalidJSONResponse(),
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        OllamaProvider().generate("verify")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"response": ""},
        {"response": 123},
    ],
)
def test_invalid_generated_response_is_rejected(
    monkeypatch,
    payload,
):
    monkeypatch.setattr(
        "engine.llm.ollama_provider.request.urlopen",
        lambda http_request, timeout: FakeHTTPResponse(payload),
    )

    with pytest.raises(RuntimeError):
        OllamaProvider().generate("verify")


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_prompt_is_rejected(prompt):
    with pytest.raises(ValueError):
        OllamaProvider().generate(prompt)