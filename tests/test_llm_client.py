"""Tests for LLM client configuration logic (offline — no API calls)."""

from src.ai.llm_client import LLMClient


def test_empty_and_placeholder_keys_are_not_configured():
    """Template/placeholder values must not put the app into 'AI mode'."""
    for key in (
        "",
        "   ",
        "your_zcode_or_glm_api_key_here",
        "sk-your-key-here",
        "CHANGEME",
        "placeholder",
    ):
        assert not LLMClient(api_key=key).is_configured, f"key {key!r} should not configure the client"


def test_real_key_is_configured():
    assert LLMClient(api_key="sk-proj-4f9a2c8e77").is_configured


def test_base_url_override_is_kept():
    client = LLMClient(api_key="sk-proj-4f9a2c8e77", base_url="https://gateway.example.com/v1")
    assert client.base_url == "https://gateway.example.com/v1"


def test_placeholder_base_url_is_ignored():
    """An unfilled secrets template must not leak placeholders into the SDK."""
    client = LLMClient(api_key="sk-proj-4f9a2c8e77", base_url="your_api_base_url_here")
    assert client.base_url is None


def test_base_url_none_for_plain_openai(monkeypatch):
    """Without OPENAI_BASE_URL set anywhere, no base_url should be passed."""
    import config.settings as settings_module

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = LLMClient(api_key="sk-proj-4f9a2c8e77")
    assert client.base_url in (None, "")
