"""Thin client around an OpenAI-compatible chat API.

Kept intentionally small so the provider can be swapped (OpenAI, Azure,
local Ollama serving an OpenAI-compatible endpoint, etc.) without touching
the analyzer.
"""

from __future__ import annotations

import logging
from typing import Union, Dict, Any

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM client is misconfigured or a call fails."""


# Values that obviously mean "not filled in yet" (templates/docs examples).
_PLACEHOLDER_KEYS = {"changeme", "placeholder", "xxx", "test"}


def _is_placeholder(value: str | None) -> bool:
    """True for empty values and obvious template/placeholder strings."""
    lowered = (value or "").strip().lower()
    if not lowered:
        return True
    return lowered in _PLACEHOLDER_KEYS or "your" in lowered or "placeholder" in lowered


class LLMClient:
    """Minimal chat-completions wrapper used for generating insights.

    Credentials resolve from settings: environment/``.env`` locally, Streamlit
    secrets on Community Cloud. Any OpenAI-compatible endpoint works — set
    ``OPENAI_BASE_URL`` for gateways (ZCode/GLM, proxies, local servers);
    omit it for plain OpenAI.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or SETTINGS.openai_api_key
        self.model = model or SETTINGS.openai_model
        # Placeholder URLs (e.g. an unfilled secrets template) must never reach
        # the OpenAI client — treat them exactly like a missing value.
        base_url_value = (base_url or SETTINGS.openai_base_url or "").strip()
        self.base_url = None if _is_placeholder(base_url_value) else base_url_value

    @property
    def is_configured(self) -> bool:
        """True when a real (non-placeholder) API key is present."""
        return not _is_placeholder(self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Union[str, Dict[str, Any], None]:
        """Send a single chat completion request and return the message text.

        Returns:
            str, dict, or None depending on API response structure.
            
        Raises:
            LLMClientError: If no API key is configured or the API call fails.
        """
        if not self.is_configured:
            raise LLMClientError(
                "No API key configured. Set OPENAI_API_KEY in your .env file (local) "
                "or in Streamlit's Secrets settings (cloud)."
            )

        # Imported lazily so the app still runs (non-AI features) without it.
        from openai import OpenAI

        client_kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK error uniformly
            logger.exception("LLM call failed")
            raise LLMClientError(f"LLM call failed: {exc}") from exc

        # --- UNIVERSAL SAFE PARSING START ---
        if response is None:
            logger.warning("LLM returned None response object")
            return None

        # Case 1: Standard OpenAI / GLM Structure
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
                if content:
                    return content
        
        # Case 2: Direct Content Attribute (Some lightweight APIs)
        if hasattr(response, 'content'):
            return response.content
            
        # Case 3: Dictionary Response (JSON based APIs)
        if isinstance(response, dict):
            # Check common keys used by various AI providers
            for key in ['text', 'result', 'answer', 'output', 'content']:
                if key in response and response[key]:
                    return response[key]
            # If dict but no standard keys, return stringified version
            return str(response)

        # Case 4: String Response
        if isinstance(response, str):
            return response
            
        # Fallback: Log warning and convert whatever we got to string
        logger.warning(f"Unexpected LLM response type: {type(response)}. Converting to string.")
        return str(response)
        # --- UNIVERSAL SAFE PARSING END ---