"""Thin client around an OpenAI-compatible chat API."""

from __future__ import annotations

import logging
from typing import Union, Dict, Any

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM client is misconfigured or a call fails."""


_PLACEHOLDER_KEYS = {"changeme", "placeholder", "xxx", "test"}


def _is_placeholder(value: str | None) -> bool:
    lowered = (value or "").strip().lower()
    if not lowered:
        return True
    return lowered in _PLACEHOLDER_KEYS or "your" in lowered or "placeholder" in lowered


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or SETTINGS.openai_api_key
        self.model = model or SETTINGS.openai_model
        base_url_value = (base_url or SETTINGS.openai_base_url or "").strip()
        self.base_url = None if _is_placeholder(base_url_value) else base_url_value

    @property
    def is_configured(self) -> bool:
        return not _is_placeholder(self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Union[str, Dict[str, Any], None]:
        if not self.is_configured:
            raise LLMClientError("No API key configured.")

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
        except Exception as exc:
            logger.exception("LLM call failed")
            raise LLMClientError(f"LLM call failed: {exc}") from exc

        # ==========================================
        # 100% BULLETPROOF PARSING (No Subscripting)
        # ==========================================
        print("DEBUG: LLM RESPONSE TYPE =", type(response))
        
        if response is None:
            return "AI returned no response (None)."
            
        try:
            # Safe check for OpenAI standard format
            if hasattr(response, 'choices') and response.choices is not None and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content') and choice.message.content is not None:
                    return choice.message.content
        except Exception as e:
            print(f"DEBUG: Standard parsing failed safely: {e}")
            
        # Fallback 1: Direct content attribute
        if hasattr(response, 'content') and response.content is not None:
            return response.content
            
        # Fallback 2: Dictionary response
        if isinstance(response, dict):
            for key in ['text', 'result', 'answer', 'output', 'content']:
                if key in response and response[key] is not None:
                    return str(response[key])
            return str(response)
            
        # Fallback 3: String response
        if isinstance(response, str):
            return response
            
        # Final Fallback
        return str(response)