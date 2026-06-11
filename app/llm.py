"""Shared LLM provider configuration.

Selects and constructs a chat model based on the LLM_PROVIDER environment
variable, so the extraction and drafting agents can run against OpenAI,
Anthropic (Claude), or a locally hosted Ollama model without code changes.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("openai", "anthropic", "ollama")


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Construct a chat model for the configured LLM_PROVIDER.

    Environment variables:
        LLM_PROVIDER: "openai" (default), "anthropic", or "ollama"
        OPENAI_MODEL: model name when LLM_PROVIDER=openai (default "gpt-4o")
        ANTHROPIC_MODEL: model name when LLM_PROVIDER=anthropic
            (default "claude-sonnet-4-6")
        OLLAMA_MODEL: model name when LLM_PROVIDER=ollama (default "llama3.1")
        OLLAMA_BASE_URL: Ollama server URL (default "http://localhost:11434")
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        f"Expected one of {SUPPORTED_PROVIDERS}."
    )
