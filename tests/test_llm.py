"""Tests for app/llm.py provider dispatch."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.llm import get_chat_model


class TestGetChatModel:
    def test_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = get_chat_model()

        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o"

    def test_openai_uses_configured_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

        model = get_chat_model()

        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"

    def test_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        model = get_chat_model()

        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-sonnet-4-6"

    def test_anthropic_uses_configured_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        model = get_chat_model()

        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-haiku-4-5-20251001"

    def test_ollama_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        model = get_chat_model()

        assert isinstance(model, ChatOllama)
        assert model.model == "llama3.1"
        assert model.base_url == "http://localhost:11434"

    def test_ollama_uses_configured_model_and_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

        model = get_chat_model()

        assert isinstance(model, ChatOllama)
        assert model.model == "qwen2.5"
        assert model.base_url == "http://ollama:11434"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "made-up-provider")

        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            get_chat_model()

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = get_chat_model()

        assert isinstance(model, ChatOpenAI)
