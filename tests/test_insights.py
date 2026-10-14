from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from src import insights
from src.insights import FALLBACK, extract_brief, generate_insights
from src.models import Changelog


def make_response(content, finish_reason="stop", reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=500, total_tokens=1400)
    return SimpleNamespace(choices=[choice], usage=usage, model="deepseek-v4-flash")


class FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def fake_openai(monkeypatch):
    def install(outcome):
        completions = FakeCompletions(outcome)

        class FakeOpenAI:
            def __init__(self, api_key, base_url):
                self.chat = SimpleNamespace(completions=completions)

        module = types.ModuleType("openai")
        module.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", module)
        return completions

    return install


def run(changelog=None):
    return generate_insights([], changelog or Changelog(first_run=True), "test-key")


def test_returns_stripped_content_on_normal_reply(fake_openai):
    completions = fake_openai(make_response("  Prices fell today. Quiet otherwise.\n"))
    assert run() == "Prices fell today. Quiet otherwise."
    assert completions.calls[0]["max_tokens"] == insights.MAX_TOKENS
    assert completions.calls[0]["model"] == insights.MODEL


def test_empty_content_with_length_finish_returns_fallback(fake_openai):
    fake_openai(
        make_response("", finish_reason="length", reasoning_content="Let me think " * 200)
    )
    assert run() == FALLBACK


def test_none_content_returns_fallback(fake_openai):
    fake_openai(make_response(None, finish_reason="length"))
    assert run() == FALLBACK


def test_whitespace_content_returns_fallback(fake_openai):
    fake_openai(make_response("   \n\t ", finish_reason="stop"))
    assert run() == FALLBACK


def test_api_exception_returns_fallback(fake_openai):
    fake_openai(RuntimeError("boom"))
    assert run() == FALLBACK


def test_missing_openai_package_returns_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    assert run() == FALLBACK


def test_length_truncation_keeps_only_complete_sentences():
    response = make_response(
        "GPT-5.2 jumped 100%. grok-4-fast-reasoning dropped 82%, from $1.67 to $0.",
        finish_reason="length",
    )
    assert extract_brief(response) == "GPT-5.2 jumped 100%."


def test_length_truncation_without_complete_sentence_returns_fallback(fake_openai):
    fake_openai(make_response("GPT-", finish_reason="length"))
    assert run() == FALLBACK


def test_inline_think_block_is_stripped():
    response = make_response("<think>reasoning here</think>\nMarket was quiet today.")
    assert extract_brief(response) == "Market was quiet today."


def test_unclosed_think_block_yields_fallback(fake_openai):
    fake_openai(make_response("<think>still reasoning when cut off", finish_reason="length"))
    assert run() == FALLBACK
