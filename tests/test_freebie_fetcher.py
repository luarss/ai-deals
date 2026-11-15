from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import requests

from src import freebie_fetcher
from src.freebie_fetcher import redact_secrets, sanitize_item, update_freebies

GENERATED_AT = datetime(2026, 11, 15, 2, 17, tzinfo=timezone.utc)


def make_raw_item(**overrides):
    item = {
        "title": "DeepSeek v4.1 Flash Key",
        "summary": "key: sk-e908e00e55541bca34b952003d9717e4 enjoy",
        "url": "https://linux.do/t/topic/123",
        "category": "ai",
        "obtain": "free",
        "source_id": "linuxdo-welfare",
        "score": 81,
        "published_at": "2026-11-14T10:00:00.000Z",
        "expires_at": None,
        "codes": ["SECRETCODE"],
        "image_url": "https://example.com/pixel.gif",
    }
    item.update(overrides)
    return item


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_redacts_api_keys_and_base64_tokens():
    text = "a sk-6810902ebc0f3256989f528a b c2stS2hpbEFHR0laM3hKeXdxSmtZbVp4 c"
    assert redact_secrets(text) == "a [redacted] b [redacted] c"


def test_keeps_hyphenated_slugs():
    assert redact_secrets("14729-global-hack-week-builders-week") == "14729-global-hack-week-builders-week"


def test_sanitize_keeps_only_whitelisted_fields():
    item = sanitize_item(make_raw_item())
    assert set(item) == {"title", "summary", "url", "category", "obtain", "source", "score", "published_at", "expires_at"}
    assert "sk-" not in item["summary"]
    assert item["source"] == "linuxdo-welfare"


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "data:text/html,x", "//evil.com", "https://a.com/\"onmouseover=x", None, 42],
)
def test_sanitize_drops_unsafe_urls(url):
    assert sanitize_item(make_raw_item(url=url)) is None


def test_sanitize_rejects_odd_short_fields_and_timestamps():
    item = sanitize_item(make_raw_item(category="<script>", published_at="tomorrow", score="99"))
    assert item["category"] is None
    assert item["published_at"] is None
    assert item["score"] is None


def test_sanitize_truncates_long_text():
    item = sanitize_item(make_raw_item(summary="word " * 200))
    assert len(item["summary"]) <= freebie_fetcher.MAX_SUMMARY_LENGTH


def test_update_writes_sanitized_payload(tmp_path, monkeypatch):
    payload = {"items": [make_raw_item(), make_raw_item(url="javascript:x"), "junk"]}
    monkeypatch.setattr(freebie_fetcher.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    path = tmp_path / "freebies.json"

    assert update_freebies(GENERATED_AT, path) == 1
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["generated_at"] == "2026-11-15 02:17 UTC"
    assert len(written["items"]) == 1


def test_update_keeps_previous_file_on_failure(tmp_path, monkeypatch):
    def raise_error(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(freebie_fetcher.requests, "get", raise_error)
    path = tmp_path / "freebies.json"
    path.write_text('{"items": ["old"]}', encoding="utf-8")

    assert update_freebies(GENERATED_AT, path) == 0
    assert path.read_text(encoding="utf-8") == '{"items": ["old"]}'


def test_update_keeps_previous_file_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(freebie_fetcher.requests, "get", lambda *args, **kwargs: FakeResponse({"items": []}))
    path = tmp_path / "freebies.json"

    assert update_freebies(GENERATED_AT, path) == 0
    assert not path.exists()
