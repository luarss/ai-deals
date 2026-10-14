import json

import pytest

from src import scraper
from src.scraper import (
    blended_price,
    extract_embedded_models,
    extract_table,
    scrape_page,
)

LINK_MODELS = [
    {"slug": "claude-opus-5-5", "name": "Claude Opus 5.5 (Max, Default Fallback)", "releaseSlug": "claude-opus-5-5"},
]

LEADERBOARD_MODELS = [
    {
        "slug": "glm-4-5v",
        "name": "GLM-4.5V (Non-reasoning)",
        "shortName": "GLM-4.5V (non-reasoning)",
        "deprecated": True,
        "isReasoning": False,
        "modelCreatorName": "Z AI",
        "contextWindowTokens": 64000,
        "intelligenceIndex": 6.69669796944385,
        "intelligenceIndexIsEstimated": True,
        "price1mInputTokens": 0.6,
        "price1mOutputTokens": 1.8,
        "cacheHitPrice": 0.11,
        "cacheWritePrice": None,
        "medianOutputTokensPerSecond": 48.5941317995809,
        "medianTimeToFirstTokenSeconds": 2.76024521400001,
    },
    {
        "slug": "claude-opus-5-5",
        "name": "Claude Opus 5.5 (Max, Default Fallback)",
        "shortName": "Claude Opus 5.5 (max with fallback)",
        "deprecated": False,
        "isReasoning": True,
        "modelCreatorName": "Anthropic",
        "contextWindowTokens": 1000000,
        "intelligenceIndex": 57.6223698102963,
        "intelligenceIndexIsEstimated": False,
        "intelligenceIndexCostPerTask": 5.982012019521066,
        "price1mInputTokens": 4,
        "price1mOutputTokens": 20,
        "cacheHitPrice": 0.2,
        "cacheWritePrice": 5,
        "medianOutputTokensPerSecond": 95.0289927001059,
        "medianTimeToFirstTokenSeconds": 673.647297053,
    },
    {
        "slug": "gpt-6-luna",
        "name": "GPT-6 Luna (Max)",
        "shortName": "GPT-6 Luna (max)",
        "deprecated": False,
        "isReasoning": True,
        "modelCreatorName": "OpenAI",
        "contextWindowTokens": 1048576,
        "intelligenceIndex": 38.1245186869738,
        "intelligenceIndexIsEstimated": False,
        "intelligenceIndexCostPerTask": 0.06782295669812205,
        "price1mInputTokens": 0.1,
        "price1mOutputTokens": 0.5,
        "cacheHitPrice": None,
        "cacheWritePrice": 0.125,
        "medianOutputTokensPerSecond": 135.046084233276,
        "medianTimeToFirstTokenSeconds": 113.292133609,
    },
    {
        "slug": "jt-35b-flash",
        "name": "JT-35B-Flash",
        "shortName": "JT-35B-Flash",
        "deprecated": False,
        "isReasoning": False,
        "modelCreatorName": "China Mobile",
        "contextWindowTokens": 256000,
        "intelligenceIndex": 18.6626376411117,
        "intelligenceIndexIsEstimated": True,
        "price1mInputTokens": None,
        "price1mOutputTokens": None,
        "cacheHitPrice": None,
        "cacheWritePrice": None,
    },
]

TABLE_HTML = """
<table>
<thead>
<tr><th colspan="1"></th><th colspan="2">Features</th><th colspan="1">Intelligence</th><th colspan="1">Price</th>
<th colspan="1">Speed</th><th colspan="1">Latency</th><th colspan="1">End-to-End Response Time</th><th colspan="1"></th></tr>
<tr><th>Model</th><th>Context Window</th><th>Creator</th><th>Artificial Analysis Intelligence Index</th>
<th>Cost per Task USD</th><th>Median Tokens/s</th><th>Latency First Chunk (s)</th><th>Total Response (s)</th>
<th>Further Analysis</th></tr>
</thead>
<tbody>
<tr>
<td><div class="font-semibold border-l-4 pl-2">Claude Opus 5.5 (max with fallback)</div></td>
<td><div class="text-center">1M</div></td>
<td><div class="flex gap-2"><img alt="Anthropic" src="/img/logos/anthropic.svg"/><span>Anthropic</span></div></td>
<td><div class="text-center">58</div></td>
<td><div class="text-center">$5.98</div></td>
<td><div class="text-center">95</div></td>
<td><div class="text-center">673.65</div></td>
<td><div class="text-center">678.91</div></td>
<td><div class="flex items-center"><a href="/models/claude-opus-5-5">Model</a><a href="/models/claude-opus-5-5/providers">Providers</a></div></td>
</tr>
<tr>
<td><div class="font-semibold border-l-4 pl-2">JT-35B-Flash</div></td>
<td><div class="text-center">256k</div></td>
<td><div class="flex gap-2"><img alt="China Mobile" src="/img/logos/chinamobile.svg"/><span>China Mobile</span></div></td>
<td><div class="text-center cursor-default underline decoration-dotted" data-state="closed">19<!-- -->*</div></td>
<td><div class="text-center">--</div></td>
<td><div class="text-center">--</div></td>
<td><div class="text-center">--</div></td>
<td><div class="text-center">--</div></td>
<td><div class="flex items-center"><a href="/models/jt-35b-flash">Model</a><a href="/models/jt-35b-flash/providers">Providers</a></div></td>
</tr>
</tbody>
</table>
"""


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _push_script(text: str) -> str:
    return f"<script>self.__next_f.push({json.dumps([1, text])})</script>"


def _leaderboard_html(models: list[dict], table_html: str = TABLE_HTML) -> str:
    link_line = "1d:" + _compact_json(["$", "$L1e", None, {"models": LINK_MODELS}]) + "\n"
    data_line = (
        "34:"
        + _compact_json(["$", "div", None, {"className": "container", "children": [["$", "$L35", None, {"models": models}]]}])
        + "\n"
    )
    split_at = len(data_line) // 2
    scripts = [
        "<script>(self.__next_f=self.__next_f||[]).push([0])</script>",
        _push_script('1:"$Sreact.fragment"\n'),
        _push_script(link_line),
        _push_script(data_line[:split_at]),
        _push_script(data_line[split_at:]),
    ]
    return "<html><body>" + table_html + "".join(scripts) + "</body></html>"


def _by_id(records):
    return {record.model_id: record for record in records}


def test_embedded_models_parse_prices_intelligence_speed_and_context():
    records = _by_id(extract_embedded_models(_leaderboard_html(LEADERBOARD_MODELS), "leaderboard"))

    opus = records["claude-opus-5-5"]
    assert opus.name == "Claude Opus 5.5 (max with fallback)"
    assert opus.creator == "Anthropic"
    assert opus.intelligence_score == 57.6
    assert opus.price_input == 4.0
    assert opus.price_output == 20.0
    assert opus.price_blended == pytest.approx(2.94)
    assert opus.output_speed_tps == 95.0
    assert opus.context_window_k == 1000
    assert opus.source_pages == ["leaderboard"]


def test_embedded_models_blend_uses_input_rate_when_no_cache_price():
    luna = _by_id(extract_embedded_models(_leaderboard_html(LEADERBOARD_MODELS), "leaderboard"))["gpt-6-luna"]

    assert luna.price_blended == pytest.approx((9 * 0.1 + 0.5) / 10)
    assert luna.context_window_k == 1049


def test_embedded_models_keep_unpriced_models_with_null_fields():
    flash = _by_id(extract_embedded_models(_leaderboard_html(LEADERBOARD_MODELS), "leaderboard"))["jt-35b-flash"]

    assert flash.intelligence_score == 18.7
    assert flash.price_input is None
    assert flash.price_output is None
    assert flash.price_blended is None
    assert flash.output_speed_tps is None
    assert flash.coding_index is None


def test_embedded_models_skip_deprecated_and_link_only_arrays():
    records = extract_embedded_models(_leaderboard_html(LEADERBOARD_MODELS), "leaderboard")

    assert [record.model_id for record in records] == ["claude-opus-5-5", "gpt-6-luna", "jt-35b-flash"]


def test_embedded_models_empty_without_flight_payload():
    assert extract_embedded_models("<html><body>" + TABLE_HTML + "</body></html>", "leaderboard") == []


def test_blended_price_requires_input_and_output():
    assert blended_price(None, 1.0, 0.1) is None
    assert blended_price(1.0, None, 0.1) is None
    assert blended_price(1.0, 3.0, 0.5) == pytest.approx((3.5 + 2.0 + 3.0) / 10)


def test_table_fallback_parses_estimated_intelligence_and_ignores_cost_per_task():
    records = _by_id(extract_table(TABLE_HTML, "leaderboard"))

    opus = records["claude-opus-5-5"]
    assert opus.intelligence_score == 58.0
    assert opus.output_speed_tps == 95.0
    assert opus.context_window_k == 1000
    assert opus.creator == "Anthropic"
    assert opus.price_blended is None

    flash = records["jt-35b-flash"]
    assert flash.intelligence_score == 19.0
    assert flash.output_speed_tps is None
    assert flash.context_window_k == 256


def test_scrape_page_prefers_embedded_payload(monkeypatch):
    monkeypatch.setattr(scraper, "fetch_html", lambda url: _leaderboard_html(LEADERBOARD_MODELS))

    records = _by_id(scrape_page("leaderboard", "https://example.test"))

    assert set(records) == {"claude-opus-5-5", "gpt-6-luna", "jt-35b-flash"}
    assert records["claude-opus-5-5"].price_blended == pytest.approx(2.94)


def test_scrape_page_falls_back_to_table(monkeypatch):
    monkeypatch.setattr(scraper, "fetch_html", lambda url: "<html><body>" + TABLE_HTML + "</body></html>")

    records = _by_id(scrape_page("leaderboard", "https://example.test"))

    assert set(records) == {"claude-opus-5-5", "jt-35b-flash"}
    assert records["claude-opus-5-5"].intelligence_score == 58.0
