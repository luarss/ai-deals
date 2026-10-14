from __future__ import annotations

import pytest
import requests

from src import arena_fetcher
from src.arena_fetcher import fetch_all_arena, parse_rows

DATASETS_SERVER_PAGES = {
    "text_style_control": [
        {
            "rows": [
                {
                    "row_idx": 0,
                    "row": {
                        "model_name": "gemini-4-argon-high",
                        "organization": "google",
                        "license": "Proprietary",
                        "rating": 1524.767968123545,
                        "rating_lower": 1515.9280340518835,
                        "rating_upper": 1533.6079021952064,
                        "variance": 20.342385025801427,
                        "vote_count": 4903.0,
                        "rank": 1,
                        "category": "overall",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                },
                {
                    "row_idx": 2,
                    "row": {
                        "model_name": "claude-opus-5.5-high",
                        "organization": "anthropic",
                        "license": "Proprietary",
                        "rating": 1508.082505388002,
                        "rating_lower": 1500.548001518931,
                        "rating_upper": 1515.6170092570728,
                        "variance": 14.777914121382453,
                        "vote_count": 7687.0,
                        "rank": 3,
                        "category": "overall",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                }
            ],
            "num_rows_total": 6,
            "num_rows_per_page": 2,
            "partial": False
        },
        {
            "rows": [
                {
                    "row_idx": 414,
                    "row": {
                        "model_name": "stablelm-tuned-alpha-7b",
                        "organization": "",
                        "license": "CC-BY-NC-SA-4.0",
                        "rating": 953.2993417393,
                        "rating_lower": 940.6740164325547,
                        "rating_upper": 965.9246670460452,
                        "variance": 41.49435059474029,
                        "vote_count": 3287.0,
                        "rank": 415,
                        "category": "overall",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                },
                {
                    "row_idx": 415,
                    "row": {
                        "model_name": "gemini-4-argon-high",
                        "organization": "google",
                        "license": "Proprietary",
                        "rating": 1588.4474265503748,
                        "rating_lower": 1556.727491588241,
                        "rating_upper": 1620.1673615125087,
                        "variance": 261.91983852118915,
                        "vote_count": 361.0,
                        "rank": 1,
                        "category": "chinese",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                }
            ],
            "num_rows_total": 6,
            "num_rows_per_page": 2,
            "partial": False
        },
        {
            "rows": [
                {
                    "row_idx": 416,
                    "row": {
                        "model_name": "claude-opus-5.5-high",
                        "organization": "anthropic",
                        "license": "Proprietary",
                        "rating": 1574.4316618951773,
                        "rating_lower": 1551.6809161806436,
                        "rating_upper": 1597.182407609711,
                        "variance": 134.73954940739046,
                        "vote_count": 704.0,
                        "rank": 2,
                        "category": "chinese",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                },
                {
                    "row_idx": 417,
                    "row": {
                        "model_name": "claude-fable-5.1-max",
                        "organization": "anthropic",
                        "license": "Proprietary",
                        "rating": 1567.3035199780754,
                        "rating_lower": 1548.3089484623306,
                        "rating_upper": 1586.2980914938203,
                        "variance": 93.92102425337264,
                        "vote_count": 1069.0,
                        "rank": 3,
                        "category": "chinese",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                }
            ],
            "num_rows_total": 6,
            "num_rows_per_page": 2,
            "partial": False
        }
    ],
    "webdev": [
        {
            "rows": [
                {
                    "row_idx": 0,
                    "row": {
                        "model_name": "claude-opus-5.5-max",
                        "organization": "anthropic",
                        "license": "Proprietary",
                        "rating": 1811.149195823145,
                        "rating_lower": 1796.5635202192325,
                        "rating_upper": 1825.7348714270574,
                        "variance": 55.38050588399346,
                        "vote_count": 2597,
                        "rank": 1,
                        "category": "overall",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                },
                {
                    "row_idx": 25,
                    "row": {
                        "model_name": "gpt-5.6-sol-xhigh (codex-harness)",
                        "organization": "openai",
                        "license": "Proprietary",
                        "rating": 1617.0327297912622,
                        "rating_lower": 1611.0585544739884,
                        "rating_upper": 1623.006905108536,
                        "variance": 9.290941901877485,
                        "vote_count": 17276,
                        "rank": 26,
                        "category": "overall",
                        "leaderboard_publish_date": "2026-10-13"
                    },
                    "truncated_cells": []
                }
            ],
            "num_rows_total": 2,
            "num_rows_per_page": 2,
            "partial": False
        }
    ]
}


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        return self.payload


class FakeDatasetsServer:
    def __init__(self):
        self.requested_offsets: dict[str, list[int]] = {}

    def __call__(self, url, params=None, headers=None, timeout=None):
        config = params["config"]
        offset = params["offset"]
        self.requested_offsets.setdefault(config, []).append(offset)
        pages = DATASETS_SERVER_PAGES[config]
        return FakeResponse(pages[offset // arena_fetcher.PAGE_SIZE])


@pytest.fixture
def fake_server(monkeypatch):
    server = FakeDatasetsServer()
    monkeypatch.setattr(arena_fetcher, "PAGE_SIZE", 2)
    monkeypatch.setattr(arena_fetcher.requests, "get", server)
    monkeypatch.setattr(arena_fetcher.time, "sleep", lambda seconds: None)
    return server


def records_from(records, source_page):
    return [record for record in records if record.source_pages == [source_page]]


def test_fetch_all_arena_reads_overall_text_and_code(fake_server):
    records = fetch_all_arena()

    text_names = [record.name for record in records_from(records, "arena-text")]
    code_names = [record.name for record in records_from(records, "arena-code")]
    assert text_names == ["gemini-4-argon-high", "claude-opus-5.5-high", "stablelm-tuned-alpha-7b"]
    assert code_names == ["claude-opus-5.5-max", "gpt-5.6-sol-xhigh"]


def test_paging_stops_after_overall_rows_end(fake_server):
    fetch_all_arena()

    assert fake_server.requested_offsets["text_style_control"] == [0, 2, 4]
    assert fake_server.requested_offsets["webdev"] == [0]


def test_text_record_fields(fake_server):
    top = records_from(fetch_all_arena(), "arena-text")[0]

    assert top.model_id == "gemini-4-argon-high"
    assert top.creator == "google"
    assert top.arena_elo == 1525
    assert top.arena_ci == 9
    assert top.arena_votes == 4903
    assert top.arena_coding_elo is None


def test_empty_organization_leaves_creator_unset(fake_server):
    last = records_from(fetch_all_arena(), "arena-text")[-1]

    assert last.name == "stablelm-tuned-alpha-7b"
    assert last.creator is None


def test_code_record_fields_and_harness_suffix_stripped(fake_server):
    code_records = records_from(fetch_all_arena(), "arena-code")

    assert code_records[0].arena_coding_elo == 1811
    assert code_records[0].arena_elo is None
    assert code_records[0].arena_ci is None
    assert code_records[0].arena_votes == 2597
    assert code_records[1].name == "gpt-5.6-sol-xhigh"
    assert code_records[1].model_id == "gpt-5-6-sol-xhigh"


def test_parse_rows_skips_non_overall_and_incomplete_rows():
    rows = [
        {"model_name": "a", "organization": "x", "rating": 1400.2, "category": "coding"},
        {"model_name": "", "organization": "x", "rating": 1400.2, "category": "overall"},
        {"model_name": "b", "organization": "x", "rating": None, "category": "overall"},
        {"model_name": "c", "organization": "N/A", "rating": 1300.6, "category": "overall"},
    ]

    records = parse_rows(rows, "text")

    assert [record.name for record in records] == ["c"]
    assert records[0].creator is None
    assert records[0].arena_elo == 1301
    assert records[0].arena_ci is None


def test_fetch_failure_skips_leaderboard(monkeypatch):
    monkeypatch.setattr(arena_fetcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        arena_fetcher.requests,
        "get",
        lambda url, params=None, headers=None, timeout=None: FakeResponse({}, status_code=503),
    )

    assert fetch_all_arena() == []
