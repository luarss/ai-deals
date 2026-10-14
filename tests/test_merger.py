from __future__ import annotations

import pytest

from src.merger import base_model_key, merge_records, normalise_name
from src.models import RawModelRecord


def aa(model_id: str, name: str, intelligence: float = 40.0, creator: str | None = None) -> RawModelRecord:
    return RawModelRecord(
        model_id=model_id,
        name=name,
        creator=creator,
        intelligence_score=intelligence,
        source_pages=["leaderboard"],
    )


def openrouter(or_id: str, name: str, price_in: float = 1.0, price_out: float = 4.0, uptime: float | None = 99.5) -> RawModelRecord:
    slug = or_id.split("/", 1)[1].lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in slug).strip("-")
    return RawModelRecord(
        model_id=slug,
        name=name,
        openrouter_model_id=or_id,
        openrouter_price_input=price_in,
        openrouter_price_output=price_out,
        uptime_1d=uptime,
        uptime_30m=uptime,
        source_pages=["openrouter"],
    )


def by_id(models):
    return {m.model_id: m for m in models}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Anthropic: Claude Haiku 5.5", "claude haiku 5 5"),
        ("Claude Haiku 5.5 (xhigh)", "claude haiku 5 5 xhigh"),
        ("Anthropic: Claude Haiku 5.5 (batch)", "claude haiku 5 5 batch"),
        ("anthropic/claude-haiku-5.5:free", "claude haiku 5 5 free"),
        ("GPT-4o Mini (Nov '24)", "gpt 4o mini"),
        ("DeepSeek V4.1 Flash", "deepseek v4 1 flash"),
        ("Qwen: Qwen3.8 Max (0902)", "qwen3 8 max"),
        ("Command A+", "command a plus"),
        ("Mi:dm K 2.5 Pro", "midm k 2 5 pro"),
    ],
)
def test_normalise_name_keeps_variants_and_drops_creator_prefix(name, expected):
    assert normalise_name(name) == expected


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        ("Claude Haiku 5.5 (xhigh)", {}, "claude haiku 5 5"),
        ("Claude Opus 5.5 (xhigh with fallback)", {}, "claude opus 5 5"),
        ("DeepSeek V4.1 Flash (non-reasoning)", {}, "deepseek v4 1 flash"),
        ("DeepSeek: DeepSeek V4.1 Flash (batch)", {}, "deepseek v4 1 flash"),
        ("deepseek/deepseek-v4.1-flash:batch", {}, "deepseek v4 1 flash"),
        ("Anthropic: Claude Haiku 5.5", {}, "claude haiku 5 5"),
        ("Gemma 4 31B (thinking)", {}, "gemma 4 31b"),
        ("Qwen3 Coder (reasoning)", {}, "qwen3 coder"),
        ("nvidia-nemotron-3-nano-30b-a3b-reasoning", {"is_slug": True}, "nemotron 3 nano 30b a3b"),
        ("claude-haiku-5-5-xhigh", {"is_slug": True}, "claude haiku 5 5"),
        ("deepseek-v4-1-flash-non-reasoning", {"is_slug": True}, "deepseek v4 1 flash"),
        ("amazon/nova-premier-v1", {}, "nova premier"),
        ("Meta: Llama 3.3 70B Instruct", {}, "llama 3 3 70b"),
        ("DeepSeek: DeepSeek V4 Flash Vision Exp", {"loose": True}, "deepseek v4 flash vision"),
        ("Mistral: Ministral 3 14B 2512", {"loose": True}, "ministral 3 14b"),
        ("Amazon: Nova 2.0 Lite", {"loose": True}, "nova 2 lite"),
    ],
)
def test_base_model_key_strips_variants(name, kwargs, expected):
    assert base_model_key(name, **kwargs) == expected


def test_base_model_key_strict_keeps_date_stamps():
    assert base_model_key("DeepSeek V4 Pro 0813 (non-reasoning)") == "deepseek v4 pro 0813"
    assert base_model_key("DeepSeek: DeepSeek V4 Pro 0423") == "deepseek v4 pro 0423"


def test_effort_variants_stay_separate_rows_but_inherit_openrouter_price():
    records = [
        aa("claude-haiku-5-5", "Claude Haiku 5.5 (max)", 45),
        aa("claude-haiku-5-5-xhigh", "Claude Haiku 5.5 (xhigh)", 44),
        aa("claude-haiku-5-5-high", "Claude Haiku 5.5 (high)", 42),
        aa("claude-haiku-5-5-low", "Claude Haiku 5.5 (low)", 35),
        openrouter("anthropic/claude-haiku-5.5", "Anthropic: Claude Haiku 5.5", 1.0, 5.0, 99.0),
        openrouter("anthropic/claude-haiku-5.5:batch", "Anthropic: Claude Haiku 5.5 (batch)", 0.5, 2.5, 97.0),
    ]
    models = by_id(merge_records(records))

    assert set(models) == {
        "claude-haiku-5-5",
        "claude-haiku-5-5-xhigh",
        "claude-haiku-5-5-high",
        "claude-haiku-5-5-low",
        "claude-haiku-5-5-batch",
    }
    for model_id in ("claude-haiku-5-5", "claude-haiku-5-5-xhigh", "claude-haiku-5-5-high", "claude-haiku-5-5-low"):
        model = models[model_id]
        assert model.openrouter_price_input == 1.0
        assert model.openrouter_price_output == 5.0
        assert model.uptime_1d == 99.0
        assert model.openrouter_model_id == "anthropic/claude-haiku-5.5"
        assert model.source_pages == ["leaderboard", "openrouter"]
    assert models["claude-haiku-5-5-xhigh"].intelligence_score == 44
    assert models["claude-haiku-5-5"].name == "Claude Haiku 5.5 (max)"
    batch = models["claude-haiku-5-5-batch"]
    assert batch.intelligence_score is None
    assert batch.openrouter_price_input == 0.5


def test_non_reasoning_variant_inherits_base_price_not_batch():
    records = [
        aa("deepseek-v4-1-flash", "DeepSeek V4.1 Flash (max)", 38),
        aa("deepseek-v4-1-flash-non-reasoning", "DeepSeek V4.1 Flash (non-reasoning)", 30),
        openrouter("deepseek/deepseek-v4.1-flash:batch", "DeepSeek: DeepSeek V4.1 Flash (batch)", 0.1, 0.2),
        openrouter("deepseek/deepseek-v4.1-flash", "DeepSeek: DeepSeek V4.1 Flash", 0.2, 0.4),
    ]
    models = by_id(merge_records(records))

    assert models["deepseek-v4-1-flash"].openrouter_price_input == 0.2
    assert models["deepseek-v4-1-flash-non-reasoning"].openrouter_price_input == 0.2
    assert models["deepseek-v4-1-flash-non-reasoning"].openrouter_model_id == "deepseek/deepseek-v4.1-flash"


def test_vision_max_links_to_experimental_openrouter_listing():
    records = [
        aa("deepseek-v4-flash-vision", "DeepSeek V4 Flash Vision (max)", 33),
        openrouter("deepseek/deepseek-v4-flash-vision-exp", "DeepSeek: DeepSeek V4 Flash Vision Exp", 0.3, 0.9),
        openrouter("~deepseek/deepseek-v4-flash-latest", "DeepSeek: DeepSeek V4 Flash Latest", 9.0, 9.0),
    ]
    models = by_id(merge_records(records))

    vision = models["deepseek-v4-flash-vision"]
    assert vision.openrouter_model_id == "deepseek/deepseek-v4-flash-vision-exp"
    assert vision.openrouter_price_input == 0.3


def test_dated_release_links_to_matching_date_not_sibling():
    records = [
        aa("deepseek-v4-pro-non-reasoning", "DeepSeek V4 Pro 0813 (non-reasoning)", 31),
        openrouter("deepseek/deepseek-v4-pro", "DeepSeek: DeepSeek V4 Pro 0423", 1.0, 3.0),
        openrouter("deepseek/deepseek-v4-pro-0813", "DeepSeek: DeepSeek V4 Pro 0813", 1.5, 4.0),
    ]
    models = by_id(merge_records(records))

    assert models["deepseek-v4-pro-non-reasoning"].openrouter_model_id == "deepseek/deepseek-v4-pro-0813"


def test_ambiguous_loose_key_does_not_link():
    records = [
        aa("deepseek-v4-flash-x", "DeepSeek V4 Flash 0601", 30),
        openrouter("deepseek/deepseek-v4-flash", "DeepSeek: DeepSeek V4 Flash 0423"),
        openrouter("deepseek/deepseek-v4-flash-0731", "DeepSeek: DeepSeek V4 Flash 0731"),
    ]
    models = by_id(merge_records(records))

    assert models["deepseek-v4-flash-x"].openrouter_price_input is None
    assert models["deepseek-v4-flash-x"].source_pages == ["leaderboard"]


def test_creator_prefixed_openrouter_name_merges_into_aa_row():
    records = [
        aa("step-5", "Step 5 Preview", 41, creator="StepFun"),
        openrouter("stepfun/step-5-preview", "StepFun: Step 5 Preview", 2.0, 6.0),
    ]
    models = merge_records(records)

    assert len(models) == 1
    merged = models[0]
    assert merged.model_id == "step-5"
    assert merged.name == "Step 5 Preview"
    assert merged.intelligence_score == 41
    assert merged.openrouter_price_input == 2.0
    assert merged.source_pages == ["leaderboard", "openrouter"]


def test_same_slug_merge_keeps_aa_id_and_priority():
    records = [
        aa("claude-haiku-5-5", "Claude Haiku 5.5 (max)", 45),
        openrouter("anthropic/claude-haiku-5.5", "Anthropic: Claude Haiku 5.5", 1.0, 5.0),
    ]
    records[1] = records[1].model_copy(update={"intelligence_score": 1.0, "creator": "Anthropic"})
    models = merge_records(records)

    assert len(models) == 1
    assert models[0].model_id == "claude-haiku-5-5"
    assert models[0].intelligence_score == 45
    assert models[0].creator == "Anthropic"


def test_same_source_records_never_collapse_by_name():
    records = [
        aa("qwen3-next-80b-a3b-reasoning", "Qwen3 Next 80B A3B", 30),
        aa("qwen3-next-80b-a3b-instruct", "Qwen3 Next 80B A3B", 25),
        openrouter("deepseek/deepseek-chat", "DeepSeek: DeepSeek V3"),
        openrouter("deepseek/deepseek-chat-v3.1", "DeepSeek: DeepSeek V3.1"),
    ]
    models = by_id(merge_records(records))

    assert models["qwen3-next-80b-a3b-reasoning"].intelligence_score == 30
    assert models["qwen3-next-80b-a3b-instruct"].intelligence_score == 25
    assert {"deepseek-chat", "deepseek-chat-v3-1"} <= set(models)


def test_existing_openrouter_data_is_not_overwritten_by_inheritance():
    records = [
        aa("gpt-6-astra", "GPT-6 Astra (max)", 55),
        aa("gpt-6-astra-pro", "GPT-6 Astra Pro (xhigh)", 57),
        openrouter("openai/gpt-6-astra", "OpenAI: GPT-6 Astra", 5.0, 20.0),
        openrouter("openai/gpt-6-astra-pro", "OpenAI: GPT-6 Astra Pro", 30.0, 120.0),
    ]
    models = by_id(merge_records(records))

    assert models["gpt-6-astra"].openrouter_price_input == 5.0
    assert models["gpt-6-astra-pro"].openrouter_price_input == 30.0


def test_merge_records_empty():
    assert merge_records([]) == []
