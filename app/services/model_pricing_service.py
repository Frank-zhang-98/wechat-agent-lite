from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests

from app.core.config import CONFIG

PRICE_CATALOG_CURRENCY = "CNY"
PRICE_CATALOG_BUILT_IN_VERSION = "2026-03-25"
PRICE_SYNC_MAX_AGE_HOURS = 12
PRICE_CACHE_PATH = CONFIG.data_dir / "pricing_catalog.json"
SOURCE_URLS = {
    "model_pricing": "https://help.aliyun.com/zh/model-studio/model-pricing",
    "text_rerank": "https://help.aliyun.com/zh/model-studio/billing-for-text-rerank",
}


@dataclass(frozen=True)
class PriceTier:
    label: str
    max_input_tokens: int | None
    prompt_price_per_million: float
    completion_price_per_million: float = 0.0


@dataclass(frozen=True)
class PriceRule:
    canonical_model: str
    billing_kind: str
    tiers: tuple[PriceTier, ...] = ()
    image_price_per_call: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class CostEstimate:
    supported: bool
    provider: str
    model: str
    canonical_model: str
    billing_kind: str
    tier_label: str
    prompt_cost: float
    completion_cost: float
    image_cost: float
    total_cost: float
    note: str = ""


_BUILT_IN_RULES: dict[str, dict[str, PriceRule]] = {
    "alibaba_bailian": {
        "qwen-plus": PriceRule(
            canonical_model="qwen-plus",
            billing_kind="text",
            tiers=(
                PriceTier(label="input<=128K", max_input_tokens=128000, prompt_price_per_million=0.8, completion_price_per_million=2.0),
                PriceTier(label="128K<input<=256K", max_input_tokens=256000, prompt_price_per_million=2.4, completion_price_per_million=20.0),
                PriceTier(label="256K<input<=1M", max_input_tokens=1_000_000, prompt_price_per_million=4.8, completion_price_per_million=48.0),
            ),
            note="Built-in fallback for qwen-plus.",
        ),
        "qwen-flash": PriceRule(
            canonical_model="qwen-flash",
            billing_kind="text",
            tiers=(
                PriceTier(label="input<=128K", max_input_tokens=128000, prompt_price_per_million=0.15, completion_price_per_million=1.5),
                PriceTier(label="128K<input<=256K", max_input_tokens=256000, prompt_price_per_million=0.6, completion_price_per_million=6.0),
            ),
            note="Built-in fallback for qwen-flash.",
        ),
        "qwen-max": PriceRule(
            canonical_model="qwen-max",
            billing_kind="text",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=2.4, completion_price_per_million=9.6),
            ),
            note="Built-in fallback for qwen-max.",
        ),
        "qwen3.5-plus": PriceRule(
            canonical_model="qwen3.5-plus",
            billing_kind="text",
            tiers=(
                PriceTier(label="input<=128K", max_input_tokens=128000, prompt_price_per_million=0.8, completion_price_per_million=4.8),
                PriceTier(label="128K<input<=256K", max_input_tokens=256000, prompt_price_per_million=2.0, completion_price_per_million=12.0),
                PriceTier(label="256K<input<=1M", max_input_tokens=1_000_000, prompt_price_per_million=4.0, completion_price_per_million=24.0),
            ),
            note="Built-in fallback for qwen3.5-plus.",
        ),
        "deepseek-v3.2": PriceRule(
            canonical_model="deepseek-v3.2",
            billing_kind="text",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=2.0, completion_price_per_million=3.0),
            ),
            note="Built-in fallback for deepseek-v3.2.",
        ),
        "kimi-k2.5": PriceRule(
            canonical_model="kimi-k2.5",
            billing_kind="text",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=4.0, completion_price_per_million=21.0),
            ),
            note="Built-in fallback for kimi-k2.5.",
        ),
        "moonshot-kimi-k2-instruct": PriceRule(
            canonical_model="moonshot-kimi-k2-instruct",
            billing_kind="text",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=4.0, completion_price_per_million=16.0),
            ),
            note="Built-in fallback for Moonshot-Kimi-K2-Instruct.",
        ),
        "glm-5": PriceRule(
            canonical_model="glm-5",
            billing_kind="text",
            tiers=(
                PriceTier(label="input<=32K", max_input_tokens=32000, prompt_price_per_million=4.0, completion_price_per_million=18.0),
                PriceTier(label="32K<input<=198K", max_input_tokens=198000, prompt_price_per_million=6.0, completion_price_per_million=22.0),
            ),
            note="Built-in fallback for glm-5.",
        ),
        "glm-4.7": PriceRule(
            canonical_model="glm-4.7",
            billing_kind="text",
            tiers=(
                PriceTier(label="input<=32K", max_input_tokens=32000, prompt_price_per_million=3.0, completion_price_per_million=14.0),
                PriceTier(label="32K<input<=166K", max_input_tokens=166000, prompt_price_per_million=4.0, completion_price_per_million=16.0),
            ),
            note="Built-in fallback for glm-4.7.",
        ),
        "qwen3-rerank": PriceRule(
            canonical_model="qwen3-rerank",
            billing_kind="rerank",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=0.5, completion_price_per_million=0.0),
            ),
            note="Built-in fallback for qwen3-rerank.",
        ),
        "qwen3-rerank-plus": PriceRule(
            canonical_model="qwen3-rerank-plus",
            billing_kind="rerank",
            tiers=(
                PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=1.0, completion_price_per_million=0.0),
            ),
            note="Built-in fallback for qwen3-rerank-plus.",
        ),
        "wanx2.0-t2i-turbo": PriceRule(
            canonical_model="wanx2.0-t2i-turbo",
            billing_kind="image",
            image_price_per_call=0.04,
            note="Built-in fallback for wanx2.0-t2i-turbo.",
        ),
        "wan2.6-t2i": PriceRule(
            canonical_model="wan2.6-t2i",
            billing_kind="image",
            image_price_per_call=0.20,
            note="Built-in fallback for wan2.6-t2i.",
        ),
    }
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "qwen-plus": ("qwen-plus", "qwen-plus-latest"),
    "qwen-flash": ("qwen-flash", "qwen-flash-latest"),
    "qwen-max": ("qwen-max", "qwen-max-latest"),
    "qwen3.5-plus": ("qwen3.5-plus", "qwen3.5-plus-latest"),
    "deepseek-v3.2": ("deepseek-v3.2", "deepseek-v3.2-exp"),
    "kimi-k2.5": ("kimi-k2.5", "kimi/kimi-k2.5", "moonshot-kimi-k2.5"),
    "moonshot-kimi-k2-instruct": ("moonshot-kimi-k2-instruct",),
    "glm-5": ("glm-5",),
    "glm-4.7": ("glm-4.7",),
    "qwen3-rerank": ("qwen3-rerank",),
    "qwen3-rerank-plus": ("qwen3-rerank-plus",),
    "wanx2.0-t2i-turbo": ("wanx2.0-t2i-turbo",),
    "wan2.6-t2i": ("wan2.6-t2i",),
}

_MODULE_CACHE: dict[str, Any] = {"catalog": None}


def pricing_catalog_meta() -> dict:
    return dict(get_pricing_catalog(auto_sync=True).get("meta", {}))


def warm_pricing_catalog() -> dict:
    return get_pricing_catalog(auto_sync=True)


def get_pricing_catalog(*, auto_sync: bool = True, force_sync: bool = False) -> dict:
    catalog = _MODULE_CACHE.get("catalog")
    if catalog is None:
        catalog = _load_catalog_from_disk() or _build_built_in_catalog()
    if force_sync or (auto_sync and _is_catalog_stale(catalog)):
        synced = sync_pricing_catalog()
        if synced is not None:
            catalog = synced
        else:
            catalog = _mark_sync_failure(catalog)
    _MODULE_CACHE["catalog"] = catalog
    return catalog


def sync_pricing_catalog() -> dict | None:
    try:
        pages = {
            name: requests.get(url, timeout=30).text
            for name, url in SOURCE_URLS.items()
        }
        synced_rules = _extract_official_rules(pages)
        catalog = _build_built_in_catalog()
        merged_rules = _dict_to_rules(catalog["rules"])
        synced_models: list[str] = []
        fallback_models: list[str] = []
        for provider, model_rules in merged_rules.items():
            for model_name in list(model_rules.keys()):
                synced_rule = synced_rules.get(provider, {}).get(model_name)
                if synced_rule is not None:
                    model_rules[model_name] = synced_rule
                    synced_models.append(f"{provider}:{model_name}")
                else:
                    fallback_models.append(f"{provider}:{model_name}")
        now = datetime.now(UTC).isoformat()
        catalog = {
            "meta": {
                "currency": PRICE_CATALOG_CURRENCY,
                "catalog_updated_at": now,
                "pricing_mode": "official_auto_sync",
                "built_in_version": PRICE_CATALOG_BUILT_IN_VERSION,
                "last_sync_at": now,
                "last_sync_error": "",
                "sync_interval_hours": PRICE_SYNC_MAX_AGE_HOURS,
                "source_urls": list(SOURCE_URLS.values()),
                "synced_models": sorted(synced_models),
                "fallback_models": sorted(fallback_models),
            },
            "rules": _rules_to_dict(merged_rules),
        }
        _write_catalog_to_disk(catalog)
        _MODULE_CACHE["catalog"] = catalog
        return catalog
    except Exception as exc:
        catalog = _load_catalog_from_disk() or _build_built_in_catalog()
        catalog = _mark_sync_failure(catalog, str(exc))
        _MODULE_CACHE["catalog"] = catalog
        return None


def estimate_call_cost(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    call_count: int = 1,
) -> CostEstimate:
    normalized_provider = _normalize_provider(provider)
    normalized_model = (model or "").strip().lower()
    canonical_model = _resolve_canonical_model(normalized_provider, normalized_model)
    catalog = get_pricing_catalog(auto_sync=True)
    rules = _dict_to_rules(catalog.get("rules", {}))
    rule = rules.get(normalized_provider, {}).get(canonical_model)
    if not canonical_model or rule is None:
        return CostEstimate(
            supported=False,
            provider=normalized_provider,
            model=model,
            canonical_model=canonical_model or normalized_model or "unknown",
            billing_kind="unknown",
            tier_label="uncovered",
            prompt_cost=0.0,
            completion_cost=0.0,
            image_cost=0.0,
            total_cost=0.0,
            note="The current pricing catalog does not cover this model.",
        )

    if rule.billing_kind == "image":
        image_cost = round(rule.image_price_per_call * max(call_count, 1), 6)
        return CostEstimate(
            supported=True,
            provider=normalized_provider,
            model=model,
            canonical_model=canonical_model,
            billing_kind="image",
            tier_label="per_call",
            prompt_cost=0.0,
            completion_cost=0.0,
            image_cost=image_cost,
            total_cost=image_cost,
            note=rule.note,
        )

    tier = _resolve_tier(rule.tiers, max(prompt_tokens, 0))
    if tier is None:
        return CostEstimate(
            supported=False,
            provider=normalized_provider,
            model=model,
            canonical_model=canonical_model,
            billing_kind=rule.billing_kind,
            tier_label="uncovered",
            prompt_cost=0.0,
            completion_cost=0.0,
            image_cost=0.0,
            total_cost=0.0,
            note="The current pricing catalog does not cover this token tier.",
        )
    prompt_cost = round(max(prompt_tokens, 0) / 1_000_000 * tier.prompt_price_per_million, 6)
    completion_cost = 0.0
    if rule.billing_kind == "text":
        completion_cost = round(max(completion_tokens, 0) / 1_000_000 * tier.completion_price_per_million, 6)
    total_cost = round(prompt_cost + completion_cost, 6)
    return CostEstimate(
        supported=True,
        provider=normalized_provider,
        model=model,
        canonical_model=canonical_model,
        billing_kind=rule.billing_kind,
        tier_label=tier.label,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        image_cost=0.0,
        total_cost=total_cost,
        note=rule.note,
    )


def _build_built_in_catalog() -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "meta": {
            "currency": PRICE_CATALOG_CURRENCY,
            "catalog_updated_at": PRICE_CATALOG_BUILT_IN_VERSION,
            "pricing_mode": "built_in_fallback",
            "built_in_version": PRICE_CATALOG_BUILT_IN_VERSION,
            "last_sync_at": None,
            "last_sync_error": "",
            "sync_interval_hours": PRICE_SYNC_MAX_AGE_HOURS,
            "source_urls": list(SOURCE_URLS.values()),
            "synced_models": [],
            "fallback_models": sorted(f"{provider}:{model}" for provider, rules in _BUILT_IN_RULES.items() for model in rules),
            "created_at": now,
        },
        "rules": _rules_to_dict(_BUILT_IN_RULES),
    }


def _mark_sync_failure(catalog: dict, error: str = "") -> dict:
    meta = dict(catalog.get("meta", {}))
    meta["last_sync_error"] = error or meta.get("last_sync_error") or "official sync failed"
    if meta.get("pricing_mode") != "official_auto_sync":
        meta["pricing_mode"] = "built_in_fallback"
    updated = dict(catalog)
    updated["meta"] = meta
    return updated


def _load_catalog_from_disk() -> dict | None:
    try:
        if not PRICE_CACHE_PATH.exists():
            return None
        raw = json.loads(PRICE_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "rules" not in raw or "meta" not in raw:
            return None
        return raw
    except Exception:
        return None


def _write_catalog_to_disk(catalog: dict) -> None:
    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICE_CACHE_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_catalog_stale(catalog: dict) -> bool:
    meta = catalog.get("meta", {}) if isinstance(catalog, dict) else {}
    raw_ts = meta.get("last_sync_at") or meta.get("catalog_updated_at")
    if meta.get("pricing_mode") != "official_auto_sync":
        return True
    if not raw_ts:
        return True
    try:
        timestamp = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
    except Exception:
        return True
    return datetime.now(UTC) - timestamp.astimezone(UTC) >= timedelta(hours=PRICE_SYNC_MAX_AGE_HOURS)


def _rules_to_dict(rules: dict[str, dict[str, PriceRule]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        provider: {
            model: {
                "canonical_model": rule.canonical_model,
                "billing_kind": rule.billing_kind,
                "tiers": [asdict(tier) for tier in rule.tiers],
                "image_price_per_call": rule.image_price_per_call,
                "note": rule.note,
            }
            for model, rule in model_rules.items()
        }
        for provider, model_rules in rules.items()
    }


def _dict_to_rules(raw: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, PriceRule]]:
    output: dict[str, dict[str, PriceRule]] = {}
    for provider, model_rules in (raw or {}).items():
        output[provider] = {}
        for model, item in (model_rules or {}).items():
            tiers = tuple(
                PriceTier(
                    label=str(tier.get("label", "standard")),
                    max_input_tokens=tier.get("max_input_tokens"),
                    prompt_price_per_million=float(tier.get("prompt_price_per_million", 0.0) or 0.0),
                    completion_price_per_million=float(tier.get("completion_price_per_million", 0.0) or 0.0),
                )
                for tier in item.get("tiers", [])
            )
            output[provider][model] = PriceRule(
                canonical_model=str(item.get("canonical_model", model)),
                billing_kind=str(item.get("billing_kind", "unknown")),
                tiers=tiers,
                image_price_per_call=float(item.get("image_price_per_call", 0.0) or 0.0),
                note=str(item.get("note", "")),
            )
    return output


def _resolve_tier(tiers: Iterable[PriceTier], prompt_tokens: int) -> PriceTier | None:
    for tier in tiers:
        if tier.max_input_tokens is None or prompt_tokens <= tier.max_input_tokens:
            return tier
    return None


def _normalize_provider(provider: str) -> str:
    raw = (provider or "").strip().lower()
    if raw in {"alibaba-bailian", "alibaba_bailian", "dashscope", "aliyun_bailian"}:
        return "alibaba_bailian"
    return raw or "unknown"


def _resolve_canonical_model(provider: str, model: str) -> str:
    if provider != "alibaba_bailian":
        return model
    for canonical, aliases in _ALIASES.items():
        if any(model == alias or model.startswith(f"{alias}-") for alias in aliases):
            return canonical
    return model


def _extract_official_rules(pages: dict[str, str]) -> dict[str, dict[str, PriceRule]]:
    model_page = pages.get("model_pricing", "")
    rerank_page = pages.get("text_rerank", "")
    rules: dict[str, dict[str, PriceRule]] = {"alibaba_bailian": {}}

    def add(rule: PriceRule) -> None:
        rules["alibaba_bailian"][rule.canonical_model] = rule

    add(_parse_text_tier_rule(model_page, "qwen-plus", [(r"0<Token≤128K", 128000), (r"128K<Token≤256K", 256000), (r"256K<Token≤1M", 1_000_000)]))
    add(_parse_text_tier_rule(model_page, "qwen-flash", [(r"0<Token≤128K", 128000), (r"128K<Token≤256K", 256000)]))
    add(_parse_text_flat_rule(model_page, "qwen-max"))
    add(_parse_text_tier_rule(model_page, "qwen3.5-plus", [(r"0<Token≤128K", 128000), (r"128K<Token≤256K", 256000), (r"256K<Token≤1M", 1_000_000)]))
    add(_parse_text_flat_rule(model_page, "deepseek-v3.2"))
    add(_parse_text_flat_rule(model_page, "kimi-k2.5"))
    add(_parse_text_flat_rule(model_page, "moonshot-kimi-k2-instruct"))
    add(_parse_text_tier_rule(model_page, "glm-5", [(r"0<Token≤32K", 32000), (r"32K<Token≤198K", 198000)]))
    add(_parse_text_tier_rule(model_page, "glm-4.7", [(r"0<Token≤32K", 32000), (r"32K<Token≤166K", 166000)]))
    add(_parse_rerank_rule(rerank_page, "qwen3-rerank"))
    add(_parse_image_rule(model_page, "wanx2.0-t2i-turbo"))
    add(_parse_image_rule(model_page, "wan2.6-t2i"))

    return rules


def _parse_text_flat_rule(page_html: str, canonical_model: str) -> PriceRule:
    fragment = _extract_clean_fragment(page_html, canonical_model, 1800)
    prompt_price, completion_price = _first_price_pair(fragment)
    return PriceRule(
        canonical_model=canonical_model,
        billing_kind="text",
        tiers=(
            PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=prompt_price, completion_price_per_million=completion_price),
        ),
        note=f"Official synced from {SOURCE_URLS['model_pricing']}",
    )


def _parse_text_tier_rule(page_html: str, canonical_model: str, tiers: list[tuple[str, int]]) -> PriceRule:
    fragment = _extract_clean_fragment(page_html, canonical_model, 2600)
    parsed_tiers: list[PriceTier] = []
    for marker, max_tokens in tiers:
        prompt_price, completion_price = _price_pair_after(fragment, marker)
        parsed_tiers.append(
            PriceTier(
                label=marker.replace("≤", "<=").replace("<", "<"),
                max_input_tokens=max_tokens,
                prompt_price_per_million=prompt_price,
                completion_price_per_million=completion_price,
            )
        )
    return PriceRule(
        canonical_model=canonical_model,
        billing_kind="text",
        tiers=tuple(parsed_tiers),
        note=f"Official synced from {SOURCE_URLS['model_pricing']}",
    )


def _parse_rerank_rule(page_html: str, canonical_model: str) -> PriceRule:
    fragment = _extract_clean_fragment(page_html, canonical_model, 1200)
    prompt_price = _first_single_price(fragment)
    return PriceRule(
        canonical_model=canonical_model,
        billing_kind="rerank",
        tiers=(PriceTier(label="standard", max_input_tokens=None, prompt_price_per_million=prompt_price, completion_price_per_million=0.0),),
        note=f"Official synced from {SOURCE_URLS['text_rerank']}",
    )


def _parse_image_rule(page_html: str, canonical_model: str) -> PriceRule:
    fragment = _extract_clean_fragment(page_html, canonical_model, 900)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*元/张", fragment)
    if not match:
        raise ValueError(f"image price not found for {canonical_model}")
    return PriceRule(
        canonical_model=canonical_model,
        billing_kind="image",
        image_price_per_call=float(match.group(1)),
        note=f"Official synced from {SOURCE_URLS['model_pricing']}",
    )


def _extract_clean_fragment(page_html: str, keyword: str, extra_chars: int) -> str:
    match = re.search(re.escape(keyword), page_html, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"keyword not found: {keyword}")
    fragment = page_html[match.start(): match.start() + extra_chars]
    fragment = re.sub(r'<span class="help-letter-space"></span>', '', fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment)
    fragment = re.sub(r"\s+", " ", fragment)
    return fragment.strip()


def _first_price_pair(fragment: str) -> tuple[float, float]:
    prices = _extract_yuan_numbers(fragment)
    if len(prices) < 2:
        raise ValueError("price pair not found")
    return prices[0], prices[1]


def _price_pair_after(fragment: str, marker: str) -> tuple[float, float]:
    match = re.search(marker, fragment)
    if not match:
        raise ValueError(f"tier marker not found: {marker}")
    prices = _extract_yuan_numbers(fragment[match.end():])
    if len(prices) < 2:
        raise ValueError(f"price pair not found after marker: {marker}")
    return prices[0], prices[1]


def _first_single_price(fragment: str) -> float:
    prices = _extract_yuan_numbers(fragment)
    if not prices:
        raise ValueError("single price not found")
    return prices[0]


def _extract_yuan_numbers(fragment: str) -> list[float]:
    values = [float(item) for item in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*元", fragment)]
    return values
