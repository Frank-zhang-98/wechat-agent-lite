from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from app.services.fetch_service import FetchService
from app.services.llm_gateway import LLMGateway
from app.services.search_providers.anspire_provider import AnspireSearchProvider
from app.services.search_providers.base import SearchHit, SearchProvider
from app.services.settings_service import SettingsService


class WebEnrichService:
    def __init__(self, settings: SettingsService, fetch: FetchService) -> None:
        self.settings = settings
        self.fetch = fetch

    def is_enabled(self) -> bool:
        return self.settings.get_bool("web_enrich.enabled", True)

    def build_search_plan(
        self,
        *,
        run_id: str,
        topic: dict[str, Any],
        source_pack: dict[str, Any],
        source_structure: dict[str, Any],
        content_type: str,
        evidence_score: float,
        llm: LLMGateway,
    ) -> dict[str, Any]:
        threshold = float(self.settings.get_float("web_enrich.min_evidence_score_to_skip", 60.0))
        if not self.is_enabled():
            return {
                "enabled": False,
                "should_search": False,
                "reason": "web_enrich_disabled",
                "queries": [],
                "official_domains": [],
            }
        if evidence_score >= threshold:
            return {
                "enabled": True,
                "should_search": False,
                "reason": f"evidence_score_{evidence_score}_above_threshold_{threshold}",
                "queries": [],
                "official_domains": [],
            }

        prompt = (
            "You are a search planner for factual article enrichment.\n"
            "Return strict JSON only.\n"
            "Decide whether web search is needed, then propose at most 3 focused queries.\n"
            "Search goals: verify official facts, find official docs, and collect context only when needed.\n"
            "Do not suggest speculative internal architecture searches unless the source already mentions them.\n"
            "Output schema:\n"
            "{\n"
            '  "should_search": true,\n'
            '  "reason": "...",\n'
            '  "official_domains": ["example.com"],\n'
            '  "queries": [\n'
            '    {"q": "...", "intent": "...", "source_type": "official|context", "must_include": [], "must_exclude": []}\n'
            "  ]\n"
            "}\n\n"
            f"Topic:\n{json.dumps(topic, ensure_ascii=False)}\n"
            f"Content Type: {content_type}\n"
            f"Evidence Score: {evidence_score}\n"
            f"Source Pack:\n{json.dumps(source_pack, ensure_ascii=False)[:2500]}\n"
            f"Source Structure:\n{json.dumps(source_structure, ensure_ascii=False)[:2500]}"
        )
        result = llm.call(run_id, "WEB_SEARCH_PLAN", "decision", prompt, temperature=0.1)
        parsed = self._parse_plan(result.text)
        if parsed:
            parsed["enabled"] = True
            return parsed
        fallback_query = str(topic.get("title", "") or "").strip()
        return {
            "enabled": True,
            "should_search": bool(fallback_query),
            "reason": "fallback_topic_title_query",
            "official_domains": [],
            "queries": (
                [{"q": fallback_query, "intent": "verify public facts", "source_type": "context", "must_include": [], "must_exclude": []}]
                if fallback_query
                else []
            ),
        }

    def fetch_search_results(self, *, plan: dict[str, Any]) -> dict[str, Any]:
        if not self.is_enabled():
            return {"status": "skipped", "reason": "web_enrich_disabled", "official_sources": [], "context_sources": [], "queries": []}
        if not plan.get("should_search"):
            return {"status": "skipped", "reason": plan.get("reason", "planner_skip"), "official_sources": [], "context_sources": [], "queries": []}
        provider = self._build_provider()
        if provider is None or not provider.is_available():
            return {"status": "skipped", "reason": "search_provider_unavailable", "official_sources": [], "context_sources": [], "queries": []}

        max_queries = max(1, self.settings.get_int("web_enrich.max_queries", 3))
        max_results = max(1, self.settings.get_int("web_enrich.max_results_per_query", 5))
        max_fetch_per_query = max(1, self.settings.get_int("web_enrich.max_fetch_per_query", 2))
        official_domains = {str(item).strip().lower() for item in (plan.get("official_domains") or []) if str(item).strip()}

        official_sources: list[dict[str, Any]] = []
        context_sources: list[dict[str, Any]] = []
        audit_queries: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for query_item in list(plan.get("queries") or [])[:max_queries]:
            query = str(query_item.get("q", "") or "").strip()
            if not query:
                continue
            source_type = str(query_item.get("source_type", "context") or "context").strip().lower()
            hits = provider.search(query, limit=max_results)
            accepted: list[SearchHit] = []
            for hit in hits:
                if not hit.url or hit.url in seen_urls:
                    continue
                if source_type == "official" and official_domains:
                    if not any(hit.domain.endswith(domain) for domain in official_domains):
                        continue
                seen_urls.add(hit.url)
                accepted.append(hit)
                if len(accepted) >= max_fetch_per_query:
                    break

            normalized_hits: list[dict[str, Any]] = []
            for hit in accepted:
                extract = self.fetch.extract_article_content(hit.url, max_chars=2500)
                entry = {
                    "title": hit.title,
                    "url": hit.url,
                    "snippet": hit.snippet,
                    "domain": hit.domain,
                    "source_type": source_type,
                    "status": extract.get("status", "failed"),
                    "reason": extract.get("reason", ""),
                    "content_text": extract.get("content_text", ""),
                }
                normalized_hits.append(entry)
                if source_type == "official":
                    official_sources.append(entry)
                else:
                    context_sources.append(entry)

            audit_queries.append(
                {
                    "q": query,
                    "intent": str(query_item.get("intent", "") or ""),
                    "source_type": source_type,
                    "accepted": len(normalized_hits),
                }
            )

        return {
            "status": "ok" if (official_sources or context_sources) else "empty",
            "reason": "",
            "official_sources": official_sources,
            "context_sources": context_sources,
            "queries": audit_queries,
        }

    def _build_provider(self) -> SearchProvider | None:
        provider_name = self.settings.get("search.provider", "anspire").strip().lower()
        if provider_name == "anspire":
            return AnspireSearchProvider(
                api_key=self.settings.get("search.anspire.api_key", ""),
                base_url=self.settings.get("search.anspire.base_url", ""),
                timeout_seconds=self.settings.get_int("search.request_timeout_seconds", 20),
            )
        return None

    @staticmethod
    def _parse_plan(text: str) -> dict[str, Any] | None:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            data = json.loads(text[start : end + 1])
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        queries: list[dict[str, Any]] = []
        for item in data.get("queries", []) or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q", "") or "").strip()
            if not q:
                continue
            queries.append(
                {
                    "q": q,
                    "intent": str(item.get("intent", "") or "").strip(),
                    "source_type": str(item.get("source_type", "context") or "context").strip().lower(),
                    "must_include": [str(x).strip() for x in (item.get("must_include") or []) if str(x).strip()],
                    "must_exclude": [str(x).strip() for x in (item.get("must_exclude") or []) if str(x).strip()],
                }
            )
        return {
            "should_search": bool(data.get("should_search", False)),
            "reason": str(data.get("reason", "") or ""),
            "official_domains": [
                urlparse(str(item).strip()).netloc.lower().split("@")[-1] if "://" in str(item) else str(item).strip().lower()
                for item in (data.get("official_domains") or [])
                if str(item).strip()
            ],
            "queries": queries[:3],
        }
