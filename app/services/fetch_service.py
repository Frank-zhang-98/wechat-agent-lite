from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests
import yaml

from app.core.config import CONFIG


def _parse_time(entry: Any) -> datetime:
    published = entry.get("published_parsed")
    if published:
        return datetime(*published[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class FetchService:
    def __init__(self, all_proxy: str | None = None):
        self.all_proxy = all_proxy or ""

    @staticmethod
    def sources_path() -> Path:
        path = Path(CONFIG.data_dir).parents[0] / "config" / "sources.yaml"
        if not path.exists():
            # fallback for dev in project root
            path = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
        return path

    def _request(self, url: str, timeout: int = 15) -> requests.Response:
        proxies = None
        if self.all_proxy:
            proxies = {"http": self.all_proxy, "https": self.all_proxy}
        headers = {"User-Agent": "wechat-agent-lite/1.0 (+rss-fetcher)"}
        try:
            return requests.get(url, timeout=timeout, headers=headers, proxies=proxies)
        except Exception as exc:
            if proxies and "SOCKS" in str(exc).upper():
                return requests.get(url, timeout=timeout, headers=headers)
            raise

    def load_sources(self) -> dict[str, Any]:
        path = self.sources_path()
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_sources(self, payload: dict[str, Any]) -> None:
        path = self.sources_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        dumped = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        temp_path.write_text(dumped, encoding="utf-8", newline="\n")
        temp_path.replace(path)

    def extract_article_content(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        if not url.strip():
            return {"url": url, "status": "failed", "reason": "empty_url", "title": "", "content_text": "", "paragraphs": []}
        try:
            response = self._request(url, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            return {
                "url": url,
                "status": "failed",
                "reason": f"request_failed: {exc}",
                "title": "",
                "content_text": "",
                "paragraphs": [],
            }

        content_type = (response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type and "xml" not in content_type and "text" not in content_type:
            return {
                "url": url,
                "status": "failed",
                "reason": f"unsupported_content_type: {content_type or '-'}",
                "title": "",
                "content_text": "",
                "paragraphs": [],
            }

        html_text = response.text or ""
        title = self._extract_html_title(html_text)
        content_text, paragraphs = self._extract_main_text(html_text, max_chars=max_chars)
        if not content_text:
            return {
                "url": url,
                "status": "failed",
                "reason": "content_not_found",
                "title": title,
                "content_text": "",
                "paragraphs": [],
            }
        return {
            "url": url,
            "status": "ok",
            "reason": "",
            "title": title,
            "content_text": content_text,
            "paragraphs": paragraphs,
        }

    def fetch_rss(self, source: dict[str, Any], max_age_hours: int, max_items: int) -> list[dict[str, Any]]:
        response = self._request(source["url"], timeout=15)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items: list[dict[str, Any]] = []
        for entry in feed.entries[:max_items]:
            published = _parse_time(entry)
            if published < cutoff:
                continue
            items.append(
                {
                    "title": entry.get("title", "").strip(),
                    "url": entry.get("link", "").strip(),
                    "summary": (entry.get("summary", "") or "").strip()[:500],
                    "published": published.isoformat(),
                    "source": source["name"],
                    "source_weight": float(source.get("weight", 0.7)),
                    "type": "rss",
                }
            )
        return items

    def fetch_html_list(self, source: dict[str, Any], max_items: int, scrapling: Any | None = None) -> list[dict[str, Any]]:
        if scrapling is None:
            raise RuntimeError("scrapling_fallback_unavailable")
        return scrapling.build_html_list_items(
            url=str(source.get("url", "") or "").strip(),
            source_name=str(source.get("name", "") or ""),
            source_weight=float(source.get("weight", 0.7) or 0.7),
            max_items=max_items,
        )

    def fetch_source(
        self,
        source: dict[str, Any],
        *,
        max_age_hours: int,
        max_items: int,
        scrapling: Any | None = None,
    ) -> list[dict[str, Any]]:
        mode = str(source.get("mode", "rss") or "rss").strip().lower()
        if mode == "html_list":
            return self.fetch_html_list(source, max_items=max_items, scrapling=scrapling)
        return self.fetch_rss(source, max_age_hours=max_age_hours, max_items=max_items)

    def fetch_github(self, github_cfg: dict[str, Any], max_age_hours: int) -> list[dict[str, Any]]:
        if not github_cfg.get("enabled", True):
            return []
        languages = " ".join(f"language:{x}" for x in github_cfg.get("languages", ["python"]))
        topics = " ".join(f"topic:{x}" for x in github_cfg.get("topics", ["llm"]))
        min_stars = int(github_cfg.get("min_stars", 10))
        query = f"{languages} {topics} stars:>={min_stars}"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": int(github_cfg.get("max_results", 20)),
        }
        proxies = None
        if self.all_proxy:
            proxies = {"http": self.all_proxy, "https": self.all_proxy}
        response = requests.get(
            "https://api.github.com/search/repositories",
            params=params,
            timeout=30,
            proxies=proxies,
            headers={"User-Agent": "wechat-agent-lite/1.0 (+github-fetcher)"},
        )
        response.raise_for_status()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        payload = response.json()
        out: list[dict[str, Any]] = []
        for repo in payload.get("items", []):
            updated = datetime.strptime(repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if updated < cutoff:
                continue
            out.append(
                {
                    "title": repo["name"],
                    "url": repo["html_url"],
                    "summary": (repo.get("description") or "")[:500],
                    "published": updated.isoformat(),
                    "source": "GitHub Trending",
                    "source_weight": 0.85,
                    "type": "github",
                    "stars": int(repo.get("stargazers_count", 0)),
                }
            )
        return out

    @staticmethod
    def dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        output: list[dict[str, Any]] = []
        for item in items:
            url = item.get("url", "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            output.append(item)
        return output

    @staticmethod
    def dump_debug(items: list[dict[str, Any]], run_id: str) -> None:
        target = CONFIG.data_dir / "runs" / run_id
        target.mkdir(parents=True, exist_ok=True)
        with (target / "hotspots.json").open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_html_title(html_text: str) -> str:
        patterns = [
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\'](.*?)["\']',
            r"<title[^>]*>(.*?)</title>",
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return FetchService._clean_text(match.group(1))
        return ""

    @staticmethod
    def _extract_main_text(html_text: str, max_chars: int = 8000) -> tuple[str, list[str]]:
        lowered = html_text.lower()
        candidates = [
            FetchService._match_first_block(html_text, lowered, "article"),
            FetchService._match_first_block(html_text, lowered, "main"),
            FetchService._match_body(html_text, lowered),
            html_text,
        ]
        best_blocks: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            blocks = FetchService._html_to_blocks(candidate)
            if len(" ".join(blocks)) > len(" ".join(best_blocks)):
                best_blocks = blocks

        filtered: list[str] = []
        seen: set[str] = set()
        total_len = 0
        for block in best_blocks:
            cleaned = FetchService._clean_text(block)
            if len(cleaned) < 24:
                continue
            lowered_block = cleaned.lower()
            if any(noise in lowered_block for noise in ("cookie", "subscribe", "privacy policy", "all rights reserved")):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            filtered.append(cleaned)
            total_len += len(cleaned)
            if total_len >= max_chars:
                break

        content_text = "\n\n".join(filtered)
        return content_text[:max_chars].strip(), filtered[:24]

    @staticmethod
    def _match_first_block(html_text: str, lowered: str, tag_name: str) -> str:
        start = lowered.find(f"<{tag_name}")
        if start < 0:
            return ""
        end = lowered.find(f"</{tag_name}>", start)
        if end < 0:
            return ""
        end += len(tag_name) + 3
        return html_text[start:end]

    @staticmethod
    def _match_body(html_text: str, lowered: str) -> str:
        start = lowered.find("<body")
        if start < 0:
            return ""
        end = lowered.find("</body>", start)
        if end < 0:
            return ""
        return html_text[start : end + len("</body>")]

    @staticmethod
    def _html_to_blocks(fragment: str) -> list[str]:
        text = fragment
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"<(script|style|svg|noscript|iframe|form|button|nav|footer|header|aside)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|section|article|main|li|ul|ol|h1|h2|h3|h4|h5|h6)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        blocks = re.split(r"\n+", text)
        return [FetchService._clean_text(block) for block in blocks if FetchService._clean_text(block)]

    @staticmethod
    def _clean_text(value: str) -> str:
        text = re.sub(r"\s+", " ", html.unescape(value or "")).strip()
        return text
