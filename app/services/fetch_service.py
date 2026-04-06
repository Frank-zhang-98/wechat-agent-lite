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

    def extract_article_structure(self, url: str, max_chars: int = 12000) -> dict[str, Any]:
        if not url.strip():
            return {
                "url": url,
                "status": "failed",
                "reason": "empty_url",
                "title": "",
                "lead": "",
                "sections": [],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": [],
            }
        try:
            response = self._request(url, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            return {
                "url": url,
                "status": "failed",
                "reason": f"request_failed: {exc}",
                "title": "",
                "lead": "",
                "sections": [],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": [],
            }

        content_type = (response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type and "xml" not in content_type and "text" not in content_type:
            return {
                "url": url,
                "status": "failed",
                "reason": f"unsupported_content_type: {content_type or '-'}",
                "title": "",
                "lead": "",
                "sections": [],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": [],
            }

        html_text = response.text or ""
        title = self._extract_html_title(html_text)
        structure = self._build_article_structure(html_text, title=title, max_chars=max_chars)
        structure["url"] = url
        return structure

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

    def fetch_html_list(
        self,
        source: dict[str, Any],
        *,
        max_age_hours: int,
        max_items: int,
        scrapling: Any | None = None,
    ) -> list[dict[str, Any]]:
        if scrapling is None:
            raise RuntimeError("scrapling_fallback_unavailable")
        raw_items = scrapling.build_html_list_items(
            url=str(source.get("url", "") or "").strip(),
            source_name=str(source.get("name", "") or ""),
            source_weight=float(source.get("weight", 0.7) or 0.7),
            max_items=max_items,
        )
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items: list[dict[str, Any]] = []
        for item in raw_items:
            normalized = self._normalize_html_list_item(item=item, source=source)
            if not normalized:
                continue
            try:
                published_dt = datetime.fromisoformat(str(normalized["published"]))
            except Exception:
                continue
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
            if published_dt < cutoff:
                continue
            items.append(normalized)
            if len(items) >= max_items:
                break
        return items

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
            return self.fetch_html_list(source, max_age_hours=max_age_hours, max_items=max_items, scrapling=scrapling)
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

    def extract_article_metadata(self, url: str) -> dict[str, str]:
        if not url.strip():
            return {"title": "", "published": "", "summary": ""}
        try:
            response = self._request(url, timeout=20)
            response.raise_for_status()
        except Exception:
            return {"title": "", "published": "", "summary": ""}
        content_type = (response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type and "xml" not in content_type and "text" not in content_type:
            return {"title": "", "published": "", "summary": ""}
        html_text = response.text or ""
        title = self._extract_html_title(html_text)
        published = self._extract_html_published(html_text)
        content_text, paragraphs = self._extract_main_text(html_text, max_chars=1200)
        summary = ""
        for paragraph in paragraphs:
            cleaned = self._clean_text(paragraph)
            if len(cleaned) >= 24 and cleaned != title:
                summary = cleaned[:500]
                break
        if not summary and content_text:
            summary = self._clean_text(content_text)[:500]
        return {"title": title, "published": published, "summary": summary}

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

    @classmethod
    def _extract_html_published(cls, html_text: str) -> str:
        patterns = [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+property=["\']og:published_time["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\'](?:publishdate|pubdate|date|article:published_time)["\'][^>]+content=["\'](.*?)["\']',
            r'<time[^>]+datetime=["\'](.*?)["\']',
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'"dateCreated"\s*:\s*"([^"]+)"',
            r'"dateModified"\s*:\s*"([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            normalized = cls._normalize_published_text(match.group(1))
            if normalized:
                return normalized
        # Fallback: scan visible text near the article header/body for date strings.
        visible = cls._clean_text(cls._match_first_block(html_text, html_text.lower(), "article") or cls._match_body(html_text, html_text.lower()) or html_text)
        for pattern in [
            r"\b(20\d{2}-\d{1,2}-\d{1,2})\b",
            r"\b(20\d{2}/\d{1,2}/\d{1,2})\b",
            r"\b(20\d{2}\.\d{1,2}\.\d{1,2})\b",
            r"(20\d{2}年\d{1,2}月\d{1,2}日)",
        ]:
            match = re.search(pattern, visible)
            if not match:
                continue
            normalized = cls._normalize_published_text(match.group(1))
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _normalize_published_text(value: str) -> str:
        raw = FetchService._clean_text(value)
        if not raw:
            return ""
        candidates = [raw]
        if raw.endswith("Z"):
            candidates.append(raw[:-1] + "+00:00")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            candidates.append(raw + "T00:00:00+00:00")
        if re.fullmatch(r"\d{4}/\d{2}/\d{2}", raw):
            candidates.append(raw.replace("/", "-") + "T00:00:00+00:00")
        if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", raw):
            candidates.append(raw.replace(".", "-") + "T00:00:00+00:00")
        cn_match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw)
        if cn_match:
            year, month, day = cn_match.groups()
            candidates.append(f"{int(year):04d}-{int(month):02d}-{int(day):02d}T00:00:00+00:00")
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                continue
        return ""

    def _normalize_html_list_item(self, *, item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
        url = str(item.get("url", "") or "").strip()
        if not url:
            return None
        title = str(item.get("title", "") or "").strip()
        summary = str(item.get("summary", "") or "").strip()[:500]
        published = str(item.get("published", "") or "").strip()
        needs_meta = self._looks_like_url_title(title, url) or not published or not summary
        if needs_meta:
            meta = self.extract_article_metadata(url)
            if self._looks_like_url_title(title, url):
                title = str(meta.get("title", "") or "").strip()
            if not published:
                published = str(meta.get("published", "") or "").strip()
            if not summary:
                summary = str(meta.get("summary", "") or "").strip()[:500]
        if self._looks_like_url_title(title, url):
            return None
        if not published:
            return None
        return {
            "title": title,
            "url": url,
            "summary": summary,
            "published": published,
            "source": str(source.get("name", "") or ""),
            "source_weight": float(source.get("weight", 0.7) or 0.7),
            "type": "html_list",
        }

    @staticmethod
    def _looks_like_url_title(title: str, url: str) -> bool:
        raw_title = str(title or "").strip()
        raw_url = str(url or "").strip()
        if not raw_title:
            return True
        lowered = raw_title.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return True
        return raw_title == raw_url

    @staticmethod
    def _build_article_structure(html_text: str, *, title: str, max_chars: int = 12000) -> dict[str, Any]:
        fragment = (
            FetchService._match_first_block(html_text, html_text.lower(), "article")
            or FetchService._match_first_block(html_text, html_text.lower(), "main")
            or FetchService._match_body(html_text, html_text.lower())
            or html_text
        )
        sanitized = re.sub(r"<!--.*?-->", " ", fragment, flags=re.DOTALL)
        sanitized = re.sub(
            r"<(script|style|svg|noscript|iframe|form|button|nav|footer|header|aside)[^>]*>.*?</\1>",
            " ",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        code_blocks: list[dict[str, Any]] = []
        code_placeholders: dict[str, str] = {}

        def stash_code(match: re.Match[str]) -> str:
            idx = len(code_blocks)
            attrs = match.group(1) or ""
            inner = match.group(2) or ""
            language_match = re.search(r"language-([a-z0-9_+-]+)", attrs, flags=re.IGNORECASE)
            text = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
            language = (language_match.group(1).lower() if language_match else "")
            code_blocks.append(
                {
                    "language": language,
                    "code_excerpt": text[:1200],
                    "code_text": text[:12000],
                    "line_count": len([line for line in text.splitlines() if line.strip()]),
                    "kind": FetchService._classify_code_block(text=text, language=language),
                }
            )
            token = f"__CODE_BLOCK_{idx}__"
            code_placeholders[token] = text[:1200]
            return f"\n{token}\n"

        sanitized = re.sub(
            r"<pre([^>]*)>(.*?)</pre>",
            stash_code,
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        heading_pattern = re.compile(r"<(h[1-4])[^>]*>(.*?)</\1>", flags=re.IGNORECASE | re.DOTALL)
        paragraph_pattern = re.compile(r"<p[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
        list_pattern = re.compile(r"<(ul|ol)[^>]*>(.*?)</\1>", flags=re.IGNORECASE | re.DOTALL)
        li_pattern = re.compile(r"<li[^>]*>(.*?)</li>", flags=re.IGNORECASE | re.DOTALL)
        table_pattern = re.compile(r"<table[^>]*>(.*?)</table>", flags=re.IGNORECASE | re.DOTALL)
        tr_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", flags=re.IGNORECASE | re.DOTALL)

        sections: list[dict[str, Any]] = []
        current_section: dict[str, Any] | None = None
        lead_parts: list[str] = []
        lists: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        coverage_checklist: list[str] = []
        consumed_chars = 0

        token_pattern = re.compile(
            r"(<h[1-4][^>]*>.*?</h[1-4]>|<p[^>]*>.*?</p>|<(?:ul|ol)[^>]*>.*?</(?:ul|ol)>|<table[^>]*>.*?</table>|__CODE_BLOCK_\d+__)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for token in token_pattern.findall(sanitized):
            code_match = re.fullmatch(r"__CODE_BLOCK_(\d+)__", token.strip())
            if code_match:
                if current_section is not None:
                    code_idx = int(code_match.group(1))
                    if code_idx not in current_section["code_refs"]:
                        current_section["code_refs"].append(code_idx)
                continue

            heading_match = heading_pattern.match(token)
            if heading_match:
                level = int(heading_match.group(1)[1])
                heading_text = FetchService._clean_text(re.sub(r"<[^>]+>", " ", heading_match.group(2)))
                if not heading_text:
                    continue
                if title and FetchService._clean_text(heading_text) == FetchService._clean_text(title):
                    continue
                current_section = {"heading": heading_text, "level": level, "paragraphs": [], "code_refs": [], "list_refs": [], "table_refs": []}
                sections.append(current_section)
                if heading_text not in coverage_checklist:
                    coverage_checklist.append(heading_text)
                consumed_chars += len(heading_text)
                if consumed_chars >= max_chars:
                    break
                continue

            paragraph_match = paragraph_pattern.match(token)
            if paragraph_match:
                text = re.sub(r"<[^>]+>", " ", paragraph_match.group(1))
                for placeholder, code_text in code_placeholders.items():
                    text = text.replace(placeholder, code_text)
                cleaned = FetchService._clean_text(text)
                if not cleaned:
                    continue
                if current_section is None:
                    lead_parts.append(cleaned)
                else:
                    current_section["paragraphs"].append(cleaned)
                    for idx, code in enumerate(code_blocks):
                        excerpt = str(code.get("code_excerpt", "") or "")
                        if excerpt and excerpt[:80] in cleaned and idx not in current_section["code_refs"]:
                            current_section["code_refs"].append(idx)
                consumed_chars += len(cleaned)
                if consumed_chars >= max_chars:
                    break
                continue

            list_match = list_pattern.match(token)
            if list_match:
                items = [
                    FetchService._clean_text(re.sub(r"<[^>]+>", " ", match))
                    for match in li_pattern.findall(list_match.group(2))
                ]
                items = [item for item in items if item]
                if items:
                    lists.append({"type": list_match.group(1).lower(), "items": items[:8]})
                    if current_section is not None:
                        current_section["list_refs"].append(len(lists) - 1)
                consumed_chars += sum(len(item) for item in items[:4])
                if consumed_chars >= max_chars:
                    break
                continue

            table_match = table_pattern.match(token)
            if table_match:
                rows = []
                for tr in tr_pattern.findall(table_match.group(1)):
                    cells = [FetchService._clean_text(re.sub(r"<[^>]+>", " ", cell)) for cell in cell_pattern.findall(tr)]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        rows.append(cells[:6])
                if rows:
                    tables.append({"rows": rows[:6]})
                    if current_section is not None:
                        current_section["table_refs"].append(len(tables) - 1)
                consumed_chars += sum(len(cell) for row in rows[:2] for cell in row)
                if consumed_chars >= max_chars:
                    break

        lead = "\n\n".join(lead_parts[:2]).strip()
        normalized_sections: list[dict[str, Any]] = []
        for section in sections[:12]:
            paragraphs = list(section.get("paragraphs") or [])
            normalized_sections.append(
                {
                    "heading": section.get("heading", ""),
                    "level": section.get("level", 2),
                    "summary": " ".join(paragraphs[:2])[:600],
                    "paragraphs": paragraphs[:4],
                    "code_refs": list(section.get("code_refs") or [])[:3],
                    "list_refs": list(section.get("list_refs") or [])[:2],
                    "table_refs": list(section.get("table_refs") or [])[:2],
                }
            )

        return {
            "status": "ok",
            "reason": "",
            "title": title,
            "lead": lead,
            "sections": normalized_sections,
            "code_blocks": code_blocks[:10],
            "lists": lists[:10],
            "tables": tables[:4],
            "coverage_checklist": coverage_checklist[:12],
        }

    @staticmethod
    def _classify_code_block(*, text: str, language: str) -> str:
        lowered_language = str(language or "").strip().lower()
        if lowered_language in {"bash", "shell", "sh", "zsh", "powershell", "ps1", "cmd", "console", "terminal"}:
            return "command"
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return "code"
        command_like = 0
        for line in lines[:8]:
            if line.startswith(("$ ", "# ", "PS>", "sudo ", "curl ", "ollama ", "pip ", "python ", "npm ", "uv ", "git ")):
                command_like += 1
        return "command" if command_like >= max(1, min(3, len(lines))) else "code"
