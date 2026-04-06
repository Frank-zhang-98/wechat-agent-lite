from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any
from urllib.parse import urljoin, urlparse


class ScraplingFallbackService:
    DEFAULT_REPO_CANDIDATES = (
        "F:/projects/Scrapling-main",
        "F:\\projects\\Scrapling-main",
    )
    EXCLUDED_PATH_PARTS = {
        "",
        "about",
        "account",
        "api",
        "author",
        "authors",
        "careers",
        "case-study",
        "case-studies",
        "category",
        "contact",
        "docs",
        "jobs",
        "legal",
        "login",
        "pricing",
        "privacy",
        "product",
        "products",
        "register",
        "rss",
        "search",
        "sitemap",
        "signup",
        "tag",
        "tags",
        "terms",
        "topic",
        "topics",
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        repo_path: str = "",
        timeout_seconds: int = 20,
        proxy: str = "",
        max_concurrency: int = 1,
    ):
        self.enabled = enabled
        self.repo_path = repo_path.strip()
        self.timeout_seconds = max(5, timeout_seconds)
        self.proxy = proxy.strip()
        self._semaphore = BoundedSemaphore(max(1, int(max_concurrency or 1)))
        self._fetcher_cls: Any | None = None
        self._load_error = ""

    @property
    def load_error(self) -> str:
        return self._load_error

    def is_available(self) -> bool:
        return self._get_fetcher_cls() is not None

    def discover_page(self, url: str, max_articles: int = 12, timeout_seconds: float | None = None) -> dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "available": False,
            "used": False,
            "page_url": url,
            "feed_links": [],
            "articles": [],
            "error": "",
        }
        if not self.enabled:
            result["error"] = "disabled"
            return result

        fetcher_cls = self._get_fetcher_cls()
        if not fetcher_cls:
            result["error"] = self._load_error or "scrapling_unavailable"
            return result

        try:
            request_timeout = max(1.0, float(timeout_seconds)) if timeout_seconds is not None else self.timeout_seconds
            with self._semaphore:
                page = fetcher_cls.get(
                    url,
                    timeout=request_timeout,
                    follow_redirects=True,
                    stealthy_headers=True,
                    retries=1,
                    retry_delay=0,
                    proxy=self.proxy or None,
                )
        except Exception as exc:
            result["available"] = True
            result["error"] = f"fetch_failed: {exc}"
            return result

        result["available"] = True
        result["used"] = True
        result["page_url"] = str(getattr(page, "url", "") or url)
        status = int(getattr(page, "status", 0) or 0)
        if status >= 400:
            result["error"] = f"http_{status}"
            return result

        result["feed_links"] = self._extract_feed_links(page=page, page_url=result["page_url"])
        result["articles"] = self._extract_article_links(page=page, page_url=result["page_url"], limit=max_articles)
        return result

    def build_html_list_items(
        self,
        *,
        url: str,
        source_name: str,
        source_weight: float,
        max_items: int,
    ) -> list[dict[str, Any]]:
        discovered = self.discover_page(url=url, max_articles=max_items * 3)
        if not discovered.get("articles"):
            raise RuntimeError(discovered.get("error") or "no_article_links_found")
        items: list[dict[str, Any]] = []
        for article in list(discovered["articles"])[:max_items]:
            items.append(
                {
                    "title": str(article.get("title", "") or "").strip(),
                    "url": str(article.get("url", "") or "").strip(),
                    "summary": str(article.get("summary", "") or "").strip()[:500],
                    "published": str(article.get("published", "") or "").strip(),
                    "source": source_name,
                    "source_weight": float(source_weight),
                    "type": "html_list",
                }
            )
        return [item for item in items if item["url"]]

    def _get_fetcher_cls(self) -> Any | None:
        if self._fetcher_cls is not None:
            return self._fetcher_cls
        if not self.enabled:
            self._load_error = "disabled"
            return None
        try:
            self._fetcher_cls = importlib.import_module("scrapling").Fetcher
            return self._fetcher_cls
        except Exception:
            pass

        for candidate in self._repo_candidates():
            try:
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                self._fetcher_cls = importlib.import_module("scrapling").Fetcher
                return self._fetcher_cls
            except Exception as exc:
                self._load_error = str(exc)
                continue
        if not self._load_error:
            self._load_error = "scrapling_import_failed"
        return None

    def _repo_candidates(self) -> list[str]:
        candidates: list[str] = []
        env_path = os.getenv("WAL_SCRAPLING_PATH", "").strip()
        for candidate in (self.repo_path, env_path, *self.DEFAULT_REPO_CANDIDATES):
            if not candidate:
                continue
            normalized = str(Path(candidate))
            if normalized not in candidates and Path(normalized).exists():
                candidates.append(normalized)
        return candidates

    def _extract_feed_links(self, *, page: Any, page_url: str) -> list[str]:
        links: list[str] = []
        nodes = self._safe_css(page, "link[href], a[href]")
        for node in nodes:
            href = str(getattr(node, "attrib", {}).get("href", "") or "").strip()
            if not href:
                continue
            link_type = str(getattr(node, "attrib", {}).get("type", "") or "").lower()
            rel = str(getattr(node, "attrib", {}).get("rel", "") or "").lower()
            text = str(getattr(node, "text", "") or "").lower()
            if not any(token in f"{href.lower()} {link_type} {rel} {text}" for token in ("rss", "feed", "atom", "xml")):
                continue
            resolved = urljoin(page_url, href)
            if self._is_same_host(page_url, resolved):
                links.append(resolved)
        return self._dedup_preserve_order(links)

    def _extract_article_links(self, *, page: Any, page_url: str, limit: int) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for node in self._safe_css(page, "a[href]"):
            href = str(getattr(node, "attrib", {}).get("href", "") or "").strip()
            if not href:
                continue
            resolved = urljoin(page_url, href)
            if resolved in seen or not self._is_same_host(page_url, resolved):
                continue
            score = self._article_url_score(page_url=page_url, article_url=resolved)
            if score <= 0:
                continue
            seen.add(resolved)
            title = self._extract_node_title(node)
            summary = self._extract_node_summary(node, title=title)
            candidates.append(
                {
                    "url": resolved,
                    "title": title or resolved,
                    "summary": summary,
                    "published": "",
                    "score": score,
                }
            )
        candidates.sort(key=lambda item: (-int(item.get("score", 0) or 0), item.get("url", "")))
        return [{key: value for key, value in item.items() if key != "score"} for item in candidates[:limit]]

    def _article_url_score(self, *, page_url: str, article_url: str) -> int:
        page_path = urlparse(page_url).path.rstrip("/")
        parsed = urlparse(article_url)
        path = parsed.path.rstrip("/")
        if not path or path == page_path:
            return 0
        lowered = path.lower()
        if any(token in lowered for token in ("/rss", "/feed", "/atom", "/category/", "/tag/", "/author/", "/topic/", "/topics/")):
            return 0
        parts = [part for part in lowered.split("/") if part]
        if not parts:
            return 0
        if len(parts) == 1 and parts[0] in self.EXCLUDED_PATH_PARTS:
            return 0

        score = 0
        if len(parts) >= 2:
            score += 2
        if any(part.isdigit() and len(part) == 4 for part in parts):
            score += 3
        if any(token in lowered for token in ("/blog/", "/news/", "/article/", "/articles/", "/post/", "/posts/", "/updates/")):
            score += 4
        if "-" in parts[-1] or len(parts[-1]) >= 12:
            score += 2
        if parsed.query:
            score -= 2
        return score

    @staticmethod
    def _safe_css(page: Any, selector: str) -> list[Any]:
        try:
            return list(page.css(selector) or [])
        except Exception:
            return []

    @staticmethod
    def _extract_node_title(node: Any) -> str:
        text = str(getattr(node, "text", "") or "").strip()
        return " ".join(text.split())

    @staticmethod
    def _extract_node_summary(node: Any, *, title: str) -> str:
        parent = getattr(node, "parent", None)
        if parent is None:
            return ""
        parent_text = str(getattr(parent, "text", "") or "").strip()
        parent_text = " ".join(parent_text.split())
        if not parent_text:
            return ""
        if title and parent_text.startswith(title):
            parent_text = parent_text[len(title) :].strip(" -:|")
        return parent_text[:240]

    @staticmethod
    def _dedup_preserve_order(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _normalize_host(url: str) -> str:
        host = urlparse(url).netloc.lower().split("@")[-1]
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host

    @classmethod
    def _is_same_host(cls, base_url: str, candidate_url: str) -> bool:
        host_a = cls._normalize_host(base_url)
        host_b = cls._normalize_host(candidate_url)
        if not host_a or not host_b:
            return False
        return host_a == host_b or host_a.endswith(f".{host_b}") or host_b.endswith(f".{host_a}")
