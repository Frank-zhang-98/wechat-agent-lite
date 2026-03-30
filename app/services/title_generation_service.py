from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.llm_gateway import LLMGateway


@dataclass
class TitlePlan:
    article_title: str
    wechat_title: str
    source: str = "heuristic"
    debug: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "article_title": self.article_title,
            "wechat_title": self.wechat_title,
            "source": self.source,
            "debug": self.debug,
        }


class TitleGenerationService:
    ARTICLE_TITLE_MAX_CHARS = 80
    WECHAT_TITLE_MAX_CHARS = 32
    WECHAT_TITLE_MAX_BYTES = 96

    CLICKBAIT_WORDS = ("震惊", "必看", "惊呆", "不看后悔", "全网最全", "保姆级")

    PRODUCT_HINTS = (
        "Claude Code",
        "Claude",
        "OpenAI",
        "GPT",
        "ChatGPT",
        "Gemini",
        "Anthropic",
        "Cursor",
        "Copilot",
        "LangChain",
        "Perplexity",
        "DeepSeek",
        "Qwen",
        "Midjourney",
    )

    ENGLISH_STOPWORDS = {
        "how",
        "what",
        "why",
        "when",
        "where",
        "who",
        "i",
        "my",
        "your",
        "our",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "from",
        "in",
        "on",
    }

    ENGLISH_PHRASE_MAP = {
        "multi-agent systems": "多代理系统",
        "software development": "软件开发",
        "claude code session": "Claude Code 会话",
        "claude code": "Claude Code",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "reshaping": "重塑",
        "analyzed": "复盘",
        "analyze": "拆解",
        "session": "会话",
        "workflow": "工作流",
        "developer": "开发者",
        "coding": "编程",
        "tool": "工具",
        "tools": "工具",
        "agent": "Agent",
        "agents": "Agents",
    }

    def generate(
        self,
        *,
        run_id: str,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_compress: dict[str, Any],
        content_type: str,
        llm: LLMGateway,
    ) -> TitlePlan:
        fallback = self._generate_fallback(topic=topic, fact_pack=fact_pack, fact_compress=fact_compress, content_type=content_type)
        llm_plan = self._generate_with_llm(
            run_id=run_id,
            topic=topic,
            fact_pack=fact_pack,
            fact_compress=fact_compress,
            content_type=content_type,
            llm=llm,
            fallback=fallback,
        )
        plan = llm_plan or fallback
        article_title = self._clean_article_title(plan.article_title or fallback.article_title)
        wechat_title = self._clean_wechat_title(plan.wechat_title or article_title or fallback.wechat_title)
        if not article_title:
            article_title = fallback.article_title
        if not wechat_title:
            wechat_title = self._clean_wechat_title(article_title)
        return TitlePlan(
            article_title=article_title,
            wechat_title=wechat_title,
            source=plan.source,
            debug=plan.debug,
        )

    def _generate_with_llm(
        self,
        *,
        run_id: str,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_compress: dict[str, Any],
        content_type: str,
        llm: LLMGateway,
        fallback: TitlePlan,
    ) -> TitlePlan | None:
        prompt = self._build_prompt(
            topic=topic,
            fact_pack=fact_pack,
            fact_compress=fact_compress,
            content_type=content_type,
            fallback=fallback,
        )
        result = llm.call(run_id, "WRITE", "writer", prompt, temperature=0.35)
        parsed = self._parse_llm_result(result.text)
        if not parsed:
            return None
        article_title = self._clean_article_title(parsed.get("article_title", ""))
        wechat_title = self._clean_wechat_title(parsed.get("wechat_title", ""))
        if not article_title or not wechat_title:
            return None
        return TitlePlan(
            article_title=article_title,
            wechat_title=wechat_title,
            source="llm",
            debug={
                "prompt": prompt[:4000],
                "response": result.text[:2000],
                "fallback_article_title": fallback.article_title,
                "fallback_wechat_title": fallback.wechat_title,
            },
        )

    def _build_prompt(
        self,
        *,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_compress: dict[str, Any],
        content_type: str,
        fallback: TitlePlan,
    ) -> str:
        return (
            "你是一名微信公众号标题编辑，请为同一篇 AI 技术解读文章生成两层标题，并只返回 JSON。\n\n"
            "返回格式：\n"
            "{\n"
            '  "article_title": "...",\n'
            '  "wechat_title": "...",\n'
            '  "reason": "一句话说明为什么这么写"\n'
            "}\n\n"
            "要求：\n"
            "- 输出简体中文标题，可保留关键英文产品名，如 Claude Code、OpenAI，可保留关键英文术语，如token、agent等。\n"
            "- article_title 用于站内/邮件，信息要完整，建议 18-34 个字，不能空泛，不要标题党。\n"
            "- wechat_title 专门用于公众号草稿，必须更短，建议 12-26 个字，绝对不要生硬截断的英文半句话。\n"
            "- 如果原始标题是英文，请翻成自然中文，保留关键专有名词。\n"
            "- 不要编造数字；只有事实包里明确有数字时才能写数字。\n"
            "- 优先突出：它是什么、为什么值得关注、对谁有价值。\n"
            f"- 如果你拿不准，请参考这个兜底方向：article_title={fallback.article_title} | wechat_title={fallback.wechat_title}\n\n"
            f"文章类型：{content_type}\n"
            f"原始标题：{topic.get('title', '')}\n"
            f"原始摘要：{topic.get('summary', '')}\n"
            f"一句话总结：{fact_compress.get('one_sentence_summary', '')}\n"
            f"关键机制：{json.dumps(fact_compress.get('key_mechanisms', []), ensure_ascii=False)}\n"
            f"典型场景：{json.dumps(fact_compress.get('concrete_scenarios', []), ensure_ascii=False)}\n"
            f"数字信息：{json.dumps(fact_compress.get('numbers', []), ensure_ascii=False)}\n"
            f"关键点：{json.dumps(fact_pack.get('key_points', []), ensure_ascii=False)}\n"
        )

    def _generate_fallback(
        self,
        *,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_compress: dict[str, Any],
        content_type: str,
    ) -> TitlePlan:
        core_title = self._extract_core_title(topic)
        tool_name = self._extract_tool_name(topic)
        benefit = self._extract_benefit(topic, fact_pack, fact_compress)
        number = self._extract_number(topic, fact_pack, fact_compress)
        localized_title = self._localize_title(core_title)
        templates_article = self._article_templates(content_type=content_type, tool_name=tool_name, benefit=benefit, number=number, core_title=localized_title)
        templates_wechat = self._wechat_templates(tool_name=tool_name, benefit=benefit, number=number, core_title=localized_title)
        article_title = self._pick_best_title(templates_article, prefer_short=False) or self._default_article_title(localized_title, tool_name, benefit)
        wechat_title = self._pick_best_title(templates_wechat, prefer_short=True) or self._default_wechat_title(article_title, tool_name)
        return TitlePlan(
            article_title=self._clean_article_title(article_title),
            wechat_title=self._clean_wechat_title(wechat_title),
            source="heuristic",
            debug={
                "tool_name": tool_name,
                "benefit": benefit,
                "number": number,
                "localized_title": localized_title,
            },
        )

    def _article_templates(self, *, content_type: str, tool_name: str, benefit: str, number: str, core_title: str) -> list[str]:
        if not tool_name:
            return [
                f"{core_title}：实战解读",
                f"{core_title}：为什么值得关注",
                f"{number}个重点看懂{core_title}",
            ]
        if "workflow" in content_type:
            return [
                f"{tool_name or core_title}：{number}个关键信号看懂它如何提升{benefit}",
                f"{tool_name or core_title}：为什么它正在重塑工作流",
                f"{core_title}：实战解读与落地建议",
            ]
        if "tutorial" in content_type:
            return [
                f"{tool_name or core_title}上手：{number}个重点看懂{benefit}",
                f"{tool_name or core_title}实战：从入门到落地",
                f"{core_title}：实战解读",
            ]
        return [
            f"{tool_name or core_title}：{benefit}实战解读",
            f"{tool_name or core_title}值不值得用？{number}个重点看懂",
            f"{core_title}：为什么值得关注",
            f"{core_title}：实战解读",
        ]

    def _wechat_templates(self, *, tool_name: str, benefit: str, number: str, core_title: str) -> list[str]:
        if not tool_name:
            return [
                f"{core_title}解读",
                core_title,
                f"{number}个重点看懂{core_title}",
            ]
        return [
            f"{tool_name}：{benefit}" if tool_name else "",
            f"{tool_name}值不值得用" if tool_name else "",
            f"{tool_name}{number}个重点" if tool_name and number else "",
            f"{core_title}解读",
            core_title,
        ]

    def _pick_best_title(self, candidates: list[str], *, prefer_short: bool) -> str:
        best_title = ""
        best_score = -1
        for candidate in candidates:
            title = self._normalize_spaces(candidate)
            if not title:
                continue
            score = self._score_title(title, prefer_short=prefer_short)
            if score > best_score:
                best_score = score
                best_title = title
        return best_title

    def _score_title(self, title: str, *, prefer_short: bool) -> int:
        score = 50
        length = len(title)
        if prefer_short:
            if 10 <= length <= 24:
                score += 20
            elif length <= 32:
                score += 10
            else:
                score -= 20
        else:
            if 14 <= length <= 30:
                score += 20
            elif length <= 40:
                score += 10
            else:
                score -= 15
        if any(char.isdigit() for char in title):
            score += 6
        if any(kw in title for kw in ("效率", "上手", "实战", "重点", "落地", "解读", "工作流")):
            score += 8
        if self._mostly_ascii(title):
            score -= 12
        if len(title.encode("utf-8")) > (self.WECHAT_TITLE_MAX_BYTES if prefer_short else 180):
            score -= 20
        return score

    def _default_article_title(self, core_title: str, tool_name: str, benefit: str) -> str:
        if tool_name:
            return f"{tool_name}：{benefit}实战解读"
        return f"{core_title}：实战解读"

    def _default_wechat_title(self, article_title: str, tool_name: str) -> str:
        if tool_name:
            return f"{tool_name}实战解读"
        simplified = re.sub(r"[:：].*$", "", article_title).strip()
        return simplified or article_title

    def _extract_core_title(self, topic: dict[str, Any]) -> str:
        title = self._normalize_spaces(str(topic.get("title", "") or "AI 热点"))
        title = re.sub(r"\s*[|丨｜]\s*.*$", "", title).strip()
        title = re.sub(r"\s+-\s+.*$", "", title).strip()
        return title[:60]

    def _extract_tool_name(self, topic: dict[str, Any]) -> str:
        title = self._extract_core_title(topic)
        for product in self.PRODUCT_HINTS:
            if product.lower() in title.lower():
                return product
        if not self._mostly_ascii(title):
            return title[:18]
        return ""

    def _extract_benefit(self, topic: dict[str, Any], fact_pack: dict[str, Any], fact_compress: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(topic.get("title", "") or ""),
                str(topic.get("summary", "") or ""),
                str(fact_compress.get("one_sentence_summary", "") or ""),
                " ".join(str(item) for item in fact_pack.get("key_points", [])[:4]),
            ]
        ).lower()
        mapping = [
            ("效率", ("efficiency", "productivity", "提效", "效率", "省时", "save time")),
            ("工作流", ("workflow", "工作流", "协同", "自动化", "automation")),
            ("开发效率", ("developer", "coding", "code", "开发", "编程", "写代码")),
            ("落地速度", ("launch", "ship", "上线", "落地", "部署")),
            ("团队协作", ("team", "collaboration", "协作", "团队")),
        ]
        for benefit, keywords in mapping:
            if any(keyword in text for keyword in keywords):
                return benefit
        return "效率提升"

    def _extract_number(self, topic: dict[str, Any], fact_pack: dict[str, Any], fact_compress: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(topic.get("title", "") or ""),
                str(topic.get("summary", "") or ""),
                json.dumps(fact_compress.get("numbers", []), ensure_ascii=False),
                json.dumps(fact_pack.get("numbers", []), ensure_ascii=False),
            ]
        )
        numbers = re.findall(r"\b([3-9]|10)\b", text)
        return numbers[0] if numbers else "3"

    def _localize_title(self, title: str) -> str:
        normalized = title.replace("-", " ")
        lower_title = normalized.lower()
        if "multi agent systems" in lower_title and "software development" in lower_title:
            return "多代理系统重塑软件开发"
        if "claude code" in lower_title and "session" in lower_title:
            if "mistake" in lower_title or "repeat" in lower_title:
                return "Claude Code 会话复盘"
            return "Claude Code 会话拆解"

        localized = normalized
        for src, target in sorted(self.ENGLISH_PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
            if src in lower_title:
                localized = re.sub(src, target, localized, flags=re.IGNORECASE)
        localized = self._normalize_spaces(localized)
        localized = localized.replace("  ", " ").strip(" -|：:")
        if self._mostly_ascii(localized):
            words = [
                word
                for word in re.findall(r"[A-Za-z0-9]+", localized)
                if word.lower() not in self.ENGLISH_STOPWORDS
            ]
            if words:
                localized = " ".join(words[:4]).strip()
        if self._mostly_ascii(localized):
            tool_name = self._extract_tool_name({"title": localized})
            if tool_name:
                return f"{tool_name} 相关变化"
        return localized

    @staticmethod
    def _parse_llm_result(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _clean_article_title(self, title: str) -> str:
        cleaned = self._normalize_spaces(title)
        for word in self.CLICKBAIT_WORDS:
            cleaned = cleaned.replace(word, "")
        cleaned = cleaned.strip("：:- ")
        cleaned = cleaned[: self.ARTICLE_TITLE_MAX_CHARS].strip()
        return cleaned or "AI 热点：实战解读"

    def _clean_wechat_title(self, title: str) -> str:
        cleaned = self._normalize_spaces(title)
        for word in self.CLICKBAIT_WORDS:
            cleaned = cleaned.replace(word, "")
        cleaned = re.sub(r"(：实战解读|：深度解读|：落地建议|：完整指南)$", "", cleaned).strip()
        cleaned = self._truncate_word_boundary(cleaned, self.WECHAT_TITLE_MAX_CHARS)
        cleaned = self._truncate_utf8_bytes(cleaned, self.WECHAT_TITLE_MAX_BYTES)
        cleaned = cleaned.strip("：:- ,.;")
        if not cleaned:
            cleaned = "AI 热点解读"
        return cleaned

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()

    @staticmethod
    def _mostly_ascii(text: str) -> bool:
        content = str(text or "").strip()
        if not content:
            return False
        ascii_count = sum(1 for char in content if ord(char) < 128)
        return ascii_count / max(len(content), 1) >= 0.75

    def _truncate_word_boundary(self, text: str, max_chars: int) -> str:
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        if self._mostly_ascii(value):
            chunk = value[: max_chars + 1]
            if " " in chunk:
                candidate = chunk.rsplit(" ", 1)[0].strip()
                if len(candidate) >= max(8, max_chars // 2):
                    return candidate
        return value[:max_chars].strip()

    @staticmethod
    def _truncate_utf8_bytes(text: str, max_bytes: int) -> str:
        value = str(text or "").strip()
        if len(value.encode("utf-8")) <= max_bytes:
            return value
        output: list[str] = []
        used_bytes = 0
        for char in value:
            char_bytes = len(char.encode("utf-8"))
            if used_bytes + char_bytes > max_bytes:
                break
            output.append(char)
            used_bytes += char_bytes
        return "".join(output).strip()
