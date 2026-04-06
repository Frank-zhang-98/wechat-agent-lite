from __future__ import annotations

import re


class LocalizationService:
    _DIRECT_MAP = {
        "introduction": "引言",
        "what you'll learn": "你将学到什么",
        "what you’ll learn": "你将学到什么",
        "prerequisites": "前置条件",
        "project background": "项目背景",
        "project introduction": "项目介绍",
        "author/team introduction": "作者 / 团队介绍",
        "project stats": "项目概览",
        "main features": "核心特性",
        "core purpose": "核心目标",
        "use cases": "适用场景",
        "use case setup": "用例设置",
        "quick start": "快速开始",
        "core features": "核心能力",
        "code and engineering details": "代码与工程细节",
        "how does pageindex work?": "PageIndex 如何工作？",
        "why this works so well?": "为什么它效果这么好？",
        "why this is difficult to scale?": "为什么它难以规模化？",
        "engineering a better 检索器 — proxy-pointer rag": "Proxy-Pointer RAG：更好的检索器工程",
        "engineering a better 检索器 - proxy-pointer rag": "Proxy-Pointer RAG：更好的检索器工程",
        "engineering a better retriever - proxy-pointer rag": "Proxy-Pointer RAG：更好的检索器工程",
        "build a skeleton tree": "构建骨架树",
        "install": "安装方式",
        "installation": "安装方式",
        "why this matters": "为什么这件事值得关注",
        "the numbers": "关键数据一览",
        "free security tool": "免费安全工具",
        "free website audit": "免费网站审计",
        "install ollama": "安装 Ollama",
        "run your first model": "运行第一个模型",
        "session architecture": "会话架构",
        "tool layer": "工具层",
        "agent calls": "智能体调用",
        "related reading": "延伸阅读",
        "further reading": "延伸阅读",
        "references": "参考资料",
        "summary": "总结",
        "conclusion": "结论",
        "overview": "总览",
    }

    _PHRASE_MAP = [
        ("What You'll Learn", "你将学到什么"),
        ("What You’ll Learn", "你将学到什么"),
        ("Project Background", "项目背景"),
        ("Project Introduction", "项目介绍"),
        ("Author/Team Introduction", "作者 / 团队介绍"),
        ("Project Stats", "项目概览"),
        ("Main Features", "核心特性"),
        ("Core Purpose", "核心目标"),
        ("Use Cases", "适用场景"),
        ("Use Case Setup", "用例设置"),
        ("Quick Start", "快速开始"),
        ("Core Features", "核心能力"),
        ("Code and engineering details", "代码与工程细节"),
        ("Build a Skeleton Tree", "构建骨架树"),
        ("Engineering a Better", "更好的工程化方案"),
        ("Vectorless RAG", "无向量 RAG"),
        ("Flat Vector RAG", "扁平向量 RAG"),
        ("Vectorless", "无向量 RAG"),
        ("How does", "如何理解"),
        ("Why this works so well", "为什么它效果这么好"),
        ("Why this is difficult to scale", "为什么它难以规模化"),
        ("Indexing", "索引"),
        ("Retrieval", "检索"),
        ("once per document", "每份文档一次"),
        ("Per Query", "每次查询"),
        ("per query", "每次查询"),
        ("Why this matters", "为什么这件事值得关注"),
        ("The numbers", "关键数据一览"),
        ("Free security tool", "免费安全工具"),
        ("Free website audit", "免费网站审计"),
        ("Install Ollama", "安装 Ollama"),
        ("Run your first model", "运行第一个模型"),
        ("Session architecture", "会话架构"),
        ("Tool layer", "工具层"),
        ("Agent calls", "智能体调用"),
        ("Agent Runtime", "智能体运行时"),
        ("AI Agent", "AI 智能体"),
        ("AI Agents", "AI 智能体"),
        ("dual-time model", "双时间模型"),
        ("hybrid retrieval", "混合检索"),
        ("knowledge graph", "知识图谱"),
        ("time-aware", "时间感知"),
        ("real-time", "实时"),
        ("workflow", "流程"),
        ("architecture", "架构"),
        ("comparison", "对比"),
        ("tutorial", "教程"),
        ("install", "安装"),
        ("sandbox", "沙箱"),
        ("gateway", "网关"),
        ("policy engine", "策略引擎"),
        ("session store", "会话存储"),
        ("wallet adapter", "钱包适配层"),
        ("memory layer", "记忆层"),
        ("data source", "数据源"),
        ("retriever", "检索器"),
        ("parser", "解析器"),
        ("scheduler", "调度器"),
        ("queue", "队列"),
        ("input", "输入"),
        ("output", "输出"),
        ("core system", "核心系统"),
        ("module", "模块"),
        ("step", "步骤"),
        ("infrastructure", "基础设施"),
        ("plugin", "插件"),
        ("agent", "智能体"),
        ("overview", "总览"),
        ("summary", "总结"),
        ("conclusion", "结论"),
        ("references", "参考资料"),
    ]

    _COMMAND_PREFIXES = (
        "pip ",
        "pip3 ",
        "npm ",
        "npx ",
        "curl ",
        "wget ",
        "uv ",
        "python ",
        "python3 ",
        "docker ",
        "ollama ",
        "claude ",
        "git ",
        "scp ",
        "ssh ",
        "irm ",
        "ps>",
        "$ ",
    )

    @classmethod
    def localize_heading_text(cls, text: str) -> str:
        return cls._localize_text(text)

    @classmethod
    def looks_like_heading_text(cls, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        normalized = cls._normalize_key(raw)
        if normalized in cls._DIRECT_MAP:
            return True
        if cls._match_common_pattern(raw):
            return True
        return False

    @classmethod
    def localize_visual_text(cls, text: str) -> str:
        return cls._localize_text(text)

    @classmethod
    def localize_visual_items(cls, items: list[str] | tuple[str, ...]) -> list[str]:
        return [cls.localize_visual_text(str(item).strip()) for item in items if str(item).strip()]

    @classmethod
    def _localize_text(cls, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return raw
        direct = cls._DIRECT_MAP.get(cls._normalize_key(raw))
        if direct:
            return direct
        pattern_hit = cls._match_common_pattern(raw)
        if pattern_hit:
            return pattern_hit
        if cls._looks_like_command(raw):
            return raw

        result = raw
        for source, target in sorted(cls._PHRASE_MAP, key=lambda item: len(item[0]), reverse=True):
            result = cls._replace_phrase(result, source, target)
        return cls._cleanup_text(result)

    @staticmethod
    def _normalize_key(text: str) -> str:
        normalized = str(text or "").strip()
        normalized = normalized.replace("’", "'").replace("`", "")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    @classmethod
    def _match_common_pattern(cls, text: str) -> str | None:
        normalized = cls._normalize_key(text)
        whats_new = re.match(r"^what['’]s new in (.+)$", normalized)
        if whats_new:
            return f"{whats_new.group(1)} 更新了什么"

        phase_pattern = re.match(r"^phase\s*(\d+)\s*:\s*(.+?)(?:\s*\((.+)\))?$", text, flags=re.IGNORECASE)
        if phase_pattern:
            stage_no = phase_pattern.group(1)
            stage_title = cls._localize_text(phase_pattern.group(2))
            stage_note = cls._localize_text(str(phase_pattern.group(3) or "").strip()) if phase_pattern.group(3) else ""
            return f"阶段 {stage_no}：{stage_title}" + (f"（{stage_note}）" if stage_note else "")

        how_pattern = re.match(r"^how does (.+?) work\??$", text, flags=re.IGNORECASE)
        if how_pattern:
            subject = cls._localize_text(how_pattern.group(1))
            return f"{subject} 如何工作？"

        why_pattern = re.match(r"^why this works so well\??$", text, flags=re.IGNORECASE)
        if why_pattern:
            return "为什么它效果这么好？"

        scale_pattern = re.match(r"^why this is difficult to scale\??$", text, flags=re.IGNORECASE)
        if scale_pattern:
            return "为什么它难以规模化？"

        compare_pattern = re.match(r"^对比\s+of\s+(.+?)\s+vs\s+(.+)$", text, flags=re.IGNORECASE)
        if compare_pattern:
            left = cls._normalize_compare_term(compare_pattern.group(1))
            right = cls._normalize_compare_term(compare_pattern.group(2))
            return f"{left} 与 {right} 对比"

        generic_vs_pattern = re.match(r"^(.+?)\s+vs\s+(.+)$", text, flags=re.IGNORECASE)
        if generic_vs_pattern:
            left = cls._normalize_compare_term(generic_vs_pattern.group(1))
            right = cls._normalize_compare_term(generic_vs_pattern.group(2))
            return f"{left} 与 {right} 对比"

        better_pattern = re.match(r"^engineering a better\s+(.+?)\s*[—-]\s*(.+)$", text, flags=re.IGNORECASE)
        if better_pattern:
            left = cls._localize_text(better_pattern.group(1))
            right = cls._localize_text(better_pattern.group(2))
            return f"{right}：更好的{left}工程"

        infra_pattern = re.match(r"^(?P<subject>.+?) is infrastructure, not a plugin\.?$", text, flags=re.IGNORECASE)
        if infra_pattern:
            subject = cls._localize_text(infra_pattern.group("subject"))
            return f"{subject}不是插件，而是基础设施。"

        replacement_pattern = re.match(
            r"^(?P<subject>.+?) is not a replacement for (?P<left>.+?), but (?P<right>.+?)\.?$",
            text,
            flags=re.IGNORECASE,
        )
        if replacement_pattern:
            subject = cls._localize_text(replacement_pattern.group("subject"))
            left = cls._localize_text(replacement_pattern.group("left"))
            right = cls._localize_text(replacement_pattern.group("right"))
            return f"{subject}不是{left}的替代品，而是{right}。"
        return None

    @classmethod
    def _normalize_compare_term(cls, text: str) -> str:
        raw = str(text or "").strip()
        normalized = cls._normalize_key(raw)
        if normalized == "vectorless":
            return "无向量 RAG"
        if normalized == "flat vector rag":
            return "扁平向量 RAG"
        return cls._localize_text(raw)

    @staticmethod
    def _looks_like_command(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if lowered.startswith(LocalizationService._COMMAND_PREFIXES):
            return True
        if "http://" in lowered or "https://" in lowered:
            return True
        if re.search(r"(^|\s)--?[a-z0-9_-]+", lowered):
            return True
        if re.search(r"[\\/][\\w.-]+", lowered) and " " not in lowered:
            return True
        return False

    @staticmethod
    def _replace_phrase(text: str, source: str, target: str) -> str:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])", flags=re.IGNORECASE)
        return pattern.sub(target, text)

    @staticmethod
    def _cleanup_text(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
        cleaned = re.sub(r"\s+([，。；：、）》】])", r"\1", cleaned)
        cleaned = re.sub(r"([（《【])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()
