from __future__ import annotations

import base64
import html
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.config import CONFIG


@dataclass
class RenderedArticle:
    layout_name: str
    layout_label: str
    html: str
    block_count: int
    description: str = ""
    source: str = "rule"


class ArticleRenderService:
    def __init__(self) -> None:
        self.config = self._load_layouts()

    def _load_layouts(self) -> dict[str, Any]:
        path = Path(CONFIG.data_dir).parents[0] / "config" / "article_layouts.yaml"
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "config" / "article_layouts.yaml"
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def resolve_layout(self, *, content_type: str, explicit_layout: str = "") -> dict[str, Any]:
        layouts = dict(self.config.get("layouts") or {})
        default_layout = str(self.config.get("default_layout", "clean_reading") or "clean_reading")
        if explicit_layout and explicit_layout in layouts:
            name = explicit_layout
            source = "explicit"
        else:
            mapped = str((self.config.get("content_type_map") or {}).get(content_type, "") or "").strip()
            if mapped and mapped in layouts:
                name = mapped
                source = "content_type_rule"
            elif default_layout in layouts:
                name = default_layout
                source = "default"
            elif layouts:
                name = next(iter(layouts))
                source = "fallback"
            else:
                name = "clean_reading"
                source = "fallback"
        layout = dict(layouts.get(name) or {})
        layout["name"] = name
        layout["source"] = source
        layout["label"] = str(layout.get("label", name) or name)
        layout["description"] = str(layout.get("description", "") or "")
        return layout

    def render(
        self,
        markdown_text: str,
        *,
        article_title: str,
        content_type: str,
        target_audience: str = "",
        layout_name: str = "",
        illustrations: list[dict[str, Any]] | None = None,
    ) -> RenderedArticle:
        layout = self.resolve_layout(content_type=content_type, explicit_layout=layout_name)
        blocks = self._parse_blocks(markdown_text=markdown_text, article_title=article_title)
        html_blocks = self._render_blocks(
            blocks=blocks,
            layout=layout,
            content_type=content_type,
            target_audience=target_audience,
            illustrations=illustrations or [],
        )
        html_output = (
            f'<div style="{self._page_style(layout)}">'
            f'<div style="{self._card_style(layout)}">'
            f'{"".join(html_blocks)}'
            f"</div>"
            f"</div>"
        )
        return RenderedArticle(
            layout_name=layout["name"],
            layout_label=layout["label"],
            html=html_output,
            block_count=len(blocks),
            description=layout["description"],
            source=layout["source"],
        )

    @staticmethod
    def save_html(rendered: RenderedArticle, run_id: str) -> str:
        target = CONFIG.data_dir / "runs" / run_id
        target.mkdir(parents=True, exist_ok=True)
        path = target / "article.html"
        path.write_text(rendered.html, encoding="utf-8")
        return str(path)

    def _parse_blocks(self, *, markdown_text: str, article_title: str) -> list[dict[str, Any]]:
        lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[dict[str, Any]] = []
        i = 0
        first_heading_skipped = False
        while i < len(lines):
            raw = lines[i].rstrip()
            stripped = raw.strip()
            if not stripped:
                i += 1
                continue

            if not first_heading_skipped and stripped.startswith("# "):
                heading_text = stripped[2:].strip()
                if self._normalized_title(heading_text) == self._normalized_title(article_title):
                    first_heading_skipped = True
                    i += 1
                    continue
            first_heading_skipped = True

            if stripped.startswith("```"):
                fence = stripped[:3]
                language = stripped[3:].strip()
                code_lines: list[str] = []
                i += 1
                while i < len(lines):
                    candidate = lines[i].rstrip("\n")
                    if candidate.strip().startswith(fence):
                        break
                    code_lines.append(candidate)
                    i += 1
                blocks.append({"type": "code", "language": language, "text": "\n".join(code_lines).strip("\n")})
                i += 1
                continue

            if stripped in {"---", "***", "___"}:
                blocks.append({"type": "hr"})
                i += 1
                continue

            if stripped.startswith("### "):
                blocks.append({"type": "h3", "text": stripped[4:].strip()})
                i += 1
                continue
            if stripped.startswith("## "):
                blocks.append({"type": "h2", "text": stripped[3:].strip()})
                i += 1
                continue
            if stripped.startswith("# "):
                blocks.append({"type": "h1", "text": stripped[2:].strip()})
                i += 1
                continue

            if stripped.startswith(">"):
                quote_lines: list[str] = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    quote_lines.append(lines[i].strip()[1:].lstrip())
                    i += 1
                callout_type = ""
                if quote_lines and re.match(r"^\[![A-Za-z]+\]", quote_lines[0]):
                    marker = quote_lines.pop(0)
                    callout_type = marker[3:].strip("]!").lower()
                quote_text = "\n".join(line for line in quote_lines if line).strip()
                blocks.append({"type": "callout" if callout_type else "blockquote", "callout_type": callout_type or "", "text": quote_text})
                continue

            if self._looks_like_table(lines, i):
                header = self._split_table_row(lines[i])
                i += 2  # skip separator
                rows: list[list[str]] = []
                while i < len(lines):
                    candidate = lines[i].strip()
                    if not candidate or "|" not in candidate:
                        break
                    rows.append(self._split_table_row(lines[i]))
                    i += 1
                blocks.append({"type": "table", "header": header, "rows": rows})
                continue

            if re.match(r"^[-*]\s+", stripped):
                items: list[str] = []
                while i < len(lines):
                    candidate = lines[i].strip()
                    if not re.match(r"^[-*]\s+", candidate):
                        break
                    items.append(re.sub(r"^[-*]\s+", "", candidate).strip())
                    i += 1
                blocks.append({"type": "ul", "items": items})
                continue

            if re.match(r"^\d+\.\s+", stripped):
                items = []
                while i < len(lines):
                    candidate = lines[i].strip()
                    if not re.match(r"^\d+\.\s+", candidate):
                        break
                    items.append(re.sub(r"^\d+\.\s+", "", candidate).strip())
                    i += 1
                blocks.append({"type": "ol", "items": items})
                continue

            paragraph_lines = [stripped]
            i += 1
            while i < len(lines):
                candidate = lines[i].strip()
                if not candidate:
                    break
                if self._starts_new_block(candidate, lines, i):
                    break
                paragraph_lines.append(candidate)
                i += 1
            blocks.append({"type": "p", "text": " ".join(paragraph_lines).strip()})

        return blocks

    def _render_blocks(
        self,
        *,
        blocks: list[dict[str, Any]],
        layout: dict[str, Any],
        content_type: str,
        target_audience: str,
        illustrations: list[dict[str, Any]],
    ) -> list[str]:
        rendered: list[str] = []
        pending = [item for item in illustrations if isinstance(item, dict)]
        inserted_indexes: set[int] = set()
        section_illustrations: list[dict[str, Any]] = []
        section_indexes: list[int] = []
        section_plan = self._plan_illustration_sections(blocks=blocks, illustrations=pending)

        lede_used = False
        for block_idx, block in enumerate(blocks):
            block_type = block["type"]
            if block_type in {"h1", "h2", "h3"} and section_illustrations:
                rendered.extend(self._render_illustration_group(items=section_illustrations, layout=layout))
                inserted_indexes.update(section_indexes)
                section_illustrations = []
                section_indexes = []
            if block_type == "p" and not lede_used and layout.get("use_lede", False):
                rendered.append(f'<p style="{self._lede_style(layout)}">{self._inline(block["text"], layout)}</p>')
                lede_used = True
            elif block_type == "p":
                rendered.append(f'<p style="{self._paragraph_style(layout)}">{self._inline(block["text"], layout)}</p>')
            elif block_type == "h1":
                rendered.append(f'<h1 style="{self._heading_style(layout, 1)}">{self._inline(block["text"], layout)}</h1>')
            elif block_type == "h2":
                rendered.append(f'<h2 style="{self._heading_style(layout, 2)}">{self._inline(block["text"], layout)}</h2>')
            elif block_type == "h3":
                rendered.append(f'<h3 style="{self._heading_style(layout, 3)}">{self._inline(block["text"], layout)}</h3>')
            elif block_type == "blockquote":
                rendered.append(f'<blockquote style="{self._quote_style(layout)}">{self._inline(block["text"], layout)}</blockquote>')
            elif block_type == "callout":
                rendered.append(
                    f'<div style="{self._callout_style(layout, block.get("callout_type", ""))}">'
                    f'{self._inline(block["text"], layout)}'
                    f"</div>"
                )
            elif block_type == "ul":
                items = "".join(f'<li style="{self._li_style(layout)}">{self._inline(item, layout)}</li>' for item in block["items"])
                rendered.append(f'<ul style="{self._list_style(layout)}">{items}</ul>')
            elif block_type == "ol":
                items = "".join(f'<li style="{self._li_style(layout)}">{self._inline(item, layout)}</li>' for item in block["items"])
                rendered.append(f'<ol style="{self._list_style(layout)}">{items}</ol>')
            elif block_type == "code":
                rendered.append(self._render_code_block(block=block, layout=layout))
            elif block_type == "table":
                rendered.append(self._render_table(block=block, layout=layout))
            elif block_type == "hr":
                rendered.append(f'<hr style="{self._hr_style(layout)}" />')

            if block_type in {"h1", "h2", "h3"}:
                section_indexes = [idx for idx in section_plan.get(block_idx, []) if idx not in inserted_indexes]
                section_illustrations = [pending[idx] for idx in section_indexes]
            elif section_illustrations and self._is_illustration_anchor_block(block_type):
                rendered.extend(self._render_illustration_group(items=section_illustrations, layout=layout))
                inserted_indexes.update(section_indexes)
                section_illustrations = []
                section_indexes = []

        if section_illustrations:
            rendered.extend(self._render_illustration_group(items=section_illustrations, layout=layout))
            inserted_indexes.update(section_indexes)
        for idx, item in enumerate(pending):
            if idx in inserted_indexes:
                continue
            rendered.extend(self._render_illustration_group(items=[item], layout=layout))
        return rendered

    def _render_code_block(self, *, block: dict[str, Any], layout: dict[str, Any]) -> str:
        language = str(block.get("language", "") or "").strip().lower()
        code_text = str(block.get("text", "") or "")
        lines = code_text.split("\n")
        if not lines:
            lines = [""]
        language_badge = ""
        if language:
            language_badge = (
                f'<div style="margin:0 0 10px;color:{layout["muted_color"]};font-size:11px;line-height:1.2;'
                f'text-transform:uppercase;letter-spacing:0.08em;font-family:Consolas,Monaco,monospace;">'
                f"{html.escape(language)}"
                f"</div>"
            )
        code_html = "<br/>".join(self._render_code_line(line) for line in lines)
        return (
            f'<section style="{self._code_style(layout)}">'
            f"{language_badge}"
            f'<code style="display:block;margin:0;white-space:normal;word-break:break-word;overflow-wrap:anywhere;'
            f'font-family:Consolas,Monaco,monospace;">{code_html}</code>'
            f"</section>"
        )

    @staticmethod
    def _render_code_line(line: str) -> str:
        expanded = str(line or "").replace("\t", "    ")
        match = re.match(r"^( +)", expanded)
        if not expanded:
            return "&nbsp;"
        leading = match.group(1) if match else ""
        body = expanded[len(leading) :]
        prefix = "&nbsp;" * len(leading)
        escaped_body = html.escape(body)
        return f"{prefix}{escaped_body}" if (prefix or escaped_body) else "&nbsp;"

    def _render_illustration(self, *, item: dict[str, Any], layout: dict[str, Any]) -> str:
        src = self._resolve_illustration_src(str(item.get("path", "") or "").strip())
        if not src:
            return ""
        title = html.escape(str(item.get("title", "") or "").strip())
        caption = html.escape(str(item.get("caption", "") or "").strip())
        caption_html = ""
        if title or caption:
            title_html = ""
            if title:
                title_html = (
                    f'<div style="margin-bottom:4px;color:{layout["heading_color"]};font-size:15px;line-height:1.5;font-weight:700;">'
                    f"图解：{title}"
                    f"</div>"
                )
            caption_html = (
                f'<figcaption style="margin-top:12px;padding:0 4px;color:{layout["muted_color"]};font-size:13px;line-height:1.75;">'
                f"{title_html}"
                f"{caption}"
                f"</figcaption>"
            )
        return (
            f'<figure style="margin:22px auto 30px;max-width:960px;">'
            f'<img src="{src}" alt="{title or "illustration"}" style="display:block;width:100%;height:auto;border-radius:18px;border:1px solid {layout["border_color"]};background:{layout["card_background"]};box-shadow:0 10px 30px rgba(15,23,42,0.08);" />'
            f"{caption_html}"
            f"</figure>"
        )

    def _render_illustration_group(self, *, items: list[dict[str, Any]], layout: dict[str, Any]) -> list[str]:
        rendered: list[str] = []
        bridge = self._render_illustration_bridge(items=items, layout=layout)
        if bridge:
            rendered.append(bridge)
        for item in items:
            block = self._render_illustration(item=item, layout=layout)
            if block:
                rendered.append(block)
        return rendered

    def _render_illustration_bridge(self, *, items: list[dict[str, Any]], layout: dict[str, Any]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            text = self._single_illustration_bridge(items[0])
        else:
            text = self._multi_illustration_bridge(items)
        return f'<p style="{self._paragraph_style(layout)}margin-top:8px;">{text}</p>'

    def _single_illustration_bridge(self, item: dict[str, Any]) -> str:
        focus = html.escape(self._bridge_focus_text(item))
        title = str(item.get("title", "") or "").strip()
        caption = str(item.get("caption", "") or "").strip()
        signal = " ".join([title, caption])
        if re.search(r"(流程|链路|步骤|阶段|工作流)", signal):
            return f"下面这张图把 {focus} 的顺序和衔接关系梳理出来了，适合和前文对照着看。"
        if re.search(r"(架构|模块|分层|节点|支撑)", signal):
            return f"前面提到的 {focus}，放到图里看层次和分工会更直观。"
        if re.search(r"(对比|差异|vs|VS)", signal):
            return f"如果只看文字不够直观，下面这张图把 {focus} 的差异并排展开了。"
        templates = [
            f"前面讲到的 {focus}，可以直接对照下面这张图来看。",
            f"下面这张图把 {focus} 单独拎了出来，读到这里看会更顺手。",
            f"这一段涉及的 {focus}，用图来理解会更快一些。",
            f"如果想更快抓住这一节的重点，先看下面这张 {focus} 图就够了。",
        ]
        return templates[self._stable_variant_index(focus, len(templates))]

    def _multi_illustration_bridge(self, items: list[dict[str, Any]]) -> str:
        focuses = [self._bridge_focus_text(item) for item in items if self._bridge_focus_text(item)]
        if len(focuses) >= 2:
            pair = "、".join(html.escape(text) for text in focuses[:2])
            return f"这一节里有两个关键点最值得配合图来看：{pair}。顺着下面几张图往下看，会更容易把关系串起来。"
        return "下面几张图分别对应这一节里最关键的结构和流程，建议和前文放在一起对照着看。"

    @staticmethod
    def _bridge_focus_text(item: dict[str, Any]) -> str:
        for key in ("title", "section", "caption"):
            value = str(item.get(key, "") or "").strip()
            if value:
                return value[:26] if len(value) > 26 else value
        return "这部分内容"

    @staticmethod
    def _stable_variant_index(text: str, count: int) -> int:
        if count <= 0:
            return 0
        total = sum(ord(ch) for ch in str(text or ""))
        return total % count

    @staticmethod
    def _is_illustration_anchor_block(block_type: str) -> bool:
        return block_type in {"p", "ul", "ol", "blockquote", "callout", "table", "code"}

    def _plan_illustration_sections(self, *, blocks: list[dict[str, Any]], illustrations: list[dict[str, Any]]) -> dict[int, list[int]]:
        sections = self._build_sections(blocks)
        plan: dict[int, list[int]] = {}
        for idx, item in enumerate(illustrations):
            best_heading_idx = None
            best_score = 0
            for section in sections:
                score = self._score_illustration_to_section(item=item, section=section)
                if score > best_score:
                    best_score = score
                    best_heading_idx = section.get("heading_idx")
            if best_heading_idx is None or best_score < 18:
                continue
            plan.setdefault(int(best_heading_idx), []).append(idx)
        return plan

    def _build_sections(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for idx, block in enumerate(blocks):
            if block.get("type") in {"h1", "h2", "h3"}:
                if current:
                    sections.append(current)
                heading_text = str(block.get("text", "") or "").strip()
                current = {
                    "heading_idx": idx,
                    "heading_text": heading_text,
                    "search_text": heading_text,
                }
                continue
            if not current:
                continue
            block_text = self._block_search_text(block)
            if block_text:
                current["search_text"] = f"{current['search_text']} {block_text}".strip()
        if current:
            sections.append(current)
        return sections

    def _score_illustration_to_section(self, *, item: dict[str, Any], section: dict[str, Any]) -> int:
        heading_text = str(section.get("heading_text", "") or "")
        search_text = str(section.get("search_text", "") or "")
        heading_norm = self._normalized_title(heading_text)
        section_norm = self._normalized_title(str(item.get("section", "") or ""))
        title_norm = self._normalized_title(str(item.get("title", "") or ""))
        caption_norm = self._normalized_title(str(item.get("caption", "") or ""))
        score = 0
        for candidate in (section_norm, title_norm):
            if candidate and heading_norm:
                if candidate == heading_norm:
                    score += 120
                elif len(candidate) >= 4 and (candidate in heading_norm or heading_norm in candidate):
                    score += 72
        search_norm = self._normalized_title(search_text)
        for candidate in (section_norm, title_norm, caption_norm):
            if candidate and search_norm and candidate in search_norm:
                score += 38
        item_tokens = self._match_tokens(
            " ".join(
                [
                    str(item.get("section", "") or ""),
                    str(item.get("title", "") or ""),
                    str(item.get("caption", "") or ""),
                ]
            )
        )
        section_tokens = self._match_tokens(search_text)
        score += len(item_tokens & section_tokens) * 8
        return score

    @staticmethod
    def _block_search_text(block: dict[str, Any]) -> str:
        block_type = str(block.get("type", "") or "")
        if block_type in {"p", "h1", "h2", "h3", "blockquote", "callout"}:
            return str(block.get("text", "") or "").strip()
        if block_type in {"ul", "ol"}:
            return " ".join(str(item).strip() for item in (block.get("items") or []) if str(item).strip())
        if block_type == "table":
            values = list(block.get("header") or [])
            for row in (block.get("rows") or []):
                values.extend(row)
            return " ".join(str(item).strip() for item in values if str(item).strip())
        if block_type == "code":
            return str(block.get("text", "") or "")[:300]
        return ""

    @staticmethod
    def _match_tokens(text: str) -> set[str]:
        stopwords = {
            "图解", "这一节", "这一部分", "关键", "结构", "流程", "系统", "模块", "说明", "建议",
            "实现", "完整", "链路", "部分", "用于", "相关", "统一", "技术", "方案",
        }
        tokens = re.findall(r"[A-Za-z0-9@._/-]{2,}|[\u4e00-\u9fff]{2,8}", str(text or ""))
        return {token.lower() for token in tokens if token.lower() not in stopwords}

    @staticmethod
    def _resolve_illustration_src(path: str) -> str:
        raw = str(path or "").strip().replace("\\", "/")
        if not raw:
            return ""
        if raw.startswith("data:image/") or raw.startswith("http://") or raw.startswith("https://"):
            return raw
        candidate = Path(raw)
        if not candidate.exists() or not candidate.is_file():
            return raw
        mime_type, _ = mimetypes.guess_type(candidate.name)
        mime_type = mime_type or "image/png"
        encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _render_table(self, *, block: dict[str, Any], layout: dict[str, Any]) -> str:
        header_cells = "".join(
            f'<th style="{self._th_style(layout)}">{self._inline(cell, layout)}</th>' for cell in block.get("header", [])
        )
        row_html = []
        for row in block.get("rows", []):
            cells = "".join(f'<td style="{self._td_style(layout)}">{self._inline(cell, layout)}</td>' for cell in row)
            row_html.append(f"<tr>{cells}</tr>")
        return (
            f'<div style="{self._table_wrap_style(layout)}">'
            f'<table style="{self._table_style(layout)}">'
            f"<thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{''.join(row_html)}</tbody>"
            f"</table>"
            f"</div>"
        )

    @staticmethod
    def _normalized_title(value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _starts_new_block(candidate: str, lines: list[str], index: int) -> bool:
        if candidate.startswith(("# ", "## ", "### ", "```", ">")):
            return True
        if candidate in {"---", "***", "___"}:
            return True
        if re.match(r"^[-*]\s+", candidate):
            return True
        if re.match(r"^\d+\.\s+", candidate):
            return True
        return ArticleRenderService._looks_like_table(lines, index)

    @staticmethod
    def _looks_like_table(lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        current = lines[index].strip()
        next_line = lines[index + 1].strip()
        if "|" not in current or "|" not in next_line:
            return False
        return bool(re.match(r"^\|?[\s:-]+\|[\s|:-]*$", next_line))

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        text = line.strip().strip("|")
        return [cell.strip() for cell in text.split("|")]

    def _inline(self, value: str, layout: dict[str, Any]) -> str:
        tokens: list[str] = []

        def store_token(text: str) -> str:
            tokens.append(text)
            return f"__TOKEN_{len(tokens) - 1}__"

        text = str(value or "")
        text = re.sub(
            r"`([^`]+)`",
            lambda match: store_token(
                f'<code style="background:rgba(15,23,42,0.06);padding:2px 6px;border-radius:6px;color:{layout["heading_color"]};font-family:Consolas,Monaco,monospace;font-size:0.92em;">{html.escape(match.group(1))}</code>'
            ),
            text,
        )
        text = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            lambda match: store_token(
                f'<a href="{html.escape(match.group(2), quote=True)}" style="color:{layout["accent_color"]};text-decoration:none;border-bottom:1px solid {layout["accent_color"]};">{html.escape(match.group(1))}</a>'
            ),
            text,
        )
        text = html.escape(text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

        for idx, token in enumerate(tokens):
            text = text.replace(f"__TOKEN_{idx}__", token)
        return text

    @staticmethod
    def _page_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:0;padding:24px 0;background:{layout['page_background']};"
            f"font-family:{layout['font_family']};color:{layout['text_color']};"
        )

    @staticmethod
    def _card_style(layout: dict[str, Any]) -> str:
        return (
            f"max-width:{layout['max_width']};margin:0 auto;background:{layout['card_background']};"
            f"border:1px solid {layout['border_color']};border-radius:22px;padding:28px 24px;"
            f"box-sizing:border-box;"
        )

    @staticmethod
    def _lede_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:0 0 18px;color:{layout['heading_color']};font-size:18px;line-height:1.9;"
            f"font-weight:500;"
        )

    @staticmethod
    def _paragraph_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:0 0 16px;color:{layout['text_color']};font-size:15px;line-height:1.95;"
        )

    @staticmethod
    def _heading_style(layout: dict[str, Any], level: int) -> str:
        size_map = {1: "30px", 2: "24px", 3: "19px"}
        margin_map = {1: "0 0 16px", 2: "30px 0 14px", 3: "24px 0 12px"}
        weight_map = {1: "800", 2: "750", 3: "700"}
        border = f"padding-bottom:8px;border-bottom:1px solid {layout['border_color']};" if level == 2 else ""
        return (
            f"margin:{margin_map[level]};color:{layout['heading_color']};font-size:{size_map[level]};"
            f"line-height:1.35;font-weight:{weight_map[level]};font-family:{layout['heading_font_family']};"
            f"{border}"
        )

    @staticmethod
    def _quote_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:18px 0;padding:14px 16px;background:{layout['quote_background']};"
            f"border-left:4px solid {layout['quote_border']};border-radius:0 14px 14px 0;"
            f"color:{layout['text_color']};font-size:15px;line-height:1.9;"
        )

    @staticmethod
    def _callout_style(layout: dict[str, Any], callout_type: str) -> str:
        accent = layout["accent_color"]
        if callout_type in {"warning", "warn"}:
            accent = "#d97706"
        elif callout_type in {"danger", "error"}:
            accent = "#dc2626"
        elif callout_type in {"tip", "success"}:
            accent = "#059669"
        return (
            f"margin:18px 0;padding:14px 16px;background:{layout['quote_background']};"
            f"border:1px solid {accent};border-radius:16px;color:{layout['text_color']};"
            f"font-size:15px;line-height:1.9;"
        )

    @staticmethod
    def _list_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:0 0 18px;padding-left:22px;color:{layout['text_color']};font-size:15px;line-height:1.95;"
        )

    @staticmethod
    def _li_style(layout: dict[str, Any]) -> str:
        return f"margin:0 0 8px;color:{layout['text_color']};"

    @staticmethod
    def _code_style(layout: dict[str, Any]) -> str:
        return (
            f"margin:18px 0;padding:16px;background:{layout['code_background']};color:{layout['code_color']};"
            f"border-radius:16px;overflow:auto;font-size:13px;line-height:1.75;font-family:Consolas,Monaco,monospace;"
            f"border:1px solid {layout['border_color']};box-sizing:border-box;"
        )

    @staticmethod
    def _hr_style(layout: dict[str, Any]) -> str:
        return f"margin:26px 0;border:none;border-top:1px solid {layout['border_color']};"

    @staticmethod
    def _table_wrap_style(layout: dict[str, Any]) -> str:
        return "margin:18px 0;overflow:auto;"

    @staticmethod
    def _table_style(layout: dict[str, Any]) -> str:
        return (
            f"width:100%;border-collapse:collapse;border:1px solid {layout['border_color']};"
            f"border-radius:14px;overflow:hidden;background:{layout['card_background']};"
        )

    @staticmethod
    def _th_style(layout: dict[str, Any]) -> str:
        return (
            f"padding:10px 12px;background:{layout['table_header_background']};color:{layout['heading_color']};"
            f"border-bottom:1px solid {layout['border_color']};font-size:13px;text-align:left;"
        )

    @staticmethod
    def _td_style(layout: dict[str, Any]) -> str:
        return (
            f"padding:10px 12px;color:{layout['text_color']};border-bottom:1px solid {layout['border_color']};"
            f"font-size:14px;line-height:1.8;vertical-align:top;"
        )
