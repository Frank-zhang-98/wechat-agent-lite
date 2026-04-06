import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.programmatic_visual_service import ProgrammaticVisualService


class ProgrammaticVisualServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ProgrammaticVisualService()

    def test_render_body_illustration_outputs_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "body.png"
            result = self.service.render_body_illustration(
                article_title="Ollama Tutorial",
                brief={
                    "type": "workflow_diagram",
                    "title": "Ollama 本地运行流程",
                    "caption": "安装、拉模型、开始对话",
                    "must_show": ["安装 Ollama", "拉取模型", "本地对话"],
                },
                output_path=path,
                size="1024*1024",
            )

        self.assertEqual(result["status"], "generated")
        self.assertTrue(path.exists() or result["path"].endswith(".png"))

    def test_render_cover_outputs_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cover.png"
            result = self.service.render_cover(
                article_title="Graphiti：为 AI Agent 构建时间感知知识图谱",
                strategy={
                    "cover_family": "structure",
                    "cover_brief": {
                        "main_claim": "时间感知图谱不是静态 RAG 的替代品，而是动态 Agent 记忆层。",
                        "must_show": ["Graphiti", "双时间模型", "混合检索"],
                    },
                },
                cover_5d={},
                output_path=path,
                size="1280*720",
            )

        self.assertEqual(result["status"], "generated")
        self.assertTrue(path.exists() or result["path"].endswith(".png"))

    def test_overlay_cover_title_outputs_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.png"
            output = Path(tmpdir) / "final.png"
            Image.new("RGB", (1280, 720), (18, 42, 66)).save(source, format="PNG")

            result = self.service.overlay_cover_title(
                base_image_path=source,
                article_title="为 Python AI 智能体设计的 @observe 装饰器",
                output_path=output,
                size="1280*720",
            )

        self.assertEqual(result["status"], "generated")
        self.assertTrue(output.exists() or result["path"].endswith(".png"))

    def test_overlay_cover_title_accepts_safe_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.png"
            output = Path(tmpdir) / "final-top.png"
            Image.new("RGB", (1280, 720), (32, 56, 88)).save(source, format="PNG")

            result = self.service.overlay_cover_title(
                base_image_path=source,
                article_title="Graphiti 为 AI Agent 构建时间感知知识图谱",
                output_path=output,
                size="1280*720",
                title_safe_zone="left_top",
            )

        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["title_safe_zone"], "left_top")
        self.assertTrue(output.exists() or result["path"].endswith(".png"))

    def test_font_candidates_prioritize_cjk_fonts(self) -> None:
        regular = self.service._font_candidates(bold=False)
        bold = self.service._font_candidates(bold=True)

        self.assertTrue(any("noto" in item.lower() or "wqy" in item.lower() or "msyh" in item.lower() for item in regular))
        self.assertTrue(any("bold" in item.lower() or "bd" in item.lower() or "wqy" in item.lower() for item in bold))

    def test_wrap_text_breaks_long_mixed_token_within_width(self) -> None:
        image = Image.new("RGB", (800, 400), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = self.service._font(48, bold=True)
        text = "为Python AI智能体设计的@observe装饰器显著提升本地可观测性"

        lines = self.service._wrap_text(draw, text, font, 320, max_lines=4)

        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all(draw.textlength(line, font=font) <= 320 for line in lines))
        self.assertTrue(all("…" not in line for line in lines))

    def test_body_detail_items_uses_caption_to_increase_density(self) -> None:
        details = self.service._body_detail_items(
            title="装饰器拦截与数据采集架构",
            caption="从函数调用入口到 SQLite 写入的完整链路：时间戳捕获、异常分类、结构化记录、持久化存储",
            must_show=["@observe 装饰器总览"],
            limit=8,
        )

        self.assertIn("@observe 装饰器总览", details)
        self.assertTrue(any("时间戳捕获" in item for item in details))
        self.assertTrue(any("异常分类" in item for item in details))
        self.assertGreaterEqual(len(details), 4)
        self.assertTrue(all(len(item) <= 14 for item in details))

    def test_fit_text_lines_shrinks_instead_of_ellipsis(self) -> None:
        image = Image.new("RGB", (1280, 720), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = self.service._font(62, bold=True)
        text = "Proxy-Pointer RAG技术解读：无向量检索如何实现98.7%准确率并兼顾向量RAG召回能力"

        fitted_font, lines = self.service._fit_text_lines(
            draw=draw,
            text=text,
            font=font,
            max_width=520,
            max_lines=4,
            max_height=250,
            min_size=24,
        )

        self.assertLessEqual(fitted_font.size, font.size)
        self.assertTrue(all("…" not in line for line in lines))
        self.assertTrue(all(draw.textlength(line, font=fitted_font) <= 520 for line in lines))

    def test_short_phrase_prefers_concise_visual_label(self) -> None:
        shortened = self.service._short_phrase("从函数调用入口到 SQLite 写入的完整链路")
        self.assertLessEqual(len(shortened), 14)
        self.assertNotEqual(shortened, "从函数调用入口到 SQLite 写入的完整链路")
        self.assertEqual(self.service._short_phrase("时间戳捕获"), "时间戳捕获")


if __name__ == "__main__":
    unittest.main()
