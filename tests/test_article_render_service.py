import tempfile
import unittest
from pathlib import Path

from app.services.article_render_service import ArticleRenderService


class ArticleRenderServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ArticleRenderService()

    def test_resolve_layout_uses_content_type_mapping(self) -> None:
        layout = self.service.resolve_layout(content_type="tutorial")

        self.assertEqual(layout["name"], "practical_tutorial")
        self.assertEqual(layout["source"], "content_type_rule")

    def test_resolve_layout_maps_technical_walkthrough_to_tutorial_layout(self) -> None:
        layout = self.service.resolve_layout(content_type="technical_walkthrough")

        self.assertEqual(layout["name"], "practical_tutorial")
        self.assertEqual(layout["source"], "content_type_rule")

    def test_render_skips_duplicate_h1_and_outputs_html(self) -> None:
        rendered = self.service.render(
            "# 测试标题\n\n第一段导语。\n\n## 小节\n- 要点一\n- 要点二",
            article_title="测试标题",
            content_type="tutorial",
            target_audience="ai_builder",
        )

        self.assertEqual(rendered.layout_name, "practical_tutorial")
        self.assertIn("<h2", rendered.html)
        self.assertIn("<ul", rendered.html)
        self.assertNotIn(">测试标题</h1>", rendered.html)

    def test_render_inserts_illustration_after_matching_heading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "illus.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x17\x9b"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            rendered = self.service.render(
                "## 总览\n\n正文A\n\n## 细节\n\n正文B",
                article_title="测试",
                content_type="technical_walkthrough",
                illustrations=[
                    {
                        "section": "细节",
                        "title": "细节图",
                        "caption": "用于说明细节链路",
                        "path": str(image_path),
                    }
                ],
            )

        self.assertIn("<figure", rendered.html)
        self.assertIn("data:image/png;base64,", rendered.html)
        self.assertIn("用于说明细节链路", rendered.html)
        self.assertLess(rendered.html.index("正文B"), rendered.html.index("<figure"))
        self.assertLess(rendered.html.index("<figure"), rendered.html.rindex("细节"))

    def test_render_inserts_illustration_before_next_heading_when_section_has_no_paragraph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "illus.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x17\x9b"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            rendered = self.service.render(
                "## 细节\n\n## 后续\n\n正文B",
                article_title="测试",
                content_type="technical_walkthrough",
                illustrations=[
                    {
                        "section": "细节",
                        "title": "细节图",
                        "caption": "用于说明细节链路",
                        "path": str(image_path),
                    }
                ],
            )

        detail_heading_index = rendered.html.index(">细节</h2>")
        figure_index = rendered.html.index("<figure")
        next_heading_index = rendered.html.index(">后续</h2>")
        self.assertLess(detail_heading_index, figure_index)
        self.assertLess(figure_index, next_heading_index)
        self.assertIn("图解：细节图", rendered.html)

    def test_render_fuzzy_matches_illustration_to_section_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "illus.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x17\x9b"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            rendered = self.service.render(
                "## 模式设计\n\n这一节重点讨论六种多智能体协作模式架构对比，以及 Supervisor 和 Swarm 的分工。\n\n## 可执行建议\n\n正文B",
                article_title="测试",
                content_type="technical_walkthrough",
                illustrations=[
                    {
                        "section": "六种多智能体协作模式架构对比",
                        "title": "六种多智能体协作模式架构对比",
                        "caption": "每种模式为独立可插拔模块。",
                        "path": str(image_path),
                    }
                ],
            )

        section_index = rendered.html.index(">模式设计</h2>")
        figure_index = rendered.html.index("<figure")
        advice_index = rendered.html.index(">可执行建议</h2>")
        self.assertLess(section_index, figure_index)
        self.assertLess(figure_index, advice_index)
        self.assertIn("六种多智能体协作模式架构对比", rendered.html)
        self.assertTrue(
            "下面这张图" in rendered.html
            or "可以直接对照下面这张图来看" in rendered.html
            or "放到图里看层次和分工会更直观" in rendered.html
        )

    def test_render_code_block_preserves_multiline_layout_for_wechat(self) -> None:
        rendered = self.service.render(
            "## 示例\n\n```bash\npip install graphiti-core[neptune]\nuv add graphiti-core\n```",
            article_title="测试标题",
            content_type="technical_walkthrough",
            target_audience="ai_builder",
        )

        self.assertIn("<section style=", rendered.html)
        self.assertIn("<code style=", rendered.html)
        self.assertIn("pip install graphiti-core[neptune]<br/>uv add graphiti-core", rendered.html)
        self.assertIn("text-transform:uppercase", rendered.html)


if __name__ == "__main__":
    unittest.main()
