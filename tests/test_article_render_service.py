import unittest

from app.services.article_render_service import ArticleRenderService


class ArticleRenderServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ArticleRenderService()

    def test_resolve_layout_uses_content_type_mapping(self) -> None:
        layout = self.service.resolve_layout(content_type="tutorial")

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
        self.assertNotIn("模板：", rendered.html)
        self.assertNotIn("类型：", rendered.html)
        self.assertNotIn("受众：", rendered.html)


if __name__ == "__main__":
    unittest.main()
