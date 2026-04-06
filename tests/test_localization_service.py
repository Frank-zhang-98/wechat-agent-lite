import unittest

from app.services.localization_service import LocalizationService


class LocalizationServiceTests(unittest.TestCase):
    def test_localize_heading_text_supports_direct_and_mixed_headings(self) -> None:
        self.assertEqual(LocalizationService.localize_heading_text("Introduction"), "引言")
        self.assertEqual(LocalizationService.localize_heading_text("Core Features 实现拆解"), "核心能力实现拆解")
        self.assertEqual(LocalizationService.localize_heading_text("What's new in v3.0.0"), "v3.0.0 更新了什么")
        self.assertEqual(LocalizationService.localize_heading_text("Use Case Setup"), "用例设置")
        self.assertEqual(LocalizationService.localize_heading_text("How does PageIndex work?"), "PageIndex 如何工作？")
        self.assertEqual(
            LocalizationService.localize_heading_text("Phase 1: Indexing (once per document)"),
            "阶段 1：索引（每份文档一次）",
        )
        self.assertEqual(
            LocalizationService.localize_heading_text("对比 of Vectorless vs Flat Vector RAG"),
            "无向量 RAG 与 扁平向量 RAG 对比",
        )
        self.assertEqual(
            LocalizationService.localize_heading_text("Engineering a Better 检索器 — Proxy-Pointer RAG"),
            "Proxy-Pointer RAG：更好的检索器工程",
        )

    def test_localize_visual_text_preserves_commands_and_translates_labels(self) -> None:
        self.assertEqual(
            LocalizationService.localize_visual_text("Agent Runtime is infrastructure, not a plugin."),
            "智能体运行时不是插件，而是基础设施。",
        )
        self.assertEqual(
            LocalizationService.localize_visual_items(["Agent", "Sandbox", "npx 0nmcp@latest"]),
            ["智能体", "沙箱", "npx 0nmcp@latest"],
        )


if __name__ == "__main__":
    unittest.main()
