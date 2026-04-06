import unittest

from app.services.writing_template_service import WritingTemplateService


class WritingTemplateServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = WritingTemplateService()

    def test_infer_content_type_prefers_technical_walkthrough_for_dense_structure(self) -> None:
        result = self.service.infer_content_type(
            {
                "title": "WAIaaS session management for AI agents",
                "summary": "TTL, renewals, absolute lifetime, session architecture and implementation details.",
                "source": "dev.to",
                "url": "https://example.com/session-management",
            },
            source_structure={
                "lead": "The article breaks down session TTL, renewal rules, lifecycle caps, and service components.",
                "coverage_checklist": [
                    "TTL timeout",
                    "Renewal control",
                    "Absolute lifetime",
                    "Session headers",
                ],
                "sections": [
                    {"heading": "Problem and threat model", "summary": "Why AI agents need wallet session controls."},
                    {"heading": "Session architecture", "summary": "Gateway, wallet policy, and session store."},
                    {"heading": "Step 1: TTL", "summary": "How idle timeout is enforced."},
                    {"heading": "Step 2: Renewals", "summary": "How renewal budget is tracked."},
                    {"heading": "Step 3: Absolute lifetime", "summary": "How hard expiry terminates sessions."},
                ],
                "code_blocks": [{"language": "ts", "code_excerpt": "const session = createSession({...})"}],
            },
            primary_source={
                "content_text": "This implementation uses TTL, renewal budget, session headers, and lifecycle guards.",
            },
        )

        self.assertEqual(result, "technical_walkthrough")

    def test_build_fact_pack_includes_structure_and_preserved_code_blocks(self) -> None:
        ctx = {
            "selected_topic": {
                "title": "Building Sourcing Intel",
                "summary": "A supply chain intelligence platform with LangGraph, MCP, and RAG.",
                "url": "https://example.com/article",
                "source": "dev.to",
                "published": "2026-04-01T00:00:00+00:00",
            },
            "source_pack": {
                "primary": {
                    "status": "ok",
                    "content_text": "lead text",
                    "paragraphs": ["para1", "para2", "para3"],
                },
                "related": [],
            },
            "top_k": [],
            "source_structure": {
                "lead": "This system uses LangGraph and MCP to coordinate sourcing.",
                "coverage_checklist": ["LangGraph orchestration", "MCP integration", "RAG layer"],
                "sections": [
                    {
                        "heading": "Step 1: LangGraph orchestration",
                        "summary": "Use LangGraph to control sourcing flow.",
                        "paragraphs": ["State machine design", "Retry behavior"],
                        "code_refs": [0],
                    },
                    {
                        "heading": "MCP integration",
                        "summary": "Expose tools through MCP.",
                        "paragraphs": ["Tool layer", "Agent calls"],
                        "code_refs": [],
                    },
                ],
                "code_blocks": [
                    {
                        "language": "python",
                        "code_excerpt": "graph = StateGraph(State)",
                        "code_text": "graph = StateGraph(State)\ngraph.add_node('fetch', fetch_node)",
                        "kind": "code",
                        "line_count": 2,
                    }
                ],
            },
        }

        fact_pack = self.service.build_fact_pack(ctx, audience_key="ai_builder")

        self.assertTrue(fact_pack["section_blueprint"])
        self.assertTrue(fact_pack["implementation_steps"])
        self.assertTrue(fact_pack["architecture_points"])
        self.assertTrue(fact_pack["code_artifacts"])
        self.assertTrue(fact_pack["preserved_code_blocks"])
        self.assertEqual(fact_pack["coverage_checklist"], ["LangGraph orchestration", "MCP integration", "RAG layer"])
        self.assertIn("graph.add_node", fact_pack["preserved_code_blocks"][0]["code_text"])

    def test_build_write_prompt_includes_code_preservation_and_verbatim_blocks(self) -> None:
        fact_pack = {
            "topic_source": "realpython.com",
            "published": "2026-04-01T00:00:00+00:00",
            "key_points": ["Ollama runs local models without an API key."],
            "related_topics": [],
            "numbers": [],
            "keywords": ["Ollama", "local models"],
            "source_lead": "This tutorial explains how to install and run Ollama locally.",
            "section_blueprint": [
                {"heading": "Install Ollama", "summary": "Use the install script and verify the version."},
                {"heading": "Run your first model", "summary": "Pull a model and open chat."},
            ],
            "implementation_steps": [
                {"title": "Install Ollama", "summary": "Use the install script.", "details": ["Run curl installer"]},
            ],
            "architecture_points": [],
            "code_artifacts": [
                {
                    "section": "Install Ollama",
                    "language": "bash",
                    "summary": "Install Ollama.",
                    "code_text": "curl -fsSL https://ollama.com/install.sh | sh",
                    "kind": "command",
                    "line_count": 1,
                    "preserve_verbatim": True,
                },
            ],
            "preserved_command_blocks": [
                {
                    "section": "Install Ollama",
                    "language": "bash",
                    "summary": "Install Ollama.",
                    "code_text": "curl -fsSL https://ollama.com/install.sh | sh",
                    "kind": "command",
                    "line_count": 1,
                    "preserve_verbatim": True,
                }
            ],
            "preserved_code_blocks": [
                {
                    "section": "Run your first model",
                    "language": "bash",
                    "summary": "Pull a model and start chat.",
                    "code_text": "ollama pull llama3.2:latest\nollama chat llama3.2:latest",
                    "kind": "command",
                    "line_count": 2,
                    "preserve_verbatim": True,
                }
            ],
            "coverage_checklist": ["Install Ollama", "Run your first model"],
            "grounded_hard_facts": ["Ollama is a local LLM runtime."],
            "grounded_official_facts": [],
            "grounded_context_facts": [],
            "industry_context_points": ["Recent toolchain launches show the same shift toward agent-native workflows."],
            "soft_inferences": [],
            "unknowns": [],
            "forbidden_claims": [],
            "evidence_mode": "tutorial",
        }

        prompt = self.service.build_write_prompt(
            topic={
                "title": "Ollama Tutorial",
                "summary": "How to install Ollama and run a local model.",
                "url": "https://realpython.com/ollama/",
            },
            fact_pack=fact_pack,
            audience_key="ai_builder",
            content_type="tutorial",
        )

        self.assertIn("[Code Preservation]", prompt)
        self.assertIn("[Industry Context Integration]", prompt)
        self.assertIn("Do not output a standalone section titled", prompt)
        self.assertIn("Keep command blocks and code blocks verbatim", prompt)
        self.assertIn("curl -fsSL https://ollama.com/install.sh | sh", prompt)
        self.assertIn("ollama pull llama3.2:latest", prompt)


if __name__ == "__main__":
    unittest.main()
