import unittest
from unittest.mock import patch

from app.services.wechat_service import WeChatService


class DummySettings:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class WeChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = WeChatService(
            DummySettings(
                {
                    "wechat.app_id": "app-id",
                    "wechat.app_secret": "app-secret",
                    "wechat.author": "\u4f5c\u8005\u540d",
                }
            )
        )

    def test_json_dumps_keeps_utf8_characters(self) -> None:
        digest = "\u8fd9\u662f\u4e2d\u6587\u6458\u8981"

        body = self.service._json_dumps({"digest": digest})

        self.assertIn(digest, body)
        self.assertNotIn("\\u8fd9", body)

    def test_digest_uses_character_limit(self) -> None:
        markdown = "# title\n\n" + ("\u4e2d" * 160)

        digest = self.service._digest(markdown)

        self.assertLessEqual(len(digest), 54)
        self.assertLessEqual(len(digest.encode("utf-8")), 120)

    @patch("app.services.wechat_service.requests.post")
    @patch("app.services.wechat_service.requests.get")
    def test_publish_draft_sends_utf8_json_body(self, get_mock, post_mock) -> None:
        get_mock.return_value = FakeResponse({"access_token": "token"})
        post_mock.return_value = FakeResponse({"media_id": "draft-id"})
        title = "\u4e2d\u6587\u6807\u9898"
        markdown = "# {title}\n\n{body}".format(
            title=title,
            body="\u8fd9\u662f\u4e00\u6bb5\u4e2d\u6587\u6458\u8981\u3002",
        )

        with patch.object(self.service, "_resolve_thumb_media_id", return_value=("thumb-123", "")):
            result = self.service.publish_draft(title=title, markdown_content=markdown)

        self.assertTrue(result.success)
        self.assertEqual(result.draft_id, "draft-id")

        _, kwargs = post_mock.call_args
        self.assertIn("data", kwargs)
        self.assertNotIn("json", kwargs)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json; charset=utf-8")

        body = kwargs["data"].decode("utf-8")
        self.assertIn(title, body)
        self.assertIn("\u4f5c\u8005\u540d", body)
        self.assertNotIn("\\u4e2d", body)

    @patch("app.services.wechat_service.requests.post")
    @patch("app.services.wechat_service.requests.get")
    def test_publish_draft_prefers_supplied_html_content(self, get_mock, post_mock) -> None:
        get_mock.return_value = FakeResponse({"access_token": "token"})
        post_mock.return_value = FakeResponse({"media_id": "draft-id"})

        custom_html = "<div><p>custom-rendered-html</p></div>"
        with patch.object(self.service, "_resolve_thumb_media_id", return_value=("thumb-123", "")):
            result = self.service.publish_draft(
                title="中文标题",
                markdown_content="# 中文标题\n\n正文",
                html_content=custom_html,
            )

        self.assertTrue(result.success)
        body = post_mock.call_args.kwargs["data"].decode("utf-8")
        self.assertIn("custom-rendered-html", body)

    @patch("app.services.wechat_service.requests.post")
    @patch("app.services.wechat_service.requests.get")
    def test_publish_draft_retries_without_digest_on_45004(self, get_mock, post_mock) -> None:
        get_mock.return_value = FakeResponse({"access_token": "token"})
        post_mock.side_effect = [
            FakeResponse({"errcode": 45004, "errmsg": "description size out of limit rid: test-rid"}),
            FakeResponse({"media_id": "draft-id"}),
        ]

        with patch.object(self.service, "_resolve_thumb_media_id", return_value=("thumb-123", "")):
            result = self.service.publish_draft(
                title="\u4e2d\u6587\u6807\u9898",
                markdown_content="# \u4e2d\u6587\u6807\u9898\n\n" + ("\u6b63\u6587" * 80),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.draft_id, "draft-id")
        self.assertEqual(result.sent_digest, "")
        self.assertIn("自动改为不传摘要重试", result.reason)
        self.assertEqual(post_mock.call_count, 2)

        first_body = post_mock.call_args_list[0].kwargs["data"].decode("utf-8")
        second_body = post_mock.call_args_list[1].kwargs["data"].decode("utf-8")
        self.assertIn("\"digest\":", first_body)
        self.assertNotIn("\"digest\":", second_body)


if __name__ == "__main__":
    unittest.main()
