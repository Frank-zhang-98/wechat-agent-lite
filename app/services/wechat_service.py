from __future__ import annotations

import base64
import html
import json
import mimetypes
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from app.services.settings_service import SettingsService


@dataclass
class WeChatDraftResult:
    success: bool
    draft_id: str
    reason: str
    thumb_media_id: str = ""
    sent_title: str = ""
    sent_digest: str = ""
    debug_info: dict = field(default_factory=dict)


class WeChatService:
    BASE = "https://api.weixin.qq.com/cgi-bin"
    SUPPORTED_THUMB_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    TITLE_MAX_CHARS = 32
    AUTHOR_MAX_CHARS = 16
    DIGEST_MAX_CHARS = 54
    DIGEST_MAX_BYTES = 120

    def __init__(self, settings: SettingsService):
        self.settings = settings

    def publish_draft(
        self,
        title: str,
        markdown_content: str,
        html_content: str = "",
        source_url: str = "",
        cover_image_path: str = "",
    ) -> WeChatDraftResult:
        app_id = self.settings.get("wechat.app_id", "").strip()
        app_secret = self.settings.get("wechat.app_secret", "").strip()
        configured_thumb_media_id = self.settings.get("wechat.thumb_media_id", "").strip()
        author = self.settings.get("wechat.author", "").strip()
        default_source = self.settings.get("wechat.content_source_url", "").strip()
        content_source_url = source_url or default_source

        if not app_id or not app_secret:
            return WeChatDraftResult(False, "", "缺少微信公众号 app_id/app_secret")

        token = self._get_access_token(app_id, app_secret)
        thumb_media_id, thumb_error = self._resolve_thumb_media_id(
            token=token,
            configured_thumb_media_id=configured_thumb_media_id,
            cover_image_path=cover_image_path,
        )
        if not thumb_media_id:
            return WeChatDraftResult(False, "", thumb_error)

        sent_title = self._prepare_title(title)
        sent_author = self._prepare_author(author)
        sent_digest = self._digest(markdown_content)
        final_html = str(html_content or "").strip() or self._markdown_to_html(markdown_content)
        final_html = self._prepare_html_images_for_wechat(token=token, html_content=final_html)
        article = {
            "title": sent_title,
            "author": sent_author,
            "digest": sent_digest,
            "content": final_html,
            "content_source_url": content_source_url,
            "thumb_media_id": thumb_media_id,
            "need_open_comment": 0,
            "only_fans_can_comment": 0,
        }
        data = self._draft_add(token=token, article=article)
        final_digest = sent_digest
        result_note = ""
        if self._is_digest_limit_error(data) and article.get("digest"):
            fallback_article = dict(article)
            fallback_article.pop("digest", None)
            data = self._draft_add(token=token, article=fallback_article)
            final_digest = ""
            result_note = "（已自动改为不传摘要重试）"
        if int(data.get("errcode", 0)) != 0:
            debug_info = self._build_debug_info(token=token, data=data)
            return WeChatDraftResult(
                False,
                "",
                f"微信草稿发布失败: {data.get('errcode')} {data.get('errmsg', '')}{result_note}",
                thumb_media_id=thumb_media_id,
                sent_title=sent_title,
                sent_digest=final_digest,
                debug_info=debug_info,
            )
        draft_id = str(data.get("media_id", "")).strip()
        return WeChatDraftResult(
            True,
            draft_id,
            "ok" if not result_note else f"ok{result_note}",
            thumb_media_id=thumb_media_id,
            sent_title=sent_title,
            sent_digest=final_digest,
        )

    @staticmethod
    def _digest(markdown_text: str) -> str:
        lines = []
        for raw in markdown_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# "):
                continue
            if line.startswith("## "):
                continue
            if line.startswith("### "):
                continue
            if line.startswith("!["):
                continue
            if line.startswith("- "):
                line = line[2:]
            line = re.sub(r"^\d+\.\s+", "", line)
            line = line.replace("**", "").replace("*", "").replace("`", "").strip()
            if line:
                lines.append(line)
        plain = WeChatService._normalize_text(" ".join(lines))
        plain = WeChatService._truncate_chars(plain, WeChatService.DIGEST_MAX_CHARS)
        return WeChatService._truncate_utf8_bytes(plain, WeChatService.DIGEST_MAX_BYTES)

    @classmethod
    def _prepare_title(cls, title: str) -> str:
        value = cls._normalize_text(title)
        return cls._truncate_chars(value, cls.TITLE_MAX_CHARS)

    @classmethod
    def _prepare_author(cls, author: str) -> str:
        value = cls._normalize_text(author)
        return cls._truncate_chars(value, cls.AUTHOR_MAX_CHARS)

    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        lines = [x.strip() for x in markdown_text.splitlines() if x.strip()]
        out = []
        for line in lines:
            if line.startswith("### "):
                out.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("## "):
                out.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("# "):
                out.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("- "):
                out.append(f"<p>• {html.escape(line[2:])}</p>")
            elif line[0:2].isdigit() and line[1:3] == ". ":
                out.append(f"<p>{html.escape(line)}</p>")
            else:
                out.append(f"<p>{html.escape(line)}</p>")
        return "\n".join(out)

    def _prepare_html_images_for_wechat(self, *, token: str, html_content: str) -> str:
        content = str(html_content or "").strip()
        if not content:
            return content

        img_pattern = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', flags=re.IGNORECASE)
        cache: dict[str, str] = {}

        def replace(match: re.Match[str]) -> str:
            prefix, src, suffix = match.groups()
            original_src = str(src or "").strip()
            if not original_src:
                return match.group(0)
            if original_src in cache:
                return f"{prefix}{cache[original_src]}{suffix}"
            if self._is_wechat_image_url(original_src):
                cache[original_src] = original_src
                return match.group(0)
            try:
                uploaded_url = self._upload_article_image(token=token, src=original_src)
            except Exception:
                return match.group(0)
            cache[original_src] = uploaded_url
            return f"{prefix}{uploaded_url}{suffix}"

        return img_pattern.sub(replace, content)

    def _upload_article_image(self, *, token: str, src: str) -> str:
        value = str(src or "").strip()
        if value.startswith("data:image/"):
            return self._upload_data_image(token=token, data_url=value)
        if value.startswith("http://") or value.startswith("https://"):
            return self._upload_remote_image(token=token, url=value)
        return self._upload_local_image(token=token, image_path=Path(value))

    def _upload_data_image(self, *, token: str, data_url: str) -> str:
        match = re.match(r"^data:(image/[^;]+);base64,(.+)$", data_url, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise RuntimeError("invalid data image url")
        mime_type = match.group(1).strip().lower()
        payload = base64.b64decode(match.group(2).strip())
        suffix = mimetypes.guess_extension(mime_type) or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            temp_path = Path(tmp.name)
        try:
            return self._upload_image_for_article(token=token, image_path=temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _upload_remote_image(self, *, token: str, url: str) -> str:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "").split(";")[0].strip() or "image/png"
        suffix = mimetypes.guess_extension(mime_type) or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(response.content)
            temp_path = Path(tmp.name)
        try:
            return self._upload_image_for_article(token=token, image_path=temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _upload_local_image(self, *, token: str, image_path: Path) -> str:
        if not image_path.exists() or not image_path.is_file():
            raise RuntimeError(f"image not found: {image_path}")
        return self._upload_image_for_article(token=token, image_path=image_path)

    def _upload_image_for_article(self, *, token: str, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        with image_path.open("rb") as file_obj:
            response = requests.post(
                f"{self.BASE}/media/uploadimg",
                params={"access_token": token},
                files={"media": (image_path.name, file_obj, mime_type or "image/png")},
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        if int(data.get("errcode", 0)) != 0:
            raise RuntimeError(f"{data.get('errcode')} {data.get('errmsg', '')}".strip())
        url = str(data.get("url", "")).strip()
        if not url:
            raise RuntimeError("wechat uploadimg succeeded but no url returned")
        return url

    @staticmethod
    def _is_wechat_image_url(url: str) -> bool:
        lowered = str(url or "").strip().lower()
        return "mmbiz.qpic.cn" in lowered or "mmbiz.qlogo.cn" in lowered

    def _get_access_token(self, app_id: str, app_secret: str) -> str:
        response = requests.get(
            f"{self.BASE}/token",
            params={"grant_type": "client_credential", "appid": app_id, "secret": app_secret},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        token = str(data.get("access_token", "")).strip()
        if not token:
            raise RuntimeError(f"获取 access_token 失败: {data.get('errcode')} {data.get('errmsg', '')}")
        return token

    def _resolve_thumb_media_id(
        self,
        *,
        token: str,
        configured_thumb_media_id: str,
        cover_image_path: str,
    ) -> tuple[str, str]:
        if configured_thumb_media_id:
            return configured_thumb_media_id, ""

        if not cover_image_path:
            return "", "缺少 wechat.thumb_media_id，且当前运行没有可用封面图，无法创建草稿"

        image_path = Path(cover_image_path)
        if not image_path.exists():
            return "", f"缺少 wechat.thumb_media_id，且封面图不存在: {image_path}"
        if not image_path.is_file():
            return "", f"缺少 wechat.thumb_media_id，且封面路径不是文件: {image_path}"
        if image_path.suffix.lower() not in self.SUPPORTED_THUMB_SUFFIXES:
            return "", f"缺少 wechat.thumb_media_id，且本次封面不是可上传图片文件: {image_path.name}"

        try:
            media_id = self._upload_image_material(token, image_path)
        except Exception as exc:
            return "", f"封面素材上传失败，无法创建草稿: {exc}"
        return media_id, ""

    def _upload_image_material(self, token: str, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        with image_path.open("rb") as file_obj:
            response = requests.post(
                f"{self.BASE}/material/add_material",
                params={"access_token": token, "type": "image"},
                files={"media": (image_path.name, file_obj, mime_type or "image/png")},
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        if int(data.get("errcode", 0)) != 0:
            raise RuntimeError(f"{data.get('errcode')} {data.get('errmsg', '')}".strip())
        media_id = str(data.get("media_id", "")).strip()
        if not media_id:
            raise RuntimeError("微信素材上传成功但未返回 media_id")
        return media_id

    @staticmethod
    def _normalize_text(text: str) -> str:
        value = str(text or "")
        value = value.replace("\r", " ").replace("\n", " ")
        return " ".join(value.split()).strip()

    @staticmethod
    def _truncate_chars(text: str, max_chars: int) -> str:
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].strip()

    @staticmethod
    def _json_dumps(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _post_json(cls, url: str, *, params: dict[str, str], payload: dict, timeout: int) -> requests.Response:
        # WeChat draft fields must be sent as raw UTF-8, not \uXXXX escapes.
        body = cls._json_dumps(payload).encode("utf-8")
        return requests.post(
            url,
            params=params,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=timeout,
        )

    def _draft_add(self, *, token: str, article: dict) -> dict:
        payload = {"articles": [article]}
        response = self._post_json(
            f"{self.BASE}/draft/add",
            params={"access_token": token},
            payload=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _is_digest_limit_error(data: dict) -> bool:
        try:
            errcode = int(data.get("errcode", 0))
        except Exception:
            errcode = 0
        errmsg = str(data.get("errmsg", "") or "").lower()
        return errcode == 45004 or "description size out of limit" in errmsg

    def _build_debug_info(self, *, token: str, data: dict) -> dict:
        debug_info = {
            "errcode": data.get("errcode", 0),
            "errmsg": data.get("errmsg", ""),
        }
        rid = self._extract_rid(str(data.get("errmsg", "") or ""))
        if not rid:
            return debug_info
        debug_info["rid"] = rid
        rid_info = self._lookup_rid_info(token=token, rid=rid)
        if rid_info:
            debug_info["rid_info"] = rid_info
        return debug_info

    def _lookup_rid_info(self, *, token: str, rid: str) -> dict:
        try:
            response = self._post_json(
                f"{self.BASE}/openapi/rid/get",
                params={"access_token": token},
                payload={"rid": rid},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {"lookup_failed": str(exc)}
        request = data.get("request") if isinstance(data.get("request"), dict) else {}
        return {
            "invoke_time": request.get("invoke_time"),
            "request_url": request.get("request_url", ""),
            "request_body": self._clip_debug_text(request.get("request_body", "")),
            "response_body": self._clip_debug_text(request.get("response_body", "")),
            "client_ip": request.get("client_ip", ""),
        }

    @staticmethod
    def _extract_rid(errmsg: str) -> str:
        match = re.search(r"rid:\s*([0-9A-Za-z\-]+)", str(errmsg or ""))
        return match.group(1) if match else ""

    @staticmethod
    def _clip_debug_text(value: str, limit: int = 1200) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated]"

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
