from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from sqlalchemy.orm import Session

from app.models import LLMCall
from app.services.settings_service import SettingsService


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    estimated: bool
    provider: str
    model: str


class LLMGateway:
    """
    Unified model gateway.
    - Text generation uses OpenAI-compatible /chat/completions.
    - Native DashScope rerank models use the rerank API.
    - Native DashScope image models use the text-to-image async API.
    - Missing config falls back to deterministic mock output.
    """

    def __init__(self, session: Session, settings: SettingsService):
        self.session = session
        self.settings = settings
        self.runtime_meta: dict[str, list[dict[str, Any]]] = {}

    def call(self, run_id: str, step_name: str, role: str, prompt: str, temperature: float = 0.4) -> LLMResult:
        start = time.perf_counter()
        cfg = self._role_cfg(role)
        timeout_seconds, complexity_tier = self._resolve_timeout_plan(role=role, step_name=step_name, prompt_text=prompt)

        if cfg["base_url"] and cfg["api_key"] and cfg["model_id"]:
            result = self._call_openai_compatible(
                cfg,
                prompt=prompt,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                role=role,
                step_name=step_name,
            )
        else:
            result = self._mock_result(role=role, prompt=prompt)
        self._record_runtime_meta(
            step_name=step_name,
            role=role,
            timeout_seconds=timeout_seconds,
            complexity_tier=complexity_tier,
            provider=result.provider,
            model=result.model,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        result.latency_ms = max(result.latency_ms, latency_ms)
        self._record_call(
            run_id=run_id,
            step_name=step_name,
            role=role,
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            estimated=result.estimated,
        )
        return result

    def rerank_documents(
        self,
        run_id: str,
        step_name: str,
        role: str,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self._role_cfg(role)
        top_n = max(1, min(top_n or len(documents) or 1, len(documents) or 1))
        documents_text = "\n".join(documents[:8])
        timeout_seconds, complexity_tier = self._resolve_timeout_plan(
            role=role,
            step_name=step_name,
            prompt_text=f"{query}\n{documents_text}",
            document_count=len(documents),
        )
        start = time.perf_counter()

        if not documents:
            return []

        if cfg["base_url"] and cfg["api_key"] and cfg["model_id"] and self._is_native_rerank_model(cfg["model_id"]):
            results, usage = self._call_dashscope_rerank(
                cfg,
                query=query,
                documents=documents,
                top_n=top_n,
                timeout_seconds=timeout_seconds,
                role=role,
                step_name=step_name,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            total_tokens = int(usage.get("total_tokens", 0) or 0)
            self._record_call(
                run_id=run_id,
                step_name=step_name,
                role=role,
                provider="alibaba-bailian",
                model=cfg["model_id"],
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                estimated=False,
            )
            self._record_runtime_meta(
                step_name=step_name,
                role=role,
                timeout_seconds=timeout_seconds,
                complexity_tier=complexity_tier,
                provider="alibaba-bailian",
                model=cfg["model_id"],
            )
            return results

        prompt = self._build_chat_rerank_prompt(query=query, documents=documents)
        chat_result = self.call(run_id, step_name, role, prompt, temperature=0.1)
        parsed = self._parse_chat_rerank_output(chat_result.text, len(documents))
        return parsed[:top_n] if parsed else self._fallback_rerank(documents)

    def generate_cover_image(
        self,
        run_id: str,
        step_name: str,
        role: str,
        *,
        prompt: str,
        output_dir: Path,
        size: str = "1280*720",
    ) -> dict[str, Any]:
        cfg = self._role_cfg(role)
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = output_dir / "cover_prompt.txt"
        prompt_path.write_text(prompt.strip(), encoding="utf-8")

        timeout_seconds = max(10, self.settings.get_int("model.request_timeout_seconds", 45))
        start = time.perf_counter()

        if cfg["base_url"] and cfg["api_key"] and cfg["model_id"] and self._is_native_image_model(cfg["model_id"]):
            resolved_timeout, complexity_tier = self._resolve_timeout_plan(
                role=role,
                step_name=step_name,
                prompt_text=prompt,
                base_timeout=timeout_seconds,
            )
            image_info = self._call_dashscope_text_to_image(
                cfg,
                prompt=prompt,
                size=size,
                timeout_seconds=max(
                    90,
                    resolved_timeout * 3,
                ),
                role=role,
                step_name=step_name,
            )
            file_ext = self._infer_file_extension(image_info.get("url", ""), image_info.get("content_type", ""))
            image_path = output_dir / f"cover{file_ext}"
            self._download_binary(image_info["url"], image_path)
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._record_call(
                run_id=run_id,
                step_name=step_name,
                role=role,
                provider="alibaba-bailian",
                model=cfg["model_id"],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                estimated=False,
            )
            self._record_runtime_meta(
                step_name=step_name,
                role=role,
                timeout_seconds=max(90, resolved_timeout * 3),
                complexity_tier=complexity_tier,
                provider="alibaba-bailian",
                model=cfg["model_id"],
            )
            return {
                "status": "generated",
                "size": size.replace("*", "x"),
                "prompt": prompt[:2000],
                "path": str(image_path).replace("\\", "/"),
                "prompt_path": str(prompt_path).replace("\\", "/"),
                "task_id": image_info.get("task_id", ""),
                "remote_url": image_info.get("url", ""),
                "model": cfg["model_id"],
            }

        chat_result = self.call(run_id, step_name, role, prompt, temperature=0.4)
        fallback_path = output_dir / "cover.txt"
        fallback_path.write_text(chat_result.text.strip(), encoding="utf-8")
        return {
            "status": "prompt_only",
            "size": size.replace("*", "x"),
            "prompt": prompt[:2000],
            "path": str(fallback_path).replace("\\", "/"),
            "prompt_path": str(prompt_path).replace("\\", "/"),
            "model": chat_result.model,
            "note": "当前配置不是原生出图模型，已回退为文本结果。",
        }

    def _role_cfg(self, role: str) -> dict[str, str]:
        key = f"model.{role}"
        return {
            "base_url": self.settings.get(f"{key}.base_url", "").strip(),
            "api_key": self.settings.get(f"{key}.api_key", "").strip(),
            "model_id": self.settings.get(f"{key}.model_id", "").strip(),
        }

    def _call_openai_compatible(
        self,
        cfg: dict[str, str],
        prompt: str,
        temperature: float,
        timeout_seconds: int,
        role: str,
        step_name: str,
    ) -> LLMResult:
        endpoint = self._resolve_chat_endpoint(cfg["base_url"])
        payload = {
            "model": cfg["model_id"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = self._request_with_timeout_retry(
            "post",
            endpoint,
            timeout=timeout_seconds,
            role=role,
            step_name=step_name,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", 0))
        estimated = False
        if total_tokens <= 0:
            prompt_tokens = max(8, len(prompt) // 4)
            completion_tokens = max(12, len(text) // 4)
            total_tokens = prompt_tokens + completion_tokens
            estimated = True
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=0,
            estimated=estimated,
            provider="alibaba-bailian",
            model=cfg["model_id"],
        )

    def _call_dashscope_rerank(
        self,
        cfg: dict[str, str],
        *,
        query: str,
        documents: list[str],
        top_n: int,
        timeout_seconds: int,
        role: str,
        step_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        endpoint = f"{self._resolve_dashscope_api_root(cfg['base_url'])}/services/rerank/text-rerank/text-rerank"
        model_id = cfg["model_id"].strip()
        lowered_model_id = model_id.lower()
        if lowered_model_id == "qwen3-vl-rerank":
            payload: dict[str, Any] = {
                "model": model_id,
                "input": {
                    "query": {"text": query},
                    "documents": [{"text": text} for text in documents],
                },
                "parameters": {
                    "top_n": top_n,
                    "return_documents": True,
                },
            }
        else:
            payload = {
                "model": model_id,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": top_n,
                    "return_documents": True,
                },
            }
        response = self._request_with_timeout_retry(
            "post",
            endpoint,
            timeout=timeout_seconds,
            role=role,
            step_name=step_name,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not response.ok:
            try:
                err_payload = response.json()
            except Exception:
                err_payload = response.text
            if isinstance(err_payload, dict):
                err_code = err_payload.get("code") or response.status_code
                err_msg = err_payload.get("message") or err_payload.get("Message") or json.dumps(err_payload, ensure_ascii=False)
                raise RuntimeError(f"DashScope rerank 请求失败：{err_code} | {err_msg}")
            raise RuntimeError(f"DashScope rerank 请求失败：HTTP {response.status_code} | {str(err_payload).strip()}")
        data = response.json()
        if isinstance(data, dict) and data.get("code"):
            raise RuntimeError(data.get("message") or data["code"])
        output = data.get("output", {}) if isinstance(data, dict) else {}
        raw_results = output.get("results", []) if isinstance(output, dict) else []
        results: list[dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                continue
            score = float(item.get("relevance_score", 0.0) or 0.0)
            doc_payload = item.get("document")
            if isinstance(doc_payload, dict):
                document_text = str(doc_payload.get("text", ""))
            else:
                document_text = documents[index]
            results.append(
                {
                    "index": index,
                    "relevance_score": max(0.0, min(score, 1.0)),
                    "document": document_text,
                    "reason": "",
                }
            )
        return results, data.get("usage", {}) if isinstance(data, dict) else {}

    def _call_dashscope_text_to_image(
        self,
        cfg: dict[str, str],
        *,
        prompt: str,
        size: str,
        timeout_seconds: int,
        role: str,
        step_name: str,
    ) -> dict[str, Any]:
        api_root = self._resolve_dashscope_api_root(cfg["base_url"])
        create_endpoint = f"{api_root}/services/aigc/text2image/image-synthesis"
        create_payload = {
            "model": cfg["model_id"],
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size,
                "n": 1,
                "style": "<auto>",
            },
        }
        create_resp = self._request_with_timeout_retry(
            "post",
            create_endpoint,
            timeout=min(timeout_seconds, 60),
            role=role,
            step_name=step_name,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json=create_payload,
        )
        create_resp.raise_for_status()
        create_data = create_resp.json()
        if isinstance(create_data, dict) and create_data.get("code"):
            raise RuntimeError(create_data.get("message") or create_data["code"])

        output = create_data.get("output", {}) if isinstance(create_data, dict) else {}
        task_id = str(output.get("task_id", "") or "").strip()
        if not task_id:
            raise RuntimeError("图像生成未返回 task_id")

        task_endpoint = f"{api_root}/tasks/{task_id}"
        deadline = time.time() + timeout_seconds
        last_status = str(output.get("task_status", "PENDING") or "PENDING")
        while time.time() < deadline:
            task_resp = self._request_with_timeout_retry(
                "get",
                task_endpoint,
                timeout=min(timeout_seconds, 30),
                role=role,
                step_name=step_name,
                headers={"Authorization": f"Bearer {cfg['api_key']}"},
            )
            task_resp.raise_for_status()
            task_data = task_resp.json()
            task_output = task_data.get("output", {}) if isinstance(task_data, dict) else {}
            task_status = str(task_output.get("task_status", "") or "").upper()
            if task_status == "SUCCEEDED":
                results = task_output.get("results", []) if isinstance(task_output, dict) else []
                if not results or not isinstance(results[0], dict) or not results[0].get("url"):
                    raise RuntimeError("图像生成成功，但未返回图片地址")
                image_url = str(results[0]["url"])
                head_resp = requests.head(image_url, timeout=20, allow_redirects=True)
                content_type = head_resp.headers.get("Content-Type", "") if head_resp.ok else ""
                return {
                    "task_id": task_id,
                    "url": image_url,
                    "content_type": content_type,
                    "image_count": int((task_data.get("usage", {}) or {}).get("image_count", 1) or 1),
                }
            if task_status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise RuntimeError(task_output.get("message") or f"图像生成失败：{task_status}")
            last_status = task_status or last_status
            time.sleep(2.5)

        raise RuntimeError(f"图像生成等待超时，当前状态：{last_status}")

    def _download_binary(self, url: str, output_path: Path) -> None:
        response = self._request_with_timeout_retry(
            "get",
            url,
            timeout=60,
            role="cover_image",
            step_name="COVER_GEN",
        )
        response.raise_for_status()
        output_path.write_bytes(response.content)

    def _resolve_timeout_plan(
        self,
        *,
        role: str,
        step_name: str,
        prompt_text: str = "",
        document_count: int = 0,
        base_timeout: int | None = None,
    ) -> tuple[int, str]:
        timeout_seconds = max(10, base_timeout or self.settings.get_int("model.request_timeout_seconds", 45))
        if step_name == "SOURCE_MAINTENANCE":
            maintenance_timeout = max(
                10,
                self.settings.get_int("source_maintenance.llm_timeout_seconds", min(timeout_seconds, 20)),
            )
            tier = self._resolve_complexity_tier(
                role=role,
                step_name=step_name,
                prompt_text=prompt_text,
                document_count=document_count,
            )
            return maintenance_timeout, tier
        multiplier = 1.0
        if role == "writer":
            multiplier = 2.0
        elif role == "decision":
            multiplier = 1.6
        elif role == "rerank":
            multiplier = 1.5
        elif role == "cover_prompt":
            multiplier = 1.4
        if step_name in {"WRITE", "FACT_COMPRESS"}:
            multiplier = max(multiplier, 2.0)
        elif step_name in {"RERANK", "SELECT"}:
            multiplier = max(multiplier, 1.5)
        rule_timeout = int(max(10, round(timeout_seconds * multiplier)))
        if not self.settings.get_bool("model.timeout_complexity_enabled", True):
            return rule_timeout, "rule"

        tier = self._resolve_complexity_tier(
            role=role,
            step_name=step_name,
            prompt_text=prompt_text,
            document_count=document_count,
        )
        tier_multiplier = {
            "low": 1.0,
            "medium": 1.35,
            "high": 1.8,
            "xhigh": 2.4,
        }.get(tier, 1.0)
        tier_timeout = int(max(10, round(timeout_seconds * tier_multiplier)))
        return max(rule_timeout, tier_timeout), tier

    @staticmethod
    def _resolve_complexity_tier(
        *,
        role: str,
        step_name: str,
        prompt_text: str,
        document_count: int = 0,
    ) -> str:
        prompt_len = len(prompt_text or "")
        if role == "writer" or step_name == "WRITE":
            if prompt_len >= 12000:
                return "xhigh"
            if prompt_len >= 7000:
                return "high"
            if prompt_len >= 3000:
                return "medium"
            return "low"
        if step_name == "FACT_COMPRESS" or role == "decision":
            if prompt_len >= 10000:
                return "xhigh"
            if prompt_len >= 6000:
                return "high"
            if prompt_len >= 2500:
                return "medium"
            return "low"
        if step_name == "RERANK" or role == "rerank":
            if document_count >= 8 or prompt_len >= 10000:
                return "high"
            if document_count >= 4 or prompt_len >= 4000:
                return "medium"
            return "low"
        if role in {"cover_prompt", "cover_image"}:
            if prompt_len >= 4000:
                return "high"
            if prompt_len >= 1800:
                return "medium"
            return "low"
        if prompt_len >= 8000:
            return "high"
        if prompt_len >= 3000:
            return "medium"
        return "low"

    def _request_with_timeout_retry(
        self,
        method: str,
        url: str,
        *,
        timeout: int,
        role: str,
        step_name: str,
        **kwargs: Any,
    ) -> requests.Response:
        if step_name == "SOURCE_MAINTENANCE":
            retry_count = max(0, self.settings.get_int("source_maintenance.llm_retry_count", 0))
            backoff_seconds = max(1, self.settings.get_int("source_maintenance.llm_retry_backoff_seconds", 2))
        else:
            retry_count = max(0, self.settings.get_int("model.timeout_retry_count", 2))
            backoff_seconds = max(1, self.settings.get_int("model.timeout_retry_backoff_seconds", 3))
        last_exc: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                return requests.request(method=method, url=url, timeout=timeout, **kwargs)
            except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt >= retry_count:
                    break
                time.sleep(backoff_seconds * (attempt + 1))
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt >= retry_count:
                    break
                time.sleep(backoff_seconds * (attempt + 1))
        raise RuntimeError(
            f"{step_name}/{role} 模型请求超时或连接失败，已重试 {retry_count} 次，最终超时阈值 {timeout}s: {last_exc}"
        )

    def _record_runtime_meta(
        self,
        *,
        step_name: str,
        role: str,
        timeout_seconds: int,
        complexity_tier: str,
        provider: str,
        model: str,
    ) -> None:
        entries = self.runtime_meta.setdefault(step_name, [])
        entries.append(
            {
                "role": role,
                "timeout_seconds": timeout_seconds,
                "complexity_tier": complexity_tier,
                "provider": provider,
                "model": model,
            }
        )

    def get_step_runtime_meta(self, step_name: str) -> list[dict[str, Any]]:
        return list(self.runtime_meta.get(step_name, []))

    def _record_call(
        self,
        *,
        run_id: str,
        step_name: str,
        role: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: int,
        estimated: bool,
    ) -> None:
        self.session.add(
            LLMCall(
                run_id=run_id,
                step_name=step_name,
                role=role,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                estimated=estimated,
            )
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        try:
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = [x.get("text", "") for x in content if isinstance(x, dict)]
                    return "\n".join(x for x in parts if x).strip()
                return str(content).strip()
        except Exception:
            pass
        return json.dumps(data, ensure_ascii=False)[:1000]

    @staticmethod
    def _resolve_chat_endpoint(base_url: str) -> str:
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        if re.search(r"/v\d+$", url):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    @staticmethod
    def _resolve_dashscope_api_root(base_url: str) -> str:
        normalized = base_url.strip()
        if not normalized:
            return "https://dashscope.aliyuncs.com/api/v1"
        parsed = urlparse(normalized)
        if not parsed.scheme:
            parsed = urlparse(f"https://{normalized}")
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/api/v1"
        return normalized.rstrip("/")

    @staticmethod
    def _is_native_rerank_model(model_id: str) -> bool:
        lowered = model_id.strip().lower()
        return "rerank" in lowered

    @staticmethod
    def _is_native_image_model(model_id: str) -> bool:
        lowered = model_id.strip().lower()
        return lowered.startswith("wan") or lowered.startswith("wanx") or "-t2i" in lowered or lowered.startswith("qwen-image")

    @staticmethod
    def _build_chat_rerank_prompt(query: str, documents: list[str]) -> str:
        lines = []
        for idx, text in enumerate(documents):
            lines.append(f"[{idx}] {text[:1200]}")
        joined = "\n".join(lines)
        return (
            "你是公众号选题编辑。请根据与微信公众号文章潜力的相关性，对候选主题进行重新排序。\n"
            "评价维度：时效性、传播性、可写性、实用价值。\n"
            "请严格输出 JSON 数组，不要输出任何额外解释。\n"
            "每个元素必须包含：index（原始序号）、score（0到100的数字）、reason（不超过30字）。\n"
            "按 score 从高到低排序。\n"
            f"筛选目标：{query}\n"
            f"候选列表：\n{joined}"
        )

    @classmethod
    def _parse_chat_rerank_output(cls, text: str, doc_count: int) -> list[dict[str, Any]]:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        used: set[int] = set()
        results: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= doc_count or index in used:
                continue
            score = float(item.get("score", 0) or 0)
            used.add(index)
            results.append(
                {
                    "index": index,
                    "relevance_score": max(0.0, min(score / 100.0, 1.0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        return results

    @staticmethod
    def _fallback_rerank(documents: list[str]) -> list[dict[str, Any]]:
        results = []
        base = 0.92
        for idx, _ in enumerate(documents):
            results.append(
                {
                    "index": idx,
                    "relevance_score": max(0.0, base - idx * 0.06),
                    "reason": "启发式兜底",
                }
            )
        return results

    @staticmethod
    def _infer_file_extension(url: str, content_type: str) -> str:
        lowered_type = content_type.lower()
        if "jpeg" in lowered_type or "jpg" in lowered_type:
            return ".jpg"
        if "webp" in lowered_type:
            return ".webp"
        if "png" in lowered_type:
            return ".png"
        path = urlparse(url).path.lower()
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            if path.endswith(suffix):
                return ".jpg" if suffix == ".jpeg" else suffix
        return ".png"

    @staticmethod
    def _mock_result(role: str, prompt: str) -> LLMResult:
        prefix = {
            "decision": "Decision result",
            "rerank": "Rerank result",
            "writer": "Writer result",
            "cover_prompt": "Cover prompt result",
            "cover_image": "Cover image result",
        }.get(role, "Model result")
        tail = prompt[:120].replace("\n", " ")
        text = f"{prefix}: {tail}"
        prompt_tokens = max(8, len(prompt) // 4)
        completion_tokens = max(12, len(text) // 4)
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=random.randint(20, 80),
            estimated=True,
            provider="alibaba-bailian",
            model="mock-model",
        )
