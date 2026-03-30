from __future__ import annotations

import base64
import json
import os
import socket
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse


@dataclass
class ProxyNode:
    scheme: str
    host: str
    port: int
    summary: str
    transport: str = ""
    security: str = ""


class ProxyLinkService:
    DEFAULT_SOCKS = "socks5h://127.0.0.1:10808"
    DEFAULT_SOCKS_PORTS = [10808, 1080, 7890, 7891, 7898, 7899, 20170, 2080, 9050]
    DEFAULT_HTTP_PORTS = [7897, 8080, 8001, 8118]

    @classmethod
    def parse_share_link(cls, share_link: str) -> ProxyNode:
        link = share_link.strip()
        if not link:
            raise ValueError("empty share link")

        if link.startswith("vless://"):
            return cls._parse_vless(link)
        if link.startswith("trojan://"):
            return cls._parse_trojan(link)
        if link.startswith("vmess://"):
            return cls._parse_vmess(link)
        if link.startswith("ss://"):
            return cls._parse_ss(link)
        raise ValueError("unsupported share link scheme")

    @classmethod
    def derive_settings(cls, share_link: str, current_all_proxy: str) -> dict[str, str]:
        node = cls.parse_share_link(share_link)
        all_proxy = cls.resolve_all_proxy(current_all_proxy=current_all_proxy)
        return {
            "proxy.enabled": "true",
            "proxy.all_proxy": all_proxy,
            "proxy.node_scheme": node.scheme,
            "proxy.node_host": node.host,
            "proxy.node_port": str(node.port),
            "proxy.node_transport": node.transport,
            "proxy.node_security": node.security,
            "proxy.node_summary": node.summary,
        }

    @classmethod
    def resolve_all_proxy(cls, current_all_proxy: str) -> str:
        current = (current_all_proxy or "").strip()
        if current:
            # If user explicitly set a non-local proxy, trust it.
            parsed = cls._parse_proxy_url(current)
            if parsed and not cls._is_local_host(parsed[1]):
                return current
            # If a local proxy is configured and reachable, keep it.
            if cls._probe_proxy_url(current):
                return current

        # Try process env proxy values if available and reachable.
        for env_name in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            env_val = (os.getenv(env_name) or "").strip()
            if env_val and cls._probe_proxy_url(env_val):
                return env_val

        # Auto-detect local inbound proxy ports.
        detected = cls._detect_local_proxy()
        if detected:
            return detected

        # Fallbacks
        return current or cls.DEFAULT_SOCKS

    @classmethod
    def _detect_local_proxy(cls) -> str:
        socks_ports = cls._candidate_ports("WECHAT_AGENT_SOCKS_PORTS", cls.DEFAULT_SOCKS_PORTS)
        for port in socks_ports:
            if cls._probe_socks5("127.0.0.1", port):
                return f"socks5h://127.0.0.1:{port}"

        http_ports = cls._candidate_ports("WECHAT_AGENT_HTTP_PROXY_PORTS", cls.DEFAULT_HTTP_PORTS)
        for port in http_ports:
            if cls._probe_http_proxy("127.0.0.1", port):
                return f"http://127.0.0.1:{port}"
        return ""

    @staticmethod
    def _candidate_ports(env_name: str, defaults: list[int]) -> list[int]:
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            return defaults
        ports: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                port = int(part)
            except ValueError:
                continue
            if 1 <= port <= 65535:
                ports.append(port)
        return ports or defaults

    @classmethod
    def _probe_proxy_url(cls, proxy_url: str) -> bool:
        parsed = cls._parse_proxy_url(proxy_url)
        if not parsed:
            return False
        scheme, host, port = parsed
        if scheme in {"socks5", "socks5h", "socks4", "socks4a"}:
            return cls._probe_socks5(host, port)
        if scheme in {"http", "https"}:
            return cls._probe_http_proxy(host, port)
        return False

    @staticmethod
    def _parse_proxy_url(proxy_url: str) -> tuple[str, str, int] | None:
        try:
            u = urlparse(proxy_url)
            if not u.scheme or not u.hostname or not u.port:
                return None
            return u.scheme.lower(), u.hostname, int(u.port)
        except Exception:
            return None

    @staticmethod
    def _is_local_host(host: str) -> bool:
        host_l = (host or "").lower()
        return host_l in {"127.0.0.1", "localhost", "::1"}

    @staticmethod
    def _probe_socks5(host: str, port: int, timeout: float = 0.7) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(b"\x05\x01\x00")
                resp = sock.recv(2)
                # 0x05 means SOCKS5 server; method can be 0x00/0x02/0xFF.
                return len(resp) == 2 and resp[0] == 0x05
        except Exception:
            return False

    @staticmethod
    def _probe_http_proxy(host: str, port: int, timeout: float = 0.8) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                req = (
                    b"CONNECT example.com:443 HTTP/1.1\r\n"
                    b"Host: example.com:443\r\n"
                    b"Proxy-Connection: Keep-Alive\r\n\r\n"
                )
                sock.sendall(req)
                resp = sock.recv(32)
                return resp.startswith(b"HTTP/")
        except Exception:
            return False

    @staticmethod
    def _parse_vless(link: str) -> ProxyNode:
        u = urlparse(link)
        q = parse_qs(u.query)
        transport = (q.get("type", [""])[0] or "").strip()
        security = (q.get("security", [""])[0] or "").strip()
        host = u.hostname or ""
        port = int(u.port or 0)
        if not host or not port:
            raise ValueError("invalid vless link")
        summary = f"vless://{host}:{port} security={security} transport={transport}"
        return ProxyNode(
            scheme="vless",
            host=host,
            port=port,
            summary=summary,
            transport=transport,
            security=security,
        )

    @staticmethod
    def _parse_trojan(link: str) -> ProxyNode:
        u = urlparse(link)
        q = parse_qs(u.query)
        security = (q.get("security", ["tls"])[0] or "").strip()
        transport = (q.get("type", ["tcp"])[0] or "").strip()
        host = u.hostname or ""
        port = int(u.port or 0)
        if not host or not port:
            raise ValueError("invalid trojan link")
        summary = f"trojan://{host}:{port} security={security} transport={transport}"
        return ProxyNode(
            scheme="trojan",
            host=host,
            port=port,
            summary=summary,
            transport=transport,
            security=security,
        )

    @staticmethod
    def _parse_vmess(link: str) -> ProxyNode:
        raw = link[len("vmess://") :].strip()
        # support url-safe and missing padding base64
        padded = raw + "=" * ((4 - len(raw) % 4) % 4)
        try:
            payload = json.loads(base64.b64decode(padded).decode("utf-8", errors="ignore"))
        except Exception as exc:
            raise ValueError("invalid vmess link") from exc
        host = (payload.get("add") or "").strip()
        port = int(str(payload.get("port") or "0"))
        transport = (payload.get("net") or "").strip()
        security = (payload.get("tls") or "").strip()
        if not host or not port:
            raise ValueError("invalid vmess link payload")
        summary = f"vmess://{host}:{port} security={security} transport={transport}"
        return ProxyNode(
            scheme="vmess",
            host=host,
            port=port,
            summary=summary,
            transport=transport,
            security=security,
        )

    @staticmethod
    def _parse_ss(link: str) -> ProxyNode:
        # format may be:
        # ss://base64(method:password@host:port)#tag
        # ss://method:password@host:port#tag
        body = link[len("ss://") :]
        body = body.split("#", 1)[0]
        if "@" not in body:
            padded = body + "=" * ((4 - len(body) % 4) % 4)
            body = base64.b64decode(padded).decode("utf-8", errors="ignore")
        if "@" not in body:
            raise ValueError("invalid ss link")
        auth, endpoint = body.split("@", 1)
        endpoint = unquote(endpoint)
        if ":" not in endpoint:
            raise ValueError("invalid ss endpoint")
        host, port_text = endpoint.rsplit(":", 1)
        port = int(port_text or "0")
        if not host or not port:
            raise ValueError("invalid ss host/port")
        method = auth.split(":", 1)[0] if ":" in auth else ""
        summary = f"ss://{host}:{port} method={method}"
        return ProxyNode(
            scheme="ss",
            host=host,
            port=port,
            summary=summary,
            transport="",
            security=method,
        )
