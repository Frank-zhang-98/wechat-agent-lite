from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.services.settings_service import SettingsService


class MailService:
    def __init__(self, settings: SettingsService):
        self.settings = settings

    def is_enabled(self) -> bool:
        to_addresses = self.settings.get("mail.to_addresses", "").strip()
        daily_enabled = self.settings.get_bool("mail.daily_enabled", True)
        return daily_enabled and bool(to_addresses)

    def send_daily(self, subject: str, html_body: str) -> dict:
        if not self.settings.get_bool("mail.daily_enabled", True):
            return {"sent": False, "skipped": True, "reason": "日报邮件未启用"}
        to_addresses_raw = self.settings.get("mail.to_addresses", "").strip()
        if not to_addresses_raw:
            return {"sent": False, "skipped": True, "reason": "收件邮箱为空，默认不发送邮件"}
        return self._send(subject=subject, html_body=html_body, to_addresses_raw=to_addresses_raw)

    def send_test(self, subject: str, html_body: str) -> dict:
        to_addresses_raw = self.settings.get("mail.to_addresses", "").strip()
        if not to_addresses_raw:
            return {"sent": False, "reason": "收件邮箱为空，请先填写 To Address(es)"}
        return self._send(subject=subject, html_body=html_body, to_addresses_raw=to_addresses_raw)

    def _send(self, subject: str, html_body: str, to_addresses_raw: str) -> dict:
        host = self.settings.get("mail.smtp_host", "").strip()
        user = self.settings.get("mail.smtp_user", "").strip()
        password = self.settings.get("mail.smtp_password", "")
        from_addr = self.settings.get("mail.from_address", user).strip()
        port = self.settings.get_int("mail.smtp_port", 587)
        use_tls = self.settings.get_bool("mail.smtp_tls", True)
        if not host or not user or not password:
            return {"sent": False, "reason": "SMTP 配置不完整"}

        to_list = [x.strip() for x in to_addresses_raw.split(",") if x.strip()]
        if not to_list:
            return {"sent": False, "reason": "收件邮箱为空，请先填写 To Address(es)"}

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_list)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        smtp_cls = smtplib.SMTP_SSL if self._use_implicit_ssl(port=port, use_tls=use_tls) else smtplib.SMTP
        with smtp_cls(host=host, port=port, timeout=20) as server:
            server.ehlo()
            if use_tls and smtp_cls is smtplib.SMTP:
                server.starttls()
                server.ehlo()
            server.login(user, password)
            server.sendmail(from_addr, to_list, msg.as_string())
        return {"sent": True, "count": len(to_list)}

    @staticmethod
    def _use_implicit_ssl(*, port: int, use_tls: bool) -> bool:
        return use_tls and port == 465
