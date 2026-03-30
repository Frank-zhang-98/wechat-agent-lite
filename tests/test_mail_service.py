import unittest
from unittest.mock import patch

from app.services.mail_service import MailService


class DummySettings:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)

    def get_bool(self, key: str, default: bool) -> bool:
        raw = self.values.get(key, "true" if default else "false")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def get_int(self, key: str, default: int) -> int:
        return int(self.values.get(key, default))


class FakeSMTPBase:
    last_instance = None

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.logged_in = False
        self.sent = False
        type(self).last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def ehlo(self):
        return None

    def starttls(self):
        self.started_tls = True

    def login(self, user: str, password: str):
        self.logged_in = True

    def sendmail(self, from_addr: str, to_list: list[str], message: str):
        self.sent = True


class FakeSMTP(FakeSMTPBase):
    pass


class FakeSMTPSSL(FakeSMTPBase):
    pass


class MailServiceTests(unittest.TestCase):
    def test_send_uses_smtp_ssl_on_port_465(self) -> None:
        service = MailService(
            DummySettings(
                {
                    "mail.smtp_host": "smtp.yeah.net",
                    "mail.smtp_port": "465",
                    "mail.smtp_tls": "true",
                    "mail.smtp_user": "demo@yeah.net",
                    "mail.smtp_password": "secret",
                    "mail.from_address": "demo@yeah.net",
                    "mail.to_addresses": "user@example.com",
                }
            )
        )

        with patch("app.services.mail_service.smtplib.SMTP", FakeSMTP):
            with patch("app.services.mail_service.smtplib.SMTP_SSL", FakeSMTPSSL):
                result = service.send_test("subject", "<p>hello</p>")

        self.assertTrue(result["sent"])
        self.assertIsNotNone(FakeSMTPSSL.last_instance)
        self.assertTrue(FakeSMTPSSL.last_instance.logged_in)
        self.assertTrue(FakeSMTPSSL.last_instance.sent)
        self.assertFalse(FakeSMTPSSL.last_instance.started_tls)

    def test_send_uses_starttls_on_port_587(self) -> None:
        service = MailService(
            DummySettings(
                {
                    "mail.smtp_host": "smtp.example.com",
                    "mail.smtp_port": "587",
                    "mail.smtp_tls": "true",
                    "mail.smtp_user": "demo@example.com",
                    "mail.smtp_password": "secret",
                    "mail.from_address": "demo@example.com",
                    "mail.to_addresses": "user@example.com",
                }
            )
        )

        with patch("app.services.mail_service.smtplib.SMTP", FakeSMTP):
            with patch("app.services.mail_service.smtplib.SMTP_SSL", FakeSMTPSSL):
                result = service.send_daily("subject", "<p>hello</p>")

        self.assertTrue(result["sent"])
        self.assertIsNotNone(FakeSMTP.last_instance)
        self.assertTrue(FakeSMTP.last_instance.started_tls)
        self.assertTrue(FakeSMTP.last_instance.logged_in)
        self.assertTrue(FakeSMTP.last_instance.sent)


if __name__ == "__main__":
    unittest.main()
