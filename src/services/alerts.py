"""
HAAM Alert Service
==================
Sends email and/or Slack alerts when agent risk exceeds threshold.
Config is read from config/alerts.json (created with defaults if missing).
"""

import json
import logging
import smtplib
import urllib.request
import urllib.error
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "alerts.json"
_DEFAULT_CONFIG = {
    "email":  {"enabled": False, "smtp_host": "smtp.gmail.com", "smtp_port": 587,
               "sender": "", "password": "", "recipients": []},
    "slack":  {"enabled": False, "webhook_url": ""},
    "risk_threshold": 0.6,
}


class AlertService:
    def __init__(self):
        self.config = self._load_config()

    # ── Config I/O ─────────────────────────────────────────────────────────────
    def _load_config(self) -> dict:
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load alerts.json: {e}")
        return dict(_DEFAULT_CONFIG)

    def save_config(self, new_config: dict):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(new_config, f, indent=2)
        self.config = new_config
        logger.info("Alert config saved.")

    def get_config(self) -> dict:
        self.config = self._load_config()
        # Never return the password in plaintext to the API
        safe = json.loads(json.dumps(self.config))
        if safe.get("email", {}).get("password"):
            safe["email"]["password"] = "••••••••"
        return safe

    # ── Core Alert Dispatch ────────────────────────────────────────────────────
    def check_and_alert(self, agent_id: str, risk_score: float, summary: dict):
        """
        Called after every call. If risk >= threshold, fires alerts.
        summary: dict with keys: dominant_emotion, call_id, transcript_excerpt, etc.
        """
        threshold = float(self.config.get("risk_threshold", 0.6))
        if risk_score < threshold:
            return

        logger.info(f"⚠️ Risk threshold triggered for agent {agent_id}: {risk_score:.2f}")
        subject = f"⚠️ HAAM Alert — High Risk Detected for Agent {agent_id}"
        body    = self._format_body(agent_id, risk_score, summary)

        if self.config.get("email", {}).get("enabled"):
            try:
                self.send_email(subject, body)
            except Exception as e:
                logger.error(f"Email alert failed: {e}")

        if self.config.get("slack", {}).get("enabled"):
            try:
                self.send_slack(subject, body, risk_score)
            except Exception as e:
                logger.error(f"Slack alert failed: {e}")

    def _format_body(self, agent_id, risk_score, summary) -> str:
        emotion    = summary.get("dominant_emotion", "N/A")
        call_id    = summary.get("call_id", "N/A")
        excerpt    = summary.get("transcript_excerpt", "")
        confidence = summary.get("confidence", 0)
        return (
            f"HAAM Risk Alert\n"
            f"{'─'*40}\n"
            f"Agent:           {agent_id}\n"
            f"Risk Score:      {risk_score * 100:.1f}%\n"
            f"Call ID:         {call_id}\n"
            f"Dominant Emotion:{emotion}\n"
            f"Confidence:      {confidence * 100:.1f}%\n"
            f"\nTranscript:\n{excerpt[:300]}\n"
            f"{'─'*40}\n"
            f"Review in the HAAM Dashboard for details."
        )

    # ── Email ──────────────────────────────────────────────────────────────────
    def send_email(self, subject: str, body: str):
        cfg = self.config.get("email", {})
        if not cfg.get("sender") or not cfg.get("recipients"):
            raise ValueError("Email sender/recipients not configured.")

        msg = MIMEMultipart()
        msg["From"]    = cfg["sender"]
        msg["To"]      = ", ".join(cfg["recipients"])
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["sender"], cfg["password"])
            server.sendmail(cfg["sender"], cfg["recipients"], msg.as_string())
        logger.info(f"Email sent to {cfg['recipients']}")

    # ── Slack ──────────────────────────────────────────────────────────────────
    def send_slack(self, subject: str, body: str, risk_score: float):
        cfg = self.config.get("slack", {})
        webhook = cfg.get("webhook_url", "")
        if not webhook:
            raise ValueError("Slack webhook_url not configured.")

        pct   = round(risk_score * 100, 1)
        color = "#ff4444" if risk_score >= 0.7 else "#ff8800"
        payload = json.dumps({
            "text": subject,
            "attachments": [{
                "color":  color,
                "text":   body,
                "footer": f"Risk Score: {pct}% | HAAM Hybrid Model",
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            webhook, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Slack returned HTTP {resp.status}")
        logger.info("Slack alert sent.")


# Module-level singleton
alert_service = AlertService()
