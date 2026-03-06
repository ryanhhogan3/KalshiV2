"""
Email notification system for PredictionShift alerts.

Uses AWS SES (already in requirements.txt via boto3).
The EC2 instance must have an IAM role with ses:SendEmail permission,
or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars.

Required env vars:
    ALERT_FROM_EMAIL    — verified SES sender (e.g. alerts@predictionshift.com)
    AWS_REGION          — SES region (default: us-east-1)

Optional:
    ALERT_DRY_RUN=1     — log emails instead of sending (for testing)
"""

import logging
import os
from typing import Dict, Any, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_SES_CLIENT = None


def _get_ses_client():
    global _SES_CLIENT
    if _SES_CLIENT is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        _SES_CLIENT = boto3.client("ses", region_name=region)
    return _SES_CLIENT


def _format_price(value, platform: str) -> str:
    """Format price for display: Kalshi uses integer ticks, Poly uses 0–1 decimal."""
    if value is None:
        return "N/A"
    if platform == "kalshi":
        return f"{int(value)}¢"
    else:
        return f"{float(value) * 100:.1f}%"


def _format_move(value, platform: str) -> str:
    """Format a move value with sign."""
    if value is None:
        return "N/A"
    if platform == "kalshi":
        sign = "+" if float(value) >= 0 else ""
        return f"{sign}{int(value)} ticks"
    else:
        sign = "+" if float(value) >= 0 else ""
        return f"{sign}{float(value) * 100:.1f}pp"


def _build_subject(alert: Dict[str, Any]) -> str:
    """Build the email subject line."""
    title = alert.get("title") or alert.get("market_key", "Unknown")
    # Truncate long titles
    if len(title) > 60:
        title = title[:57] + "..."
    return f"PredictionShift Alert: {title}"


def _build_body(alert: Dict[str, Any]) -> str:
    """Build the plain-text email body."""
    platform = alert.get("platform", "kalshi")
    platform_label = "Kalshi" if platform == "kalshi" else "Polymarket"
    title = alert.get("title") or alert.get("market_key", "Unknown")
    alert_type = alert.get("alert_type", "PRICE_MOVE")

    price_now = _format_price(alert.get("price_now"), platform)
    price_prev = _format_price(alert.get("price_prev"), platform)
    move = _format_move(alert.get("move"), platform)

    body = f"""
{title}

{alert_type.replace("_", " ").title()} Alert Triggered

Platform:    {platform_label}
Market:      {alert.get("market_key", "N/A")}
Move:        {move}
Current:     {price_now}
Previous:    {price_prev}
Threshold:   {alert.get("threshold", "N/A")}
Actual:      {alert.get("actual_value", "N/A")}

---
PredictionShift — predictionshift.com
To stop this alert, deactivate it via the API.
""".strip()

    return body


def send_alert_email(alert: Dict[str, Any]) -> bool:
    """Send a single alert email via AWS SES.

    Returns True if sent successfully, False otherwise.
    Logs errors but does not raise.
    """
    from_email = os.environ.get("ALERT_FROM_EMAIL", "")
    dry_run = os.environ.get("ALERT_DRY_RUN", "0") == "1"

    if not from_email and not dry_run:
        logger.error("ALERT_FROM_EMAIL not set — cannot send email")
        return False

    to_email = alert.get("user_email")
    if not to_email:
        logger.error("No user_email in alert — skipping")
        return False

    subject = _build_subject(alert)
    body = _build_body(alert)

    if dry_run:
        logger.info("DRY RUN — would send to %s:\nSubject: %s\n%s", to_email, subject, body)
        return True

    try:
        ses = _get_ses_client()
        ses.send_email(
            Source=from_email,
            Destination={"ToAddresses": [to_email]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": body, "Charset": "UTF-8"},
                },
            },
        )
        logger.info("Email sent to %s for market %s", to_email, alert.get("market_key"))
        return True

    except ClientError as e:
        logger.error("SES send failed for %s: %s", to_email, e)
        return False
    except Exception as e:
        logger.error("Unexpected email error for %s: %s", to_email, e)
        return False


def send_triggered_alerts(triggered: List[Dict[str, Any]], cur) -> int:
    """Send emails for all triggered alerts and update history.

    Returns the number of successfully sent emails.
    """
    from src.data.alerts import mark_alert_sent

    sent_count = 0
    for alert in triggered:
        history_id = alert.get("history_id")
        success = send_alert_email(alert)

        if history_id:
            mark_alert_sent(cur, history_id, success,
                            None if success else "send failed")

        if success:
            sent_count += 1

    logger.info("Sent %d / %d alert emails", sent_count, len(triggered))
    return sent_count
