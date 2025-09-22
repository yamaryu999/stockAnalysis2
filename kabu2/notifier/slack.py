from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from kabu2.models import ScoredItem


logger = logging.getLogger(__name__)

HISTORY_PATH = Path.home() / ".cache" / "kabu2" / "slack_history.json"
HISTORY_RETENTION_HOURS = 168  # keep 1 week of notification history


@dataclass
class SlackResult:
    status: Optional[int]
    delivered: int
    skipped: int
    payload: Dict[str, object]
    dry_run: bool


def _load_history() -> Dict[str, str]:
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _save_history(history: Dict[str, str]) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.debug("failed to persist slack history", exc_info=True)


def _parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _filter_items(
    items: Sequence[ScoredItem], min_score: int, cooldown_minutes: int
) -> Tuple[List[ScoredItem], Dict[str, str], int]:
    now = datetime.now(timezone.utc)
    history = _load_history()
    eligible: List[ScoredItem] = []
    skipped = 0
    cooldown_delta = timedelta(minutes=max(cooldown_minutes, 0)) if cooldown_minutes else timedelta(0)
    cutoff = now - timedelta(hours=HISTORY_RETENTION_HOURS)

    for item in items:
        if item.score < min_score:
            skipped += 1
            continue
        last_ts = _parse_ts(history.get(item.news.id))
        if last_ts:
            if last_ts < cutoff:
                history.pop(item.news.id, None)
            elif cooldown_delta and now - last_ts < cooldown_delta:
                skipped += 1
                continue
        eligible.append(item)
    return eligible, history, skipped


def _format_timestamp(dt: datetime | None) -> str:
    if not isinstance(dt, datetime):
        return "-"
    try:
        local = dt.astimezone(timezone(timedelta(hours=9)))
    except Exception:
        return dt.isoformat(timespec="minutes")
    return local.strftime("%m/%d %H:%M")


def build_payload(items: Sequence[ScoredItem]) -> Dict[str, object]:
    if not items:
        return {"text": "No signals"}

    blocks: List[Dict[str, object]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "kabu2 シグナル"},
        }
    ]

    for s in items:
        n = s.news
        score_line = f"[{s.score}] {n.ticker or '-'} {n.company_name or '-'}"
        reasons = " / ".join(s.reasons) if s.reasons else "-"
        hold = "スイング" if s.hold == "swing" else "デイ"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{score_line}*\n<{n.link}|{n.title}>\n{reasons}",
                },
                "fields": [
                    {"type": "mrkdwn", "text": f"*ソース*\n{n.source}"},
                    {"type": "mrkdwn", "text": f"*想定ホールド*\n{hold}"},
                    {"type": "mrkdwn", "text": f"*発表*\n{_format_timestamp(n.published_at)}"},
                ],
            }
        )
        blocks.append({"type": "divider"})

    text_fallback = "\n".join(
        f"[{s.score}] {s.news.ticker or '-'} {s.news.company_name or '-'} | {s.news.title} -> {s.news.link}" for s in items
    )
    if blocks and blocks[-1].get("type") == "divider":
        blocks = blocks[:-1]
    return {"text": text_fallback, "blocks": blocks}


def post_slack(
    webhook: str,
    items: Iterable[ScoredItem],
    *,
    min_score: int = 0,
    cooldown_minutes: int = 0,
    dry_run: bool = False,
) -> SlackResult:
    candidates = list(items)
    filtered, history, skipped_below = _filter_items(candidates, min_score, cooldown_minutes)
    payload = build_payload(filtered)

    if dry_run:
        return SlackResult(
            status=None,
            delivered=len(filtered),
            skipped=skipped_below,
            payload=payload,
            dry_run=True,
        )

    if not filtered:
        return SlackResult(
            status=None,
            delivered=0,
            skipped=skipped_below,
            payload=payload,
            dry_run=False,
        )

    try:
        resp = requests.post(
            webhook,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        status = resp.status_code
        if status < 400:
            now = datetime.now(timezone.utc)
            for item in filtered:
                history[item.news.id] = now.isoformat()
            cutoff = now - timedelta(hours=HISTORY_RETENTION_HOURS)
            for key, value in list(history.items()):
                ts = _parse_ts(value)
                if ts and ts < cutoff:
                    history.pop(key, None)
            _save_history(history)
        else:
            logger.warning("Slack webhook returned status %s", status)
        return SlackResult(status=status, delivered=len(filtered), skipped=skipped_below, payload=payload, dry_run=False)
    except Exception as exc:
        logger.error("failed to post to Slack: %s", exc)
        return SlackResult(status=None, delivered=0, skipped=len(candidates), payload=payload, dry_run=False)
