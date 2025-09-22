from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from kabu2.models import NewsItem, ScoredItem
from kabu2.extractor.rules import Extracted


def score_from_tags(tags: List[str], weights: Dict[str, int]) -> int:
    score = 0
    for t in tags:
        score += int(weights.get(t, 0))
    return score


def hold_horizon(tags: List[str], combo_hits: List[str]) -> str:
    # rough mapping: strong structural events -> swing
    swing_tags = {"buyback_large", "approval_patent", "upgrade", "kpi_surprise_high"}
    if any(t in swing_tags for t in tags):
        return "swing"
    if combo_hits:
        return "swing"
    return "day"


COMBO_RULES: Tuple[Tuple[set[str], int, str], ...] = (
    ({"upgrade", "kpi_surprise_high"}, 2, "決算×KPIコンボ"),
    ({"buyback_large", "dividend_increase"}, 1, "株主還元コンボ"),
    ({"partner_tier1", "big_order"}, 1, "大手提携×大型案件"),
)


def combo_bonus(tags: List[str]) -> Tuple[int, List[str]]:
    hits: List[str] = []
    total = 0
    tag_set = set(tags)
    for required, bonus, label in COMBO_RULES:
        if required.issubset(tag_set):
            total += bonus
            hits.append(label)
    return total, hits


def freshness_adjustment(published_at: datetime | None) -> Tuple[int, Optional[str]]:
    if not isinstance(published_at, datetime):
        return 0, None
    try:
        aware = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff_hours = (now - aware).total_seconds() / 3600.0
    except Exception:
        return 0, None

    if diff_hours < 0:
        return 1, "発表前速報"
    if diff_hours <= 3:
        return 2, "発表直後"
    if diff_hours <= 6:
        return 1, "当日フレッシュ"
    if diff_hours <= 24:
        return 0, None
    if diff_hours <= 48:
        return -1, "鮮度低下"
    return -2, "鮮度大幅低下"


def build_scored(item: NewsItem, ext: Extracted, weights: Dict[str, int]) -> ScoredItem:
    base_score = score_from_tags(ext.tags, weights)
    combo_score, combo_reasons = combo_bonus(ext.tags)
    freshness_score, freshness_reason = freshness_adjustment(item.published_at)
    score = base_score + combo_score + freshness_score

    reasons = list(ext.reasons)
    reasons.extend(combo_reasons)
    if freshness_reason:
        reasons.append(freshness_reason)

    hold = hold_horizon(ext.tags, combo_reasons)
    # propagate company/ticker to news
    item.company_name = ext.company_name or item.company_name
    item.ticker = ext.ticker or item.ticker
    return ScoredItem(news=item, score=score, reasons=reasons, hold=hold, tags=list(ext.tags))
