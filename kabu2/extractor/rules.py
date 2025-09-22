from __future__ import annotations

import csv
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from kabu2.models import NewsItem


@dataclass
class Extracted:
    tags: List[str]
    reasons: List[str]
    company_name: Optional[str]
    ticker: Optional[str]


def load_name_map(csv_path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip()
                aliases = str(row.get("aliases", "")).strip()
                if not code or not name:
                    continue
                mp[name] = code
                if aliases:
                    for a in re.split(r"[;,]", aliases):
                        a = a.strip()
                        if a:
                            mp[a] = code
    except FileNotFoundError:
        pass
    return mp


def guess_company(text: str, name_map: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    text = text or ""
    # simple substring matching (exact/alias)
    for alias, code in name_map.items():
        if alias and alias in text:
            return alias, code

    code_match = CODE_PATTERN.search(text)
    if code_match:
        code = code_match.group(1)
        for alias, mapped in name_map.items():
            if mapped == code and len(alias) >= 2:
                return alias, code
        return None, code

    candidates = [name for name in name_map.keys() if name]
    if not candidates:
        return None, None

    fuzzy_hit = process.extractOne(text, candidates, scorer=fuzz.partial_ratio, score_cutoff=85)
    if fuzzy_hit:
        alias = fuzzy_hit[0]
        return alias, name_map.get(alias)
    return None, None


PATTERNS: Dict[str, re.Pattern[str]] = {
    "upgrade": re.compile(r"(上方修正|増額修正)"),
    "buyback": re.compile(r"(自己株式取得|自社株買い)"),
    "dividend_increase": re.compile(r"配当.*(増額|上方)"),
    "split": re.compile(r"株式分割"),
    "big_order": re.compile(r"(大口受注|大型受注|契約締結|基本合意)"),
    "partner": re.compile(r"(提携|パートナーシップ|採用)"),
    "approval_patent": re.compile(r"(承認|薬事|特許|認可)"),
    "kpi": re.compile(r"(会員|ARR|GMV|販売台数|出荷|月次|四半期).*(増|上回|好調)"),
    "negative_offering": re.compile(r"(公募|第三者割当|CB|社債|新株予約権).*(発行|実施)"),
    "lawsuit": re.compile(r"(訴訟|係争)"),
    "deficit_widen": re.compile(r"(赤字|損失).*(拡大)"),
}


JST = timezone(timedelta(hours=9))
TIER1_PARTNERS = re.compile(r"(トヨタ|NTT|任天堂|ソニー|ソフトバンク|日立|パナソニック)")
CODE_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def _is_intraday(dt: Optional[datetime]) -> bool:
    if not isinstance(dt, datetime):
        return False
    try:
        aware = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        local = aware.astimezone(JST)
    except Exception:
        return False
    if local.weekday() >= 5:
        return False
    return 9 <= local.hour < 15


def extract(item: NewsItem, name_map: Dict[str, str]) -> Extracted:
    text = f"{item.title} {item.summary or ''}"
    tags: List[str] = []
    reasons: List[str] = []

    if PATTERNS["upgrade"].search(text):
        tags.append("upgrade")
        reasons.append("上方修正")

    if PATTERNS["buyback"].search(text):
        m = re.search(r"(\d+(?:\.\d+)?)%", text)
        if m and float(m.group(1)) >= 5:
            tags.append("buyback_large")
            reasons.append(f"自社株買い {m.group(1)}%")
        else:
            tags.append("buyback_small")
            reasons.append("自社株買い")

    if PATTERNS["dividend_increase"].search(text):
        tags.append("dividend_increase")
        reasons.append("配当増額")

    if PATTERNS["split"].search(text):
        tags.append("split")
        reasons.append("株式分割")

    if PATTERNS["big_order"].search(text):
        tags.append("big_order")
        reasons.append("大型受注/契約")

    if PATTERNS["partner"].search(text):
        if TIER1_PARTNERS.search(text):
            tags.append("partner_tier1")
            reasons.append("大手との提携/採用")
        else:
            tags.append("partner")
            reasons.append("提携/採用")

    if PATTERNS["approval_patent"].search(text):
        tags.append("approval_patent")
        reasons.append("承認/特許")

    if PATTERNS["kpi"].search(text):
        if re.search(r"(過去最高|記録|大幅|二桁|前年比\s*\d+%増|\d+%増)", text):
            tags.append("kpi_surprise_high")
            reasons.append("KPIサプライズ(大)")
        else:
            tags.append("kpi_surprise_low")
            reasons.append("KPIサプライズ")

    if PATTERNS["negative_offering"].search(text):
        tags.append("negative_offering")
        reasons.append("公募増資/CB等")

    if PATTERNS["lawsuit"].search(text):
        tags.append("lawsuit")
        reasons.append("訴訟")

    if PATTERNS["deficit_widen"].search(text):
        tags.append("deficit_widen")
        reasons.append("赤字拡大")

    if _is_intraday(item.published_at):
        tags.append("intraday_bonus")
        reasons.append("場中発表")

    # company / ticker guess
    name, code = guess_company(text, name_map)

    return Extracted(tags=tags, reasons=reasons, company_name=name or item.company_name, ticker=code or item.ticker)
