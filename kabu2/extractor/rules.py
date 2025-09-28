from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html import unescape
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from kabu2.models import NewsItem


@dataclass
class Extracted:
    tags: List[str]
    reasons: List[str]
    company_name: Optional[str]
    ticker: Optional[str]


_CORP_TOKEN_PATTERN = re.compile(
    r"(株式会社|有限会社|（株）|\(株\)|㈱|（有）|\(有\)|合同会社|ホールディングス|グループ|"
    r"Co\.?\s*,?\s*Ltd\.?|Inc\.?|Corporation|Corp\.?|Company|Holdings|HD\b|ＨＤ\b)",
    re.IGNORECASE,
)


def _normalize_company_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = _CORP_TOKEN_PATTERN.sub("", text)
    normalized = re.sub(r"[\s\u3000・、,，．。/／\\\\\-‐―ー()（）\[\]{}<>『』「」【】＆&]+", "", normalized)
    return normalized.casefold()


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
    normalized_text = _normalize_company_text(text)
    # simple substring matching (exact/alias)
    for alias, code in name_map.items():
        if alias and alias in text:
            return alias, code
        simplified_alias = _normalize_company_text(alias)
        if simplified_alias and simplified_alias in normalized_text:
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
    "downside": re.compile(r"(下方修正|減額修正|通期.*未達|業績予想.*下方)"),
    "buyback": re.compile(r"(自己株式取得|自社株買い)"),
    "dividend_increase": re.compile(r"配当.*(増額|上方)"),
    "dividend_cut": re.compile(r"配当.*(減額|減配|無配|中止)"),
    "split": re.compile(r"株式分割"),
    "big_order": re.compile(r"(大口受注|大型受注|契約締結|基本合意)"),
    "partner": re.compile(r"(提携|パートナーシップ|採用)"),
    "approval_patent": re.compile(r"(承認|薬事|特許|認可)"),
    "kpi": re.compile(r"(会員|ARR|GMV|販売台数|出荷|月次|四半期).*(増|上回|好調)"),
    "negative_offering": re.compile(r"(公募|第三者割当|CB|社債|新株予約権).*(発行|実施)"),
    "lawsuit": re.compile(r"(訴訟|係争)"),
    "deficit_widen": re.compile(r"(赤字|損失).*(拡大)"),
    "delay": re.compile(r"(決算|発表|提出).*(延期|遅延|後ろ倒し)"),
    "cyber": re.compile(r"(情報漏えい|個人情報.*流出|不正アクセス|サイバー攻撃)"),
    "incident": re.compile(r"(火災|爆発|操業停止|生産停止).*(工場|プラント|設備|ライン)"),
}


JST = timezone(timedelta(hours=9))
TIER1_PARTNERS = re.compile(r"(トヨタ|NTT|任天堂|ソニー|ソフトバンク|日立|パナソニック)")
CODE_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


try:  # pragma: no cover - optional dependency
    import jpholiday  # type: ignore
except Exception:  # pragma: no cover
    jpholiday = None  # type: ignore


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u3000\xa0]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    if jpholiday and jpholiday.is_holiday(local.date()):
        return False
    if (local.month, local.day) in {(12, 31), (1, 1)}:
        return False
    return 9 <= local.hour < 15


def extract(item: NewsItem, name_map: Dict[str, str]) -> Extracted:
    title = _strip_html(item.title)
    summary = _strip_html(item.summary)
    raw_entry: Dict[str, Any] = {}
    if isinstance(item.raw, dict):
        entry = item.raw.get("entry")
        if isinstance(entry, dict):
            raw_entry = entry

    company_candidates: List[str] = []

    def _add_candidate(value: Any) -> None:
        if value is None:
            return
        text_value = str(value).strip()
        if not text_value:
            return
        if text_value not in company_candidates:
            company_candidates.append(text_value)

    if raw_entry:
        raw_keys = (
            "company",
            "dc_corp",
            "dc:corp",
            "dc_creator",
            "dc:creator",
            "dc_source",
            "dc:source",
            "dc_publisher",
            "dc:publisher",
            "author",
            "authors",
            "source",
        )
        for key in raw_keys:
            value = raw_entry.get(key)
            if isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, dict):
                        _add_candidate(v.get("name") or v.get("value"))
                    else:
                        _add_candidate(v)
            elif isinstance(value, dict):
                _add_candidate(value.get("name") or value.get("value"))
            else:
                _add_candidate(value)

    extra_text = " ".join(company_candidates)
    text = " ".join(part for part in (title, summary, extra_text) if part).strip()
    tags: List[str] = []
    reasons: List[str] = []

    if PATTERNS["upgrade"].search(text):
        tags.append("upgrade")
        reasons.append("上方修正")

    if PATTERNS["downside"].search(text):
        tags.append("downside")
        reasons.append("下方修正/予想下振れ")

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

    if PATTERNS["dividend_cut"].search(text):
        tags.append("dividend_cut")
        reasons.append("減配/無配")

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

    if PATTERNS["delay"].search(text):
        tags.append("delay")
        reasons.append("開示/決算の延期")

    if PATTERNS["cyber"].search(text):
        tags.append("cyber")
        reasons.append("情報漏えい/サイバー")

    if PATTERNS["incident"].search(text):
        tags.append("incident")
        reasons.append("事故/火災/操業停止")

    if _is_intraday(item.published_at):
        tags.append("intraday_bonus")
        reasons.append("場中発表")

    # company / ticker guess
    name, code = guess_company(text, name_map)
    display_name = name

    if company_candidates:
        for candidate in company_candidates:
            if not code:
                _, cand_code = guess_company(candidate, name_map)
                if cand_code:
                    code = cand_code
            if not display_name:
                display_name = candidate
            elif _normalize_company_text(display_name) == _normalize_company_text(candidate):
                display_name = candidate
                break
    if not display_name:
        display_name = item.company_name

    return Extracted(tags=tags, reasons=reasons, company_name=display_name or item.company_name, ticker=code or item.ticker)
