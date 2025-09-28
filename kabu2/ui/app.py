from __future__ import annotations

import json
import time
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Optional
from collections import Counter
from html import escape

import pandas as pd
import streamlit as st
from yaml import YAMLError

# Ensure repo root on sys.path when running via `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kabu2.collector import collect_from_config, parse_feed
from kabu2.config import load_config, save_config
from kabu2.extractor.rules import extract, load_name_map
from kabu2.models import NewsItem, ScoredItem
from kabu2.scorer.score import build_scored
from kabu2.notifier.slack import post_slack


@st.cache_data(show_spinner=False)
def load_jsonl(path: str) -> List[NewsItem]:
    items: List[NewsItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(NewsItem.from_dict(data))
    return items


def collect_selected(cfg: Dict[str, Any], selected: List[str]) -> List[NewsItem]:
    if not selected:
        return []

    feeds_cfg = cfg.get("feeds", {}) or {}
    collectors_cfg = cfg.get("collectors", []) or []

    selected_set = {key for key in selected if key}
    feed_subset = {key: feeds_cfg[key] for key in selected_set if key in feeds_cfg and feeds_cfg[key]}

    collector_subset: List[Dict[str, Any]] = []
    for spec in collectors_cfg:
        if not isinstance(spec, Mapping):
            continue
        name = spec.get("name")
        if not name:
            continue
        name_str = str(name)
        if name_str not in selected_set:
            continue
        collector_subset.append(deepcopy(dict(spec)))

    if not feed_subset and not collector_subset:
        return []

    sub_cfg: Dict[str, Any] = {}
    if feed_subset:
        sub_cfg["feeds"] = feed_subset
    if collector_subset:
        sub_cfg["collectors"] = collector_subset

    return collect_from_config(sub_cfg)


def normalize_config(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    base = deepcopy(cfg) if cfg else {}
    base.setdefault("weights", {})
    thresholds = base.setdefault("thresholds", {})
    thresholds.setdefault("min_score", 0)
    thresholds.setdefault("top_k", 10)
    base.setdefault("feeds", {})
    base.setdefault("collectors", [])
    base.setdefault("feeds_priority", {})
    feeds_ai = base.setdefault("feeds_ai", {})
    feeds_ai.setdefault("max_keys", 3)
    feeds_ai.setdefault("fresh_hours", 12)
    feeds_ai.setdefault("stale_penalty", 5.0)
    feeds_ai.setdefault("repeat_penalty", 2.0)
    feeds_ai.setdefault("history_weight", 0.5)
    feeds_ai.setdefault("cooldown_hours", 4)
    storage = base.setdefault("storage", {})
    storage.setdefault("path", "data/news.jsonl")
    notifier = base.setdefault("notifier", {})
    slack = notifier.setdefault("slack", {})
    slack.setdefault("enabled", False)
    slack.setdefault("webhook_env", "SLACK_WEBHOOK_URL")
    slack.setdefault("min_score", thresholds.get("min_score", 0))
    slack.setdefault("cooldown_min", 0)
    return base


def weights_to_df(weights: Dict[str, Any]) -> pd.DataFrame:
    rows = [{"タグ": key, "スコア": value} for key, value in weights.items()]
    if not rows:
        rows = [{"タグ": "", "スコア": 0}]
    return pd.DataFrame(rows)


def df_to_weights(df: pd.DataFrame) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in df.to_dict(orient="records"):
        raw_key = row.get("タグ")
        if raw_key is None:
            continue
        key = str(raw_key).strip()
        if not key or key.lower() == "nan":
            continue
        raw_score = row.get("スコア")
        try:
            score = int(float(raw_score))
        except (TypeError, ValueError):
            continue
        result[key] = score
    return result


def feeds_to_df(feeds: Dict[str, str]) -> pd.DataFrame:
    rows = [{"キー": key, "URL": value} for key, value in feeds.items()]
    if not rows:
        rows = [{"キー": "", "URL": ""}]
    return pd.DataFrame(rows)


def df_to_feeds(df: pd.DataFrame) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for row in df.to_dict(orient="records"):
        raw_key = row.get("キー")
        raw_url = row.get("URL")
        if raw_key is None or raw_url is None:
            continue
        key = str(raw_key).strip()
        url = str(raw_url).strip()
        if not key or key.lower() == "nan" or not url or url.lower() == "nan":
            continue
        result[key] = url
    return result


def priority_to_df(priority_map: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in priority_map.items():
        rows.append({"キー": key, "ブースト": value})
    if not rows:
        rows = [{"キー": "", "ブースト": 0}]
    return pd.DataFrame(rows)


def df_to_priority(df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in df.to_dict(orient="records"):
        key = str(row.get("キー", "")).strip()
        if not key or key.lower() == "nan":
            continue
        try:
            boost = float(row.get("ブースト", 0))
        except (TypeError, ValueError):
            continue
        result[key] = boost
    return result


def make_scored(items: List[NewsItem], weights: Dict[str, int], name_map_path: str) -> List[ScoredItem]:
    name_map = load_name_map(name_map_path)
    scored: List[ScoredItem] = []
    for it in items:
        ext = extract(it, name_map)
        scored.append(build_scored(it, ext, weights))
    return scored


def extract_tags(items: List[NewsItem], name_map_path: str) -> Tuple[List[List[str]], Counter]:
    name_map = load_name_map(name_map_path)
    all_tags: List[List[str]] = []
    counter: Counter = Counter()
    for it in items:
        ext = extract(it, name_map)
        all_tags.append(ext.tags)
        counter.update(ext.tags)
    return all_tags, counter


def auto_weights(base_weights: Dict[str, int], items: List[NewsItem], name_map_path: str) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int, int]]]:
    """Derive dynamic weights from recent items.

    Returns:
      new_weights, summary where summary[tag] = (count, old, new)
    """
    _, counts = extract_tags(items, name_map_path)
    new_w: Dict[str, int] = dict(base_weights)
    summary: Dict[str, Tuple[int, int, int]] = {}

    for tag, old in base_weights.items():
        c = int(counts.get(tag, 0))
        new_val = old
        if old > 0:
            # Boost rarer positive tags. Up to +2 points when unseen, +1 when rare.
            if c == 0:
                boost = 2
            elif c <= 2:
                boost = 1
            else:
                boost = 0
            new_val = old + boost
        elif old < 0:
            # Keep negative weights stable to avoid accidental overexposure.
            new_val = old
        # clamp to sensible range
        if old > 0:
            new_val = max(1, min(10, int(new_val)))
        else:
            new_val = min(-1, max(-10, int(new_val)))
        new_w[tag] = new_val
        summary[tag] = (c, int(old), int(new_val))

    return new_w, summary


def _jst_now():
    from datetime import timezone, timedelta

    return datetime.now(timezone(timedelta(hours=9)))


def auto_pick_feeds(cfg: Dict[str, Any]) -> List[str]:
    keys = list((cfg.get("feeds") or {}).keys())
    if not keys:
        return []
    now = _jst_now()
    hour = now.hour
    market_open = 9 <= hour < 15
    picked: List[str] = []
    if "tdnet" in keys:
        picked.append("tdnet")
    # 場外はPR TIMESなども拾う
    if not market_open and "prtimes" in keys:
        picked.append("prtimes")
    # その他、IR/press系を優先候補に
    for k in keys:
        if k in picked:
            continue
        if any(s in k.lower() for s in ("ir", "press", "news")):
            picked.append(k)
    # まだ少ない場合は先頭から補完
    for k in keys:
        if k not in picked:
            picked.append(k)
        if len(picked) >= 3:
            break
    # 最多3つまで
    return picked[:3]


def slack_settings(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str, int, int]:
    node = cfg.get("notifier", {}).get("slack", {}) or {}
    enabled = bool(node.get("enabled", False))
    env_name = node.get("webhook_env", "SLACK_WEBHOOK_URL")
    thresholds = cfg.get("thresholds", {}) or {}
    default_min = int(thresholds.get("min_score", 0))
    min_score = int(node.get("min_score", default_min))
    cooldown = int(node.get("cooldown_min", 0))
    return node, enabled, env_name, min_score, cooldown


def filter_recent(items: List[NewsItem], hours: int = 24) -> List[NewsItem]:
    try:
        now = _jst_now()
        cutoff = now.timestamp() - hours * 3600
        out: List[NewsItem] = []
        for it in items:
            dt = it.published_at
            ts = dt.timestamp() if hasattr(dt, "timestamp") else None
            if ts is None or ts >= cutoff:
                out.append(it)
        return out
    except Exception:
        return items


def auto_threshold(scored: List[ScoredItem], cfg_min: int, desired_top: int = 10) -> Tuple[int, int]:
    if not scored:
        return cfg_min, desired_top
    scores = sorted([s.score for s in scored], reverse=True)
    # 目標件数のスコアを閾値に（下限はcfg_min）
    if len(scores) >= desired_top:
        th = max(cfg_min, scores[desired_top - 1])
    else:
        th = max(cfg_min, scores[-1])
    return th, desired_top


NOTICE_CLASS_MAP = {
    "info": "notice notice-info",
    "warning": "notice notice-warning",
    "success": "notice notice-success",
    "error": "notice notice-error",
    "accent": "notice notice-accent",
}


def render_notice(text: str, level: str = "info", *, target=None) -> None:
    container = target or st
    cls = NOTICE_CLASS_MAP.get(level, NOTICE_CLASS_MAP["info"])
    container.markdown(f"<div class=\"{cls}\">{text}</div>", unsafe_allow_html=True)


def _score_tags(tags: List[str], weights: Dict[str, int]) -> int:
    return sum(int(weights.get(t, 0)) for t in tags)


_FEED_SUMMARY_TTL_SECONDS = 300
_FEED_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}
_FEED_HISTORY_PATH = Path.home() / ".cache" / "kabu2" / "feed_history.json"
_TAG_PRIORITY = {
    "upgrade": 2,
    "buyback_large": 3,
    "buyback_small": 1,
    "kpi_surprise_high": 2,
    "partner_tier1": 2,
}
_AI_PICK_LAST_RESULT: List[Dict[str, Any]] = []


def _load_feed_history() -> Dict[str, Any]:
    try:
        with open(_FEED_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _save_feed_history(history: Dict[str, Any]) -> None:
    try:
        _FEED_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEED_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def get_ai_pick_diagnostics() -> List[Dict[str, Any]]:
    return list(_AI_PICK_LAST_RESULT)


def _record_feed_success(scored_items: List[ScoredItem], cfg: Dict[str, Any]) -> None:
    if not scored_items:
        return
    history = _load_feed_history()
    min_score = int(cfg.get("thresholds", {}).get("min_score", 0))
    high_score = max((item.score for item in scored_items), default=min_score)
    alpha = 0.4
    now = datetime.now(timezone.utc)
    for item in scored_items:
        feed_key = item.news.source
        if not feed_key:
            continue
        hist = history.setdefault(feed_key, {})
        success_score = hist.get("success_score", 0.0)
        normalized = max(0.0, item.score)
        if high_score > 0:
            normalized = normalized / high_score * 10.0
        hist["success_score"] = alpha * normalized + (1 - alpha) * success_score
        hist["success_count"] = int(hist.get("success_count", 0)) + (1 if item.score >= min_score else 0)
        hist["last_success"] = now.isoformat()
        hist.setdefault("priority_bonus", 0.0)
    _save_feed_history(history)


def _mark_feeds_selected(feeds: List[str]) -> None:
    if not feeds:
        return
    history = _load_feed_history()
    now = datetime.now(timezone.utc).isoformat()
    for key in feeds:
        hist = history.setdefault(key, {})
        hist["last_selected"] = now
    _save_feed_history(history)


def _fetch_feed_summary(
    key: str,
    url: str,
    name_map: Dict[str, str],
    weights: Dict[str, int],
    per_feed_limit: int,
) -> Dict[str, Any]:
    now = time.time()
    cached = _FEED_SUMMARY_CACHE.get(url)
    if cached and now - cached.get("ts", 0) < _FEED_SUMMARY_TTL_SECONDS:
        return cached["data"]

    try:
        items = parse_feed(url, source=key)
    except Exception as exc:
        summary = {"error": str(exc), "items": [], "updated": None}
        _FEED_SUMMARY_CACHE[url] = {"ts": now, "data": summary}
        return summary

    name_map_local = name_map  # avoid closure mutation
    weights_local = weights
    scored_entries: List[Dict[str, Any]] = []
    for it in items[: per_feed_limit]:
        ext = extract(it, name_map_local)
        base = _score_tags(ext.tags, weights_local)
        priority = sum(_TAG_PRIORITY.get(tag, 0) for tag in ext.tags)
        published = it.published_at if isinstance(it.published_at, datetime) else None
        age_hours = None
        if isinstance(published, datetime):
            try:
                age_hours = max(0.0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).total_seconds() / 3600.0)
            except Exception:
                age_hours = None
        recency_boost = 0
        if age_hours is not None:
            if age_hours <= 3:
                recency_boost = 3
            elif age_hours <= 12:
                recency_boost = 2
            elif age_hours <= 24:
                recency_boost = 1
        scored_entries.append(
            {
                "score": base + priority + recency_boost,
                "base": base,
                "tags": ext.tags,
                "published": published,
            }
        )

    summary = {"items": scored_entries, "updated": datetime.now(timezone.utc)}
    _FEED_SUMMARY_CACHE[url] = {"ts": now, "data": summary}
    return summary


def ai_pick_feeds(cfg: Dict[str, Any], name_map_path: str, *, per_feed_limit: int = 20, max_keys: Optional[int] = None) -> List[str]:
    global _AI_PICK_LAST_RESULT
    feeds: Dict[str, str] = cfg.get("feeds", {}) or {}
    if not feeds:
        return []
    ai_cfg = cfg.get("feeds_ai", {}) or {}
    max_keys = int(ai_cfg.get("max_keys", max_keys or 3)) if max_keys is None else max_keys
    if max_keys < 1:
        max_keys = 1
    fresh_hours = float(ai_cfg.get("fresh_hours", 12))
    stale_penalty = float(ai_cfg.get("stale_penalty", 5.0))
    repeat_penalty = float(ai_cfg.get("repeat_penalty", 2.0))
    history_weight = float(ai_cfg.get("history_weight", 0.5))
    cooldown_hours = float(ai_cfg.get("cooldown_hours", 4))
    weights = cfg.get("weights", {}) or {}
    name_map = load_name_map(name_map_path)
    history = _load_feed_history()
    preferences = cfg.get("feeds_priority", {}) or {}
    scored_keys: List[Tuple[str, float, Dict[str, Any]]] = []
    diagnostics: List[Dict[str, Any]] = []
    for key, url in feeds.items():
        summary = _fetch_feed_summary(key, url, name_map, weights, per_feed_limit)
        hist = history.setdefault(key, {})
        preference_boost = float(preferences.get(key, 0) or 0)
        hist["priority_bonus"] = preference_boost
        cooldown_penalty = 0.0
        last_selected = hist.get("last_selected")
        if last_selected:
            try:
                last_dt = datetime.fromisoformat(last_selected)
                delta_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
                if delta_hours < cooldown_hours:
                    cooldown_penalty = repeat_penalty * max(0.0, (cooldown_hours - delta_hours) / max(cooldown_hours, 1e-6))
            except Exception:
                cooldown_penalty = 0.0
        if summary.get("error"):
            hist["fail_count"] = int(hist.get("fail_count", 0)) + 1
            hist["last_error"] = summary["error"]
            hist["last_updated"] = datetime.now(timezone.utc).isoformat()
            scored_keys.append((key, float("-inf"), summary))
            diagnostics.append({"feed": key, "status": "error", "message": summary["error"], "boost": preference_boost, "penalty": cooldown_penalty})
            continue
        entries = summary.get("items", [])
        if not entries:
            hist.setdefault("score_ema", -5.0)
            hist["last_score"] = -5.0
            hist["last_updated"] = datetime.now(timezone.utc).isoformat()
            stale_score = hist.get("score_ema", -5.0)
            total = stale_score + preference_boost - cooldown_penalty
            diagnostics.append({"feed": key, "status": "empty", "message": "シグナル候補なし", "boost": preference_boost, "penalty": round(cooldown_penalty, 2)})
            scored_keys.append((key, total, summary))
            continue
        scores = sorted([it["score"] for it in entries], reverse=True)
        top = scores[0]
        top3_avg = sum(scores[:3]) / min(3, len(scores))
        positive_hits = len([s for s in scores if s > 0])
        unique_tags = len({tag for item in entries for tag in item.get("tags", [])})
        fresh_hits = 0
        latest_age = None
        recency = 0
        for item in entries[:3]:
            published = item.get("published")
            if isinstance(published, datetime):
                try:
                    age = (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).total_seconds() / 3600.0
                except Exception:
                    age = None
                if age is not None:
                    if latest_age is None or age < latest_age:
                        latest_age = age
                    if age <= 6:
                        recency += 2
                    elif age <= 24:
                        recency += 1
                    if age <= fresh_hours:
                        fresh_hits += 1
        staleness = 0.0
        if latest_age is not None and latest_age > fresh_hours:
            staleness = stale_penalty * min(1.5, (latest_age - fresh_hours) / max(fresh_hours, 1e-6))
        feed_score = top * 1.4 + top3_avg * 1.0 + positive_hits * 0.8 + recency
        coverage_bonus = min(4.0, unique_tags * 0.5)
        freshness_bonus = fresh_hits * 0.7
        prev_ema = hist.get("score_ema")
        alpha = 0.5
        ema = feed_score if prev_ema is None else alpha * feed_score + (1 - alpha) * prev_ema
        success_score = hist.get("success_score", 0.0)
        hist.update(
            {
                "score_ema": ema,
                "last_score": feed_score,
                "fail_count": 0,
                "last_error": None,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        combined = feed_score * 0.5 + ema * 0.3 + success_score * history_weight
        priority_bonus = hist.get("priority_bonus", 0.0)
        total_score = combined + priority_bonus + coverage_bonus + freshness_bonus - staleness - cooldown_penalty
        diagnostics.append(
            {
                "feed": key,
                "status": "ok",
                "score": round(total_score, 2),
                "top": round(top, 2),
                "ema": round(ema, 2),
                "hits": positive_hits,
                "boost": priority_bonus,
                "coverage": coverage_bonus,
                "fresh": freshness_bonus,
                "stale": round(staleness, 2),
                "penalty": round(cooldown_penalty, 2),
            }
        )
        scored_keys.append((key, total_score, summary))

    _save_feed_history(history)
    scored_keys.sort(key=lambda x: x[1], reverse=True)
    selected = [k for k, score, _ in scored_keys[:max_keys] if score != float("-inf")]
    _AI_PICK_LAST_RESULT = diagnostics
    return selected


def render_card(item: ScoredItem, variant: str = "default") -> None:
    n = item.news
    published_raw = n.published_at.strftime("%Y-%m-%d %H:%M") if isinstance(n.published_at, datetime) else "-"
    hold_label = "スイング" if item.hold == "swing" else "デイ"
    score_display = escape(str(item.score))
    source_label = escape(n.source or "-")
    published_label = escape(published_raw)
    hold_chip = escape(hold_label)
    ticker_label = escape(n.ticker or "-")
    company_label = escape(n.company_name or "-")
    reasons = item.reasons or []
    tags_html = "".join(f"<span class=\"tag-chip\">{escape(reason)}</span>" for reason in reasons)
    if not tags_html:
        tags_html = "<span class=\"tag-chip tag-chip-empty\">シグナル待ち</span>"
    if n.link:
        title_html = f"<a href=\"{escape(n.link)}\" target=\"_blank\" rel=\"noopener noreferrer\">{escape(n.title or '-')}</a>"
    else:
        title_html = escape(n.title or "-")

    card_class = "card"
    if variant != "default":
        card_class += f" card-{variant}"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="badge">{score_display}</div>
            <div class="card-body">
                <div class="card-meta">
                    <span class="meta-slot">{published_label}</span>
                    <span class="meta-dot"></span>
                    <span class="meta-slot">{source_label}</span>
                    <span class="meta-chip">{hold_chip}</span>
                </div>
                <div class="card-title">{title_html}</div>
                <div class="card-sub">
                    <span class="sub-group">
                        <span class="sub-label">コード</span>
                        <span class="sub-code">{ticker_label}</span>
                    </span>
                    <span class="sub-divider"></span>
                    <span class="sub-group">
                        <span class="sub-label">企業</span>
                        <span class="sub-company">{company_label}</span>
                    </span>
                </div>
                <div class="card-tags">{tags_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )


def render_hero(cfg: Dict[str, Any], items: List[NewsItem], data_mode: str, min_score: int, top_k: int) -> None:
    feeds = cfg.get("feeds", {}) or {}
    weights = cfg.get("weights", {}) or {}
    feed_count = len(feeds)
    active_weights = sum(1 for value in weights.values() if int(value or 0) != 0)
    item_count = len(items)
    mode_label = "ライブ収集モード" if data_mode == "collect" else "ファイル解析モード"
    pills = [mode_label, f"スコア閾値 {min_score:+}", f"Top {top_k} 件"]
    pills_html = "".join(f"<span class=\"hero-pill\">{escape(text)}</span>" for text in pills)

    stat_defs = [
        ("登録フィード", str(feed_count), "Feeds"),
        ("アクティブルール", str(active_weights), "Weights"),
        ("読み込みニュース", str(item_count), "Items"),
    ]
    stats_html = "".join(
        f"""
        <div class=\"hero-stat\">
            <span class=\"hero-stat-value\">{escape(value)}</span>
            <span class=\"hero-stat-label\">{escape(label)}</span>
            <span class=\"hero-stat-hint\">{escape(hint)}</span>
        </div>
        """
        for label, value, hint in stat_defs
    )

    timestamp = _jst_now().strftime("%Y/%m/%d %H:%M")
    hero_html = f"""
    <div class=\"hero\">
        <div class=\"hero-inner\">
            <div class=\"hero-main\">
                <div class=\"hero-badge\">Signal Intelligence</div>
                <h1>kabu2 シグナルボード</h1>
                <p class=\"hero-lead\">最新ニュースから日本株のチャンスを素早く捕捉</p>
                <div class=\"hero-pills\">{pills_html}</div>
                <div class=\"hero-timestamp\">更新 {escape(timestamp)} JST</div>
            </div>
            <div class=\"hero-stats\">{stats_html}</div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="kabu2 シグナルボード", page_icon="📈", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');
        :root {
            color-scheme: dark;
            --accent: #38bdf8;
            --accent-strong: #0ea5e9;
            --bg-base: #060914;
            --bg-soft: rgba(15, 23, 42, 0.74);
            --border-soft: rgba(148, 163, 184, 0.22);
        }
        body, .stApp {
            font-family: 'Inter', 'Noto Sans JP', sans-serif;
            background:
                radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.18), transparent 55%),
                radial-gradient(circle at 90% 12%, rgba(129, 140, 248, 0.2), transparent 60%),
                linear-gradient(135deg, #0f172a, #020617 78%);
            color: #e2e8f0;
        }
        .stApp header, [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1200px;
            padding-top: 2.8rem;
            padding-bottom: 4rem;
        }
        .stMetric-label, .stMetric-value {
            color: rgba(226, 232, 240, 0.9) !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(6, 10, 21, 0.78);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
            backdrop-filter: blur(18px);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 2.2rem;
        }
        ::-webkit-scrollbar {
            width: 9px;
            height: 9px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(56, 189, 248, 0.35);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.45);
        }
        .hero {
            position: relative;
            margin-bottom: 2.4rem;
            padding: 1.9rem 2.4rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.32), rgba(79, 70, 229, 0.32));
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 45px 70px rgba(8, 47, 73, 0.35);
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 18% 20%, rgba(255, 255, 255, 0.12), transparent 55%);
            opacity: 0.6;
            pointer-events: none;
        }
        .hero-inner {
            position: relative;
            display: flex;
            flex-wrap: wrap;
            gap: 2.4rem;
            justify-content: space-between;
            align-items: flex-end;
            z-index: 1;
        }
        .hero-main {
            flex: 1 1 320px;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.55);
            color: #f8fafc;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .hero-main h1 {
            margin: 0;
            font-size: 1.95rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .hero-lead {
            margin: 0;
            max-width: 36rem;
            color: rgba(226, 232, 240, 0.85);
            font-size: 1.02rem;
            line-height: 1.6;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #e0f2fe;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
        }
        .hero-timestamp {
            font-size: 0.8rem;
            color: rgba(226, 232, 240, 0.65);
            letter-spacing: 0.08em;
        }
        .hero-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.9rem;
            min-width: 260px;
        }
        .hero-stat {
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .hero-stat-value {
            font-size: 1.65rem;
            font-weight: 600;
            color: #f8fafc;
        }
        .hero-stat-label {
            font-size: 0.85rem;
            color: rgba(226, 232, 240, 0.75);
        }
        .hero-stat-hint {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(148, 163, 184, 0.7);
        }
        .card {
            position: relative;
            padding: 1.4rem 1.6rem 1.2rem 1.6rem;
            border-radius: 20px;
            background: rgba(12, 17, 30, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 24px 45px rgba(8, 47, 73, 0.35);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
            overflow: hidden;
        }
        .card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(14, 165, 233, 0.08), rgba(99, 102, 241, 0.05));
            opacity: 0;
            transition: opacity 180ms ease;
        }
        .card:hover {
            transform: translateY(-3px);
            border-color: rgba(56, 189, 248, 0.35);
            box-shadow: 0 30px 60px rgba(14, 116, 144, 0.28);
        }
        .card:hover::before {
            opacity: 1;
        }
        .card.card-quick {
            background: linear-gradient(135deg, rgba(248, 250, 252, 0.96), rgba(224, 242, 254, 0.95));
            color: #0f172a;
            border: 1px solid rgba(14, 165, 233, 0.35);
            box-shadow: 0 24px 50px rgba(14, 165, 233, 0.18);
        }
        .card.card-quick .card-title a {
            color: #0f172a;
        }
        .card.card-quick .card-meta {
            color: rgba(15, 23, 42, 0.65);
        }
        .card.card-quick .card-sub {
            color: rgba(15, 23, 42, 0.82);
        }
        .card.card-quick .sub-label {
            background: rgba(14, 165, 233, 0.18);
            border-color: rgba(14, 165, 233, 0.32);
            color: #0f172a;
        }
        .card.card-quick .sub-code {
            color: #0b1120;
        }
        .card.card-quick .sub-divider {
            background: rgba(15, 23, 42, 0.35);
        }
        .card.card-quick .tag-chip {
            background: rgba(14, 165, 233, 0.14);
            border-color: rgba(14, 165, 233, 0.26);
            color: #0f172a;
        }
        .card.card-quick .badge {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: #f8fafc;
            box-shadow: 0 12px 24px rgba(14, 165, 233, 0.3);
        }
        .badge {
            position: absolute;
            top: -14px;
            left: 20px;
            background: linear-gradient(135deg, #22d3ee, #3b82f6);
            border-radius: 999px;
            padding: 0.35rem 0.78rem;
            font-weight: 700;
            color: #0b1120;
            box-shadow: 0 16px 32px rgba(34, 211, 238, 0.32);
            z-index: 2;
        }
        .card-body {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            position: relative;
            z-index: 1;
        }
        .card-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            color: rgba(226, 232, 240, 0.7);
        }
        .meta-slot {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        .meta-dot {
            display: inline-block;
            width: 4px;
            height: 4px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.6);
        }
        .meta-chip {
            margin-left: auto;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.16);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #e0f2fe;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.4;
            color: #f8fafc;
        }
        .card-title a {
            color: inherit;
            text-decoration: none;
        }
        .card-title a:hover {
            text-decoration: underline;
        }
        .card-sub {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            font-size: 0.9rem;
            color: rgba(199, 210, 254, 0.95);
            letter-spacing: 0.01em;
        }
        .sub-group {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .sub-label {
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.45rem;
            border-radius: 6px;
            background: rgba(56, 189, 248, 0.14);
            border: 1px solid rgba(56, 189, 248, 0.28);
            color: #bae6fd;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }
        .sub-code {
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            letter-spacing: 0.03em;
        }
        .sub-divider {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: rgba(148, 163, 184, 0.5);
        }
        .sub-company {
            opacity: 0.9;
        }
        .card-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .tag-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(14, 165, 233, 0.14);
            border: 1px solid rgba(14, 165, 233, 0.18);
            color: #bae6fd;
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }
        .tag-chip-empty {
            background: rgba(148, 163, 184, 0.15);
            border-color: rgba(148, 163, 184, 0.25);
            color: rgba(226, 232, 240, 0.65);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: transparent;
            color: rgba(226, 232, 240, 0.75);
            font-weight: 600;
            padding: 0.45rem 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(14, 165, 233, 0.18) !important;
            color: #38bdf8 !important;
        }
        .st-expander {
            border-radius: 14px !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            background: rgba(15, 23, 42, 0.55) !important;
        }
        .st-expander:hover {
            border-color: rgba(56, 189, 248, 0.3) !important;
        }
        .stButton button {
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.7), rgba(59, 130, 246, 0.7));
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #f8fafc;
            font-weight: 600;
            padding: 0.45rem 1.2rem;
            transition: transform 140ms ease, box-shadow 140ms ease;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 20px rgba(14, 165, 233, 0.25);
        }
        .stButton button:focus {
            outline: none;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.35);
        }
        .stButton button:active {
            transform: translateY(0);
        }
        .stCheckbox label, .stRadio label, .stSelectbox label, .stNumberInput label, .stFileUploader label {
            color: rgba(226, 232, 240, 0.85) !important;
            font-weight: 500;
        }
        .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox [role="combobox"], .stMultiSelect [data-baseweb="select"] > div {
            border-radius: 12px !important;
            background: rgba(15, 23, 42, 0.65) !important;
            border: 1px solid rgba(71, 85, 105, 0.6) !important;
            color: #e2e8f0 !important;
        }
        .stSlider > div[data-baseweb="slider"] > div {
            color: rgba(226, 232, 240, 0.85);
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background: rgba(36, 211, 255, 0.25);
        }
        .stSlider [data-baseweb="slider"] .MuiSlider-thumb, .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: linear-gradient(135deg, #22d3ee, #3b82f6);
            box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.25);
        }
        .notice {
            margin: 0.75rem 0 1.1rem;
            padding: 1rem 1.25rem;
            border-radius: 16px;
            font-size: 0.95rem;
            line-height: 1.6;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(6px);
        }
        .notice strong {
            display: inline-block;
            margin-bottom: 0.35rem;
        }
        .notice-info {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(14, 165, 233, 0.16));
            color: #dbeafe;
            border-color: rgba(59, 130, 246, 0.45);
        }
        .notice-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.18), rgba(251, 191, 36, 0.18));
            color: #fff7ed;
            border-color: rgba(251, 191, 36, 0.45);
        }
        .notice-success {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(16, 185, 129, 0.18));
            color: #dcfce7;
            border-color: rgba(34, 197, 94, 0.45);
        }
        .notice-error {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(248, 113, 113, 0.16));
            color: #fee2e2;
            border-color: rgba(239, 68, 68, 0.5);
        }
        .notice-accent {
            background: linear-gradient(135deg, #f8fafc, #e0f2fe);
            color: #0f172a;
            border-color: rgba(14, 165, 233, 0.45);
            box-shadow: 0 16px 28px rgba(14, 165, 233, 0.12);
        }
        .notice-accent strong {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ヒーローセクションでタイトルを表示するため、既存のタイトル表示は置き換え

    cfg_path = st.sidebar.text_input(
        "設定ファイルのパス",
        value="config.example.yaml",
        help="ここで指定したYAMLを読み込み・保存の対象として使います。例: config.yaml",
    )
    name_map_path = st.sidebar.text_input(
        "銘柄マスタ(CSV)",
        value="data/jpx_tickers.sample.csv",
        help="企業名・略称などから証券コードに紐づけるための辞書CSVです（列: code,name,aliases）",
    )

    if "cfg_state_version" not in st.session_state:
        st.session_state["cfg_state_version"] = 0

    warning_msg: str | None = None
    error_msg: str | None = None
    needs_load = "cfg_state" not in st.session_state or st.session_state.get("cfg_state_path") != cfg_path

    if needs_load:
        try:
            loaded_cfg = load_config(cfg_path)
        except FileNotFoundError:
            warning_msg = "設定ファイルが見つかりません。空の設定として編集できます。"
            loaded_cfg = {}
        except OSError as exc:  # pragma: no cover - ファイルアクセスエラー
            error_msg = f"設定ファイルへアクセスできません: {exc}"
            loaded_cfg = {}
        except YAMLError as exc:  # pragma: no cover - UI warning only
            error_msg = "設定ファイルの読み込みに失敗しました。YAMLの文法を確認してください。"
            loaded_cfg = {}
        st.session_state["cfg_state"] = normalize_config(loaded_cfg)
        st.session_state["cfg_state_path"] = cfg_path
        st.session_state["cfg_state_version"] += 1

    if warning_msg:
        st.sidebar.warning(warning_msg)
    if error_msg:
        st.sidebar.error(error_msg)

    if "cfg_state" not in st.session_state:
        st.session_state["cfg_state"] = normalize_config({})
        st.session_state["cfg_state_path"] = cfg_path

    cfg = normalize_config(st.session_state["cfg_state"])
    st.session_state["cfg_state"] = cfg
    cfg_version = st.session_state.get("cfg_state_version", 0)
    cfg_path_state = st.session_state.get("cfg_state_path", cfg_path)
    key_base = f"cfg_{abs(hash(cfg_path_state))}_{cfg_version}"

    mode_labels = {
        "ファイルアップロード": "file",
        "フィード/コレクタを今すぐ取得": "collect",
    }
    mode_choice = st.sidebar.radio(
        "データソースを選択",
        options=list(mode_labels.keys()),
        help="シグナル計算に使うデータの入手方法を選びます（ファイルアップロード／フィード・コレクタ即時取得）。",
    )
    data_mode = mode_labels[mode_choice]

    items: List[NewsItem] = []
    if data_mode == "file":
        uploaded = st.sidebar.file_uploader(
            "JSONLファイルをアップロード",
            type="jsonl",
            help="1行1ニュースのJSONLをアップロードすると、その場でスコアリングします。",
        )
        if uploaded is not None:
            raw = uploaded.getvalue().decode("utf-8").splitlines()
            temp_items: List[NewsItem] = []
            for line in raw:
                if not line.strip():
                    continue
                temp_items.append(NewsItem.from_dict(json.loads(line)))
            items = temp_items
            st.session_state["uploaded_items"] = items
        else:
            items = st.session_state.get("uploaded_items", [])
    else:
        feeds_cfg = cfg.get("feeds", {}) or {}
        collectors_cfg = cfg.get("collectors", []) or []
        source_types: Dict[str, str] = {}
        options: List[str] = []
        for key in feeds_cfg.keys():
            options.append(key)
            source_types[key] = "feed"
        for spec in collectors_cfg:
            if not isinstance(spec, Mapping):
                continue
            name = spec.get("name")
            if not name:
                continue
            name_str = str(name)
            if name_str not in options:
                options.append(name_str)
            source_types[name_str] = "collector"

        # Allow AI to propose feed selection for this mode as well
        ai_pick = st.sidebar.checkbox(
            "AIで取得ソースを自動選択", value=True, help="各フィードの最新ニュースを解析し、上位ソースを提案します（RSS優先）。"
        )
        session_default = st.session_state.get("collect_selected", [])
        default_selected = [key for key in session_default if key in options]
        if not default_selected:
            default_selected = options[:2]

        def _format_source(name: str) -> str:
            if source_types.get(name) == "collector":
                return f"{name} (collector)"
            return name

        selected = st.sidebar.multiselect(
            "取得対象ソース",
            options=options,
            default=default_selected,
            format_func=_format_source,
            help="設定ファイルの feeds / collectors から収集対象を選びます。",
        )
        if ai_pick and st.sidebar.button("AIで選定"):
            suggested = ai_pick_feeds(cfg, name_map_path)
            if suggested:
                st.session_state["collect_selected"] = suggested
                _mark_feeds_selected(suggested)
                render_notice(f"AIが選定: {', '.join(suggested)}", "accent", target=st.sidebar)
                selected = suggested
            else:
                render_notice("AI選定に失敗しました。設定やネットワークを確認してください。", "warning", target=st.sidebar)
            diagnostics = get_ai_pick_diagnostics()
            for diag in diagnostics:
                status = diag.get("status")
                feed_name = diag.get("feed")
                if status == "error":
                    render_notice(f"{feed_name}: 取得失敗 ({diag.get('message')})", "warning", target=st.sidebar)
                elif status == "empty":
                    render_notice(f"{feed_name}: シグナル候補なし", "info", target=st.sidebar)
        if selected and st.sidebar.button("この条件で取得", use_container_width=True):
            with st.spinner("最新データを取得中..."):
                items = collect_selected(cfg, selected)
                st.session_state["collect_selected"] = selected
                st.session_state["collected_items"] = items
        items = st.session_state.get("collected_items", [])
        st.sidebar.caption("同じ条件での再取得はセッションキャッシュを利用します")

    if data_mode != "collect" and "collected_items" in st.session_state:
        st.session_state.pop("collected_items")

    thresholds = cfg.get("thresholds", {})
    min_score_default = int(thresholds.get("min_score", 4))
    min_score_default = max(-10, min(20, min_score_default))
    top_k_default = int(thresholds.get("top_k", 10))
    top_k_default = max(1, min(50, top_k_default))
    min_score = st.sidebar.slider(
        "スコア下限",
        min_value=-10,
        max_value=20,
        value=min_score_default,
        step=1,
        help="この値未満のニュースは一覧に表示しません。重みは設定タブで変更できます。",
    )
    top_k = st.sidebar.slider(
        "表示件数",
        min_value=1,
        max_value=50,
        value=top_k_default,
        step=1,
        help="ランキング上位から何件表示するかを指定します。",
    )

    render_hero(cfg, items, data_mode, min_score, top_k)

    quick_tab, signal_tab, config_tab = st.tabs(["ワンタッチ分析", "シグナル閲覧", "設定編集"])

    with quick_tab:
        st.subheader("ワンタッチ分析")
        st.caption("フィード選択・閾値調整を自動で行い、上位シグナルだけを表示します。")
        auto_cols = st.columns([1, 1, 1, 2])
        recent_hours = auto_cols[0].selectbox("対象期間", options=[6, 12, 24, 48], index=1, help="直近のニュースだけに絞ります（時間）。")
        desired_top = auto_cols[1].selectbox("表示件数", options=[5, 10, 15, 20], index=1)
        do_notify = auto_cols[2].checkbox("Slack通知", value=False, help="Slack通知が有効化済みの場合のみ送信します。")
        auto_weighting = st.checkbox("AIで重みを自動調整", value=True, help="最新のニュース出現状況からタグ重みを自動チューニングします。")
        ai_feed_select = st.checkbox("AIでフィード自動選択", value=True, help="各フィードの最新ニュースを軽く解析し、シグナル潜在が高い順に選定します。")
        run_auto = auto_cols[3].button("ワンクリックで実行", type="primary")

        if run_auto:
            if ai_feed_select:
                picked_keys = ai_pick_feeds(cfg, name_map_path)
                if not picked_keys:
                    picked_keys = auto_pick_feeds(cfg)
            else:
                picked_keys = auto_pick_feeds(cfg)
            if not picked_keys:
                render_notice("設定内に利用可能なフィードがありません。設定編集で feeds を追加してください。", "warning")
            else:
                _mark_feeds_selected(picked_keys)
                with st.spinner(f"データ取得中…（{', '.join(picked_keys)}）"):
                    items_auto = collect_selected(cfg, picked_keys)
                items_auto = filter_recent(items_auto, hours=int(recent_hours))
                weights_in = cfg.get("weights", {})
                if auto_weighting and items_auto:
                    weights_dyn, summary = auto_weights(weights_in, items_auto, name_map_path)
                    render_notice("<strong>AI重み調整</strong><br />出現頻度の低いポジティブタグを優先するよう重みをチューニングしました。", "accent")
                    with st.expander("調整内容（タグ別）", expanded=False):
                        rows = []
                        for tag, (cnt, old, newv) in summary.items():
                            if old != newv:
                                rows.append({"タグ": tag, "件数": cnt, "旧": old, "新": newv})
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows).sort_values(["新", "旧"], ascending=False),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.caption("今回のニュースでは調整対象がありませんでした。")
                    weights_use = weights_dyn
                else:
                    weights_use = weights_in
                scored_auto = make_scored(items_auto, weights_use, name_map_path)
                th, topn = auto_threshold(scored_auto, cfg.get("thresholds", {}).get("min_score", 0), desired_top=int(desired_top))
                show_auto = [s for s in scored_auto if s.score >= th]
                show_auto.sort(key=lambda s: (-s.score, s.news.published_at))
                show_auto = show_auto[: int(desired_top)]

                st.write(f"選択フィード: {', '.join(picked_keys)} / 閾値: {th} / 表示: {len(show_auto)}件")
                diagnostics = get_ai_pick_diagnostics()
                if diagnostics:
                    for diag in diagnostics:
                        status = diag.get("status")
                        feed_name = diag.get("feed")
                        if status == "error":
                            render_notice(f"AI選定: {feed_name} の取得に失敗しました ({diag.get('message')}).", "warning")
                        elif status == "empty":
                            render_notice(f"AI選定: {feed_name} に最新シグナル候補が見つかりませんでした。", "info")
                    with st.expander("AI選定の詳細", expanded=False):
                        rows = []
                        for diag in diagnostics:
                            rows.append(
                                {
                                    "フィード": diag.get("feed"),
                                    "状態": diag.get("status"),
                                    "スコア": diag.get("score"),
                                    "トップ": diag.get("top"),
                                    "EMA": diag.get("ema"),
                                    "ヒット数": diag.get("hits"),
                                    "ブースト": diag.get("boost"),
                                    "メモ": diag.get("message"),
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if not show_auto:
                    render_notice("表示条件に一致するシグナルがありませんでした。", "info")
                else:
                    _record_feed_success(show_auto, cfg)
                    for s in show_auto:
                        render_card(s, variant="quick")
                # Slack
                if do_notify and show_auto:
                    slack_node, slack_enabled, slack_env, slack_min_score, slack_cooldown = slack_settings(cfg)
                    if not slack_enabled:
                        render_notice("Slack通知は設定で無効化されています。設定編集で有効化してください。", "info")
                    else:
                        import os
                        webhook = os.environ.get(slack_env)
                        if not webhook:
                            render_notice("環境変数にWebhook URLが設定されていません。", "warning")
                        else:
                            result = post_slack(
                                webhook,
                                show_auto,
                                min_score=slack_min_score,
                                cooldown_minutes=slack_cooldown,
                            )
                            if result.status is None:
                                render_notice(
                                    "Slack通知対象がありませんでした（スコア閾値/クールダウンによりスキップ）。",
                                    "warning",
                                )
                            else:
                                render_notice(
                                    f"Slack通知を送信しました（status={result.status} / 送信{result.delivered}件 / スキップ{result.skipped}件）。",
                                    "success",
                                )

    with signal_tab:
        if not items:
            render_notice("データを読み込むとシグナルが表示されます。", "info")
        else:
            slack_node, slack_enabled, slack_env, slack_min_score, slack_cooldown = slack_settings(cfg)
            webhook = os.environ.get(slack_env) if slack_enabled else None

            scored = make_scored(items, cfg.get("weights", {}), name_map_path)
            filtered = [s for s in scored if s.score >= min_score]
            filtered.sort(key=lambda s: (-s.score, s.news.published_at))
            top_items = filtered[:top_k]

            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("読み込み件数", len(items))
                col2.metric("条件合致件数", len(filtered))
                col3.metric("表示件数", len(top_items))

            table_rows = []
            for s in filtered:
                tags = getattr(s, "tags", []) or []
                table_rows.append(
                    {
                        "スコア": s.score,
                        "想定ホールド": "スイング" if s.hold == "swing" else "デイ",
                        "タグ": ", ".join(tags),
                        "根拠": ", ".join(s.reasons),
                        "タイトル": s.news.title,
                        "企業名": s.news.company_name,
                        "証券コード": s.news.ticker,
                        "ソース": s.news.source,
                        "発表日時": s.news.published_at,
                        "リンク": s.news.link,
                    }
                )
            df_table = pd.DataFrame(table_rows)

            st.subheader("シグナル一覧")
            if not top_items:
                st.write("条件に合致するシグナルはありません。")
            else:
                for item in top_items:
                    render_card(item)

            # Tag analytics
            if filtered:
                tag_counter = Counter()
                for s in filtered:
                    tags = getattr(s, "tags", []) or []
                    tag_counter.update(tags)
                if tag_counter:
                    top_tags = pd.DataFrame(
                        [{"タグ": tag, "件数": cnt} for tag, cnt in tag_counter.most_common()]
                    )
                    with st.expander("タグ出現頻度", expanded=False):
                        st.bar_chart(top_tags.set_index("タグ"))
                        st.dataframe(top_tags, use_container_width=True, hide_index=True)

            with st.expander("詳細テーブル", expanded=False):
                st.dataframe(df_table, use_container_width=True, hide_index=True)

            if filtered:
                csv_bytes = df_table.to_csv(index=False).encode("utf-8")
                action_cols = st.columns(3)
                send_clicked = action_cols[0].button(
                    "Slack通知を送信", use_container_width=True, key=f"{key_base}_slack_send", disabled=not slack_enabled
                )
                preview_clicked = action_cols[1].button(
                    "Slackプレビュー", use_container_width=True, key=f"{key_base}_slack_preview", disabled=not slack_enabled
                )
                action_cols[2].download_button(
                    "CSVダウンロード",
                    data=csv_bytes,
                    file_name="kabu2_signals.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_base}_csv_download",
                )

                if send_clicked:
                    if not slack_enabled:
                        render_notice("Slack通知は設定で無効化されています。", "warning")
                    elif not webhook:
                        render_notice("Slack Webhook環境変数が見つかりません。", "warning")
                    else:
                        result = post_slack(
                            webhook,
                            top_items,
                            min_score=slack_min_score,
                            cooldown_minutes=slack_cooldown,
                        )
                        if result.status is None:
                            render_notice("Slack通知対象がありません（スコア閾値/クールダウンによりスキップ）。", "warning")
                        else:
                            render_notice(
                                f"Slack通知を送信しました（status={result.status} / 送信{result.delivered}件 / スキップ{result.skipped}件）。",
                                "success",
                            )

                if preview_clicked and slack_enabled:
                    if not webhook:
                        render_notice("Slack Webhook環境変数が見つかりません。", "warning")
                    else:
                        preview = post_slack(
                            webhook,
                            top_items,
                            min_score=slack_min_score,
                            cooldown_minutes=slack_cooldown,
                            dry_run=True,
                        )
                        render_notice(
                            f"Slack送信プレビュー（対象{preview.delivered}件 / スキップ{preview.skipped}件）。", "info"
                        )
                        st.json(preview.payload)

    with config_tab:
        st.subheader("設定編集")
        st.caption(f"対象ファイル: `{cfg_path}`")

        weights_df = weights_to_df(cfg.get("weights", {}))
        st.markdown("#### ルール重み（タグごとの加点・減点）")
        st.caption(
            "ニュースから抽出したイベントに対して、加点/減点する重みです。例: 自社株買いは+3〜+5、上方修正は+4 など。"
        )
        weights_editor = st.data_editor(
            weights_df,
            num_rows="dynamic",
            key=f"{key_base}_weights",
        )
        with st.expander("タグの意味（例）", expanded=False):
            st.markdown(
                """
                - upgrade: 決算・業績予想の上方修正
                - buyback_large/small: 自社株買い（大: 5%以上、または規模大／小: 5%未満・規模不明）
                - dividend_increase: 増配・配当方針の上方
                - split: 株式分割
                - big_order: 大口受注・大型契約・基本合意
                - partner: 中小規模の提携・採用
                - partner_tier1: 大手企業との提携・採用
                - approval_patent: 規制承認・薬事承認・特許取得
                - kpi_surprise_low/high: 主要KPIの好調（小/大）
                - negative_offering: 公募増資・CB等（希薄化）
                - lawsuit: 訴訟・係争
                - deficit_widen: 赤字拡大
                - intraday_bonus: 場中発表のボーナス加点
                """
            )

        col_thr1, col_thr2 = st.columns(2)
        min_score_edit = col_thr1.number_input(
            "スコア下限",
            value=int(thresholds.get("min_score", 4)),
            step=1,
            format="%d",
            key=f"{key_base}_minscore",
            help="この値未満はランキング対象から除外します。",
        )
        top_k_edit = col_thr2.number_input(
            "表示件数(Top K)",
            value=int(thresholds.get("top_k", 10)),
            step=1,
            format="%d",
            key=f"{key_base}_topk",
            help="ランキング上位から表示する件数です。",
        )

        storage_path = st.text_input(
            "ニュース保存先 (storage.path)",
            value=cfg.get("storage", {}).get("path", "data/news.jsonl"),
            key=f"{key_base}_storage",
            help="収集コマンド（collect）で追記される既定のJSONLファイルのパスです。",
        )

        feeds_df = feeds_to_df(cfg.get("feeds", {}))
        st.markdown("#### 収集対象RSS（feeds）")
        st.caption("キーは任意の短い名前、URLはRSS/AtomフィードのURLを指定します。例: tdnet / prtimes")
        feeds_editor = st.data_editor(
            feeds_df,
            num_rows="dynamic",
            key=f"{key_base}_feeds",
        )

        priority_df = priority_to_df(cfg.get("feeds_priority", {}))
        st.markdown("#### フィード優先度（AI選定ブースト）")
        st.caption("AIフィード選定時に加点するブースト値です。0で通常、正は優先、負は抑制。")
        priority_editor = st.data_editor(
            priority_df,
            num_rows="dynamic",
            key=f"{key_base}_priority",
        )

        feeds_ai_cfg = cfg.get("feeds_ai", {})
        st.markdown("#### AIフィード選定パラメータ")
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        ai_max_keys = col_ai1.number_input(
            "選定件数",
            value=int(feeds_ai_cfg.get("max_keys", 3)),
            min_value=1,
            max_value=10,
            step=1,
            key=f"{key_base}_ai_max_keys",
        )
        ai_fresh_hours = col_ai2.number_input(
            "フレッシュ判定(時)",
            value=float(feeds_ai_cfg.get("fresh_hours", 12)),
            min_value=1.0,
            max_value=72.0,
            step=1.0,
            key=f"{key_base}_ai_fresh",
        )
        ai_history_weight = col_ai3.number_input(
            "履歴反映重み",
            value=float(feeds_ai_cfg.get("history_weight", 0.5)),
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key=f"{key_base}_ai_history",
        )
        col_ai4, col_ai5, col_ai6 = st.columns(3)
        ai_stale_penalty = col_ai4.number_input(
            "陳腐ペナルティ",
            value=float(feeds_ai_cfg.get("stale_penalty", 5.0)),
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            key=f"{key_base}_ai_stale",
        )
        ai_repeat_penalty = col_ai5.number_input(
            "連続選定ペナルティ",
            value=float(feeds_ai_cfg.get("repeat_penalty", 2.0)),
            min_value=0.0,
            max_value=10.0,
            step=0.5,
            key=f"{key_base}_ai_repeat",
        )
        ai_cooldown = col_ai6.number_input(
            "選定クールダウン(時)",
            value=float(feeds_ai_cfg.get("cooldown_hours", 4)),
            min_value=0.0,
            max_value=48.0,
            step=1.0,
            key=f"{key_base}_ai_cooldown",
        )

        slack_cfg = cfg.get("notifier", {}).get("slack", {})
        slack_enabled = st.checkbox(
            "Slack通知を有効化",
            value=bool(slack_cfg.get("enabled", False)),
            key=f"{key_base}_slack_enabled",
            help="ランキング出力をSlackに投稿します（Webhook URLは環境変数に設定）。",
        )
        slack_env = st.text_input(
            "Slack Webhook環境変数名",
            value=slack_cfg.get("webhook_env", "SLACK_WEBHOOK_URL"),
            key=f"{key_base}_slack_env",
            help="Webhook URLを格納する環境変数名。例: SLACK_WEBHOOK_URL",
        )
        col_slack_min, col_slack_cool = st.columns(2)
        slack_min_score_cfg = int(slack_cfg.get("min_score", thresholds.get("min_score", 0)))
        slack_min_score = col_slack_min.number_input(
            "Slack送信スコア閾値",
            value=slack_min_score_cfg,
            step=1,
            key=f"{key_base}_slack_min_score",
            help="このスコア未満のシグナルは通知に含めません。",
        )
        slack_cooldown = col_slack_cool.number_input(
            "Slackクールダウン(分)",
            value=int(slack_cfg.get("cooldown_min", 0)),
            step=5,
            min_value=0,
            key=f"{key_base}_slack_cooldown",
            help="同じシグナルを再通知しない最小間隔です。0で無効化。",
        )

        col_apply, col_reload, col_save = st.columns(3)

        if col_apply.button("変更を適用", use_container_width=True, key=f"{key_base}_apply"):
            new_cfg = deepcopy(cfg)
            new_cfg["weights"] = df_to_weights(weights_editor)
            new_cfg.setdefault("thresholds", {})["min_score"] = int(min_score_edit)
            new_cfg["thresholds"]["top_k"] = int(top_k_edit)
            new_cfg.setdefault("storage", {})["path"] = storage_path
            new_cfg["feeds"] = df_to_feeds(feeds_editor)
            new_cfg["feeds_priority"] = df_to_priority(priority_editor)
            new_cfg.setdefault("feeds_ai", {})["max_keys"] = int(ai_max_keys)
            new_cfg["feeds_ai"]["fresh_hours"] = float(ai_fresh_hours)
            new_cfg["feeds_ai"]["history_weight"] = float(ai_history_weight)
            new_cfg["feeds_ai"]["stale_penalty"] = float(ai_stale_penalty)
            new_cfg["feeds_ai"]["repeat_penalty"] = float(ai_repeat_penalty)
            new_cfg["feeds_ai"]["cooldown_hours"] = float(ai_cooldown)
            slack_node = new_cfg.setdefault("notifier", {}).setdefault("slack", {})
            slack_node["enabled"] = bool(slack_enabled)
            slack_node["webhook_env"] = slack_env or "SLACK_WEBHOOK_URL"
            slack_node["min_score"] = int(slack_min_score)
            slack_node["cooldown_min"] = int(slack_cooldown)
            st.session_state["cfg_state"] = normalize_config(new_cfg)
            st.session_state["cfg_state_version"] += 1
            render_notice("設定を適用しました。", "success")

        if col_reload.button("ファイルから再読込", use_container_width=True, key=f"{key_base}_reload"):
            try:
                reloaded = load_config(cfg_path)
            except FileNotFoundError:
                render_notice("設定ファイルが見つかりません。", "warning")
                reloaded = {}
            except OSError as exc:  # pragma: no cover - ファイルアクセスエラー
                render_notice(f"設定ファイルへアクセスできません: {exc}", "error")
                reloaded = {}
            except YAMLError:  # pragma: no cover - UI warning only
                render_notice("設定ファイルの読み込みに失敗しました。", "error")
                reloaded = {}
            st.session_state["cfg_state"] = normalize_config(reloaded)
            st.session_state["cfg_state_path"] = cfg_path
            st.session_state["cfg_state_version"] += 1
            render_notice("ファイルから設定を再読込しました。", "info")

        if col_save.button("設定ファイルに保存", use_container_width=True, key=f"{key_base}_save"):
            try:
                save_config(st.session_state["cfg_state"], cfg_path)
                render_notice(f"設定を保存しました: {cfg_path}", "success")
            except OSError as exc:  # pragma: no cover - filesystem error path
                render_notice(f"設定の保存に失敗しました: {exc}", "error")


if __name__ == "__main__":
    main()
