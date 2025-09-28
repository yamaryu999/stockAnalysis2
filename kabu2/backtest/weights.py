from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from kabu2.models import NewsItem

JST = timezone(timedelta(hours=9))


@dataclass
class BacktestSample:
    """Input sample used for backtesting."""

    news: NewsItem
    tags: List[str]
    score: int
    ticker: str


@dataclass
class TagStat:
    count: int
    avg_return: float
    median_return: float
    win_rate: float
    current_weight: int
    regression_weight: Optional[int]
    blended_weight: Optional[int]


@dataclass
class BacktestOutcome:
    """Result of running the weight backtest."""

    samples: List[Dict[str, object]]
    tag_summary: Dict[str, TagStat]
    regression_weights: Dict[str, int]
    intercept: float
    rmse: Optional[float]
    score_correlation: Optional[float]
    overall_average: Optional[float]


def run_weight_backtest(
    samples: Sequence[BacktestSample],
    base_weights: Dict[str, int],
    *,
    horizon: int = 3,
    min_count: int = 3,
    scale: float = 100.0,
) -> BacktestOutcome:
    """Run a price-action backtest for the provided samples."""

    if not samples:
        return BacktestOutcome(
            samples=[],
            tag_summary={},
            regression_weights={},
            intercept=0.0,
            rmse=None,
            score_correlation=None,
            overall_average=None,
        )

    prices = _load_price_history(samples, horizon)

    per_item: List[Dict[str, object]] = []
    tag_returns: Dict[str, List[float]] = {}

    for sample in samples:
        ticker = sample.ticker
        series = prices.get(ticker)
        if series is None or series.empty:
            continue
        entry_idx, exit_idx = _resolve_holding_window(series.index, sample.news.published_at, horizon)
        if entry_idx is None or exit_idx is None:
            continue
        entry_price = float(series.iloc[entry_idx])
        exit_price = float(series.iloc[exit_idx])
        if entry_price <= 0:
            continue
        ret = (exit_price / entry_price) - 1.0
        per_item.append(
            {
                "id": sample.news.id,
                "ticker": ticker,
                "entry_date": series.index[entry_idx].isoformat(),
                "exit_date": series.index[exit_idx].isoformat(),
                "return_pct": ret,
                "score": sample.score,
                "tags": list(sample.tags),
            }
        )
        for tag in sample.tags:
            tag_returns.setdefault(tag, []).append(ret)

    if not per_item:
        return BacktestOutcome(
            samples=[],
            tag_summary={},
            regression_weights={},
            intercept=0.0,
            rmse=None,
            score_correlation=None,
            overall_average=None,
        )

    regression_weights, intercept, rmse = _fit_regression(per_item, scale=scale, min_count=min_count)
    overall_avg = float(np.mean([row["return_pct"] for row in per_item])) if per_item else None
    score_corr = _score_return_correlation(per_item)

    summary: Dict[str, TagStat] = {}
    for tag, values in tag_returns.items():
        if not values:
            continue
        avg_ret = float(np.mean(values))
        med_ret = float(median(values))
        wins = sum(1 for v in values if v > 0)
        count = len(values)
        win_rate = wins / count if count else 0.0
        current = int(base_weights.get(tag, 0))
        reg_weight = regression_weights.get(tag)
        blended = None
        if reg_weight is not None:
            if current == 0:
                blended = reg_weight
            else:
                blended = _clamp_weight(int(round((current + reg_weight) / 2)))
        summary[tag] = TagStat(
            count=count,
            avg_return=avg_ret,
            median_return=med_ret,
            win_rate=win_rate,
            current_weight=current,
            regression_weight=reg_weight,
            blended_weight=blended,
        )

    return BacktestOutcome(
        samples=per_item,
        tag_summary=summary,
        regression_weights=regression_weights,
        intercept=intercept,
        rmse=rmse,
        score_correlation=score_corr,
        overall_average=overall_avg,
    )


def _load_price_history(samples: Sequence[BacktestSample], horizon: int) -> Dict[str, pd.Series]:
    tickers = {s.ticker for s in samples if s.ticker}
    if not tickers:
        return {}

    start_dt = None
    end_dt = None
    for s in samples:
        dt = s.news.published_at
        aware = _ensure_jst(dt)
        if not aware:
            continue
        if start_dt is None or aware < start_dt:
            start_dt = aware
        if end_dt is None or aware > end_dt:
            end_dt = aware
    if start_dt is None or end_dt is None:
        return {}

    start = (start_dt - timedelta(days=10)).date()
    end = (end_dt + timedelta(days=horizon + 10)).date()

    price_map: Dict[str, pd.Series] = {}
    for code in sorted(tickers):
        symbol = code if code.endswith(".T") else f"{code}.T"
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        if df.empty:
            continue
        if "Adj Close" in df.columns:
            close = df["Adj Close"].copy()
        elif "Close" in df.columns:
            close = df["Close"].copy()
        else:
            continue
        close.index = [idx.date() for idx in close.index]
        price_map[code] = close.sort_index()
    return price_map


def _ensure_jst(dt: Optional[datetime]) -> Optional[datetime]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        try:
            dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    try:
        return dt.astimezone(JST)
    except Exception:
        return None


def _resolve_holding_window(
    index: Iterable[date], published_at: Optional[datetime], horizon: int
) -> Tuple[Optional[int], Optional[int]]:
    ordered = list(index)
    if not ordered:
        return None, None
    local = _ensure_jst(published_at)
    if not local:
        return None, None
    entry_candidate = local.date()
    if local.hour >= 15:
        entry_candidate += timedelta(days=1)
    entry_idx = _find_first_ge(ordered, entry_candidate)
    if entry_idx is None:
        return None, None
    exit_idx = entry_idx + max(1, horizon)
    if exit_idx >= len(ordered):
        return None, None
    return entry_idx, exit_idx


def _find_first_ge(dates: Sequence[date], target: date) -> Optional[int]:
    for idx, d in enumerate(dates):
        if d >= target:
            return idx
    return None


def _fit_regression(
    per_item: Sequence[Dict[str, object]],
    *,
    scale: float,
    min_count: int,
) -> Tuple[Dict[str, int], float, Optional[float]]:
    tag_list = sorted({tag for row in per_item for tag in row.get("tags", [])})
    if not tag_list:
        return {}, 0.0, None
    returns = np.array([float(row["return_pct"]) for row in per_item], dtype=float)
    X = np.zeros((len(per_item), len(tag_list)), dtype=float)
    index_map = {tag: idx for idx, tag in enumerate(tag_list)}
    for i, row in enumerate(per_item):
        for tag in row.get("tags", []):
            j = index_map.get(tag)
            if j is None:
                continue
            X[i, j] = 1.0
    mask = ~np.isnan(returns)
    X = X[mask]
    y = returns[mask]
    if X.size == 0 or len(y) < 2:
        return {}, 0.0, None
    X_ext = np.hstack([X, np.ones((len(X), 1), dtype=float)])
    try:
        coef, residuals, rank, _ = np.linalg.lstsq(X_ext, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}, 0.0, None
    raw_weights = coef[:-1]
    intercept = float(coef[-1])
    rmse = None
    if residuals.size > 0 and len(y) > 0:
        rmse = float(math.sqrt(residuals[0] / len(y)))
    weights: Dict[str, int] = {}
    for tag, weight in zip(tag_list, raw_weights):
        suggested = int(round(weight * scale))
        count = sum(1 for row in per_item if tag in row.get("tags", []))
        if count < min_count:
            continue
        weights[tag] = _clamp_weight(suggested)
    return weights, intercept, rmse


def _clamp_weight(value: int) -> int:
    return max(-10, min(10, value))


def _score_return_correlation(per_item: Sequence[Dict[str, object]]) -> Optional[float]:
    if not per_item:
        return None
    scores = np.array([float(row.get("score", 0)) for row in per_item], dtype=float)
    returns = np.array([float(row.get("return_pct", np.nan)) for row in per_item], dtype=float)
    mask = ~np.isnan(returns)
    scores = scores[mask]
    returns = returns[mask]
    if len(scores) < 2:
        return None
    if np.std(scores) == 0 or np.std(returns) == 0:
        return None
    corr = float(np.corrcoef(scores, returns)[0, 1])
    return corr
