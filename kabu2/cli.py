from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

import yaml

from kabu2.backtest import BacktestSample, run_weight_backtest
from kabu2.config import get_slack_webhook, load_config
from kabu2.collector import collect_from_config
from kabu2.extractor.rules import extract, load_name_map
from kabu2.models import NewsItem, ScoredItem
from kabu2.notifier.slack import post_slack
from kabu2.scorer.score import build_scored
from kabu2.storage.store import JsonlStore


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    selected = args.feeds if args.feeds else None
    items = collect_from_config(cfg, selected=selected)
    # simple de-dup by id
    seen = set()
    out_items: List[NewsItem] = []
    for it in items:
        if it.id in seen:
            continue
        seen.add(it.id)
        out_items.append(it)

    out_path = args.out or cfg.get("storage", {}).get("path", "data/news.jsonl")
    store = JsonlStore(out_path)
    store.append(out_items)
    print(f"collected: {len(out_items)} -> {out_path}")
    return 0


def _load_jsonl(path: str) -> List[NewsItem]:
    items: List[NewsItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(NewsItem.from_dict(d))
    return items


def cmd_rank(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    weights = cfg.get("weights", {})

    # name map
    name_map_path = args.name_map or "data/jpx_tickers.sample.csv"
    name_map = load_name_map(name_map_path)

    # input
    input_path = args.input
    items = _load_jsonl(input_path)

    scored: List[ScoredItem] = []
    for it in items:
        ext = extract(it, name_map)
        s = build_scored(it, ext, weights)
        scored.append(s)

    # filter and sort
    min_score = args.min_score if args.min_score is not None else cfg.get("thresholds", {}).get("min_score", 0)
    topk = args.top if args.top is not None else cfg.get("thresholds", {}).get("top_k", 10)

    scored = [s for s in scored if s.score >= min_score]
    scored.sort(key=lambda x: (-x.score, x.news.published_at))
    top_items = scored[:topk]

    # print
    for s in top_items:
        n = s.news
        published = n.published_at.isoformat() if isinstance(n.published_at, datetime) else "-"
        print(
            f"[{s.score:>3}] {n.ticker or '-'} {n.company_name or '-'} | {n.title} | "
            f"{', '.join(s.reasons)} | {published} | {n.link}"
        )

    # notify
    if args.notify:
        slack_cfg = cfg.get("notifier", {}).get("slack", {})
        webhook = get_slack_webhook(cfg)
        if not webhook:
            print("Slack webhook not configured or disabled.")
        else:
            notify_min_score = int(slack_cfg.get("min_score", min_score))
            cooldown_min = int(slack_cfg.get("cooldown_min", 0))
            result = post_slack(
                webhook,
                top_items,
                min_score=notify_min_score,
                cooldown_minutes=cooldown_min,
            )
            if result.status is None:
                print("Slack notification skipped (no eligible items or request failed).")
            else:
                print(
                    f"Slack notified: status={result.status} delivered={result.delivered} skipped={result.skipped}"
                )
    return 0


def _resolve_store_path(args: argparse.Namespace, cfg: Dict[str, object]) -> str:
    if getattr(args, "path", None):
        return args.path
    return cfg.get("storage", {}).get("path", "data/news.jsonl")


def cmd_dedupe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store_path = _resolve_store_path(args, cfg)
    store = JsonlStore(store_path)
    removed = store.dedupe()
    print(f"deduped {removed} duplicates from {store_path}")
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store_path = _resolve_store_path(args, cfg)
    store = JsonlStore(store_path)
    removed = store.compact(keep=args.keep)
    print(f"compacted {store_path}: removed {removed} older items")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    weights = cfg.get("weights", {})

    name_map_path = args.name_map or "data/jpx_tickers.sample.csv"
    name_map = load_name_map(name_map_path)

    items = _load_jsonl(args.input)
    samples: List[BacktestSample] = []
    for it in items:
        ext = extract(it, name_map)
        if not ext.tags or not ext.ticker:
            continue
        scored = build_scored(it, ext, weights)
        ticker = scored.news.ticker or ext.ticker
        if not ticker:
            continue
        samples.append(
            BacktestSample(
                news=scored.news,
                tags=list(scored.tags),
                score=scored.score,
                ticker=ticker,
            )
        )

    if not samples:
        print("No samples with tags and tickers available for backtest.")
        return 0

    outcome = run_weight_backtest(
        samples,
        weights,
        horizon=args.horizon,
        min_count=args.min_count,
        scale=args.scale,
    )

    if not outcome.samples:
        print("No price data could be retrieved for the provided samples.")
        return 0

    print("=== Backtest summary ===")
    print(f"Samples with price data: {len(outcome.samples)}")
    if outcome.overall_average is not None:
        print(f"Average return ({args.horizon}d): {outcome.overall_average * 100:.2f}%")
    if outcome.rmse is not None:
        print(f"Regression RMSE: {outcome.rmse * 100:.2f}%")
    if outcome.score_correlation is not None:
        print(f"Score/return correlation: {outcome.score_correlation:.3f}")

    header = (
        f"{'tag':20} {'count':>5} {'avg%':>7} {'win%':>7} {'current':>7} "
        f"{'reg':>5} {'blend':>5}"
    )
    print(header)
    print("-" * len(header))

    blended_weights: Dict[str, int] = dict(weights)
    diagnostics: List[Dict[str, object]] = []
    for tag, stat in sorted(outcome.tag_summary.items(), key=lambda x: (-x[1].count, x[0])):
        reg_display = "-" if stat.regression_weight is None else f"{stat.regression_weight:>5}"
        blend_display = "-" if stat.blended_weight is None else f"{stat.blended_weight:>5}"
        print(
            f"{tag:20} {stat.count:5d} {stat.avg_return * 100:7.2f} "
            f"{stat.win_rate * 100:7.1f} {stat.current_weight:7d} "
            f"{reg_display} {blend_display}"
        )
        if stat.blended_weight is not None:
            blended_weights[tag] = stat.blended_weight
        elif stat.regression_weight is not None and tag not in blended_weights:
            blended_weights[tag] = stat.regression_weight
        diagnostics.append(
            {
                "tag": tag,
                "count": stat.count,
                "avg_return": stat.avg_return,
                "win_rate": stat.win_rate,
                "current_weight": stat.current_weight,
                "regression_weight": stat.regression_weight,
                "blended_weight": stat.blended_weight,
            }
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "horizon": args.horizon,
                    "samples": outcome.samples,
                    "tag_summary": diagnostics,
                    "regression_weights": outcome.regression_weights,
                    "overall_average": outcome.overall_average,
                    "score_correlation": outcome.score_correlation,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Diagnostics written to {args.output}")

    if args.write_config:
        updated_cfg = dict(cfg)
        updated_cfg["weights"] = blended_weights
        with open(args.write_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(updated_cfg, f, allow_unicode=True, sort_keys=False)
        print(f"Blended weights written to {args.write_config}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kabu2", description="News-driven JP stocks ranking (MVP)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Collect RSS feeds and append to JSONL store")
    p_collect.add_argument("--config", type=str, default=None)
    p_collect.add_argument("--out", type=str, default=None)
    p_collect.add_argument("--feeds", nargs="*", help="Subset of feed keys to collect")
    p_collect.set_defaults(func=cmd_collect)

    p_rank = sub.add_parser("rank", help="Rank news from JSONL and optionally notify")
    p_rank.add_argument("--config", type=str, default=None)
    p_rank.add_argument("--input", type=str, required=True)
    p_rank.add_argument("--name-map", dest="name_map", type=str, default=None)
    p_rank.add_argument("--top", type=int, default=None)
    p_rank.add_argument("--min-score", dest="min_score", type=int, default=None)
    p_rank.add_argument("--notify", action="store_true")
    p_rank.set_defaults(func=cmd_rank)

    p_dedupe = sub.add_parser("dedupe", help="Remove duplicate entries from stored news")
    p_dedupe.add_argument("--config", type=str, default=None)
    p_dedupe.add_argument("--path", type=str, default=None, help="Target JSONL path (defaults to storage.path)")
    p_dedupe.set_defaults(func=cmd_dedupe)

    p_compact = sub.add_parser("compact", help="Sort and optionally trim stored news")
    p_compact.add_argument("--config", type=str, default=None)
    p_compact.add_argument("--path", type=str, default=None, help="Target JSONL path (defaults to storage.path)")
    p_compact.add_argument("--keep", type=int, default=None, help="Keep only the newest N entries")
    p_compact.set_defaults(func=cmd_compact)

    p_backtest = sub.add_parser("backtest", help="Backtest and suggest tag weights from historical moves")
    p_backtest.add_argument("--config", type=str, default=None)
    p_backtest.add_argument("--input", type=str, required=True)
    p_backtest.add_argument("--name-map", dest="name_map", type=str, default=None)
    p_backtest.add_argument("--horizon", type=int, default=3, help="Holding horizon in trading days")
    p_backtest.add_argument(
        "--min-count", type=int, default=3, help="Minimum samples per tag to suggest regression weights"
    )
    p_backtest.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale factor for converting regression coefficients to weight points (default: 100 = 1pt per 1%)",
    )
    p_backtest.add_argument("--output", type=str, default=None, help="Write diagnostics JSON to this path")
    p_backtest.add_argument(
        "--write-config",
        type=str,
        default=None,
        help="Write blended weights to YAML file (does not overwrite unless same path specified)",
    )
    p_backtest.set_defaults(func=cmd_backtest)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
