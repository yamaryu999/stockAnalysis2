from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from kabu2.config import get_slack_webhook, load_config
from kabu2.collector.rss import collect_many
from kabu2.extractor.rules import extract, load_name_map
from kabu2.models import NewsItem, ScoredItem
from kabu2.notifier.slack import post_slack
from kabu2.scorer.score import build_scored
from kabu2.storage.store import JsonlStore


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    feeds = cfg.get("feeds", {})
    # allow specifying subset via --feeds
    if args.feeds:
        subset = {}
        for key in args.feeds:
            if key in feeds:
                subset[key] = feeds[key]
        feeds = subset
    items = collect_many(feeds)
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
        print(f"[{s.score:>3}] {n.ticker or '-'} {n.company_name or '-'} | {n.title} | {', '.join(s.reasons)} | {n.published_at.isoformat()} | {n.link}")

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

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
