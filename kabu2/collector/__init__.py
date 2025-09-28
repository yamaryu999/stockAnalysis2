from __future__ import annotations

import asyncio
import inspect
import logging
from importlib import import_module
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Sequence, Union

from kabu2.models import NewsItem

from .rss import collect_many as collect_rss_many, parse_feed

try:  # pragma: no cover - compatibility fallback
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


logger = logging.getLogger(__name__)

CollectorFunc = Callable[[Mapping[str, Any]], Union[Iterable[NewsItem], Sequence[NewsItem]]]

__all__ = [
    "CollectorFunc",
    "collect_from_config",
    "load_collector",
    "collect_rss_many",
    "parse_feed",
]


def _load_entry_point(name: str) -> CollectorFunc | None:
    try:
        eps = importlib_metadata.entry_points()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        return None

    if hasattr(eps, "select"):
        matches = eps.select(group="kabu2.collectors", name=name)
    else:  # pragma: no cover - older importlib_metadata
        matches = [ep for ep in eps.get("kabu2.collectors", []) if ep.name == name]

    for ep in matches:
        try:
            func = ep.load()
        except Exception:
            logger.exception("failed to load collector entry point '%s'", name)
            return None
        if callable(func):
            return func  # type: ignore[return-value]
    return None


def _import_callable(path: str) -> CollectorFunc:
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, attr = path, "collect"
    module = import_module(module_name)
    func = getattr(module, attr)
    if not callable(func):  # pragma: no cover - defensive
        raise TypeError(f"collector '{path}' is not callable")
    return func  # type: ignore[return-value]


def load_collector(kind: str) -> CollectorFunc:
    if kind == "rss":
        return _collect_rss_plugin

    func = _load_entry_point(kind)
    if func is not None:
        return func

    try:
        return _import_callable(kind)
    except Exception:
        logger.exception("collector '%s' could not be imported", kind)
        raise


def _ensure_list(result: Iterable[NewsItem] | Sequence[NewsItem] | Any) -> List[NewsItem]:
    if inspect.isawaitable(result):
        try:
            return _resolve_awaitable(result)
        except Exception:
            logger.exception("collector coroutine failed")
            return []

    if result is None:
        return []

    if isinstance(result, list):
        return result

    try:
        return list(result)
    except TypeError:
        logger.warning("collector returned non-iterable result: %r", result)
        return []


def _resolve_awaitable(awaitable: Any) -> List[NewsItem]:
    try:
        return asyncio.run(awaitable)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(awaitable)
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        raise


def _collect_rss_plugin(params: Mapping[str, Any]) -> List[NewsItem]:
    feeds: MutableMapping[str, str] = {}
    raw_feeds = params.get("feeds")
    if isinstance(raw_feeds, Mapping):
        for key, url in raw_feeds.items():
            if not key or not url:
                continue
            feeds[str(key)] = str(url)
    url = params.get("url")
    if url:
        name = params.get("name") or params.get("source") or str(url)
        feeds[str(name)] = str(url)
    if not feeds:
        return []
    return collect_rss_many(dict(feeds))


def collect_from_config(
    cfg: Mapping[str, Any],
    *,
    selected: Iterable[str] | None = None,
) -> List[NewsItem]:
    selected_set = {str(s) for s in selected} if selected is not None else None
    results: List[NewsItem] = []

    feeds = cfg.get("feeds", {})
    feed_map: MutableMapping[str, str] = {}
    if isinstance(feeds, Mapping):
        for key, url in feeds.items():
            if not key or not url:
                continue
            key_str = str(key)
            if selected_set is not None and key_str not in selected_set:
                continue
            feed_map[key_str] = str(url)
    if feed_map:
        results.extend(collect_rss_many(dict(feed_map)))

    collectors = cfg.get("collectors", [])
    if isinstance(collectors, Sequence):
        for idx, spec in enumerate(collectors):
            if not isinstance(spec, Mapping):
                continue
            name = spec.get("name")
            if name is not None:
                name = str(name)
            if selected_set is not None and name is not None and name not in selected_set:
                continue
            kind = spec.get("type")
            if not kind:
                logger.warning("collector entry %s missing 'type'", idx)
                continue
            try:
                func = load_collector(str(kind))
            except Exception:
                continue
            params: dict[str, Any] = {k: v for k, v in spec.items() if k not in {"type"}}
            if name is not None:
                params.setdefault("name", name)
            try:
                out = func(params)
            except Exception:
                logger.exception("collector '%s' failed during execution", name or kind)
                continue
            results.extend(_ensure_list(out))

    seen: set[str] = set()
    deduped: List[NewsItem] = []
    for item in results:
        if not isinstance(item, NewsItem):
            logger.debug("collector yielded non-NewsItem object: %r", item)
            continue
        if item.id in seen:
            continue
        seen.add(item.id)
        deduped.append(item)
    return deduped
