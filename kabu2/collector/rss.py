from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qsl, urlparse, urlunparse, urlencode

# Optional imports: defer failures until actually used
try:  # pragma: no cover - import may be unavailable at runtime
    import feedparser as _feedparser  # type: ignore
except Exception:  # pragma: no cover - handled lazily in functions
    _feedparser = None  # type: ignore[assignment]
try:  # pragma: no cover
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from kabu2.models import NewsItem


logger = logging.getLogger(__name__)


def _to_datetime(entry) -> datetime:
    # feedparser returns parsed time as struct_time in entry.published_parsed
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if t is not None:
        dt = datetime(*t[:6], tzinfo=timezone.utc)
        return dt
    # fallback: try parsed date string via dateutil
    from dateutil import parser

    s = getattr(entry, "published", None) or getattr(entry, "updated", None) or ""
    return parser.parse(s)


USER_AGENT = "kabu2/0.2 (+github.com/yourname/kabu2)"
DEFAULT_TIMEOUT = 10.0
CACHE_PATH = Path.home() / ".cache" / "kabu2" / "feed_cache.json"


def _load_cache() -> Dict[str, Dict[str, str]]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: Dict[str, Dict[str, str]]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.debug("failed to persist feed cache", exc_info=True)


_TRACKING_PARAM_PREFIXES = ("utm_", "utm-", "mc_")
_TRACKING_PARAM_KEYS = {"fbclid", "gclid", "yclid", "_ga", "_gl", "igshid"}


def _normalize_component(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _canonicalize_url(url: str | None) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    if path and path != "/":
        path = path.rstrip("/")
    raw_params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_params = []
    for key, value in raw_params:
        lowered = key.lower()
        if lowered.startswith(_TRACKING_PARAM_PREFIXES) or lowered in _TRACKING_PARAM_KEYS:
            continue
        filtered_params.append((key, value))
    query = urlencode(filtered_params, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _canonicalize_guid(guid: str | None, link: str | None) -> str:
    candidate = guid or link or ""
    candidate = _normalize_component(candidate)
    if candidate.startswith("http"):
        return _canonicalize_url(candidate)
    return candidate


def _make_item_id(source: str, guid: str | None, link: str, title: str) -> str:
    base_parts = [source or "feed", _canonicalize_guid(guid, link), _canonicalize_url(link), _normalize_component(title)]
    raw = "|".join(base_parts)
    digest = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()
    return f"{source}:{digest[:20]}"


def parse_feed(url: str, source: str | None = None, *, data: bytes | None = None) -> List[NewsItem]:
    global _feedparser
    if _feedparser is None:
        try:
            import feedparser as _feedparser  # type: ignore
        except Exception as exc:  # pragma: no cover - provide helpful error
            raise ImportError(
                "feedparser is required to parse RSS/Atom feeds.\n"
                "Install dependencies (e.g., `pip install -r requirements.txt`)"
            ) from exc
    if data is None:
        d = _feedparser.parse(url)
    else:
        d = _feedparser.parse(data)
    items: List[NewsItem] = []
    src = source or d.feed.get("title", "feed")
    if getattr(d, "bozo", False):
        logger.debug("feedparser bozo detected for %s: %s", url, getattr(d, "bozo_exception", None))
    for e in d.entries:
        link = getattr(e, "link", "")
        guid = getattr(e, "id", None) or getattr(e, "guid", None) or link
        title = getattr(e, "title", "").strip()
        summary = getattr(e, "summary", None)
        published_at = _to_datetime(e)
        # best-effort company name guess (before extraction)
        company_name = None
        canonical_link = _canonicalize_url(link) or link
        item_id = _make_item_id(src, guid, canonical_link, title)
        items.append(
            NewsItem(
                id=item_id,
                source=src,
                title=title,
                link=canonical_link,
                published_at=published_at,
                summary=summary,
                company_name=company_name,
                ticker=None,
                raw={"entry": dict(e)},
            )
        )
    return items

async def _collect_many_async(feeds: Dict[str, str]) -> List[NewsItem]:
    cache = _load_cache()
    results: List[NewsItem] = []

    if httpx is None:  # pragma: no cover - provide helpful error only when used
        raise ImportError(
            "httpx is required for asynchronous feed collection.\n"
            "Install dependencies (e.g., `pip install -r requirements.txt`)."
        )

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:

        async def fetch(name: str, url: str) -> List[NewsItem]:
            headers = {"User-Agent": USER_AGENT}
            cached = cache.get(url)
            if cached:
                etag = cached.get("etag")
                last_modified = cached.get("last_modified")
                if etag:
                    headers["If-None-Match"] = etag
                if last_modified:
                    headers["If-Modified-Since"] = last_modified
            try:
                resp = await client.get(url, headers=headers)
            except Exception as exc:
                logger.warning("feed '%s' failed to fetch: %s", name, exc)
                return []
            if resp.status_code == 304:
                logger.debug("feed '%s' not modified", name)
                return []
            if resp.status_code >= 400:
                logger.warning("feed '%s' returned status %s", name, resp.status_code)
                return []
            etag = resp.headers.get("ETag")
            last_modified = resp.headers.get("Last-Modified")
            if etag or last_modified:
                cache[url] = {"etag": etag or "", "last_modified": last_modified or ""}
            return parse_feed(url, source=name, data=resp.content)

        tasks = [fetch(name, url) for name, url in feeds.items() if url]
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for outcome in gathered:
                if isinstance(outcome, Exception):
                    logger.error(
                        "unexpected error while parsing feed",
                        exc_info=(type(outcome), outcome, outcome.__traceback__),
                    )
                    continue
                results.extend(outcome)

    if cache:
        _save_cache(cache)

    # ensure unique IDs even if feeds overlap
    deduped: List[NewsItem] = []
    seen = set()
    for it in results:
        if it.id in seen:
            continue
        seen.add(it.id)
        deduped.append(it)
    return deduped


def collect_many(feeds: Dict[str, str]) -> List[NewsItem]:
    if not feeds:
        return []
    try:
        return asyncio.run(_collect_many_async(feeds))
    except RuntimeError as exc:  # running event loop (e.g. Streamlit)
        if "asyncio.run() cannot be called" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_collect_many_async(feeds))
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        raise
