from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Generator, Iterable, List, Optional

from kabu2.models import NewsItem


class JsonlStore:
    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def append(self, items: Iterable[NewsItem]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False))
                f.write("\n")

    def read(self) -> Generator[NewsItem, None, None]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield NewsItem.from_dict(json.loads(line))

    def load(self) -> List[NewsItem]:
        return list(self.read() or [])

    def overwrite(self, items: Iterable[NewsItem]) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False))
                f.write("\n")
        os.replace(tmp_path, self.path)

    def dedupe(self) -> int:
        items = self.load()
        if not items:
            return 0
        deduped: dict[str, NewsItem] = {}
        for it in items:
            existing = deduped.get(it.id)
            if not existing:
                deduped[it.id] = it
                continue
            existing_dt = _safe_dt(existing.published_at)
            current_dt = _safe_dt(it.published_at)
            if current_dt and (not existing_dt or current_dt > existing_dt):
                deduped[it.id] = it
        deduped_items = sorted(deduped.values(), key=lambda x: _safe_dt(x.published_at) or datetime.min, reverse=True)
        removed = len(items) - len(deduped_items)
        if removed > 0:
            self.overwrite(deduped_items)
        return removed

    def compact(self, *, keep: Optional[int] = None) -> int:
        items = self.load()
        if not items:
            return 0
        items.sort(key=lambda x: _safe_dt(x.published_at) or datetime.min, reverse=True)
        original_len = len(items)
        if keep is not None and keep >= 0:
            items = items[:keep]
        if len(items) != original_len:
            self.overwrite(items)
        return original_len - len(items)


def _safe_dt(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    return None
