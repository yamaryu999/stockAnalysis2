from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class NewsItem:
    id: str
    source: str
    title: str
    link: str
    published_at: datetime
    summary: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    raw: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # datetime serialize
        d["published_at"] = self.published_at.isoformat()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NewsItem":
        dt = d.get("published_at")
        if isinstance(dt, str):
            try:
                # fromisoformat supports offset (Python 3.11 ok)
                published = datetime.fromisoformat(dt)
            except Exception:
                from dateutil import parser

                published = parser.parse(dt)
        else:
            published = dt

        return NewsItem(
            id=d.get("id", ""),
            source=d.get("source", ""),
            title=d.get("title", ""),
            link=d.get("link", ""),
            published_at=published,
            summary=d.get("summary"),
            company_name=d.get("company_name"),
            ticker=d.get("ticker"),
            raw=d.get("raw") or {},
        )


@dataclass
class ScoredItem:
    news: NewsItem
    score: int
    reasons: List[str]
    hold: str  # "day" or "swing"
    tags: List[str]
