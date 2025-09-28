from datetime import datetime, timezone

import streamlit as st

from kabu2.models import NewsItem, ScoredItem
from kabu2.ui.app import render_card


def _inject_min_styles() -> None:
    st.markdown(
        """
        <style>
        .card { position: relative; padding: 1.4rem 1.6rem 1.2rem 1.6rem; border-radius: 20px; background: rgba(12, 17, 30, 0.82); border: 1px solid rgba(148, 163, 184, 0.18); }
        .badge { position: absolute; top: -14px; left: 20px; background: linear-gradient(135deg, #22d3ee, #3b82f6); border-radius: 999px; padding: 0.35rem 0.78rem; font-weight: 700; color: #0b1120; }
        .card-body { display: flex; flex-direction: column; gap: 0.55rem; }
        .card-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; font-size: 0.82rem; color: rgba(226, 232, 240, 0.7); }
        .card-title { font-size: 1.1rem; font-weight: 600; line-height: 1.4; color: #f8fafc; }
        .card-title a { color: inherit; text-decoration: none; }
        .card-sub { display: flex; align-items: center; gap: 0.65rem; font-size: 0.9rem; color: rgba(199, 210, 254, 0.95); letter-spacing: 0.01em; }
        .sub-group { display: inline-flex; align-items: center; gap: 0.35rem; }
        .sub-label { display: inline-flex; align-items: center; padding: 0.1rem 0.45rem; border-radius: 6px; background: rgba(56, 189, 248, 0.14); border: 1px solid rgba(56, 189, 248, 0.28); color: #bae6fd; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; }
        .sub-code { font-weight: 700; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; letter-spacing: 0.03em; }
        .sub-divider { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: rgba(148, 163, 184, 0.5); }
        .sub-company { opacity: 0.9; }
        .card-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .tag-chip { display: inline-flex; align-items: center; padding: 0.25rem 0.6rem; border-radius: 999px; background: rgba(14, 165, 233, 0.14); border: 1px solid rgba(14, 165, 233, 0.18); color: #bae6fd; font-size: 0.78rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="kabu2 preview", page_icon="🔎", layout="wide")
    _inject_min_styles()

    n = NewsItem(
        id="sample-1",
        source="SampleFeed",
        title="テスト: サブ情報の視認性チェック",
        link="https://example.com/news/1",
        published_at=datetime.now(timezone.utc),
        summary="preview",
        company_name="トヨタ自動車",
        ticker="7203",
    )
    s = ScoredItem(news=n, score=12, reasons=["決算上振れ"], hold="day", tags=["決算", "需給"])
    st.write("プレビュー: カード1枚を描画しています。")
    render_card(s)


if __name__ == "__main__":
    main()

