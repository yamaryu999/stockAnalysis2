#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List

from playwright.async_api import async_playwright, ConsoleMessage, Page, Response


@dataclass
class Findings:
    console_errors: List[str]
    console_warnings: List[str]
    page_errors: List[str]
    bad_responses: List[str]


async def run(url: str = "http://localhost:8502/") -> int:
    findings = Findings(console_errors=[], console_warnings=[], page_errors=[], bad_responses=[])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        def on_console(msg: ConsoleMessage) -> None:
            try:
                txt = msg.text()
            except Exception:
                txt = "<unreadable>"
            if msg.type == "error":
                findings.console_errors.append(txt)
            elif msg.type == "warning":
                findings.console_warnings.append(txt)

        async def on_response(resp: Response) -> None:
            try:
                if resp.status >= 400:
                    findings.bad_responses.append(f"{resp.status} {resp.url}")
            except Exception:
                pass

        page.on("console", on_console)
        page.on("response", lambda r: asyncio.create_task(on_response(r)))
        page.on(
            "pageerror",
            lambda e: findings.page_errors.append(str(e)),
        )

        await page.goto(url, wait_until="domcontentloaded")
        os.makedirs("artifacts", exist_ok=True)
        await page.screenshot(path="artifacts/ui_home.png", full_page=True)

        # Try to switch to Quick tab and run analysis
        try:
            await page.get_by_role("tab", name="ワンタッチ分析").click(timeout=5000)
        except Exception:
            # Fallback: click by text
            try:
                await page.get_by_text("ワンタッチ分析", exact=True).click(timeout=3000)
            except Exception:
                pass

        # Press Run if present
        try:
            await page.get_by_role("button", name="ワンクリックで実行").click(timeout=4000)
            # Give it some time to compute/fetch
            await page.wait_for_timeout(4000)
        except Exception:
            pass

        # Capture a second screenshot
        await page.screenshot(path="artifacts/ui_quick.png", full_page=True)

        await browser.close()

    # Print a human-friendly summary
    def header(t: str) -> None:
        print("\n== " + t)

    header("Console Errors")
    if findings.console_errors:
        for m in findings.console_errors:
            print("- ", m)
    else:
        print("(none)")

    header("Console Warnings")
    if findings.console_warnings:
        for m in findings.console_warnings:
            print("- ", m)
    else:
        print("(none)")

    header("Page Errors")
    if findings.page_errors:
        for m in findings.page_errors:
            print("- ", m)
    else:
        print("(none)")

    header("HTTP >=400 Responses")
    if findings.bad_responses:
        for m in findings.bad_responses:
            print("- ", m)
    else:
        print("(none)")

    print("\nArtifacts: artifacts/ui_home.png, artifacts/ui_quick.png")

    # Return non-zero if any hard errors were found
    return 1 if findings.console_errors or findings.page_errors else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))

