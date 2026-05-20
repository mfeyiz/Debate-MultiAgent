"""Turkish fact-check archive scrapers with async operations.

Provides direct search against teyit.org and dogrulukpayi.com
for pre-existing fact-checks on a given claim.
"""

from __future__ import annotations

import asyncio
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus
import httpx


class _TeyitResultParser(HTMLParser):
    """Extract article links and titles from teyit.org search results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._in_article = False
        self._in_title = False
        self._in_excerpt = False
        self._capture_text = False
        self._text_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        cls = attr_dict.get("class", "")
        href = attr_dict.get("href", "")

        if tag == "article" or "post" in cls:
            self._in_article = True
            self._current = {"url": "", "title": "", "excerpt": ""}

        if self._in_article and tag == "a" and href and not self._current["url"]:
            self._current["url"] = href if href.startswith("http") else f"https://teyit.org{href}"

        if self._in_article and tag in {"h2", "h3", "h4"}:
            self._in_title = True
            self._capture_text = True
            self._text_buffer = []

        if self._in_article and ("excerpt" in cls or "summary" in cls or "entry-summary" in cls):
            self._in_excerpt = True
            self._capture_text = True
            self._text_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3", "h4"} and self._in_title:
            self._in_title = False
            self._capture_text = False
            if self._current:
                self._current["title"] = " ".join(self._text_buffer).strip()
            self._text_buffer = []

        if self._in_excerpt and tag in {"div", "p", "span"}:
            self._in_excerpt = False
            self._capture_text = False
            if self._current:
                self._current["excerpt"] = " ".join(self._text_buffer).strip()
            self._text_buffer = []

        if tag == "article" and self._in_article:
            self._in_article = False
            if self._current and self._current.get("url"):
                self.results.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._capture_text:
            self._text_buffer.append(data.strip())


class _DogrulukPayiResultParser(HTMLParser):
    """Extract article links and titles from dogrulukpayi.com search results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._in_card = False
        self._in_title = False
        self._capture_text = False
        self._text_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        cls = attr_dict.get("class", "")
        href = attr_dict.get("href", "")

        if "card" in cls or "item" in cls or tag == "article":
            self._in_card = True
            self._current = {"url": "", "title": "", "excerpt": ""}

        if self._in_card and tag == "a" and href and not self._current["url"]:
            self._current["url"] = href if href.startswith("http") else f"https://dogrulukpayi.com{href}"

        if self._in_card and tag in {"h2", "h3", "h4", "a"} and ("title" in cls or not self._current["title"]):
            self._in_title = True
            self._capture_text = True
            self._text_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag in {"h2", "h3", "h4", "a"}:
            self._in_title = False
            self._capture_text = False
            if self._current:
                self._current["title"] = " ".join(self._text_buffer).strip()
            self._text_buffer = []

        if self._in_card and tag in {"div", "article", "li"}:
            self._in_card = False
            if self._current and self._current.get("url"):
                self.results.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._capture_text:
            self._text_buffer.append(data.strip())


class _MalumatFurusResultParser(HTMLParser):
    """Extract article links and titles from malumatfurus.org search results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._in_item = False
        self._in_title = False
        self._in_excerpt = False
        self._capture_text = False
        self._text_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        cls = attr_dict.get("class", "")
        href = attr_dict.get("href", "")

        if "post" in cls or "entry" in cls or tag == "article":
            self._in_item = True
            self._current = {"url": "", "title": "", "excerpt": ""}

        if self._in_item and tag == "a" and href and not self._current["url"]:
            self._current["url"] = href if href.startswith("http") else f"https://malumatfurus.org{href}"

        if self._in_item and tag in {"h2", "h3", "h4", "a"} and ("title" in cls or not self._current["title"]):
            self._in_title = True
            self._capture_text = True
            self._text_buffer = []

        if self._in_item and ("excerpt" in cls or "summary" in cls or "entry-summary" in cls):
            self._in_excerpt = True
            self._capture_text = True
            self._text_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag in {"h2", "h3", "h4", "a"}:
            self._in_title = False
            self._capture_text = False
            if self._current:
                self._current["title"] = " ".join(self._text_buffer).strip()
            self._text_buffer = []

        if self._in_excerpt and tag in {"div", "p", "span"}:
            self._in_excerpt = False
            self._capture_text = False
            if self._current:
                self._current["excerpt"] = " ".join(self._text_buffer).strip()
            self._text_buffer = []

        if self._in_item and tag in {"article", "div", "li"}:
            self._in_item = False
            if self._current and self._current.get("url"):
                self.results.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._capture_text:
            self._text_buffer.append(data.strip())


class TurkishArchiveScraper:
    """Search Turkish fact-check archives for pre-existing verifications."""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 logosFactCheck/1.0"
    )

    @classmethod
    async def search_teyit(cls, query: str) -> list[dict[str, Any]]:
        """Search teyit.org for articles related to the claim."""
        url = f"https://teyit.org/?s={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers={"User-Agent": cls.USER_AGENT})
                html = response.text
        except Exception:
            return []

        parser = _TeyitResultParser()
        parser.feed(html)
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["excerpt"][:280],
                "source": "teyit.org",
                "source_domain": "teyit.org",
            }
            for r in parser.results[:3]
            if r.get("url")
        ]

    @classmethod
    async def search_dogrulukpayi(cls, query: str) -> list[dict[str, Any]]:
        """Search dogrulukpayi.com for articles related to the claim."""
        url = f"https://dogrulukpayi.com/?s={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers={"User-Agent": cls.USER_AGENT})
                html = response.text
        except Exception:
            return []

        parser = _DogrulukPayiResultParser()
        parser.feed(html)
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["excerpt"][:280] if r.get("excerpt") else r["title"],
                "source": "dogrulukpayi.com",
                "source_domain": "dogrulukpayi.com",
            }
            for r in parser.results[:3]
            if r.get("url")
        ]

    @classmethod
    async def search_malumatfurus(cls, query: str) -> list[dict[str, Any]]:
        """Search malumatfurus.org for articles related to the claim."""
        url = f"https://malumatfurus.org/?s={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers={"User-Agent": cls.USER_AGENT})
                html = response.text
        except Exception:
            return []

        parser = _MalumatFurusResultParser()
        parser.feed(html)
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["excerpt"][:280] if r.get("excerpt") else r["title"],
                "source": "malumatfurus.org",
                "source_domain": "malumatfurus.org",
            }
            for r in parser.results[:3]
            if r.get("url")
        ]

    @classmethod
    async def search_all(cls, claim_text: str) -> list[dict[str, Any]]:
        """Search all Turkish archives concurrently and return normalized results."""
        salient = " ".join(
            w for w in re.findall(r"[\wğüşöçıİĞÜŞÖÇ%]+", claim_text)
            if len(w) > 3
        )[:80]
        if not salient:
            salient = claim_text[:80]

        # Execute searches concurrently
        tasks = [
            cls.search_teyit(salient),
            cls.search_dogrulukpayi(salient),
            cls.search_malumatfurus(salient),
        ]
        teyit, dp, mf = await asyncio.gather(*tasks)

        results = []
        results.extend(teyit)
        results.extend(dp)
        results.extend(mf)
        return results

    @classmethod
    def infer_verdict_from_title(cls, title: str) -> str | None:
        """Infer a preliminary verdict from Turkish archive article title keywords."""
        lower = title.lower()
        if any(w in lower for w in ("doğru", "doğrulandı", "doğruladı", "gerçek", "evet")):
            return "support"
        if any(w in lower for w in ("yanlış", "yalan", "çürüt", "doğru değil", "hayır", "sahte")):
            return "attack"
        if any(w in lower for w in ("karmaşa", "karışık", "net değil", "belirsiz")):
            return "neutral"
        return None
