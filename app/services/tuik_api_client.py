"""TÜİK (Turkish Statistical Institute) API client.

Provides lightweight access to official Turkish statistical bulletins
and data series for numeric claim verification.
https://data.tuik.gov.tr
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus
import httpx

from app.config import Config


class TuikApiClient:
    """Query TÜİK official statistics for claim verification."""

    SEARCH_URL = "https://data.tuik.gov.tr/Bulten/Search"
    API_BASE = "https://data.tuik.gov.tr"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 logosFactCheck/1.0"
    )

    def __init__(self, api_key: str | None = None) -> None:
        # TÜİK does not currently require a key for basic bulletin search,
        # but we accept one for future-proofing.
        self.api_key = api_key if api_key is not None else Config.TUIK_API_KEY

    @property
    def available(self) -> bool:
        # TÜİK search endpoint is generally accessible without a key;
        # if a key is provided we validate it looks real.
        if not self.api_key:
            return True
        return bool(not self.api_key.startswith("your-"))

    async def search_bulletins(self, query: str) -> list[dict[str, Any]]:
        """Search TÜİK bulletins for relevant statistical reports."""
        if not self.available:
            return []
        try:
            url = f"{self.SEARCH_URL}?query={quote_plus(query)}&locale=tr"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers={"User-Agent": self.USER_AGENT})
                response.raise_for_status()
                data = response.json()
            results = []
            for item in data.get("data", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("summary", "")[:380],
                    "source_domain": "tuik.gov.tr",
                    "source_type": "knowledge_base",
                    "published_at": item.get("releaseDate", ""),
                    "is_public_data_source": True,
                })
            return results
        except Exception:
            return []

    async def query_claim(self, claim_text: str) -> list[dict[str, Any]]:
        """Detect statistical patterns and query TÜİK for supporting bulletins."""
        patterns = self._detect_statistical_patterns(claim_text)
        if not patterns["is_statistical"]:
            return []
        results = []
        for suggestion in patterns["suggested_queries"][:3]:
            bulletins = await self.search_bulletins(suggestion)
            for b in bulletins:
                b["query_context"] = suggestion
                results.append(b)
        return results[:5]

    @staticmethod
    def _detect_statistical_patterns(text: str) -> dict[str, Any]:
        """Detect Turkish statistical claim patterns."""
        lower = text.lower()
        patterns = {
            "enflasyon": r"enflasyon\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "dolar_kuru": r"dolar\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?",
            "euro_kuru": r"euro\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?",
            "faiz": r"faiz\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "issizlik": r"işsizlik\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "buyume": r"büyüme\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "nufus": r"nüfus\s*:?\s*(\d+[,.]?\d*)\s*(?:milyon|bin)?",
        }
        found = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, lower)
            if match:
                found[key] = match.group(1)
        return {
            "is_statistical": len(found) > 0,
            "extracted_values": found,
            "suggested_queries": [
                f"TÜİK {k.replace('_', ' ')} {v}" for k, v in found.items()
            ],
        }
