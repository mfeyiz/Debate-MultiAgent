"""Turkish public data source query helpers.

Provides real API-backed verification via:
- TÜİK bulletins
- TCMB EVDS (actual numeric comparison)
- Resmi Gazete legislation archive
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
import httpx

from app.services.tcmb_evds_client import TcmbEvdsClient
from app.services.tuik_api_client import TuikApiClient


class PublicDataSourceClient:
    """Query Turkish public data sources for claim verification."""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 logosFactCheck/1.0"
    )

    # --- Resmi Gazete ---
    RESMI_GAZETE_SEARCH = "https://www.resmigazete.gov.tr/eskiler"

    @classmethod
    async def query_tuik(cls, query: str) -> list[dict[str, Any]]:
        """Search TÜİK bulletins for statistical data matching the query."""
        return await TuikApiClient().search_bulletins(query)

    @classmethod
    async def query_tcmb_evds(cls, claim_text: str) -> list[dict[str, Any]]:
        """Query TCMB EVDS for actual numeric data and compare with claim."""
        return await TcmbEvdsClient().query_claim(claim_text)

    @classmethod
    async def query_resmi_gazete(cls, query: str) -> list[dict[str, Any]]:
        """Search Resmi Gazete for legislation mentions."""
        try:
            url = f"https://www.resmigazete.gov.tr/arsiv?search={httpx.URLExtractedSource}" if False else f"https://www.resmigazete.gov.tr/arsiv?search={query}"
            # Clean url or quote
            import urllib.parse
            url = f"https://www.resmigazete.gov.tr/arsiv?search={urllib.parse.quote(query)}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers={"User-Agent": cls.USER_AGENT})
                response.raise_for_status()
                html = response.text

            # Extract result links
            links = re.findall(
                r'href=["\']([^"\']*(?:eskiler/\d{4}/\d{2}|anasaya\.aspx)[^"\']*)["\']',
                html,
            )
            results = []
            for link in links[:2]:
                full = link if link.startswith("http") else f"https://www.resmigazete.gov.tr/{link.lstrip('/')}"
                results.append({
                    "title": "Resmi Gazete",
                    "url": full,
                    "snippet": f"Resmi Gazete arşivi: {query[:100]}",
                    "source_domain": "resmigazete.gov.tr",
                    "source_type": "knowledge_base",
                    "is_public_data_source": True,
                })
            return results
        except Exception:
            return []

    @classmethod
    def detect_statistical_patterns(cls, text: str) -> dict[str, Any]:
        """Detect statistical claim patterns in Turkish text."""
        patterns = {
            "enflasyon": r"enflasyon\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "dolar_kuru": r"dolar\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?",
            "euro_kuru": r"euro\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?",
            "faiz": r"faiz\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "enflasyon_yuzde": r"enflasyon.*?(?:%|yüzde)\s*(\d+[,.]?\d*)",
            "issizlik": r"işsizlik\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)",
            "büyüme": r"(?:büyüme|ekonomi|ekonomisi|gsyh|gayri safi).*(?:%|yüzde)\s*(\d+[,.]?\d*)",
            "nufus": r"nüfus\s*:?\s*(\d+[,.]?\d*)\s*(?:milyon|bin)?",
        }

        found = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                found[key] = match.group(1)

        return {
            "is_statistical": len(found) > 0,
            "extracted_values": found,
            "suggested_queries": [
                f"TÜİK {k.replace('_', ' ')} {v}" for k, v in found.items()
            ],
        }

    @classmethod
    async def query_all_for_claim(cls, claim_text: str) -> list[dict[str, Any]]:
        """Query all public data sources for a claim with numeric verification."""
        stats = cls.detect_statistical_patterns(claim_text)
        if not stats["is_statistical"]:
            return []

        results = []
        # Query TCMB EVDS concurrently with TÜİK and Resmi Gazete
        tasks = []
        
        # 1. TCMB EVDS (highest priority)
        tcmb_task = cls.query_tcmb_evds(claim_text)
        tasks.append(tcmb_task)

        # 2. TÜİK bulletins
        for query in stats["suggested_queries"][:2]:
            tasks.append(cls.query_tuik(query))

        # 3. Resmi Gazete
        for query in stats["suggested_queries"][:1]:
            tasks.append(cls.query_resmi_gazete(query))

        query_results = await asyncio.gather(*tasks)
        for qr in query_results:
            results.extend(qr)

        return results[:8]
