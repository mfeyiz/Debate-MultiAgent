"""TCMB EVDS (Electronic Data Distribution System) API client.

Queries real-time exchange rates, interest rates, and other economic
indicators from the Central Bank of the Republic of Turkey (TCMB).
https://evds2.tcmb.gov.tr
"""

from __future__ import annotations

import re
from typing import Any
import httpx

from app.config import Config


class TcmbEvdsClient:
    """Fetch actual numeric data from TCMB EVDS for claim verification."""

    API_BASE = "https://evds2.tcmb.gov.tr/service/evds"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 logosFactCheck/1.0"
    )

    # Common EVDS series codes
    SERIES_CODES = {
        "dolar_kuru": "TP.DK.USD.S.YTL",
        "euro_kuru": "TP.DK.EUR.S.YTL",
        "sterlin_kuru": "TP.DK.GBP.S.YTL",
        "faiz_orani": "TP.FG.J0",
        "tufe_yillik": "TP.TG2.Y01",
        "tufe_aylik": "TP.TG2.A01",
        "uye_aylik": "TP.TG2.A02",
        "issizlik_orani": "TP.YISGUCU2.G8",
        "büyüme_orani": "TP.YISGUCU2.G9",
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else Config.TCMB_EVDS_API_KEY

    @property
    def available(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-"))

    async def get_latest_value(self, series_code: str) -> dict[str, Any] | None:
        """Fetch the most recent value for an EVDS series."""
        if not self.available:
            return None
        # EVDS typically returns the last N days; we ask for the last 5 business days
        end_date = self._today_str()
        start_date = self._days_ago_str(7)
        url = (
            f"{self.API_BASE}/series={series_code}"
            f"&startDate={start_date}"
            f"&endDate={end_date}"
            f"&type=json"
            f"&key={self.api_key}"
        )
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.get(url, headers={"User-Agent": self.USER_AGENT})
                response.raise_for_status()
                data = response.json()
        except Exception:
            return None

        items = data.get("items", [])
        if not items:
            return None
        # Take the latest non-null entry
        for item in reversed(items):
            value = item.get(series_code)
            if value is not None and value != "":
                return {
                    "series_code": series_code,
                    "date": item.get("Tarih", item.get("DATE", "")),
                    "value": self._parse_number(value),
                    "raw_value": str(value),
                }
        return None

    async def query_claim(self, claim_text: str) -> list[dict[str, Any]]:
        """Detect statistical patterns in a claim and verify against TCMB data."""
        results = []
        patterns = self._detect_patterns(claim_text)
        for pattern_key, claimed_value in patterns.items():
            series_code = self.SERIES_CODES.get(pattern_key)
            if not series_code:
                continue
            actual = await self.get_latest_value(series_code)
            if not actual:
                continue
            comparison = self._compare_values(claimed_value, actual["value"])
            results.append({
                "title": f"TCMB {self._series_label(pattern_key)}",
                "url": "https://evds2.tcmb.gov.tr",
                "snippet": (
                    f"İddia: {claimed_value} | "
                    f"TCMB ({actual['date']}): {actual['raw_value']} "
                    f"[{comparison['label']}]"
                ),
                "source_domain": "tcmb.gov.tr",
                "source_type": "knowledge_base",
                "is_public_data_source": True,
                "claimed_value": claimed_value,
                "actual_value": actual["value"],
                "actual_raw": actual["raw_value"],
                "actual_date": actual["date"],
                "comparison": comparison,
                "series_code": series_code,
            })
        return results

    @staticmethod
    def _detect_patterns(text: str) -> dict[str, str]:
        """Detect economic claim patterns in Turkish text."""
        lower = text.lower()
        patterns: dict[str, str] = {}
        mapping = [
            ("dolar_kuru", r"dolar\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?"),
            ("euro_kuru", r"euro\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?"),
            ("sterlin_kuru", r"sterlin\s*(?:kuru|fiyatı)?\s*:?\s*(\d+[,.]?\d*)\s*(?:TL|tl|lira)?"),
            ("faiz_orani", r"faiz\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)"),
            ("tufe_yillik", r"enflasyon.*?(?:%|yüzde)\s*(\d+[,.]?\d*)"),
            ("issizlik_orani", r"işsizlik\s*(?:oranı|yüzdesi)?\s*:?\s*(%?\s*\d+[,.]?\d*)"),
            ("büyüme_orani", r"(?:büyüme|büyüdü|ekonomi|ekonomisi|gsyh|gayri safi).*?(?:%|yüzde)\s*(\d+[,.]?\d*)"),
        ]
        for key, regex in mapping:
            match = re.search(regex, lower)
            if match:
                patterns[key] = match.group(1).replace(",", ".").replace("%", "").strip()
        return patterns

    @staticmethod
    def _compare_values(claimed_raw: str, actual: float | None) -> dict[str, Any]:
        """Compare claimed value with actual TCMB value."""
        if actual is None:
            return {"label": "Bilinmiyor", "diff_percent": None, "verdict": "none"}
        try:
            claimed = float(claimed_raw.replace(",", ".").replace("%", "").strip())
        except ValueError:
            return {"label": "Karşılaştırılamadı", "diff_percent": None, "verdict": "none"}
        if actual == 0:
            diff_pct = abs(claimed - actual) * 100
        else:
            diff_pct = abs((claimed - actual) / actual) * 100
        if diff_pct <= 3:
            return {"label": "Eşleşiyor", "diff_percent": round(diff_pct, 2), "verdict": "support"}
        if diff_pct <= 10:
            return {"label": "Yakın", "diff_percent": round(diff_pct, 2), "verdict": "none"}
        return {"label": "Çelişiyor", "diff_percent": round(diff_pct, 2), "verdict": "attack"}

    @staticmethod
    def _parse_number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ".").replace("%", "").strip())
        except ValueError:
            return None

    @staticmethod
    def _today_str() -> str:
        from datetime import date
        return date.today().strftime("%d-%m-%Y")

    @staticmethod
    def _days_ago_str(days: int) -> str:
        from datetime import date, timedelta
        return (date.today() - timedelta(days=days)).strftime("%d-%m-%Y")

    @staticmethod
    def _series_label(key: str) -> str:
        labels = {
            "dolar_kuru": "Dolar Kuru",
            "euro_kuru": "Euro Kuru",
            "sterlin_kuru": "Sterlin Kuru",
            "faiz_orani": "Faiz Oranı",
            "tufe_yillik": "Yıllık Enflasyon (TÜFE)",
            "tufe_aylik": "Aylık Enflasyon (TÜFE)",
            "issizlik_orani": "İşsizlik Oranı",
            "büyüme_orani": "Büyüme Oranı",
        }
        return labels.get(key, key)
