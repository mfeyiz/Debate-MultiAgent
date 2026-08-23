"""Google Fact Check Tools API client.

Queries Google's global fact-check database for pre-existing claim reviews.
https://developers.google.com/fact-check/tools/api
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus
import httpx

from app.config import Config


class GoogleFactCheckClient:
    """Search Google's Fact Check Tools API for verified claims."""

    API_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 logosFactCheck/1.0"
    )

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else Config.GOOGLE_FACTCHECK_API_KEY

    @property
    def available(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-"))

    async def search(self, query: str, language_code: str = "tr", max_age_days: int = 365) -> list[dict[str, Any]]:
        """Search Google Fact Check for a claim and return normalized results."""
        if not self.available:
            return []

        url = (
            f"{self.API_BASE}?"
            f"query={quote_plus(query)}"
            f"&languageCode={language_code}"
            f"&maxAgeDays={max_age_days}"
            f"&pageSize=10"
            f"&key={self.api_key}"
        )
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.get(url, headers={"User-Agent": self.USER_AGENT})
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        results = []
        for claim in data.get("claims", []):
            claim_text = claim.get("text", "")
            claim_date = claim.get("claimDate", "")
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher", {})
                site = publisher.get("site", "")
                title = review.get("title", "")
                url_review = review.get("url", "")
                rating = review.get("textualRating", "")
                review_date = review.get("reviewDate", "")

                inferred = self._infer_verdict_from_rating(rating, title)
                results.append({
                    "title": title or claim_text,
                    "url": url_review,
                    "snippet": f"{rating} — {claim_text}"[:500],
                    "source": site or "Google Fact Check",
                    "source_domain": site or "factchecktools.googleapis.com",
                    "rating": rating,
                    "review_date": review_date or claim_date,
                    "inferred_verdict": inferred,
                    "is_google_factcheck": True,
                })
        return results

    @staticmethod
    def _infer_verdict_from_rating(rating: str, title: str) -> str | None:
        """Infer a verdict from the textual rating or title keywords."""
        text = f"{rating} {title}".lower()
        support_words = ("doğru", "doğrulandı", "true", "correct", "gerçek", "evet", "onaylandı")
        attack_words = ("yanlış", "yalan", "false", "incorrect", "çürütüldü", "sahte", "hayır", "misleading")
        if any(w in text for w in support_words):
            return "support"
        if any(w in text for w in attack_words):
            return "attack"
        return "none"
