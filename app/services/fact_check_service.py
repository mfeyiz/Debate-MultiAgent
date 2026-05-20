"""Fact-check article analysis service.

This module turns a pasted article or URL into a ModernBERT-centered
claim/evidence/relation map, with optional Tavily-backed source retrieval,
Turkish archive integration, public data source queries, credibility scoring,
and advanced manipulation detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from html.parser import HTMLParser
import json
import re
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import Config
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import FactCheckRun, FactClaim, FactEvidence, FactRelation
from app.services.bert_service import ArgumentComponent, ModernBERTPipeline
from app.services.turkish_archive_scraper import TurkishArchiveScraper
from app.services.public_data_client import PublicDataSourceClient
from app.services.credibility_scorer import CredibilityScorer
from app.services.claim_analyzer import ClaimAnalyzer
from app.services.google_factcheck_client import GoogleFactCheckClient


@dataclass
class SearchResult:
    """Normalized web search result."""

    title: str
    url: str
    snippet: str
    score: float
    published_at: str | None = None


class _ArticleHTMLParser(HTMLParser):
    """Small stdlib-only article text extractor."""

    _ARTICLE_HINTS = {
        "article",
        "content",
        "news",
        "haber",
        "story",
        "detail",
        "post",
        "entry",
        "main",
    }
    _NOISE_HINTS = {
        "nav",
        "menu",
        "footer",
        "header",
        "sidebar",
        "widget",
        "advert",
        "ad-",
        "reklam",
        "promo",
        "social",
        "share",
        "comment",
        "breadcrumb",
        "related",
        "popular",
    }
    _BLOCK_TAGS = {"p", "h1", "h2", "blockquote"}
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._tag_stack: list[tuple[str, str]] = []
        self._article_depth = 0
        self._saw_article_container = False
        self._current_block_tag: str | None = None
        self._current_block_article = False
        self._current_block_chunks: list[str] = []
        self._blocks: list[tuple[str, bool, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID_TAGS:
            return
        attr_text = self._attr_text(attrs)
        self._tag_stack.append((tag, attr_text))
        if tag in {"script", "style", "noscript", "svg", "form", "button", "select"}:
            self._skip_depth += 1
        if tag in {"nav", "aside", "footer", "header"} or self._has_noise_hint(attr_text):
            self._skip_depth += 1
        if tag in {"article", "main", "section", "div"} and self._has_article_hint(attr_text, tag):
            self._article_depth += 1
            self._saw_article_container = True
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS and self._current_block_tag is None:
            self._current_block_tag = tag
            self._current_block_article = self._article_depth > 0 or tag == "h1"
            self._current_block_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID_TAGS:
            return
        if tag in self._BLOCK_TAGS and self._current_block_tag == tag:
            text = re.sub(r"\s+", " ", " ".join(self._current_block_chunks)).strip()
            if text:
                self._blocks.append((tag, self._current_block_article, text))
            self._current_block_tag = None
            self._current_block_article = False
            self._current_block_chunks = []

        attr_text = ""
        found = False
        while self._tag_stack:
            t, attrs_str = self._tag_stack.pop()
            if t == tag:
                attr_text = attrs_str
                found = True
                break

        if found:
            if tag in {"script", "style", "noscript", "svg", "form", "button", "select"} and self._skip_depth:
                self._skip_depth -= 1
            if (tag in {"nav", "aside", "footer", "header"} or self._has_noise_hint(attr_text)) and self._skip_depth:
                self._skip_depth -= 1
            if tag in {"article", "main", "section", "div"} and self._has_article_hint(attr_text, tag) and self._article_depth:
                self._article_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()
            return
        if self._current_block_tag:
            self._current_block_chunks.append(cleaned)

    @property
    def text(self) -> str:
        preferred = [
            text
            for tag, is_article, text in self._blocks
            if is_article and self._is_useful_block(tag, text)
        ]
        fallback = [
            text
            for tag, _, text in self._blocks
            if self._is_useful_block(tag, text)
        ]
        blocks = preferred if preferred else fallback
        return re.sub(r"\n{3,}", "\n\n", "\n\n".join(self._dedupe(blocks))).strip()

    @classmethod
    def _has_article_hint(cls, attr_text: str, tag: str) -> bool:
        if tag in {"article", "main"}:
            return True
        return any(hint in attr_text for hint in cls._ARTICLE_HINTS)

    @classmethod
    def _has_noise_hint(cls, attr_text: str) -> bool:
        return any(hint in attr_text for hint in cls._NOISE_HINTS)

    @staticmethod
    def _attr_text(attrs: list[tuple[str, str | None]]) -> str:
        return " ".join(value or "" for key, value in attrs if key in {"class", "id", "role"}).lower()

    @classmethod
    def _is_useful_block(cls, tag: str, text: str) -> bool:
        stripped = text.strip()
        if tag != "h1" and len(stripped) < 55:
            return False
        if len(stripped.split()) < (3 if tag == "h1" else 8):
            return False
        if cls._is_boilerplate(stripped):
            return False
        return True

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        lower = text.lower()
        boilerplate_terms = [
            "çerez",
            "cookie",
            "abone ol",
            "giriş yap",
            "son dakika",
            "sıradaki haber",
            "en çok okunan",
            "sosyal medyada",
            "paylaş",
            "yorumlar",
            "reklam",
            "tüm hakları saklıdır",
        ]
        if any(term in lower for term in boilerplate_terms):
            return True
        market_terms = [
            "dolar",
            "euro",
            "sterlin",
            "altın",
            "borsa",
            "bitcoin",
            "kripto",
            "piyasa",
            "kur",
        ]
        if len(text) < 140 and any(term in lower for term in market_terms):
            has_sentence = bool(re.search(r"[.!?]\s+[A-ZÇĞİÖŞÜ]", text))
            many_numbers = len(re.findall(r"\d+[,.)]?\d*", text)) >= 2
            if not has_sentence or many_numbers:
                return True
        return False

    @staticmethod
    def _dedupe(blocks: list[str]) -> list[str]:
        seen: set[str] = set()
        unique = []
        for block in blocks:
            key = re.sub(r"\W+", "", block.lower())[:160]
            if key in seen:
                continue
            seen.add(key)
            unique.append(block)
        return unique


class TavilySearchClient:
    """Tiny Tavily /search HTTP client using the standard library."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else Config.TAVILY_API_KEY
        self.base_url = (base_url or Config.TAVILY_BASE_URL).rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-"))

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Search Tavily and normalize result objects."""
        if not self.available:
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results or Config.FACT_CHECK_SEARCH_RESULTS,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            f"{self.base_url}/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []

        results = []
        for item in data.get("results", []):
            snippet = item.get("content") or item.get("snippet") or ""
            url = item.get("url") or ""
            if not snippet.strip() and not url:
                continue
            results.append(
                SearchResult(
                    title=item.get("title") or self._domain(url),
                    url=url,
                    snippet=snippet.strip(),
                    score=float(item.get("score") or 0.0),
                    published_at=item.get("published_date"),
                )
            )
        return results

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc.replace("www.", "")


class FactCheckService:
    """Runs article fact-check analysis with ModernBERT as the core signal."""

    RELATION_THRESHOLD = 0.60
    ATTACK_THRESHOLD = 0.70
    MAX_SOURCE_COMPONENTS = 24

    _bert_lock = threading.Lock()
    _bert_instance: ModernBERTPipeline | None = None

    def __init__(self, search_client: TavilySearchClient | None = None) -> None:
        self.search_client = search_client or TavilySearchClient()
        self.google_factcheck = GoogleFactCheckClient()
        self._llm_model = None
        if Config.OPENROUTER_API_KEY and not Config.OPENROUTER_API_KEY.startswith("your-"):
            provider = OpenAIProvider(
                base_url=Config.OPENROUTER_BASE_URL,
                api_key=Config.OPENROUTER_API_KEY,
            )
            self._llm_model = OpenAIChatModel(Config.DEFAULT_MODEL, provider=provider)

    @property
    def bert(self) -> ModernBERTPipeline:
        """Load ModernBERT lazily for fact-check analysis."""
        if self.__class__._bert_instance is None:
            with self.__class__._bert_lock:
                if self.__class__._bert_instance is None:
                    self.__class__._bert_instance = ModernBERTPipeline()
        return self.__class__._bert_instance

    async def analyze(self, db: AsyncSession, text: str = "", url: str = "") -> dict:
        """Create and complete a fact-check run asynchronously."""
        article_text = text.strip()
        input_type = "text"
        title = None
        source_metadata: dict[str, Any] = {
            "search_provider": "tavily" if self.search_client.available else "none",
            "knowledge_base": {
                "mode": "web_rag",
                "trusted_domains": Config.FACT_CHECK_TRUSTED_DOMAINS,
                "fact_check_archives": Config.FACT_CHECK_ARCHIVE_DOMAINS,
            },
            "relation_threshold": self.RELATION_THRESHOLD,
            "attack_threshold": self.ATTACK_THRESHOLD,
        }

        if url.strip():
            input_type = "url"
            fetched = await asyncio.to_thread(self._fetch_article, url.strip())
            if not article_text:
                article_text = fetched["text"]
            title = fetched["title"]
            source_metadata["url_fetch"] = fetched["status"]
            if fetched.get("error"):
                source_metadata["url_error"] = fetched["error"]

        if len(article_text.split()) < 8:
            raise ValueError(
                "Analiz için yeterli haber metni çıkarılamadı. Lütfen URL yerine haber metnini yapıştırın."
            )

        # Balanced reporting analysis (article-level)
        balanced_reporting = ClaimAnalyzer.analyze_balanced_reporting(article_text, [])

        run = FactCheckRun(
            input_type=input_type,
            url=url.strip() or None,
            title=title or self._infer_title(article_text),
            raw_text=article_text,
            status="running",
            source_metadata_json=json.dumps(source_metadata),
            balanced_reporting_json=json.dumps(balanced_reporting),
        )
        db.add(run)
        await db.flush()

        try:
            components = await asyncio.to_thread(self._extract_article_components, article_text)
            # Enrich components with analysis
            components = self._enrich_components(components, article_text)
            db_components = await self._persist_components(db, run, components)
            await db.flush()
            await self._classify_internal_relations(db, run, db_components)
            await db.flush()
            # Turkish archive search
            await self._search_turkish_archives(db, run, db_components)
            await db.flush()
            # Google Fact Check search
            await self._search_google_factcheck(db, run, db_components)
            await db.flush()
            # Public data source queries
            await self._query_public_data_sources(db, run, db_components)
            await db.flush()
            # Tavily/web search
            await self._retrieve_and_classify_sources(db, run, db_components)
            await db.flush()
            await self._finalize_verdicts(db, run)
            # Build manipulation findings
            manipulation_findings = self._build_manipulation_findings(db_components)
            run.manipulation_findings_json = json.dumps(manipulation_findings)
            run.status = "completed"
            run.completed_at = datetime.datetime.utcnow()
            claims_only = [c for c in db_components if c.component_type == "claim"]
            run.summary = self._build_run_summary(claims_only)
            # Generate export token
            run.export_token = self._generate_export_token(run.id)
            await db.commit()
        except Exception:
            run.status = "failed"
            run.completed_at = datetime.datetime.utcnow()
            await db.commit()
            raise

        return await self.get_run(db, run.id)

    async def refresh(self, db: AsyncSession, run_id: int) -> dict:
        """Re-run analysis for the same article text and URL."""
        run = await db.get(FactCheckRun, run_id)
        if not run:
            raise ValueError("Teyit analizi bulunamadı")
        return await self.analyze(db, text=run.raw_text, url=run.url or "")

    async def get_run(self, db: AsyncSession, run_id: int) -> dict:
        """Return a serialized fact-check run."""
        run = await db.get(FactCheckRun, run_id)
        if not run:
            raise ValueError("Teyit analizi bulunamadı")

        stmt_claims = select(FactClaim).filter_by(run_id=run.id).order_by(FactClaim.start_idx.asc())
        result_claims = await db.execute(stmt_claims)
        claims = result_claims.scalars().all()

        stmt_evidence = select(FactEvidence).filter_by(run_id=run.id).order_by(FactEvidence.score.desc())
        result_evidence = await db.execute(stmt_evidence)
        evidence = result_evidence.scalars().all()

        stmt_relations = select(FactRelation).filter_by(run_id=run.id).order_by(FactRelation.confidence.desc())
        result_relations = await db.execute(stmt_relations)
        relations = result_relations.scalars().all()

        visible_relations = [r for r in relations if self._is_visible_relation(r)]

        return {
            "run": run.to_dict(),
            "claims": [claim.to_dict() for claim in claims],
            "evidence": [item.to_dict() for item in evidence],
            "relations": [relation.to_dict() for relation in visible_relations],
            "annotations": self._build_annotations(claims, visible_relations),
            "graph": self._build_graph(claims, evidence, visible_relations),
            "findings": self._build_findings(claims, visible_relations, run),
            "impact": self._build_impact(claims, evidence, visible_relations, run),
        }

    async def get_run_by_token(self, db: AsyncSession, token: str) -> dict:
        """Return a fact-check run by its shareable export token."""
        stmt = select(FactCheckRun).filter_by(export_token=token).limit(1)
        result = await db.execute(stmt)
        run = result.scalars().first()
        if not run:
            raise ValueError("Teyit analizi bulunamadı")
        return await self.get_run(db, run.id)

    async def export_json(self, db: AsyncSession, run_id: int) -> dict:
        """Export a full fact-check run as a structured JSON object."""
        return await self.get_run(db, run_id)

    def _generate_export_token(self, run_id: int) -> str:
        import secrets
        return secrets.token_urlsafe(32)[:48]

    def _fetch_article(self, url: str) -> dict:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 logosFactCheck/1.0"
                )
            },
        )
        try:
            with urlopen(request, timeout=12) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            return {"status": "failed", "title": None, "text": "", "error": str(exc)}

        parser = _ArticleHTMLParser()
        parser.feed(html)
        return {
            "status": "ok" if parser.text else "empty",
            "title": parser.title or None,
            "text": parser.text,
            "error": None if parser.text else "Sayfadan okunabilir metin çıkarılamadı.",
        }

    def _extract_article_components(self, text: str):
        components = self.bert.extract_components(text, default_type="claim")
        meaningful = [
            component
            for component in components
            if len(component.text.split()) >= 4 and len(component.text) >= 24
        ]
        if meaningful:
            has_claim = any(c.component_type == "claim" for c in meaningful)
            if not has_claim:
                meaningful[0].component_type = "claim"
            return meaningful[: self.MAX_SOURCE_COMPONENTS]
        return self._fallback_components(text)

    def _enrich_components(self, components: list[ArgumentComponent], article_text: str) -> list[ArgumentComponent]:
        """Enrich components with claim hash, manipulation score, statistical flag, and quote data."""
        for component in components:
            # Manipulation analysis
            manipulation = ClaimAnalyzer.analyze_manipulation(component.text)
            # Statistical detection
            statistical = ClaimAnalyzer.detect_statistical_claim(component.text)
            # Quote detection
            quotes = ClaimAnalyzer.detect_quotes(component.text)
            # Store enriched data as custom attributes on the component
            component._claim_hash = ClaimAnalyzer.compute_claim_hash(component.text)
            component._embedding_hash = ClaimAnalyzer.compute_embedding_hash(component.text)
            component._manipulation_score = manipulation["score"]
            component._is_statistical = statistical["is_statistical"]
            component._statistical_data = statistical
            component._quote_data = quotes
        return components

    @staticmethod
    def _fallback_components(text: str) -> list[ArgumentComponent]:
        """Create coarse claim candidates when the component model extracts no span."""
        components = []
        cursor = 0
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 24 or len(sentence.split()) < 4:
                cursor += len(sentence) + 1
                continue
            start = text.find(sentence, cursor)
            if start < 0:
                start = text.find(sentence)
            end = start + len(sentence)
            components.append(
                ArgumentComponent(
                    text=sentence,
                    component_type="claim",
                    start_idx=max(0, start),
                    end_idx=max(0, end),
                    confidence=0.58,
                )
            )
            cursor = end
            if len(components) >= 5:
                break
        return components

    @staticmethod
    async def _persist_components(db: AsyncSession, run: FactCheckRun, components) -> list[FactClaim]:
        persisted = []
        for component in components:
            row = FactClaim(
                run_id=run.id,
                component_type=component.component_type,
                text=component.text,
                start_idx=component.start_idx,
                end_idx=component.end_idx,
                confidence=component.confidence,
                claim_type="factual" if component.component_type == "claim" else "article_evidence",
                verdict_status=(
                    "Kaynak bulunamadı" if component.component_type == "claim" else "Metin içi destek adayı"
                ),
                claim_hash=getattr(component, "_claim_hash", None),
                embedding_hash=getattr(component, "_embedding_hash", None),
                manipulation_score=getattr(component, "_manipulation_score", None),
                is_statistical=getattr(component, "_is_statistical", False),
                statistical_data_json=json.dumps(getattr(component, "_statistical_data", {})),
                quote_data_json=json.dumps(getattr(component, "_quote_data", [])),
            )
            db.add(row)
            persisted.append(row)
        return persisted

    async def _classify_internal_relations(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        claims = [component for component in components if component.component_type == "claim"]
        sources = components[:]
        for target in claims:
            for source in sources:
                if source.id == target.id:
                    continue
                if source.start_idx > target.start_idx and source.component_type == "claim":
                    continue
                relation_type, confidence, probabilities = await asyncio.to_thread(
                    self.bert.classify_relation,
                    target.text,
                    source.text,
                )
                if relation_type == "neutral" or confidence < self.RELATION_THRESHOLD:
                    continue
                db.add(
                    FactRelation(
                        run_id=run.id,
                        source_claim_id=source.id,
                        target_claim_id=target.id,
                        relation_scope="internal",
                        relation_type=relation_type,
                        confidence=confidence,
                        probabilities_json=json.dumps(probabilities),
                    )
                )

    async def _search_turkish_archives(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        """Search Turkish fact-check archives for pre-existing verifications."""
        claims = [c for c in components if c.component_type == "claim"]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            archive_results = await TurkishArchiveScraper.search_all(claim.text)
            for result in archive_results:
                # Infer preliminary verdict from title
                inferred = TurkishArchiveScraper.infer_verdict_from_title(result["title"])
                # Add as evidence
                evidence = FactEvidence(
                    run_id=run.id,
                    claim_id=claim.id,
                    search_query=f"turkish_archive: {result['source']}",
                    url=result["url"],
                    title=result["title"][:500],
                    source_domain=result["source_domain"],
                    snippet=result["snippet"],
                    score=0.85,
                    is_turkish_archive=True,
                    archive_match_claim_text=claim.text,
                )
                db.add(evidence)
                await db.flush()
                # Add relation based on inferred verdict
                if inferred:
                    db.add(
                        FactRelation(
                            run_id=run.id,
                            evidence_id=evidence.id,
                            target_claim_id=claim.id,
                            relation_scope="external",
                            relation_type=inferred,
                            confidence=0.75,
                            probabilities_json=json.dumps({inferred: 0.75, "neutral": 0.25}),
                        )
                    )

    async def _search_google_factcheck(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        """Search Google Fact Check Tools API for pre-existing verifications."""
        if not self.google_factcheck.available:
            return
        claims = [c for c in components if c.component_type == "claim"]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            gc_results = await self.google_factcheck.search(claim.text, language_code="tr")
            for result in gc_results:
                inferred = result.get("inferred_verdict")
                evidence = FactEvidence(
                    run_id=run.id,
                    claim_id=claim.id,
                    search_query=f"google_factcheck: {result.get('source', '')}",
                    url=result["url"],
                    title=result["title"][:500],
                    source_domain=result["source_domain"],
                    snippet=result["snippet"],
                    score=0.88,
                    is_turkish_archive=False,
                    archive_match_claim_text=claim.text,
                )
                db.add(evidence)
                await db.flush()
                if inferred and inferred in {"support", "attack"}:
                    db.add(
                        FactRelation(
                            run_id=run.id,
                            evidence_id=evidence.id,
                            target_claim_id=claim.id,
                            relation_scope="external",
                            relation_type=inferred,
                            confidence=0.72,
                            probabilities_json=json.dumps({inferred: 0.72, "neutral": 0.28}),
                        )
                    )

    async def _query_public_data_sources(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        """Query Turkish public data sources for statistical claims."""
        claims = [c for c in components if c.component_type == "claim" and c.is_statistical]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            public_results = await PublicDataSourceClient.query_all_for_claim(claim.text)
            for result in public_results:
                # Determine relation from numeric comparison if available
                relation_type = "neutral"
                confidence = 0.90
                comparison = result.get("comparison")
                if comparison:
                    relation_type = comparison.get("verdict", "neutral")
                    confidence = 0.92 if relation_type in {"support", "attack"} else 0.75

                evidence = FactEvidence(
                    run_id=run.id,
                    claim_id=claim.id,
                    search_query=f"public_data: {result.get('source_domain', '')}",
                    url=result["url"],
                    title=result["title"][:500],
                    source_domain=source_domain if (source_domain := result.get("source_domain")) else "",
                    snippet=result["snippet"],
                    score=0.90,
                    is_public_data_source=True,
                )
                db.add(evidence)
                await db.flush()
                if relation_type in {"support", "attack"}:
                    db.add(
                        FactRelation(
                            run_id=run.id,
                            evidence_id=evidence.id,
                            target_claim_id=claim.id,
                            relation_scope="external",
                            relation_type=relation_type,
                            confidence=confidence,
                            probabilities_json=json.dumps({relation_type: confidence, "neutral": round(1 - confidence, 2)}),
                        )
                    )

    async def _retrieve_and_classify_sources(
        self,
        db: AsyncSession,
        run: FactCheckRun,
        components: list[FactClaim],
    ) -> None:
        claims = [component for component in components if component.component_type == "claim"]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            seen_urls: set[str] = set()
            query_cache: dict[str, list[SearchResult]] = {}
            for query, query_scope in self._query_plan_for_claim(claim.text):
                if query not in query_cache:
                    query_cache[query] = await asyncio.to_thread(self.search_client.search, query, Config.FACT_CHECK_SEARCH_RESULTS)
                for result in query_cache[query]:
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)
                    source_domain = urlparse(result.url).netloc.replace("www.", "")
                    # Get credibility score
                    credibility = CredibilityScorer.score_domain(source_domain)
                    evidence = FactEvidence(
                        run_id=run.id,
                        claim_id=claim.id,
                        search_query=f"{query_scope}: {query}",
                        url=result.url,
                        title=result.title[:500],
                        source_domain=source_domain,
                        snippet=result.snippet,
                        published_at=result.published_at,
                        score=result.score,
                        credibility_score=credibility.get("credibility_score"),
                        source_bias=credibility.get("bias"),
                    )
                    db.add(evidence)
                    await db.flush()
                    if result.snippet.strip():
                        relation_type, confidence, probabilities = await asyncio.to_thread(
                            self.bert.classify_relation,
                            claim.text,
                            result.snippet,
                        )
                        db.add(
                            FactRelation(
                                run_id=run.id,
                                evidence_id=evidence.id,
                                target_claim_id=claim.id,
                                relation_scope="external",
                                relation_type=relation_type,
                                confidence=confidence,
                                probabilities_json=json.dumps(probabilities),
                            )
                        )

    async def _check_claim_cache(self, db: AsyncSession, claim: FactClaim) -> dict[str, Any] | None:
        """Check if a semantically similar claim was analyzed recently."""
        if not claim.embedding_hash:
            return None
        from datetime import timedelta
        cutoff = datetime.datetime.utcnow() - timedelta(days=Config.CLAIM_CACHE_TTL_DAYS)
        stmt = select(FactClaim).filter(
            FactClaim.embedding_hash == claim.embedding_hash,
            FactClaim.created_at >= cutoff,
            FactClaim.id != claim.id,
        ).order_by(FactClaim.created_at.desc()).limit(1)
        result = await db.execute(stmt)
        cached = result.scalars().first()
        if cached:
            return {
                "cached_claim_id": cached.id,
                "verdict_status": cached.verdict_status,
                "explanation": cached.explanation,
            }
        return None

    async def _finalize_verdicts(self, db: AsyncSession, run: FactCheckRun) -> None:
        stmt_claims = select(FactClaim).filter_by(run_id=run.id, component_type="claim")
        result_claims = await db.execute(stmt_claims)
        claims = result_claims.scalars().all()

        stmt_relations = select(FactRelation).filter_by(run_id=run.id)
        result_relations = await db.execute(stmt_relations)
        relations = result_relations.scalars().all()

        by_target: dict[int, list[FactRelation]] = {}
        for relation in relations:
            by_target.setdefault(relation.target_claim_id, []).append(relation)

        for claim in claims:
            # Check cache first
            cached = await self._check_claim_cache(db, claim)
            if cached:
                claim.verdict_status = f"[Önbellek] {cached['verdict_status']}"
                claim.explanation = cached["explanation"]
                continue

            claim_relations = by_target.get(claim.id, [])
            external = [r for r in claim_relations if r.relation_scope == "external"]
            internal = [r for r in claim_relations if r.relation_scope == "internal"]
            archive_relations = [
                r for r in external if r.evidence and self._source_category(r.evidence.source_domain) == "fact_check_archive"
            ]
            knowledge_relations = [
                r for r in external if r.evidence and self._source_category(r.evidence.source_domain) == "knowledge_base"
            ]
            turkish_archive_relations = [
                r for r in external if r.evidence and r.evidence.is_turkish_archive
            ]
            public_data_relations = [
                r for r in external if r.evidence and r.evidence.is_public_data_source
            ]

            # Google Fact Check relations
            google_fc_relations = [
                r for r in external
                if r.evidence and r.evidence.search_query and "google_factcheck:" in r.evidence.search_query
            ]
            # TCMB EVDS numeric relations (high confidence public data)
            tcmb_numeric_relations = [
                r for r in public_data_relations
                if r.evidence and r.evidence.source_domain == "tcmb.gov.tr" and r.relation_type in ("support", "attack")
            ]

            # Enhanced verdict cascade
            if any(r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD for r in internal):
                claim.verdict_status = "Metin içinde çelişkili"
            elif turkish_archive_relations:
                # Turkish archive takes precedence
                attack_archives = [r for r in turkish_archive_relations if r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD]
                support_archives = [r for r in turkish_archive_relations if r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD]
                if attack_archives:
                    claim.verdict_status = "Türk arşivinde çürütülmüş"
                elif support_archives:
                    claim.verdict_status = "Türk arşivinde doğrulanmış"
            elif google_fc_relations:
                # Google Fact Check results
                attack_gc = [r for r in google_fc_relations if r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD]
                support_gc = [r for r in google_fc_relations if r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD]
                if attack_gc:
                    claim.verdict_status = "Google Fact Check'te çürütülmüş"
                elif support_gc:
                    claim.verdict_status = "Google Fact Check'te doğrulanmış"
            elif any(r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD for r in archive_relations):
                claim.verdict_status = "Teyit arşiviyle çelişiyor"
            elif any(r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD for r in archive_relations):
                claim.verdict_status = "Teyit arşivinde destek var"
            elif tcmb_numeric_relations:
                # TCMB EVDS numeric verification (concrete data comparison)
                attack_tcmb = [r for r in tcmb_numeric_relations if r.relation_type == "attack"]
                support_tcmb = [r for r in tcmb_numeric_relations if r.relation_type == "support"]
                if attack_tcmb:
                    claim.verdict_status = "Kamu verisiyle çelişiyor (somut)"
                elif support_tcmb:
                    claim.verdict_status = "Kamu verisiyle destekleniyor (somut)"
            elif any(r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD for r in public_data_relations):
                claim.verdict_status = "Kamu verisiyle çelişiyor"
            elif any(r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD for r in public_data_relations):
                claim.verdict_status = "Kamu verisiyle destekleniyor"
            elif any(r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD for r in knowledge_relations):
                claim.verdict_status = "Bilgi tabanıyla çelişiyor"
            elif any(r.relation_type == "attack" and r.confidence >= self.ATTACK_THRESHOLD for r in external):
                claim.verdict_status = "Kaynaklarla çelişiyor"
            elif any(r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD for r in knowledge_relations):
                claim.verdict_status = "Bilgi tabanıyla destekleniyor"
            elif any(r.relation_type == "support" and r.confidence >= self.RELATION_THRESHOLD for r in external):
                claim.verdict_status = "Kaynaklarla destekleniyor"
            elif not external:
                claim.verdict_status = "Kaynak bulunamadı"
            elif not any(r.relation_type == "support" for r in internal):
                claim.verdict_status = "Metin içinde kanıtsız"
            else:
                claim.verdict_status = "Belirsiz"
            claim.explanation = await self._synthesize_explanation(claim, claim_relations)

    async def _synthesize_explanation(self, claim: FactClaim, relations: list[FactRelation]) -> str:
        top_relations = sorted(relations, key=lambda r: r.confidence, reverse=True)[:4]
        source_lines = []
        for relation in top_relations:
            if relation.evidence:
                source_lines.append(
                    f"- {relation.relation_type.upper()} %{relation.confidence * 100:.0f}: "
                    f"{relation.evidence.title} ({relation.evidence.source_domain})"
                )
            elif relation.source_claim:
                source_lines.append(
                    f"- Metin içi {relation.relation_type.upper()} %{relation.confidence * 100:.0f}: "
                    f"{relation.source_claim.text[:180]}"
                )

        if not source_lines:
            return (
                "ModernBERT bu iddia için görünür bir destek veya çelişki ilişkisi bulamadı; "
                "bu nedenle sonuç ihtiyatlı biçimde kaynak bulunamadı olarak tutuldu."
            )

        if not self._llm_model:
            return (
                f"ModernBERT bu iddiayı '{claim.verdict_status}' olarak işaretledi. "
                "Karar, görünür kaynak/metin ilişkilerindeki support ve attack skorlarına dayanır."
            )

        prompt = (
            "Aşağıdaki iddia için Türkçe, 2-3 cümlelik ihtiyatlı bir medya okuryazarlığı "
            "açıklaması yaz. Kesin doğru/yanlış deme; yalnız bulunan kaynak ve ModernBERT "
            "ilişki skorlarına dayan.\n\n"
            f"İddia: {claim.text}\n"
            f"Durum: {claim.verdict_status}\n"
            f"İlişkiler:\n{chr(10).join(source_lines)}"
        )
        try:
            agent = PydanticAgent(
                self._llm_model,
                system_prompt="Kaynaklı, temkinli fact-check açıklamaları yazan bir editörsün.",
                model_settings={"temperature": 0.2},
            )
            res = await agent.run(prompt)
            return str(res.output)
        except Exception:
            return (
                f"ModernBERT bu iddiayı '{claim.verdict_status}' olarak işaretledi. "
                "LLM sentezi üretilemediği için karar kısa model özetiyle gösteriliyor."
            )

    @staticmethod
    def _query_plan_for_claim(claim_text: str) -> list[tuple[str, str]]:
        normalized = re.sub(r"\s+", " ", claim_text).strip()
        quoted = f'"{normalized[:180]}"'
        words = re.findall(r"[\wğüşöçıİĞÜŞÖÇ%]+", normalized, flags=re.UNICODE)
        salient = " ".join([word for word in words if len(word) > 3][:12])
        source_terms = " ".join(
            [
                "teyit",
                "doğruluk payı",
                "factcheck",
                "politifact",
                "TÜİK",
                "Merkez Bankası",
                "WHO",
                "World Bank",
            ]
        )
        queries: list[tuple[str, str]] = [
            ("knowledge_base", f"{salient or normalized[:180]} {source_terms}"),
            ("exact_claim", quoted),
        ]
        return [
            (query, scope)
            for scope, query in queries[: max(1, Config.FACT_CHECK_MAX_QUERIES_PER_CLAIM)]
        ]

    @staticmethod
    def _infer_title(text: str) -> str:
        first_line = text.strip().splitlines()[0] if text.strip() else "Haber Analizi"
        return first_line[:120]

    @staticmethod
    def _is_visible_relation(relation: FactRelation) -> bool:
        if relation.relation_type not in {"support", "attack"}:
            return False
        if relation.relation_type == "attack":
            return relation.confidence >= FactCheckService.ATTACK_THRESHOLD
        return relation.confidence >= FactCheckService.RELATION_THRESHOLD

    def _build_annotations(
        self,
        claims: list[FactClaim],
        relations: list[FactRelation],
    ) -> list[dict]:
        relation_by_source: dict[int, FactRelation] = {}
        for relation in relations:
            if relation.source_claim_id:
                previous = relation_by_source.get(relation.source_claim_id)
                if not previous or relation.confidence > previous.confidence:
                    relation_by_source[relation.source_claim_id] = relation

        annotations = []
        for claim in claims:
            relation = relation_by_source.get(claim.id)
            annotations.append(
                {
                    "id": claim.id,
                    "component_type": claim.component_type,
                    "start": claim.start_idx,
                    "end": claim.end_idx,
                    "confidence": claim.confidence,
                    "verdict_status": claim.verdict_status,
                    "relation_type": relation.relation_type if relation else None,
                    "relation_confidence": relation.confidence if relation else None,
                    "target_claim_id": relation.target_claim_id if relation else None,
                    "is_statistical": claim.is_statistical,
                    "manipulation_score": claim.manipulation_score,
                }
            )
        return annotations

    @staticmethod
    def _build_graph(
        claims: list[FactClaim],
        evidence: list[FactEvidence],
        relations: list[FactRelation],
    ) -> dict:
        nodes = []
        for claim in claims:
            nodes.append(
                {
                    "data": {
                        "id": f"c-{claim.id}",
                        "label": "İddia" if claim.component_type == "claim" else "Metin Kanıtı",
                        "kind": claim.component_type,
                        "text": claim.text,
                        "verdict": claim.verdict_status,
                        "confidence": claim.confidence,
                        "is_statistical": claim.is_statistical,
                        "manipulation_score": claim.manipulation_score,
                    }
                }
            )
        for item in evidence[:24]:
            nodes.append(
                {
                    "data": {
                        "id": f"e-{item.id}",
                        "label": item.source_domain or "Kaynak",
                        "kind": "source",
                        "sourceType": item.to_dict().get("source_type", "web"),
                        "text": item.snippet,
                        "url": item.url,
                        "title": item.title,
                        "confidence": item.score,
                        "credibility_score": item.credibility_score,
                        "source_bias": item.source_bias,
                        "is_turkish_archive": item.is_turkish_archive,
                        "is_public_data_source": item.is_public_data_source,
                    }
                }
            )

        edges = []
        for relation in relations:
            source = (
                f"e-{relation.evidence_id}"
                if relation.evidence_id
                else f"c-{relation.source_claim_id}"
            )
            edges.append(
                {
                    "data": {
                        "id": f"r-{relation.id}",
                        "source": source,
                        "target": f"c-{relation.target_claim_id}",
                        "relationType": relation.relation_type,
                        "scope": relation.relation_scope,
                        "confidence": relation.confidence,
                        "label": relation.relation_type.upper(),
                    }
                }
            )
        if not edges:
            for claim in claims:
                if claim.component_type != "claim":
                    continue
                note_id = f"n-{claim.id}"
                nodes.append(
                    {
                        "data": {
                            "id": note_id,
                            "label": "Analiz Notu",
                            "kind": "finding",
                            "text": claim.verdict_status,
                            "confidence": claim.confidence,
                        }
                    }
                )
                edges.append(
                    {
                        "data": {
                            "id": f"note-{claim.id}",
                            "source": note_id,
                            "target": f"c-{claim.id}",
                            "relationType": "unsupported",
                            "scope": "diagnostic",
                            "confidence": claim.confidence,
                            "label": "TEMELSIZ",
                        }
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def _build_manipulation_findings(self, components: list[FactClaim]) -> dict[str, Any]:
        """Build manipulation findings from claim components."""
        all_findings = []
        for claim in components:
            if claim.component_type != "claim":
                continue
            analysis = ClaimAnalyzer.analyze_manipulation(claim.text)
            if analysis["score"] >= 0.35:
                all_findings.append({
                    "claim_id": claim.id,
                    "text": claim.text,
                    "score": analysis["score"],
                    "details": analysis["findings"],
                })
        return {
            "total_claims": len([c for c in components if c.component_type == "claim"]),
            "flagged_count": len(all_findings),
            "findings": all_findings,
        }

    @staticmethod
    def _build_findings(
        claims: list[FactClaim],
        relations: list[FactRelation],
        run: FactCheckRun,
    ) -> list[dict]:
        findings = []
        relation_by_target: dict[int, list[FactRelation]] = {}
        for relation in relations:
            relation_by_target.setdefault(relation.target_claim_id, []).append(relation)

        for claim in claims:
            if claim.component_type != "claim":
                continue
            claim_relations = relation_by_target.get(claim.id, [])
            unsupported = not any(
                relation.relation_scope == "internal"
                and relation.relation_type == "support"
                and relation.confidence >= FactCheckService.RELATION_THRESHOLD
                for relation in claim_relations
            )
            manipulation = ClaimAnalyzer.analyze_manipulation(claim.text)
            quote_data = json.loads(claim.quote_data_json or "[]")

            # Main verdict findings
            if claim.verdict_status in {
                "Metin içinde çelişkili",
                "Kaynaklarla çelişiyor",
                "Teyit arşiviyle çelişiyor",
                "Bilgi tabanıyla çelişiyor",
                "Kamu verisiyle çelişiyor",
                "Kamu verisiyle çelişiyor (somut)",
                "Kamu verisiyle destekleniyor (somut)",
                "Türk arşivinde çürütülmüş",
                "Google Fact Check'te çürütülmüş",
                "Google Fact Check'te doğrulanmış",
                "Kaynak bulunamadı",
                "Metin içinde kanıtsız",
            }:
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": claim.verdict_status,
                        "detail": claim.text,
                        "severity": (
                            "high"
                            if "çeliş" in claim.verdict_status or "çürüt" in claim.verdict_status
                            else "medium"
                        ),
                        "confidence": max(
                            [relation.confidence for relation in claim_relations] or [claim.confidence]
                        ),
                    }
                )

            # Google Fact Check detail finding
            gc_relations = [
                r for r in claim_relations
                if r.evidence and r.evidence.search_query and "google_factcheck:" in r.evidence.search_query
            ]
            if gc_relations:
                top_gc = max(gc_relations, key=lambda r: r.confidence)
                snippet = top_gc.evidence.snippet if top_gc.evidence else ""
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": "Google Fact Check kaynağı",
                        "detail": f"{snippet[:200]}… ({top_gc.evidence.title if top_gc.evidence else ''})",
                        "severity": "low",
                        "confidence": top_gc.confidence,
                    }
                )

            # TCMB numeric detail finding
            tcmb_relations = [
                r for r in claim_relations
                if r.evidence and r.evidence.source_domain == "tcmb.gov.tr" and r.relation_type in ("support", "attack")
            ]
            if tcmb_relations:
                top_tcmb = max(tcmb_relations, key=lambda r: r.confidence)
                snippet = top_tcmb.evidence.snippet if top_tcmb.evidence else ""
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": "TCMB EVDS sayısal karşılaştırma",
                        "detail": snippet[:240],
                        "severity": "low",
                        "confidence": top_tcmb.confidence,
                    }
                )
            if unsupported:
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": "Temelsiz iddia",
                        "detail": "Bu iddiayı haber metni içinde destekleyen güçlü bir premise bulunamadı.",
                        "severity": "medium",
                        "confidence": claim.confidence,
                    }
                )
            if manipulation["score"] >= 0.35:
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": "Manipülatif dil sinyali",
                        "detail": f"{claim.text} — Sinyaller: {', '.join(manipulation['findings'])}",
                        "severity": "medium",
                        "confidence": manipulation["score"],
                    }
                )
            # Quote verification
            for quote in quote_data:
                if not quote.get("has_attribution"):
                    findings.append(
                        {
                            "claim_id": claim.id,
                            "title": "Kaynaksız alıntı",
                            "detail": f"\"{quote['quote'][:100]}...\" — Söyleyen kişi belirtilmemiş.",
                            "severity": "low",
                            "confidence": 0.6,
                        }
                    )
            # Statistical claim flag
            if claim.is_statistical:
                findings.append(
                    {
                        "claim_id": claim.id,
                        "title": "İstatistiksel iddia",
                        "detail": f"Bu iddia sayısal veri içeriyor. Doğrulanabilirliği yüksek: {claim.text[:120]}",
                        "severity": "low",
                        "confidence": 0.75,
                    }
                )

        # Balanced reporting finding
        balanced = json.loads(run.balanced_reporting_json or "{}")
        if balanced.get("is_imbalanced"):
            findings.append(
                {
                    "claim_id": None,
                    "title": "Dengesiz habercilik",
                    "detail": f"Karşı görüş sinyalleri: {balanced.get('counter_argument_signals', 0)}. "
                              f"Kaynak çeşitliliği: {balanced.get('source_diversity_count', 0)}.",
                    "severity": "medium",
                    "confidence": min(1.0, (100 - balanced.get('balance_score', 50)) / 100),
                }
            )

        if not run.to_dict()["source_metadata"].get("search_provider") == "tavily":
            findings.insert(
                0,
                {
                    "claim_id": None,
                    "title": "Dış kaynak araması kapalı",
                    "detail": "TAVILY_API_KEY bulunmadığı için yalnız metin içi argüman madenciliği çalıştı.",
                    "severity": "low",
                    "confidence": 1.0,
                },
            )
        return findings[:15]

    @staticmethod
    def _source_category(domain: str) -> str:
        normalized = domain.lower().replace("www.", "")
        if any(item in normalized for item in Config.FACT_CHECK_ARCHIVE_DOMAINS):
            return "fact_check_archive"
        if any(item in normalized for item in Config.FACT_CHECK_TRUSTED_DOMAINS):
            return "knowledge_base"
        return "web"

    @staticmethod
    def _manipulation_score(text: str) -> float:
        lower = text.lower()
        charged_terms = [
            "skandal",
            "şok",
            "ihanet",
            "rezalet",
            "felaket",
            "korkunç",
            "asla",
            "kesinlikle",
            "herkes",
            "hiç kimse",
            "yok ediyor",
        ]
        term_hits = sum(1 for term in charged_terms if term in lower)
        exclamation = min(text.count("!"), 3)
        all_caps = len(re.findall(r"\b[A-ZÇĞİÖŞÜ]{4,}\b", text))
        raw = (term_hits * 0.22) + (exclamation * 0.12) + min(all_caps, 3) * 0.10
        return min(1.0, raw)

    @staticmethod
    def _build_impact(
        claims: list[FactClaim],
        evidence: list[FactEvidence],
        relations: list[FactRelation],
        run: FactCheckRun,
    ) -> dict:
        claim_count = sum(1 for claim in claims if claim.component_type == "claim")
        support_count = sum(1 for r in relations if r.relation_type == "support")
        attack_count = sum(1 for r in relations if r.relation_type == "attack")
        unsupported_count = sum(
            1
            for claim in claims
            if claim.component_type == "claim"
            and claim.verdict_status in {"Kaynak bulunamadı", "Metin içinde kanıtsız"}
        )
        turkish_archive_count = sum(1 for e in evidence if e.is_turkish_archive)
        public_data_count = sum(1 for e in evidence if e.is_public_data_source)
        google_fc_count = sum(
            1 for e in evidence
            if e.search_query and "google_factcheck:" in e.search_query
        )
        tcmb_numeric_count = sum(
            1 for e in evidence
            if e.source_domain == "tcmb.gov.tr" and e.is_public_data_source
        )
        contradiction_ratio = attack_count / max(1, support_count + attack_count)
        logic_score = max(0, round((1 - contradiction_ratio) * 100) - (unsupported_count * 30))

        balanced = json.loads(run.balanced_reporting_json or "{}")
        manipulation = json.loads(run.manipulation_findings_json or "{}")

        return {
            "claim_count": claim_count,
            "text_component_count": len(claims),
            "source_count": len(evidence),
            "support_count": support_count,
            "attack_count": attack_count,
            "unsupported_count": unsupported_count,
            "turkish_archive_count": turkish_archive_count,
            "public_data_count": public_data_count,
            "google_factcheck_count": google_fc_count,
            "tcmb_numeric_count": tcmb_numeric_count,
            "logic_score": min(100, logic_score),
            "contradiction_percent": round(contradiction_ratio * 100),
            "balance_score": balanced.get("balance_score", 0),
            "manipulation_flagged": manipulation.get("flagged_count", 0),
        }

    @staticmethod
    def _build_run_summary(claims: list[FactClaim]) -> str:
        if not claims:
            return "ModernBERT haber metninde doğrulanabilir iddia tespit edemedi."
        verdict_counts: dict[str, int] = {}
        for claim in claims:
            verdict_counts[claim.verdict_status] = verdict_counts.get(claim.verdict_status, 0) + 1
        summary = ", ".join(f"{count} {status}" for status, count in verdict_counts.items())
        return f"ModernBERT {len(claims)} doğrulanabilir iddia çıkardı: {summary}."
