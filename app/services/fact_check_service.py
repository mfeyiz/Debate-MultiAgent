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
    MIN_SOURCE_RELEVANCE = 0.34
    MIN_SEARCH_SCORE = 0.20
    MANIPULATION_RELATION_SKIP = 0.55
    OFFICIAL_SOURCE_TERMS = {
        "tüik",
        "tuik",
        "tcmb",
        "merkez bankası",
        "oecd",
        "who",
        "dünya bankası",
        "world bank",
        "resmi gazete",
        "bakanlığı",
        "eurostat",
        "imf",
    }

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
            
            # Deep Argument Mining analysis (logical fallacies, thesis, tone)
            arg_structure = await ClaimAnalyzer.analyze_argument_structure(article_text, db_components, self._llm_model)
            
            # Blend overall thesis/tone/objectivity score into balanced_reporting
            balanced = json.loads(run.balanced_reporting_json or "{}")
            balanced["main_thesis"] = arg_structure.get("main_thesis")
            balanced["dialectic_tone"] = arg_structure.get("dialectic_tone")
            balanced["objectivity_score"] = arg_structure.get("objectivity_score")
            run.balanced_reporting_json = json.dumps(balanced)
            
            # Blend logical fallacies into manipulation findings
            manipulation_findings = self._build_manipulation_findings(db_components)
            manipulation_findings["fallacies"] = arg_structure.get("fallacies", [])
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
        for component in meaningful:
            component._extraction_reason = f"model_{component.component_type}"

        rescued = self._rescue_verifiable_claims(text, meaningful)
        if meaningful or rescued:
            merged = [*meaningful, *rescued]
            merged.sort(key=lambda component: (component.start_idx, component.end_idx))
            return merged[: self.MAX_SOURCE_COMPONENTS]
        return self._fallback_components(text)

    def _rescue_verifiable_claims(
        self,
        text: str,
        existing: list[ArgumentComponent],
    ) -> list[ArgumentComponent]:
        """Recover factual claims that the component model labeled as other."""
        rescued: list[ArgumentComponent] = []
        spans = [(component.start_idx, component.end_idx) for component in existing]

        text_units = getattr(self.bert, "_text_units", None)
        is_meaningful = getattr(self.bert, "_is_meaningful_component", None)
        units = text_units(text) if callable(text_units) else self._simple_text_units(text)
        for start_idx, end_idx in units:
            unit_text = text[start_idx:end_idx].strip()
            meaningful = is_meaningful(unit_text) if callable(is_meaningful) else self._simple_meaningful_component(unit_text)
            if not meaningful:
                continue
            if self._overlaps_existing_span(start_idx, end_idx, spans):
                continue

            statistical = ClaimAnalyzer.detect_statistical_claim(unit_text)
            official = self._has_official_source_signal(unit_text)
            if not statistical["is_statistical"] and not official:
                continue

            reason = "statistical_rescue" if statistical["is_statistical"] else "official_source_rescue"
            confidence = 0.74 if statistical["is_statistical"] else 0.68
            component = ArgumentComponent(
                text=unit_text[: self.bert.MAX_COMPONENT_CHARS],
                component_type="claim",
                start_idx=start_idx,
                end_idx=min(end_idx, start_idx + self.bert.MAX_COMPONENT_CHARS),
                confidence=confidence,
            )
            component._extraction_reason = reason
            rescued.append(component)
            spans.append((component.start_idx, component.end_idx))

        return rescued

    @staticmethod
    def _simple_text_units(text: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        for match in re.finditer(r"[^.!?;\n]+(?:[.!?;]+|$)", text, flags=re.UNICODE):
            start_idx, end_idx = match.span()
            while start_idx < end_idx and text[start_idx].isspace():
                start_idx += 1
            while end_idx > start_idx and text[end_idx - 1].isspace():
                end_idx -= 1
            if end_idx > start_idx:
                units.append((start_idx, end_idx))
        return units

    @staticmethod
    def _simple_meaningful_component(text: str) -> bool:
        return len(text.strip()) >= 24 and len(re.findall(r"[\wğüşöçıİĞÜŞÖÇ%]+", text, flags=re.UNICODE)) >= 4

    @staticmethod
    def _overlaps_existing_span(start_idx: int, end_idx: int, spans: list[tuple[int, int]]) -> bool:
        for left, right in spans:
            overlap = max(0, min(end_idx, right) - max(start_idx, left))
            if overlap / max(1, end_idx - start_idx) >= 0.6:
                return True
        return False

    @classmethod
    def _has_official_source_signal(cls, text: str) -> bool:
        lower = text.lower()
        return any(term in lower for term in cls.OFFICIAL_SOURCE_TERMS)

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
                extraction_reason=getattr(component, "_extraction_reason", f"model_{component.component_type}"),
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
                if self._should_skip_internal_source(source):
                    continue
                relation_type, confidence, probabilities = await asyncio.to_thread(
                    self.bert.classify_relation,
                    target.text,
                    source.text,
                )
                if relation_type in {"neutral", "none"} or confidence < self.RELATION_THRESHOLD:
                    continue
                probabilities = {
                    ("none" if key == "neutral" else key): value
                    for key, value in (probabilities or {}).items()
                }
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

    @classmethod
    def _should_skip_internal_source(cls, source: FactClaim) -> bool:
        """Do not let manipulative unsupported language become a strong semantic attack."""
        if source.component_type == "claim":
            return False
        manipulation = source.manipulation_score or 0.0
        if manipulation < cls.MANIPULATION_RELATION_SKIP:
            return False
        if source.is_statistical or cls._has_official_source_signal(source.text):
            return False
        return True

    async def _search_turkish_archives(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        """Search Turkish fact-check archives for pre-existing verifications."""
        claims = [c for c in components if c.component_type == "claim"]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            archive_results = await TurkishArchiveScraper.search_all(claim.text)
            for result in archive_results:
                # Infer preliminary verdict from title
                inferred = TurkishArchiveScraper.infer_verdict_from_title(result["title"])
                relevance = self._score_source_candidate(
                    claim.text,
                    title=result["title"],
                    url=result["url"],
                    snippet=result["snippet"],
                    search_score=0.85,
                    source_domain=result["source_domain"],
                    is_archive=True,
                )
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
                    relevance_score=relevance["relevance_score"],
                    source_quality=relevance["source_quality"],
                    accepted_for_verdict=relevance["accepted_for_verdict"],
                    is_turkish_archive=True,
                    archive_match_claim_text=claim.text,
                )
                db.add(evidence)
                await db.flush()
                # Add relation based on inferred verdict
                if inferred and evidence.accepted_for_verdict:
                    db.add(
                        FactRelation(
                            run_id=run.id,
                            evidence_id=evidence.id,
                            target_claim_id=claim.id,
                            relation_scope="external",
                            relation_type=inferred,
                            confidence=0.75,
                            probabilities_json=json.dumps({inferred: 0.75, "none": 0.25}),
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
                relevance = self._score_source_candidate(
                    claim.text,
                    title=result["title"],
                    url=result["url"],
                    snippet=result["snippet"],
                    search_score=0.88,
                    source_domain=result["source_domain"],
                    is_archive=True,
                )
                evidence = FactEvidence(
                    run_id=run.id,
                    claim_id=claim.id,
                    search_query=f"google_factcheck: {result.get('source', '')}",
                    url=result["url"],
                    title=result["title"][:500],
                    source_domain=result["source_domain"],
                    snippet=result["snippet"],
                    score=0.88,
                    relevance_score=relevance["relevance_score"],
                    source_quality=relevance["source_quality"],
                    accepted_for_verdict=relevance["accepted_for_verdict"],
                    is_turkish_archive=False,
                    archive_match_claim_text=claim.text,
                )
                db.add(evidence)
                await db.flush()
                if inferred and inferred in {"support", "attack"} and evidence.accepted_for_verdict:
                    db.add(
                        FactRelation(
                            run_id=run.id,
                            evidence_id=evidence.id,
                            target_claim_id=claim.id,
                            relation_scope="external",
                            relation_type=inferred,
                            confidence=0.72,
                            probabilities_json=json.dumps({inferred: 0.72, "none": 0.28}),
                        )
                    )

    async def _query_public_data_sources(self, db: AsyncSession, run: FactCheckRun, components: list[FactClaim]) -> None:
        """Query Turkish public data sources for statistical claims."""
        claims = [c for c in components if c.component_type == "claim" and c.is_statistical]
        for claim in claims[: Config.FACT_CHECK_MAX_CLAIMS]:
            public_results = await PublicDataSourceClient.query_all_for_claim(claim.text)
            for result in public_results:
                # Determine relation from numeric comparison if available
                relation_type = "none"
                confidence = 0.90
                comparison = result.get("comparison")
                if comparison:
                    relation_type = comparison.get("verdict", "none")
                    if relation_type == "neutral":
                        relation_type = "none"
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
                    relevance_score=1.0,
                    source_quality="public_data",
                    accepted_for_verdict=True,
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
                            probabilities_json=json.dumps({relation_type: confidence, "none": round(1 - confidence, 2)}),
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
                    relevance = self._score_source_candidate(
                        claim.text,
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        search_score=result.score,
                        source_domain=source_domain,
                        is_archive=self._source_category(source_domain) == "fact_check_archive",
                    )
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
                        relevance_score=relevance["relevance_score"],
                        source_quality=relevance["source_quality"],
                        accepted_for_verdict=relevance["accepted_for_verdict"],
                    )
                    db.add(evidence)
                    await db.flush()
                    if result.snippet.strip() and evidence.accepted_for_verdict:
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

    @classmethod
    def _score_source_candidate(
        cls,
        claim_text: str,
        *,
        title: str,
        url: str,
        snippet: str,
        search_score: float,
        source_domain: str,
        is_archive: bool = False,
    ) -> dict[str, Any]:
        """Strict source gate used before an external result can affect verdicts."""
        title = title or ""
        snippet = snippet or ""
        parsed = urlparse(url or "")
        path = parsed.path.strip("/")
        text = f"{title} {snippet}"
        coverage = cls._claim_term_coverage(claim_text, text)
        boilerplate = cls._is_boilerplate_source_text(title, snippet)
        homepage = path in {"", "#"} or path.lower() in {"tr", "en", "homepage", "anasayfa"}
        trusted = cls._source_category(source_domain) in {"knowledge_base", "fact_check_archive"}
        official = cls._has_official_source_signal(f"{source_domain} {title} {snippet}")
        valid_url = TurkishArchiveScraper.is_valid_source_url(url)

        relevance = round(
            min(
                1.0,
                (coverage * 0.72)
                + (min(max(search_score, 0.0), 1.0) * 0.18)
                + (0.10 if trusted or official else 0.0),
            ),
            3,
        )

        accepted = True
        quality = "accepted"
        if boilerplate:
            accepted = False
            quality = "boilerplate"
        elif homepage and not official:
            accepted = False
            quality = "homepage"
        elif not valid_url:
            accepted = False
            quality = "invalid_url_format"
        elif search_score < cls.MIN_SEARCH_SCORE and not trusted:
            accepted = False
            quality = "low_search_score"
        elif relevance < cls.MIN_SOURCE_RELEVANCE:
            accepted = False
            quality = "low_relevance"
        elif is_archive and coverage < 0.18 and not official:
            accepted = False
            quality = "archive_low_coverage"

        return {
            "relevance_score": relevance,
            "source_quality": quality,
            "accepted_for_verdict": accepted,
            "claim_coverage": coverage,
        }

    @staticmethod
    def _claim_terms(text: str) -> set[str]:
        stopwords = {
            "bir", "ve", "ile", "için", "olan", "olarak", "göre", "buna", "rağmen",
            "haberde", "iddia", "iddiası", "oldu", "olduğu", "sonunda", "yılında",
            "the", "and", "for", "with", "from", "this", "that",
        }
        return {
            word
            for word in re.findall(r"[\wğüşöçıİĞÜŞÖÇ%]+", text.lower(), flags=re.UNICODE)
            if len(word) >= 4 and word not in stopwords
        }

    @classmethod
    def _claim_term_coverage(cls, claim_text: str, source_text: str) -> float:
        claim_terms = cls._claim_terms(claim_text)
        if not claim_terms:
            return 0.0
        source_terms = cls._claim_terms(source_text)
        return round(len(claim_terms & source_terms) / len(claim_terms), 3)

    @staticmethod
    def _is_boilerplate_source_text(title: str, snippet: str) -> bool:
        text = re.sub(r"\s+", " ", f"{title} {snippet}").strip().lower()
        if len(text) < 24:
            return True
        exact_noise = {
            "hakkında",
            "tüm yazılar",
            "şehir efsaneleri",
            "fact check",
            "politifact",
        }
        if text in exact_noise:
            return True
        noise_terms = [
            "menu",
            "sign up",
            "read more",
            "newsletter",
            "membership",
            "recent articles",
            "anasayfa",
            "çerez",
            "giriş yap",
            "abone ol",
        ]
        hits = sum(1 for term in noise_terms if term in text)
        link_markers = text.count("http") + text.count("[") + text.count("]")
        return hits >= 2 or link_markers >= 8

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
            external = [
                r
                for r in claim_relations
                if r.relation_scope == "external"
                and (not r.evidence or r.evidence.accepted_for_verdict)
            ]
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
            claim.explanation = await self._synthesize_explanation(claim, [*internal, *external])

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
        lower = normalized.lower()
        words = re.findall(r"[\wğüşöçıİĞÜŞÖÇ%]+", normalized, flags=re.UNICODE)
        
        # Stopwords to keep search query focused
        stopwords = {
            "dedi", "söyledi", "açıkladı", "iddia", "edildi", "oldu", "olacak", "yapıldı",
            "yapacak", "geldi", "gitti", "verdi", "aldı", "başladı", "bitirdi", "tarafından",
            "yönelik", "ilişkin", "hakkında", "üzerine", "karşı", "sonra", "önce", "olan",
            "olarak", "göre"
        }
        filtered_words = [w for w in words if w.lower() not in stopwords and len(w) > 3]
        salient = " ".join(filtered_words[:8])
        if not salient:
            salient = " ".join(words[:8])
        if not salient:
            salient = normalized[:120]

        # 1. Clean direct query for finding news reports / evidence
        q_direct = salient
        
        # 2. Fact checking archives search
        q_verify = f"{salient} (teyit OR \"doğruluk payı\" OR malumatfuruş)"
        
        # 3. Official data search (if statistical or economic/demographic)
        is_stat = ClaimAnalyzer.detect_statistical_claim(normalized)["is_statistical"]
        has_econ = any(w in lower for w in ["enflasyon", "büyüme", "faiz", "işsizlik", "dolar", "türk lirası", "tcmb", "tüik", "nüfus", "emekli", "maaş", "bütçe"])
        
        queries = []
        if is_stat or has_econ:
            q_official = f"{salient} (TÜİK OR \"Merkez Bankası\" OR TCMB OR Resmi Gazete)"
            queries.append((q_official, "official_data"))
            
        queries.append((q_verify, "verification"))
        queries.append((q_direct, "direct_claim"))
        
        # Deduplicate and return planned queries based on maximum allowed config
        seen_queries = set()
        planned = []
        for q, scope in queries:
            if q not in seen_queries:
                seen_queries.add(q)
                planned.append((q, scope))
                
        return planned[:max(1, Config.FACT_CHECK_MAX_QUERIES_PER_CLAIM)]

    @staticmethod
    def _infer_title(text: str) -> str:
        first_line = text.strip().splitlines()[0] if text.strip() else "Haber Analizi"
        return first_line[:120]

    @staticmethod
    def _is_visible_relation(relation: FactRelation) -> bool:
        if relation.relation_type not in {"support", "attack"}:
            return False
        if relation.relation_scope == "external" and relation.evidence and not relation.evidence.accepted_for_verdict:
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
                    "extraction_reason": claim.extraction_reason,
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
                        "extraction_reason": claim.extraction_reason,
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
                        "relevance_score": item.relevance_score,
                        "source_quality": item.source_quality,
                        "accepted_for_verdict": item.accepted_for_verdict,
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
            analysis = ClaimAnalyzer.analyze_manipulation(claim.text)
            if analysis["score"] >= 0.35:
                all_findings.append({
                    "claim_id": claim.id,
                    "component_type": claim.component_type,
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
                manipulation = ClaimAnalyzer.analyze_manipulation(claim.text)
                if manipulation["score"] >= 0.35:
                    findings.append(
                        {
                            "claim_id": claim.id,
                            "title": "Manipülasyon uyarısı",
                            "detail": (
                                "Bu metin parçası doğrulanabilir kanıt yerine manipülatif/belirsiz "
                                f"kaynak dili içeriyor: {claim.text[:160]}"
                            ),
                            "severity": "medium",
                            "confidence": manipulation["score"],
                        }
                    )
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

        # Logical fallacies findings integration
        try:
            manip_data = json.loads(run.manipulation_findings_json or "{}")
            for fallacy in manip_data.get("fallacies", []):
                findings.append(
                    {
                        "claim_id": None,
                        "title": f"Mantıksal Safsata: {fallacy['title']}",
                        "detail": fallacy["detail"],
                        "severity": fallacy["severity"],
                        "confidence": fallacy["confidence"],
                    }
                )
        except Exception:
            pass

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
        archive_domains = set(Config.FACT_CHECK_ARCHIVE_DOMAINS) | {"malumatfurus.org", "malumatfurus"}
        if any(item in normalized for item in archive_domains):
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
        low_relevance_count = sum(1 for e in evidence if not e.accepted_for_verdict)
        accepted_source_count = sum(1 for e in evidence if e.accepted_for_verdict)
        numeric_claim_count = sum(1 for claim in claims if claim.component_type == "claim" and claim.is_statistical)
        filtered_source_count = low_relevance_count
        contradiction_ratio = attack_count / max(1, support_count + attack_count)
        logic_score = max(0, round((1 - contradiction_ratio) * 100) - (unsupported_count * 30))

        balanced = json.loads(run.balanced_reporting_json or "{}")
        manipulation = json.loads(run.manipulation_findings_json or "{}")

        return {
            "claim_count": claim_count,
            "text_component_count": len(claims),
            "source_count": len(evidence),
            "accepted_source_count": accepted_source_count,
            "filtered_source_count": filtered_source_count,
            "low_relevance_source_count": low_relevance_count,
            "numeric_claim_count": numeric_claim_count,
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
