"""Fact-check lab API tests with async database operations."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
import app.database
from app.config import Config
from app.services.bert_service import ArgumentComponent
from app.services.fact_check_service import (
    FactCheckService,
    SearchResult,
    TavilySearchClient,
    _ArticleHTMLParser,
)


class FakeBert:
    """Deterministic ModernBERT stand-in for fact-check tests."""

    def extract_components(self, text: str, default_type: str) -> list[ArgumentComponent]:
        claim_text = "Uzaktan çalışma verimliliği artırır."
        evidence_text = "Bağımsız çalışma verileri bu artışı desteklemektedir."
        claim_start = text.index(claim_text)
        evidence_start = text.index(evidence_text)
        return [
            ArgumentComponent(
                text=claim_text,
                component_type="claim",
                start_idx=claim_start,
                end_idx=claim_start + len(claim_text),
                confidence=0.91,
            ),
            ArgumentComponent(
                text=evidence_text,
                component_type="evidence",
                start_idx=evidence_start,
                end_idx=evidence_start + len(evidence_text),
                confidence=0.86,
            ),
        ]

    def classify_relation(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> tuple[str, float, dict[str, float]]:
        if "desteklemektedir" in evidence_text.lower() or "verimlilik" in evidence_text.lower():
            return "support", 0.93, {"support": 0.93, "attack": 0.03, "neutral": 0.04}
        return "neutral", 0.80, {"support": 0.10, "attack": 0.10, "neutral": 0.80}

    def analyze(self, source_text: str, target_text: str, source_type: str, target_type: str):
        from app.services.bert_service import AnalysisResult, RelationPrediction
        return AnalysisResult(
            overall_strength=0.85,
            components=[],
            relations=[RelationPrediction(
                source_text=source_text,
                target_text=target_text,
                relation_type="support",
                confidence=0.85,
                probabilities={"support": 0.85, "attack": 0.05, "neutral": 0.10}
            )],
            feedback="İyi bir destekleme kurulmuş."
        )


class FakeSearchClient:
    """Fake Tavily client returning one source per query."""

    available = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        self.calls.append(query)
        return [
            SearchResult(
                title="Verimlilik Araştırması",
                url="https://example.org/verimlilik",
                snippet="Verimlilik araştırması uzaktan çalışma verimliliği artırır sonucunu desteklemektedir.",
                score=0.87,
            )
        ]


@pytest.fixture(scope="function")
async def test_env():
    """Fixture to set up temporary database, mock BERT, and create test app."""
    # Store old configs
    old_db_uri = Config.SQLALCHEMY_DATABASE_URI
    old_tavily_key = Config.TAVILY_API_KEY
    old_openrouter_key = Config.OPENROUTER_API_KEY
    old_max_queries = Config.FACT_CHECK_MAX_QUERIES_PER_CLAIM
    old_bert = FactCheckService._bert_instance

    # Mock BERT
    FactCheckService._bert_instance = FakeBert()

    # Configure test environment
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "test.db"
    Config.SQLALCHEMY_DATABASE_URI = f"sqlite+aiosqlite:///{db_path}"
    Config.TAVILY_API_KEY = ""
    Config.OPENROUTER_API_KEY = ""
    Config.FACT_CHECK_MAX_QUERIES_PER_CLAIM = 1

    # Swap database engine and session factory globally
    old_engine = app.database.engine
    old_session_factory = app.database.async_session_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 30})
    app.database.engine = test_engine
    app.database.async_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Initialize schema
    await app.database.init_db()

    # Create FastAPI app
    app_instance = create_app()

    # Mock search client
    search_client = FakeSearchClient()
    app_instance.state.fact_svc.search_client = search_client
    app_instance.state.fact_svc._llm_model = None

    yield app_instance, search_client

    # Cleanup
    await test_engine.dispose()
    app.database.engine = old_engine
    app.database.async_session_factory = old_session_factory
    Config.SQLALCHEMY_DATABASE_URI = old_db_uri
    Config.TAVILY_API_KEY = old_tavily_key
    Config.OPENROUTER_API_KEY = old_openrouter_key
    Config.FACT_CHECK_MAX_QUERIES_PER_CLAIM = old_max_queries
    FactCheckService._bert_instance = old_bert
    tmpdir.cleanup()


@pytest.mark.asyncio
async def test_fact_check_text_input_returns_claims_evidence_relations(test_env) -> None:
    app_instance, search_client = test_env
    article = (
        "Uzaktan çalışma verimliliği artırır. "
        "Bağımsız çalışma verileri bu artışı desteklemektedir. "
        "Bu nedenle haber iddiası kaynaklı biçimde incelenmelidir."
    )

    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        response = await ac.post(
            "/api/fact-checks",
            json={"text": article},
        )

        assert response.status_code == 201, f"Failed with {response.status_code}: {response.text}"
        payload = response.json()
        assert payload["run"]["status"] == "completed"
        assert len(payload["claims"]) >= 2
        assert len(payload["evidence"]) >= 1
        assert len(payload["relations"]) >= 1
        assert "annotations" in payload
        assert "graph" in payload
        assert len(payload["graph"]["edges"]) >= 1
        assert payload["impact"]["claim_count"] == 1
        assert len(search_client.calls) <= 1

        run_id = payload["run"]["id"]
        get_response = await ac.get(f"/api/fact-checks/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["run"]["id"] == run_id


@pytest.mark.asyncio
async def test_fact_check_url_failure_returns_actionable_error(test_env) -> None:
    app_instance, _ = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        response = await ac.post(
            "/api/fact-checks",
            json={"url": "https://127.0.0.1:9/not-a-news-page"},
        )

        assert response.status_code == 400
        assert "haber metnini yapıştırın" in response.json()["error"].lower()


def test_tavily_client_without_key_returns_empty_results() -> None:
    client = TavilySearchClient(api_key="")
    assert client.search("uzaktan çalışma") == []


def test_article_parser_prefers_news_body_over_market_widgets() -> None:
    parser = _ArticleHTMLParser()
    parser.feed(
        """
        <html>
          <body>
            <aside class="sidebar widget">
              <p>Dolar kuru 32,10 Euro 35,40 Altın 2400</p>
              <p>En çok okunan haberler ve reklam alanı</p>
            </aside>
            <main class="news-detail article-content">
              <h1>Uzaktan çalışma araştırması yayımlandı</h1>
              <p>Stanford Üniversitesi tarafından yapılan çalışmada uzaktan çalışan ekiplerin belirli koşullarda daha yüksek verimlilik bildirdiği aktarıldı.</p>
              <p>Araştırmaya göre sessiz çalışma ortamı, daha az kesinti ve çalışan memnuniyeti bu artışı açıklayan temel etkenler arasında gösterildi.</p>
            </main>
          </body>
        </html>
        """
    )

    assert "Stanford Üniversitesi" in parser.text
    assert "sessiz çalışma ortamı" in parser.text
    assert "Dolar kuru" not in parser.text
    assert "En çok okunan" not in parser.text
