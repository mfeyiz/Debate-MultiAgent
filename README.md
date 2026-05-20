# LogicFlow — Multi-Agent Debate Platform

A modular, real-time debate platform for autonomous LLM agents with an integrated **ModernBERT** argument mining pipeline. Agents debate a topic; the pipeline extracts claims and evidences, scores their argumentative strength, and triggers regeneration with targeted feedback when responses are weak.

## Features

- **Multi-Agent Debates** — Configure proponent, opponent, and moderator agents backed by any OpenRouter-compatible LLM (OpenAI, Anthropic, Google, etc.).
- **ModernBERT Evaluation Pipeline** — Real fine-tuned ModernBERT models that extract argument components (claim/evidence) and classify relations (support/attack/neutral), scoring argumentative strength after every pair of exchanged messages.
- **Regeneration Loop** — When a response scores below the configurable threshold, the system pauses the debate, broadcasts feedback, and regenerates a stronger version.
- **Versioned Messages** — Every logical message slot supports multiple versions so the UI can compare V1 vs V2 side-by-side.
- **Real-Time UI** — Server-Sent Events (SSE) stream messages, analyses, and feedback to the browser without polling.
- **SQLite Persistence** — All debates, agents, messages, and analyses are stored locally.
- **Teyit Laboratuvarı (Fact-Check Lab)** — Paste a Turkish news article or URL to get an automated fact-check report:
  - **Argument Graph** — Interactive Cytoscape graph with support/attack relations (dagre layout, mini + modal views, click-to-sync with article text).
  - **Turkish Archive Integration** — Direct scraping of `teyit.org`, `dogrulukpayi.com`, and `malumatfurus.org`.
  - **Global Fact Check** — Google Fact Check Tools API integration for worldwide claim reviews.
  - **Public Data Verification** — TCMB EVDS real-time numeric comparison (e.g., claimed dollar rate vs actual Central Bank data) and TÜİK statistical bulletins.
  - **Source Credibility Scoring** — Hand-curated Turkish media bias & credibility database.
  - **Manipulation Detection** — Clickbait, emotional language, generalization, and missing-source heuristics.
  - **Balanced Reporting Analysis** — Detects one-sided journalism and missing counter-arguments.
  - **Chrome Extension** — One-click analysis from any news page.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + SQLAlchemy 2.0 (async) |
| LLM Orchestration | Pydantic-AI + OpenRouter |
| Argument Mining | Fine-tuned ModernBERT (real weights) |
| Frontend | Jinja2 + Tailwind CSS CDN |
| Real-Time | Server-Sent Events (SSE) |
| Database | SQLite (PostgreSQL-ready) |

## Quick Start

1. **Install dependencies** (uv is recommended):
   ```bash
   uv sync
   ```

2. **Configure environment variables** in `.env`:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...
   DEFAULT_MODEL=openai/gpt-4o-mini
   SECRET_KEY=change-me-in-production

   # Optional: Enhanced Fact-Check sources (free APIs)
   GOOGLE_FACTCHECK_API_KEY=your-google-api-key      # https://developers.google.com/fact-check/tools/api
   TCMB_EVDS_API_KEY=your-tcmb-evds-key              # https://evds2.tcmb.gov.tr (free registration)
   TUIK_API_KEY=your-tuik-key                        # https://data.tuik.gov.tr (optional, many endpoints work without key)
   ```

3. **Run the application**:
   ```bash
   uv run python main.py
   ```

4. **Open the dashboard** at http://127.0.0.1:5000

## Chrome Extension

The unpacked extension lives in `extension/`.

1. Start the API with `uv run python main.py`.
2. Open `chrome://extensions`, enable Developer mode, and load the `extension/` folder.
3. Use the popup on a news page to analyze the page or selected text. The popup links back to `/fact-check?run_id=...` for the full graph.

> **Note:** If `OPENROUTER_API_KEY` is missing or set to the placeholder value, the app falls back to deterministic mock responses so you can explore the UI and pipeline logic without an API key.

## Architecture

```
app/
├── __init__.py           # FastAPI app factory + lifespan (seeds default agents)
├── config.py             # Central configuration (reads .env)
├── database.py           # Async SQLAlchemy 2.0 engine/session factory
├── models.py             # 15 declarative ORM models
├── routers/
│   ├── pages.py          # HTML page routes (Jinja2 templates)
│   ├── api.py            # REST API endpoints
│   └── stream.py         # SSE streaming endpoint
├── services/
│   ├── agent_service.py  # Pydantic-AI + OpenRouter integration
│   ├── debate_service.py # Debate lifecycle, auto-advance, argument map builder
│   ├── bert_service.py   # Real ModernBERT pipeline (component + relation classifiers)
│   ├── fact_check_service.py  # Turkish fact-check lab (Teyit Laboratuvarı)
│   └── ...               # Archive scrapers, public data clients, credibility scorer
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── live_arena.html
    ├── agents.html
    ├── analytics.html
    └── fact_check.html
```

## ModernBERT Pipeline Interface

The `ModernBERTPipeline.analyze(source_text, target_text)` method returns a `PipelineResult` containing:

- `components` — Extracted claims / evidences
- `relations` — Support / attack / neutral relations with confidence
- `overall_strength` — Float 0.0–1.0
- `feedback` — Actionable critique if strength < threshold
- `needs_regeneration` — Boolean flag

Replace the mock internals in `app/services/bert_service.py` with your actual ModernBERT inference code. The rest of the system (models, debate service, UI) consumes this interface and does not depend on the implementation details.

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(empty)* | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | Default LLM model string |
| `TAVILY_API_KEY` | *(empty)* | Enables external source retrieval for fact-checking |
| `GOOGLE_FACTCHECK_API_KEY` | *(empty)* | Google Fact Check Tools API for global claim verification |
| `TCMB_EVDS_API_KEY` | *(empty)* | TCMB EVDS API for real-time numeric economic data verification |
| `TUIK_API_KEY` | *(empty)* | TÜİK API for official Turkish statistical bulletins |
| `FACT_CHECK_MAX_CLAIMS` | `3` | Maximum claims sent to Tavily per analysis |
| `FACT_CHECK_MAX_QUERIES_PER_CLAIM` | `1` | Tavily requests per claim |
| `FACT_CHECK_SEARCH_RESULTS` | `2` | Results requested from Tavily per query |
| `EVIDENCE_STRENGTH_THRESHOLD` | `0.70` | Strength below which regeneration triggers |
| `MAX_DEBATE_ROUNDS` | `5` | Maximum rounds per debate |
| `DATABASE_URL` | `sqlite:///<project>/debate_platform.db` | SQLite connection URI |

## API Overview

- `GET /api/agents` — List all agents
- `POST /api/agents` — Create an agent
- `GET /api/debates` — List all debates
- `POST /api/debates` — Create a debate
- `POST /api/debates/<id>/start` — Start debate (opening claim)
- `POST /api/debates/<id>/advance` — Next turn or regenerate
- `POST /api/debates/<id>/resolve` — Force resolve
- `GET /stream/<id>` — SSE stream for live updates

## License

MIT
