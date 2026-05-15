# LogicFlow — Multi-Agent Debate Platform

A modular, real-time debate platform for autonomous LLM agents with an integrated **ModernBERT** argument mining pipeline. Agents debate a topic; the pipeline extracts claims and evidences, scores their argumentative strength, and triggers regeneration with targeted feedback when responses are weak.

## Features

- **Multi-Agent Debates** — Configure proponent, opponent, and moderator agents backed by any OpenRouter-compatible LLM (OpenAI, Anthropic, Google, etc.).
- **ModernBERT Evaluation Pipeline** — Mock pipeline (swappable for real weights) that classifies relations (support / attack / neutral) and scores evidence strength after every pair of exchanged messages.
- **Regeneration Loop** — When a response scores below the configurable threshold, the system pauses the debate, broadcasts feedback, and regenerates a stronger version.
- **Versioned Messages** — Every logical message slot supports multiple versions so the UI can compare V1 vs V2 side-by-side.
- **Real-Time UI** — Server-Sent Events (SSE) stream messages, analyses, and feedback to the browser without polling.
- **SQLite Persistence** — All debates, agents, messages, and analyses are stored locally.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask + Flask-SQLAlchemy |
| LLM Orchestration | Pydantic-AI + OpenRouter |
| Argument Mining | ModernBERT (mock interface ready for real weights) |
| Frontend | Jinja2 + Tailwind CSS CDN |
| Real-Time | Server-Sent Events (SSE) |
| Database | SQLite |

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
   ```

3. **Run the application**:
   ```bash
   uv run python main.py
   ```

4. **Open the dashboard** at http://127.0.0.1:5000

> **Note:** If `OPENROUTER_API_KEY` is missing or set to the placeholder value, the app falls back to deterministic mock responses so you can explore the UI and pipeline logic without an API key.

## Architecture

```
app/
├── __init__.py           # Flask app factory
├── config.py             # Central configuration (reads .env)
├── extensions.py         # Flask-SQLAlchemy instance
├── models.py             # Declarative ORM models
├── routers/
│   ├── pages.py          # Jinja2 HTML routes
│   ├── api.py            # REST endpoints (agents, debates)
│   └── stream.py         # SSE event streams
├── services/
│   ├── agent_service.py  # Pydantic-AI + OpenRouter integration
│   ├── debate_service.py # Debate lifecycle & orchestration
│   └── bert_service.py   # ModernBERT pipeline (mock / real)
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── live_arena.html
    ├── agents.html
    └── analytics.html
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
