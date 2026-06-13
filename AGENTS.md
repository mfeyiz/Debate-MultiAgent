# Agent Instructions — Logos

## Environment

- **Python**: 3.13+
- **Package Manager**: `uv` (preferred) or `pip`
- **Virtual Environment**: `.venv` (auto-created by `uv`)

## Running the App

```bash
# Install / sync dependencies
uv sync

# Run development server
uv run python main.py

# Server starts on http://127.0.0.1:5000 with Uvicorn
```

## Adding Real ModernBERT Weights

1. Open `app/services/bert_service.py`
2. Replace the mock methods (`_extract_components`, `_infer_relation`, `_evaluate_strength`, `_generate_feedback`) with your actual model inference.
3. Keep the public `analyze(source_text, target_text, ...)` signature unchanged — the rest of the app depends only on this interface.

## Connecting a Real LLM

1. Add your OpenRouter API key to `.env`:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
2. Optionally change `DEFAULT_MODEL` to any OpenRouter-supported model (e.g. `anthropic/claude-3-opus`, `google/gemini-1.5-pro`).
3. Restart the server. The mock fallback automatically disables itself when a real key is present.

## Enhanced Fact-Check APIs (Optional, Free)

The Teyit Laboratuvarı can verify claims against multiple free APIs:

1. **Google Fact Check Tools API** — Global claim reviews from IFCN-certified outlets.
   - Get a key from https://developers.google.com/fact-check/tools/api
   - Add to `.env`: `GOOGLE_FACTCHECK_API_KEY=...`

2. **TCMB EVDS API** — Real-time numeric verification of economic claims (exchange rates, inflation, interest).
   - Register free at https://evds2.tcmb.gov.tr
   - Add to `.env`: `TCMB_EVDS_API_KEY=...`

3. **TÜİK API** — Official Turkish statistical bulletins.
   - Visit https://data.tuik.gov.tr
   - Add to `.env`: `TUIK_API_KEY=...` (many endpoints work without a key)

All three fall back gracefully when keys are missing.

## Database

SQLite is used by default. The database file (`debate_platform.db`) is created automatically on first run. To reset:

```bash
rm debate_platform.db
```

The app will re-seed two default agents (Agent Alpha & Agent Beta) on the next startup.

## Code Style

- Type hints on all public methods
- Docstrings on all modules, classes, and public functions
- Services are stateless except for config-driven initialization
- Database access is centralized in `DebateService`; routers only call service methods

## Project Structure Decisions

- **FastAPI + Pydantic-AI**: Pydantic-AI agents expose a synchronous `run_sync()` method; FastAPI runs the existing sync service layer behind ASGI endpoints.
- **SSE instead of WebSockets**: SSE is unidirectional (server → client), requires no extra libraries, and is sufficient for streaming debate events.
- **Jinja2 + Tailwind CDN**: Keeps the frontend lightweight and avoids a build step. The Tailwind config from the designs is inlined in `base.html`.
- **MessageVersion table**: Supports versioning natively. When an agent regenerates based on feedback, a new `MessageVersion` row is created rather than overwriting history.

## Troubleshooting

- **Missing Authentication header (401)**: `OPENROUTER_API_KEY` is empty or invalid. The app falls back to mock mode, but if you see this error it means the key was set incorrectly.
- **Database locked**: SQLite does not support high concurrency. For production, migrate to PostgreSQL.
- **Port already in use**: Change `PORT` in `.env` or the shell before running `uv run python main.py`.
