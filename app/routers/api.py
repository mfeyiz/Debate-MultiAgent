"""FastAPI router for REST API endpoints."""

from __future__ import annotations

from typing import Any

import anyio
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Agent
from app.services.debate_service import DebateService
from app.services.fact_check_service import FactCheckService


router = APIRouter(prefix="/api")


def _json_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------


@router.get("/agents")
async def list_agents(request: Request, db: AsyncSession = Depends(get_db)) -> list[dict]:
    svc: DebateService = request.app.state.debate_svc
    agents = await svc.list_agents(db)
    return [a.to_dict() for a in agents]


@router.post("/agents", status_code=201)
async def create_agent(
    request: Request,
    data: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    try:
        agent = await svc.create_agent(
            db=db,
            name=data["name"],
            model_name=data.get("model_name", ""),
            role=data.get("role", "proponent"),
            system_prompt=data.get("system_prompt", ""),
            temperature=float(data.get("temperature", 0.7)),
        )
        return agent.to_dict()
    except Exception as exc:
        return _json_error(str(exc), 400)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: int, db: AsyncSession = Depends(get_db)) -> Any:
    agent = await db.get(Agent, agent_id)
    if not agent:
        return _json_error("Bulunamadı", 404)
    return agent.to_dict()


@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: int,
    data: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
) -> Any:
    agent = await db.get(Agent, agent_id)
    if not agent:
        return _json_error("Bulunamadı", 404)
    agent.name = data.get("name", agent.name)
    agent.model_name = data.get("model_name", agent.model_name)
    agent.role = data.get("role", agent.role)
    agent.system_prompt = data.get("system_prompt", agent.system_prompt)
    agent.temperature = float(data.get("temperature", agent.temperature))
    await db.commit()
    return agent.to_dict()


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int, db: AsyncSession = Depends(get_db)) -> Any:
    agent = await db.get(Agent, agent_id)
    if not agent:
        return _json_error("Bulunamadı", 404)
    await db.delete(agent)
    await db.commit()
    return {"deleted": True}


# ------------------------------------------------------------------
# Debates
# ------------------------------------------------------------------


@router.get("/debates")
async def list_debates(request: Request, db: AsyncSession = Depends(get_db)) -> list[dict]:
    svc: DebateService = request.app.state.debate_svc
    debates = await svc.list_debates(db)
    return [d.to_dict() for d in debates]


@router.post("/debates", status_code=201)
async def create_debate(
    request: Request,
    data: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    try:
        debate = await svc.create_debate(
            db=db,
            topic=data["topic"],
            agent_ids=data["agent_ids"],
            max_rounds=int(data.get("max_rounds", 5)),
        )
        return debate.to_dict()
    except Exception as exc:
        return _json_error(str(exc), 400)


@router.get("/debates/{debate_id}")
async def get_debate(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    debate = await svc.get_debate(db, debate_id)
    if not debate:
        return _json_error("Bulunamadı", 404)
    return debate.to_dict()


@router.get("/debates/{debate_id}/messages")
async def get_messages(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    debate = await svc.get_debate(db, debate_id)
    if not debate:
        return _json_error("Bulunamadı", 404)
    return [m.to_dict() for m in debate.messages]


@router.post("/debates/{debate_id}/start")
async def start_debate(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "start")
        await svc.start_debate(db, debate_id)
        return {"started": True}
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


@router.post("/debates/{debate_id}/advance")
async def advance_debate(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "advance")
        await svc.advance_debate(db, debate_id)
        return {"advanced": True}
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


@router.post("/debates/{debate_id}/resolve")
async def resolve_debate(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "resolve")
        await svc.force_resolution(db, debate_id)
        return {"resolved": True}
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


@router.post("/debates/{debate_id}/analyze")
async def analyze_debate(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "analyze")
        res = await svc.analyze_debate(db, debate_id)
        return res
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


@router.get("/debates/{debate_id}/argument-map")
async def get_argument_map(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    try:
        res = await svc.get_argument_map(db, debate_id)
        return res
    except Exception as exc:
        return _json_error(str(exc), 404)


@router.post("/debates/{debate_id}/messages/{message_id}/regenerate")
async def regenerate_message(
    request: Request,
    debate_id: int,
    message_id: int,
    db: AsyncSession = Depends(get_db),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "regenerate")
        res = await svc.regenerate_message(db, debate_id, message_id)
        return res
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


@router.post("/debates/{debate_id}/auto-advance")
async def toggle_auto_advance(
    request: Request,
    debate_id: int,
    data: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    try:
        enabled = bool(data.get("enabled", False))
        debate = await svc.toggle_auto_advance(db, debate_id, enabled)
        return {"auto_advance": debate.auto_advance, "enabled": enabled}
    except Exception as exc:
        return _json_error(str(exc), 400)


@router.post("/debates/{debate_id}/run-full")
async def run_full_debate(
    request: Request,
    debate_id: int,
    db: AsyncSession = Depends(get_db),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    lock = None
    try:
        lock = await svc.acquire_operation_lock(debate_id, "run-full")
        await svc.run_full_debate(db, debate_id)
        return {"status": "running", "message": "Tartışma otomatik olarak ilerletiliyor"}
    except Exception as exc:
        return _json_error(str(exc), 400)
    finally:
        if lock and lock.locked():
            lock.release()


# ------------------------------------------------------------------
# Fact-check lab
# ------------------------------------------------------------------


@router.post("/fact-checks", status_code=201)
async def create_fact_check(
    request: Request,
    data: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
) -> Any:
    fact_svc: FactCheckService = request.app.state.fact_svc
    try:
        res = await fact_svc.analyze(
            db=db,
            text=data.get("text", ""),
            url=data.get("url", ""),
        )
        return res
    except Exception as exc:
        return _json_error(str(exc), 400)


@router.get("/fact-checks/{run_id}")
async def get_fact_check(
    request: Request, run_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    fact_svc: FactCheckService = request.app.state.fact_svc
    try:
        res = await fact_svc.get_run(db, run_id)
        return res
    except Exception as exc:
        return _json_error(str(exc), 404)


@router.post("/fact-checks/{run_id}/refresh")
async def refresh_fact_check(
    request: Request, run_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    fact_svc: FactCheckService = request.app.state.fact_svc
    try:
        res = await fact_svc.refresh(db, run_id)
        return res
    except Exception as exc:
        return _json_error(str(exc), 400)


@router.get("/fact-checks/token/{token}")
async def get_fact_check_by_token(
    request: Request, token: str, db: AsyncSession = Depends(get_db)
) -> Any:
    fact_svc: FactCheckService = request.app.state.fact_svc
    try:
        res = await fact_svc.get_run_by_token(db, token)
        return res
    except Exception as exc:
        return _json_error(str(exc), 404)


@router.get("/fact-checks/{run_id}/export")
async def export_fact_check(
    request: Request, run_id: int, db: AsyncSession = Depends(get_db)
) -> Any:
    fact_svc: FactCheckService = request.app.state.fact_svc
    try:
        data = await fact_svc.export_json(db, run_id)
        return JSONResponse(
            content=data,
            headers={
                "Content-Disposition": f'attachment; filename="fact-check-{run_id}.json"'
            },
        )
    except Exception as exc:
        return _json_error(str(exc), 404)


# ------------------------------------------------------------------
# ModernBERT Lab
# ------------------------------------------------------------------


@router.post("/modernbert/compare")
async def compare_modernbert(
    request: Request,
    data: dict = Body(default={}),
) -> Any:
    svc: DebateService = request.app.state.debate_svc
    try:
        target_text = data.get("target_text", "").strip()
        source_text = data.get("source_text", "").strip()
        if not target_text or not source_text:
            return _json_error("Hedef iddia ve yanıt metinleri boş olamaz.", 400)
            
        res = await anyio.to_thread.run_sync(
            svc.bert.compare_models,
            source_text,
            target_text,
        )
        return res
    except Exception as exc:
        return _json_error(str(exc), 400)


# ------------------------------------------------------------------
# Config Status for UI settings
# ------------------------------------------------------------------


@router.get("/config-status")
async def get_config_status() -> dict:
    from app.config import Config
    return {
        "OPENROUTER_API_KEY": bool(Config.OPENROUTER_API_KEY),
        "TAVILY_API_KEY": bool(Config.TAVILY_API_KEY),
        "GOOGLE_FACTCHECK_API_KEY": bool(Config.GOOGLE_FACTCHECK_API_KEY),
        "TCMB_EVDS_API_KEY": bool(Config.TCMB_EVDS_API_KEY),
        "TUIK_API_KEY": bool(Config.TUIK_API_KEY),
        "default_model": Config.DEFAULT_MODEL
    }


# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe. Returns 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db)) -> Any:
    """Readiness probe. Returns 200 only if the database is connectable."""
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        return JSONResponse(
            {"status": "not ready", "error": str(exc)},
            status_code=503,
        )
