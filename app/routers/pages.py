"""FastAPI router for HTML pages."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.templating import Jinja2Templates

from app.database import get_db
from app.models import AnalysisRun
from app.services.debate_service import DebateService


router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _template(
    request: Request,
    name: str,
    context: dict | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    payload = context or {}
    # Demo (presentation) mode: a cookie toggle that re-points the live debate
    # and fact-check nav links to pre-seeded, fully-analyzed demo data so the
    # real pages render instantly with no model/API calls.
    demo_state = getattr(request.app.state, "demo", None) or {}
    payload.setdefault("demo_active", request.cookies.get("demo_mode") == "1")
    payload.setdefault("demo_debate_id", demo_state.get("debate_id"))
    payload.setdefault("demo_fact_token", demo_state.get("fact_token"))
    return templates.TemplateResponse(request, name, payload, status_code=status_code)


@router.get("/", response_class=HTMLResponse, name="pages.dashboard")
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    debates = await svc.list_debates(db)
    agents = await svc.list_agents(db)
    stats = {
        "active_debates": len([d for d in debates if d.status == "active"]),
        "total_debates": len(debates),
        "total_agents": len(agents),
    }
    return _template(
        request,
        "dashboard.html",
        {
            "debates": debates,
            "agents": agents,
            "stats": stats,
            "active_page": "dashboard",
        },
    )


@router.get("/debate/{debate_id}", response_class=HTMLResponse, name="pages.live_arena")
async def live_arena(
    request: Request, debate_id: int, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    debate = await svc.get_debate(db, debate_id)
    if not debate:
        all_debates = await svc.list_debates(db)
        return _template(request, "404.html", {"debates": all_debates}, 404)
    agents = {p.agent_id: p.agent for p in debate.participants}
    messages = debate.messages.all()
    all_debates = await svc.list_debates(db)
    return _template(
        request,
        "live_arena.html",
        {
            "debate": debate,
            "debates": all_debates,
            "agents": agents,
            "messages": messages,
            "active_page": "live_arena",
        },
    )


@router.get("/agents", response_class=HTMLResponse, name="pages.agent_management")
async def agent_management(request: Request, db: AsyncSession = Depends(get_db)) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    agents = await svc.list_agents(db)
    debates = await svc.list_debates(db)
    return _template(
        request,
        "agents.html",
        {
            "agents": agents,
            "debates": debates,
            "active_page": "agent_management",
        },
    )


@router.get("/analytics", response_class=HTMLResponse, name="pages.pipeline_analytics")
async def pipeline_analytics(
    request: Request, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    debates = await svc.list_debates(db)
    analyses = []

    stmt = select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(10)
    result = await db.execute(stmt)
    analysis_runs = result.scalars().all()

    for debate in debates:
        for msg in debate.messages:
            cv = msg.current_version
            if cv:
                analyses.extend(cv.analyses.all())
    return _template(
        request,
        "analytics.html",
        {
            "debates": debates,
            "analyses": analyses,
            "analysis_runs": analysis_runs,
            "active_page": "pipeline_analytics",
        },
    )


@router.get("/fact-check", response_class=HTMLResponse, name="pages.fact_check_lab")
async def fact_check_lab(request: Request, db: AsyncSession = Depends(get_db)) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    debates = await svc.list_debates(db)
    run_token = request.query_params.get("run_token")
    return _template(
        request,
        "fact_check.html",
        {
            "debates": debates,
            "active_page": "fact_check_lab",
            "run_token": run_token,
        },
    )


@router.get("/modernbert-lab", response_class=HTMLResponse, name="pages.modernbert_lab")
async def modernbert_lab(request: Request, db: AsyncSession = Depends(get_db)) -> HTMLResponse:
    svc: DebateService = request.app.state.debate_svc
    debates = await svc.list_debates(db)
    return _template(
        request,
        "modernbert.html",
        {
            "debates": debates,
            "active_page": "modernbert_lab",
        },
    )


@router.get("/demo", name="pages.demo")
async def demo_toggle(request: Request) -> RedirectResponse:
    """Toggle presentation/demo mode (a cookie). When on, the real Canlı Tartışma
    and Teyit Lab tabs render the pre-seeded, fully-analyzed demo data."""
    currently_on = request.cookies.get("demo_mode") == "1"
    response = RedirectResponse(url=str(request.url_for("pages.dashboard")), status_code=303)
    if currently_on:
        response.delete_cookie("demo_mode")
    else:
        response.set_cookie("demo_mode", "1", max_age=60 * 60 * 24 * 7, samesite="lax")
    return response
