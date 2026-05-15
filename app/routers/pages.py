"""HTML page routes."""

from flask import Blueprint, render_template

from app.extensions import db
from app.models import Agent, Debate
from app.services.debate_service import DebateService

pages_bp = Blueprint("pages", __name__)
svc = DebateService()


@pages_bp.route("/")
def dashboard():
    debates = svc.list_debates()
    agents = svc.list_agents()
    stats = {
        "active_debates": len([d for d in debates if d.status == "active"]),
        "total_debates": len(debates),
        "total_agents": len(agents),
    }
    return render_template("dashboard.html", debates=debates, agents=agents, stats=stats)


@pages_bp.route("/debate/<int:debate_id>")
def live_arena(debate_id: int):
    debate = svc.get_debate(debate_id)
    if not debate:
        return render_template("404.html"), 404
    agents = {p.agent_id: p.agent for p in debate.participants}
    messages = debate.messages.all()
    return render_template("live_arena.html", debate=debate, agents=agents, messages=messages)


@pages_bp.route("/agents")
def agent_management():
    agents = svc.list_agents()
    return render_template("agents.html", agents=agents)


@pages_bp.route("/analytics")
def pipeline_analytics():
    debates = svc.list_debates()
    analyses = []
    for debate in debates:
        for msg in debate.messages:
            cv = msg.current_version
            if cv:
                analyses.extend(cv.analyses.all())
    return render_template("analytics.html", debates=debates, analyses=analyses)
