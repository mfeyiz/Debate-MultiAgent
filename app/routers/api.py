"""REST API routes."""

from flask import Blueprint, jsonify, request

from app.extensions import db
from app.models import Agent, Debate
from app.services.debate_service import DebateService

api_bp = Blueprint("api", __name__)
svc = DebateService()


# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------

@api_bp.route("/agents", methods=["GET"])
def list_agents():
    agents = svc.list_agents()
    return jsonify([a.to_dict() for a in agents])


@api_bp.route("/agents", methods=["POST"])
def create_agent():
    data = request.get_json() or {}
    try:
        agent = svc.create_agent(
            name=data["name"],
            model_name=data.get("model_name", ""),
            role=data.get("role", "proponent"),
            system_prompt=data.get("system_prompt", ""),
            temperature=float(data.get("temperature", 0.7)),
        )
        return jsonify(agent.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/agents/<int:agent_id>", methods=["GET"])
def get_agent(agent_id: int):
    agent = db.session.get(Agent, agent_id)
    if not agent:
        return jsonify({"error": "Bulunamadı"}), 404
    return jsonify(agent.to_dict())


@api_bp.route("/agents/<int:agent_id>", methods=["PUT"])
def update_agent(agent_id: int):
    agent = db.session.get(Agent, agent_id)
    if not agent:
        return jsonify({"error": "Bulunamadı"}), 404
    data = request.get_json() or {}
    agent.name = data.get("name", agent.name)
    agent.model_name = data.get("model_name", agent.model_name)
    agent.role = data.get("role", agent.role)
    agent.system_prompt = data.get("system_prompt", agent.system_prompt)
    agent.temperature = float(data.get("temperature", agent.temperature))
    db.session.commit()
    return jsonify(agent.to_dict())


@api_bp.route("/agents/<int:agent_id>", methods=["DELETE"])
def delete_agent(agent_id: int):
    agent = db.session.get(Agent, agent_id)
    if not agent:
        return jsonify({"error": "Bulunamadı"}), 404
    db.session.delete(agent)
    db.session.commit()
    return jsonify({"deleted": True})


# ------------------------------------------------------------------
# Debates
# ------------------------------------------------------------------

@api_bp.route("/debates", methods=["GET"])
def list_debates():
    debates = svc.list_debates()
    return jsonify([d.to_dict() for d in debates])


@api_bp.route("/debates", methods=["POST"])
def create_debate():
    data = request.get_json() or {}
    try:
        debate = svc.create_debate(
            topic=data["topic"],
            agent_ids=data["agent_ids"],
            max_rounds=int(data.get("max_rounds", 5)),
        )
        return jsonify(debate.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/debates/<int:debate_id>", methods=["GET"])
def get_debate(debate_id: int):
    debate = svc.get_debate(debate_id)
    if not debate:
        return jsonify({"error": "Bulunamadı"}), 404
    return jsonify(debate.to_dict())


@api_bp.route("/debates/<int:debate_id>/messages", methods=["GET"])
def get_messages(debate_id: int):
    debate = svc.get_debate(debate_id)
    if not debate:
        return jsonify({"error": "Bulunamadı"}), 404
    return jsonify([m.to_dict() for m in debate.messages])


@api_bp.route("/debates/<int:debate_id>/start", methods=["POST"])
def start_debate(debate_id: int):
    try:
        svc.start_debate(debate_id)
        return jsonify({"started": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/debates/<int:debate_id>/advance", methods=["POST"])
def advance_debate(debate_id: int):
    try:
        svc.advance_debate(debate_id)
        return jsonify({"advanced": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/debates/<int:debate_id>/resolve", methods=["POST"])
def resolve_debate(debate_id: int):
    try:
        svc.force_resolution(debate_id)
        return jsonify({"resolved": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------

@api_bp.route("/healthz", methods=["GET"])
def healthz():
    """Liveness probe. Returns 200 if the process is alive."""
    return jsonify({"status": "ok"}), 200


@api_bp.route("/ready", methods=["GET"])
def ready():
    """Readiness probe. Returns 200 only if the database is connectable."""
    try:
        from sqlalchemy import text
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        return jsonify({"status": "not ready", "error": str(e)}), 503
