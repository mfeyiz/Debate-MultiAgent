"""Flask application factory."""

from flask import Flask
from flask_migrate import Migrate

from app.config import Config
from app.extensions import db

migrate = Migrate()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from app.routers.pages import pages_bp
    from app.routers.api import api_bp
    from app.routers.stream import stream_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(stream_bp, url_prefix="/stream")

    # Create tables (dev only) and seed defaults
    with app.app_context():
        if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
            db.create_all()
        _seed_defaults()

    return app


def _seed_defaults() -> None:
    """Seed default agents if the database is empty."""
    from app.models import Agent
    from app.extensions import db

    if Agent.query.first():
        return

    defaults = [
        Agent(
            name="Agent Alpha",
            model_name="openai/gpt-4o-mini",
            role="proponent",
            system_prompt=(
                "You are a rigorous proponent. Your job is to construct well-supported "
                "claims using evidence, data, and logical reasoning. Always ground your "
                "arguments in facts. Be concise and specific."
            ),
            temperature=0.7,
        ),
        Agent(
            name="Agent Beta",
            model_name="openai/gpt-4o-mini",
            role="opponent",
            system_prompt=(
                "You are a critical opponent. Your job is to identify weaknesses, "
                "contradictions, and logical gaps in the opposing view. Attack claims "
                "with precision and cite counter-evidence when possible. Be concise and specific."
            ),
            temperature=0.7,
        ),
    ]
    db.session.add_all(defaults)
    db.session.commit()
