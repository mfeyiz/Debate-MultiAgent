"""Server-Sent Events (SSE) streaming routes."""

import json
import queue
import time

from flask import Blueprint, Response

from app.services.debate_service import EventBroker

stream_bp = Blueprint("stream", __name__)


def _event_stream(debate_id: int):
    """Yield SSE events for a specific debate."""
    q: queue.Queue = EventBroker.subscribe(debate_id)
    try:
        while True:
            try:
                event = q.get(timeout=15)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    finally:
        EventBroker.unsubscribe(debate_id, q)


@stream_bp.route("/<int:debate_id>")
def stream(debate_id: int):
    return Response(
        _event_stream(debate_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
