"""FastAPI router for Server-Sent Events (SSE) streaming."""

from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.debate_service import EventBroker


router = APIRouter()


async def event_stream(debate_id: int) -> AsyncIterator[str]:
    """Yield SSE events for a specific debate."""
    q: queue.Queue = EventBroker.subscribe(debate_id)
    try:
        while True:
            try:
                # Run the blocking queue get in a worker thread to keep event loop free
                event = await asyncio.to_thread(q.get, True, 15)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    finally:
        EventBroker.unsubscribe(debate_id, q)


@router.get("/stream/{debate_id}")
def stream(debate_id: int) -> StreamingResponse:
    """Stream debate progress events using Server-Sent Events."""
    return StreamingResponse(
        event_stream(debate_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
