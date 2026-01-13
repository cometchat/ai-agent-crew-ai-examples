from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv

# Load .env before any other imports that might need env vars
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
load_dotenv()  # Also try .env in cwd

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent_builder import create_product_hunt_crew
from .config import ProductHuntSettings, get_settings
from .schemas import AgentStreamRequest, ChatRequest, MessagePayload
from .services import (
    get_top_products_by_timeframe,
    get_top_products_by_votes,
    get_top_products_this_week,
    parse_timeframe,
    search_products,
)


def create_app() -> FastAPI:
    app = FastAPI(title="CrewAI Product Hunt Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = get_settings()

    settings_dependency = lambda: settings

    def _format_conversation(messages: List[MessagePayload]) -> str:
        lines: List[str] = []
        for msg in messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)

    def _stringify_result(result: object) -> str:
        if result is None:
            return ""
        for attr in ("raw", "output", "result"):
            if hasattr(result, attr):
                try:
                    return str(getattr(result, attr))
                except Exception:
                    continue
        return str(result)

    def _chunk_text(content: str, *, size: int = 320) -> List[str]:
        if not content:
            return []
        content = content.strip()
        return [content[i : i + size] for i in range(0, len(content), size)]

    @app.get("/healthz")
    async def healthcheck() -> dict:
        return {"status": "ok"}

    @app.get("/api/top")
    async def fetch_top_votes(
        limit: int = Query(default=3, ge=1, le=10),
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        posts = await get_top_products_by_votes(limit, cfg)
        return {"posts": posts, "first": limit, "order": "VOTES"}

    @app.get("/api/top-week")
    async def fetch_top_week(
        limit: int = Query(default=3, ge=1, le=10),
        days: int = Query(default=7, ge=1, le=31),
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        posts = await get_top_products_this_week(limit, days, cfg)
        return {"posts": posts, "first": limit, "days": days, "order": "RANKING", "window": "rolling-week"}

    @app.get("/api/top-range")
    async def fetch_top_timeframe(
        timeframe: Optional[str] = Query(default="today"),
        tz: Optional[str] = Query(default=None),
        limit: int = Query(default=3, ge=1, le=10),
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        posts = await get_top_products_by_timeframe(
            first=limit,
            timeframe=timeframe,
            tz=tz,
            settings=cfg,
        )
        window = parse_timeframe(timeframe, tz, settings=cfg)
        return {
            "posts": posts,
            "first": limit,
            "timeframe": timeframe or "today",
            "tz": tz or cfg.default_timezone,
            "order": "RANKING",
            "window": window,
        }

    @app.get("/api/search")
    async def search_endpoint(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=10, ge=1, le=50),
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        hits = await search_products(q, limit=limit, settings=cfg)
        return {"hits": hits, "q": q, "limit": limit}

    @app.post("/api/chat")
    async def chat_endpoint(
        payload: ChatRequest,
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        messages: List[MessagePayload] = []
        if payload.messages:
            messages.extend(payload.messages)
        if payload.message:
            messages.append(MessagePayload(role="user", content=payload.message))

        if not messages:
            raise HTTPException(status_code=400, detail="Provide message or messages with textual content.")

        crew = create_product_hunt_crew(cfg)
        conversation = _format_conversation(messages)
        latest = messages[-1].content

        result = await asyncio.to_thread(
            crew.kickoff,
            inputs={"conversation": conversation, "question": latest},
        )
        return {"content": _stringify_result(result)}

    @app.post("/kickoff")
    async def kickoff_endpoint(
        request: AgentStreamRequest,
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        if not request.messages:
            raise HTTPException(status_code=400, detail="Provide a non-empty array of messages.")

        thread_id = request.thread_id
        run_id = request.run_id

        async def ndjson_generator():
            message_id = f"msg_{uuid4().hex[:8]}"
            try:
                yield (
                    json.dumps(
                        {"type": "text_start", "message_id": message_id, "thread_id": thread_id, "run_id": run_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")

                crew = create_product_hunt_crew(cfg)
                conversation = _format_conversation(request.messages)
                latest = request.messages[-1].content

                result = await asyncio.to_thread(
                    crew.kickoff,
                    inputs={"conversation": conversation, "question": latest},
                )
                content = _stringify_result(result).strip()

                for chunk in _chunk_text(content, size=320):
                    yield (
                        json.dumps(
                            {
                                "type": "text_delta",
                                "message_id": message_id,
                                "thread_id": thread_id,
                                "run_id": run_id,
                                "content": chunk,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8")

                yield (
                    json.dumps(
                        {"type": "text_end", "message_id": message_id, "thread_id": thread_id, "run_id": run_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
                yield (json.dumps({"type": "done", "thread_id": thread_id, "run_id": run_id}) + "\n").encode("utf-8")
            except Exception as exc:
                error_payload = {
                    "type": "error",
                    "message": str(exc),
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "message_id": message_id,
                }
                yield (json.dumps(error_payload, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(
            ndjson_generator(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/stream")
    async def agent_stream(
        request: AgentStreamRequest,
        cfg: ProductHuntSettings = Depends(settings_dependency),
    ):
        if not request.messages:
            raise HTTPException(status_code=400, detail="Provide a non-empty array of messages.")

        thread_id = request.thread_id or f"thread_{uuid4().hex[:8]}"
        run_id = request.run_id or f"run_{uuid4().hex[:8]}"

        async def event_generator():
            try:
                crew = create_product_hunt_crew(cfg)
                conversation = _format_conversation(request.messages)
                latest = request.messages[-1].content

                result = await asyncio.to_thread(
                    crew.kickoff,
                    inputs={"conversation": conversation, "question": latest},
                )
                content = _stringify_result(result)

                streamed = False
                for chunk in _chunk_text(content, size=320):
                    payload = {"type": "text_delta", "content": chunk, "thread_id": thread_id, "run_id": run_id}
                    yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                    streamed = True
                if streamed:
                    yield (
                        json.dumps({"type": "text_done", "thread_id": thread_id, "run_id": run_id}, ensure_ascii=False)
                        + "\n"
                    ).encode("utf-8")
                yield (json.dumps({"type": "done", "thread_id": thread_id, "run_id": run_id}) + "\n").encode("utf-8")
            except Exception as exc:
                error_payload = {
                    "type": "error",
                    "message": str(exc),
                    "thread_id": thread_id,
                    "run_id": run_id,
                }
                yield (json.dumps(error_payload, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("product_hunt_agent.main:app", host="0.0.0.0", port=8001, reload=False)
