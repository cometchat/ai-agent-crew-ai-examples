from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv

# Load .env before any other imports that might need env vars
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
load_dotenv()  # Also try .env in cwd

from crewai.types.streaming import CrewStreamingOutput, StreamChunk, StreamChunkType
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

    def _detect_ranking_intent(text: str) -> bool:
        if not text:
            return False
        advice_phrases = [
            "best time",
            "best day",
            "best hour",
            "best practices",
            "best strategy",
            "best way",
            "how to launch",
            "launch strategy",
            "launch plan",
        ]
        if any(phrase in text for phrase in advice_phrases):
            return False
        ranking_keywords = [
            "top",
            "trending",
            "leaderboard",
            "rank",
            "ranking",
            "highest",
            "best",
            "most upvoted",
            "most voted",
            "product of the day",
        ]
        context_words = ["product hunt", "product", "post", "launch", "votes", "upvotes", "score"]
        if not any(word in text for word in context_words):
            return False
        return any(keyword in text for keyword in ranking_keywords)

    def _extract_limit_hint(text: str) -> Optional[int]:
        match = re.search(r"(?:top|first|no\\.?|number|#)\\s*(\\d{1,2})", text)
        if not match:
            match = re.search(r"\\b(\\d{1,2})\\s+(?:top|best|trending|ranked|ranking|products?|posts?)\\b", text)
        if match:
            value = int(match.group(1))
            return max(1, min(10, value))

        singleton_patterns = [
            r"\\bhighest\\b",
            r"\\btop\\s+spot\\b",
            r"\\b#\\s*1\\b",
            r"\\bno\\.?\\s*1\\b",
            r"\\bnumber\\s+1\\b",
            r"\\bproduct\\s+of\\s+the\\s+day\\b",
            r"\\bwinner\\b",
        ]
        if any(re.search(pattern, text) for pattern in singleton_patterns):
            return 1
        return None

    def _extract_timeframe_hint(text: str) -> Optional[str]:
        phrases = [
            "today",
            "yesterday",
            "this week",
            "last week",
            "this month",
            "last month",
        ]
        for phrase in phrases:
            if phrase in text:
                return phrase
        match = re.search(r"(?:past|last)\\s+\\d{1,2}\\s+days?", text)
        if match:
            return match.group(0)
        match = re.search(r"(\\d{4}-\\d{2}-\\d{2})", text)
        if match:
            return match.group(1)
        match = re.search(r"from[:=]\\s*\\d{4}-\\d{2}-\\d{2}.*to[:=]\\s*\\d{4}-\\d{2}-\\d{2}", text)
        if match:
            return match.group(0)
        return None

    def _extract_timezone_hint(text: str) -> Optional[str]:
        match = re.search(r"\\b([A-Za-z]+/[A-Za-z_]+)\\b", text)
        return match.group(1) if match else None

    def _build_intent_hint(question: str) -> str:
        if not question:
            return "none"
        lower = question.lower()
        hints: List[str] = []
        ranking_intent = _detect_ranking_intent(lower)
        search_intent = any(keyword in lower for keyword in ["search", "find", "look up", "lookup"])

        if ranking_intent:
            hints.append("Intent: rankings.")
            limit_hint = _extract_limit_hint(lower)
            if limit_hint:
                hints.append(f"Limit: {limit_hint}.")

        if search_intent:
            hints.append("Intent: search.")

        timeframe_hint = _extract_timeframe_hint(lower)
        if timeframe_hint:
            hints.append(f"Timeframe: {timeframe_hint}.")

        timezone_hint = _extract_timezone_hint(question)
        if timezone_hint:
            hints.append(f"Timezone: {timezone_hint}.")

        return " ".join(hints) if hints else "none"

    def _is_greeting(text: str) -> bool:
        if not text:
            return False
        cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        if not cleaned:
            return False
        tokens = [token for token in cleaned.split() if token]
        greetings = {
            "hi",
            "hello",
            "hey",
            "hiya",
            "yo",
            "sup",
            "morning",
            "afternoon",
            "evening",
            "greetings",
            "howdy",
        }
        fillers = {"there", "everyone", "team", "folks", "friend", "please"}
        if not any(token in greetings for token in tokens):
            return False
        return all(token in greetings or token in fillers for token in tokens)

    def _ndjson(payload: Dict[str, object]) -> bytes:
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

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

        latest = messages[-1].content
        if _is_greeting(latest):
            return {"content": "Hi! How can I help with Product Hunt today?"}

        crew = create_product_hunt_crew(cfg, stream=False)
        conversation = _format_conversation(messages)
        intent_hint = _build_intent_hint(latest)

        result = await asyncio.to_thread(
            crew.kickoff,
            inputs={"conversation": conversation, "question": latest, "intent_hint": intent_hint},
        )
        return {"content": _stringify_result(result)}

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
            message_id = f"msg_{uuid4().hex[:8]}"

            try:
                latest = request.messages[-1].content
                if _is_greeting(latest):
                    yield _ndjson(
                        {"type": "text_start", "message_id": message_id, "thread_id": thread_id, "run_id": run_id}
                    )
                    yield _ndjson(
                        {
                            "type": "text_delta",
                            "message_id": message_id,
                            "content": "Hi! How can I help with Product Hunt today?",
                            "thread_id": thread_id,
                            "run_id": run_id,
                        }
                    )
                    yield _ndjson(
                        {"type": "text_end", "message_id": message_id, "thread_id": thread_id, "run_id": run_id}
                    )
                    yield _ndjson({"type": "done", "thread_id": thread_id, "run_id": run_id})
                    return

                crew = create_product_hunt_crew(cfg, stream=True)
                conversation = _format_conversation(request.messages)
                intent_hint = _build_intent_hint(latest)

                streaming: CrewStreamingOutput = await asyncio.to_thread(
                    crew.kickoff,
                    inputs={"conversation": conversation, "question": latest, "intent_hint": intent_hint},
                )

                yield _ndjson({"type": "text_start", "message_id": message_id, "thread_id": thread_id, "run_id": run_id})

                buffer = ""
                final_answer_started = False
                streamed_any_text = False
                thought_markers = ["Thought:", "Action:", "Action Input:", "Observation:"]

                for chunk in streaming:
                    if not isinstance(chunk, StreamChunk):
                        continue

                    if chunk.chunk_type == StreamChunkType.TOOL_CALL and chunk.tool_call:
                        tool_call_id = (
                            getattr(chunk.tool_call, "tool_call_id", None)
                            or getattr(chunk.tool_call, "id", None)
                            or f"tool_{uuid4().hex[:8]}"
                        )
                        tool_name = getattr(chunk.tool_call, "tool_name", None) or getattr(chunk.tool_call, "name", None)

                        if tool_name:
                            yield _ndjson(
                                {
                                    "type": "tool_call_start",
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                    "thread_id": thread_id,
                                    "run_id": run_id,
                                }
                            )

                        args = None
                        if getattr(chunk.tool_call, "arguments", None):
                            try:
                                raw_args = chunk.tool_call.arguments
                                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            except json.JSONDecodeError:
                                args = None
                        if args is not None:
                            yield _ndjson(
                                {
                                    "type": "tool_call_args",
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                    "args": args,
                                    "thread_id": thread_id,
                                    "run_id": run_id,
                                }
                            )

                        if tool_name:
                            yield _ndjson(
                                {
                                    "type": "tool_call_end",
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                    "thread_id": thread_id,
                                    "run_id": run_id,
                                }
                            )

                        if getattr(chunk.tool_call, "result", None) is not None:
                            yield _ndjson(
                                {
                                    "type": "tool_result",
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                    "result": chunk.tool_call.result,
                                    "thread_id": thread_id,
                                    "run_id": run_id,
                                }
                            )

                        continue

                    if chunk.chunk_type == StreamChunkType.TEXT:
                        content = chunk.content or ""
                        if not content:
                            continue
                        buffer += content

                        if not final_answer_started:
                            if "Final Answer:" in buffer:
                                final_answer_started = True
                                after = buffer.split("Final Answer:", 1)[1].lstrip()
                                if after:
                                    yield _ndjson(
                                        {
                                            "type": "text_delta",
                                            "message_id": message_id,
                                            "content": after,
                                            "thread_id": thread_id,
                                            "run_id": run_id,
                                        }
                                    )
                                    streamed_any_text = True
                                buffer = ""
                            elif any(marker in buffer for marker in thought_markers):
                                buffer = ""
                        else:
                            if any(marker in content for marker in thought_markers):
                                final_answer_started = False
                                break
                            yield _ndjson(
                                {
                                    "type": "text_delta",
                                    "message_id": message_id,
                                    "content": content,
                                    "thread_id": thread_id,
                                    "run_id": run_id,
                                }
                            )
                            streamed_any_text = True

                if not streamed_any_text:
                    fallback = getattr(streaming, "result", None)
                    if isinstance(fallback, str) and "Final Answer:" in fallback:
                        fallback = fallback.split("Final Answer:", 1)[1].lstrip()
                    final_text = fallback or buffer.strip()
                    if final_text:
                        yield _ndjson(
                            {
                                "type": "text_delta",
                                "message_id": message_id,
                                "content": final_text,
                                "thread_id": thread_id,
                                "run_id": run_id,
                            }
                        )

                yield _ndjson({"type": "text_end", "message_id": message_id, "thread_id": thread_id, "run_id": run_id})
                yield _ndjson({"type": "done", "thread_id": thread_id, "run_id": run_id})
            except Exception as exc:
                error_payload = {
                    "type": "error",
                    "message": str(exc),
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "run_id": run_id,
                }
                yield _ndjson(error_payload)

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("product_hunt_agent.main:app", host="0.0.0.0", port=8001, reload=False)
