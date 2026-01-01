from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import KnowledgeAgentSettings, get_settings
from .ingestion import normalize_namespace
from .knowledge_manager import KnowledgeManager
from .schemas import AgentStreamRequest, GenerateRequest, IngestRequest, MessagePayload, SearchRequest


def create_app() -> FastAPI:
    app = FastAPI(title="CrewAI Knowledge Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = get_settings()
    manager = KnowledgeManager(settings)

    settings_dependency = lambda: settings
    manager_dependency = lambda: manager

    @app.get("/healthz")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    def _parse_sources_field(raw: Optional[Any]) -> List[Any]:
        if raw is None:
            return []
        if isinstance(raw, list):
            results: List[Any] = []
            for item in raw:
                if isinstance(item, str):
                    try:
                        results.append(json.loads(item))
                    except json.JSONDecodeError:
                        results.append(item)
                else:
                    results.append(item)
            return results
        if isinstance(raw, str):
            value = raw.strip()
            if not value:
                return []
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except json.JSONDecodeError:
                return [value]
        return [raw]

    def _resolve_namespace(request: AgentStreamRequest) -> Optional[str]:
        if isinstance(request.namespace, str) and request.namespace.strip():
            return normalize_namespace(request.namespace)
        tool_params = request.tool_params or {}
        namespace = tool_params.get("namespace") if isinstance(tool_params, dict) else None
        if isinstance(namespace, str) and namespace.strip():
            return normalize_namespace(namespace)
        return None

    def _chunk_text(content: str, *, size: int = 300) -> List[str]:
        if not content:
            return []
        chunks: List[str] = []
        text = content.strip()
        for start in range(0, len(text), size):
            chunks.append(text[start : start + size])
        return chunks

    @app.post("/api/tools/ingest")
    async def ingest_sources(
        request: Request,
        files: Optional[List[UploadFile]] = File(default=None),
        namespace: Optional[str] = Form(default=None),
        sources: Optional[str] = Form(default=None),
        settings: KnowledgeAgentSettings = Depends(settings_dependency),
        manager: KnowledgeManager = Depends(manager_dependency),
    ):
        raw_sources: List[Any] = []
        payload_namespace: Optional[str] = namespace

        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            body = await request.json()
            try:
                ingest_request = IngestRequest.model_validate(body)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            raw_sources = list(ingest_request.sources or [])
            payload_namespace = ingest_request.namespace or payload_namespace
        else:
            if sources is not None:
                raw_sources = _parse_sources_field(sources)
            else:
                form = await request.form()
                payload_namespace = payload_namespace or form.get("namespace")
                raw_sources = _parse_sources_field(form.getlist("sources"))

        result = await manager.ingest(
            namespace=payload_namespace,
            raw_sources=raw_sources,
            uploads=files,
        )

        status_code = 200
        if result["errors"] and not result["saved"]:
            status_code = 400
        elif result["errors"]:
            status_code = 207
        return JSONResponse(status_code=status_code, content=result)

    @app.post("/api/tools/searchDocs")
    async def search_docs(
        request: SearchRequest,
        manager: KnowledgeManager = Depends(manager_dependency),
    ):
        result = await manager.search(
            namespace=request.namespace,
            query=request.query,
            max_results=request.max_results,
        )
        if result.get("error"):
            return JSONResponse(status_code=400, content=result)
        return result

    @app.post("/api/agents/knowledge/generate")
    async def generate_completion(
        request: GenerateRequest,
        manager: KnowledgeManager = Depends(manager_dependency),
    ):
        if not request.messages:
            raise HTTPException(status_code=400, detail="Provide at least one message.")
        content = await manager.run_agent(namespace=request.namespace, messages=request.messages)
        return {"content": content}

    @app.post("/stream")
    async def agent_stream(
        request: AgentStreamRequest,
        manager: KnowledgeManager = Depends(manager_dependency),
    ):
        if not request.messages:
            raise HTTPException(status_code=400, detail="Provide at least one message.")

        namespace = _resolve_namespace(request)
        thread_id = request.thread_id or f"thread_{uuid4().hex[:8]}"
        run_id = request.run_id or f"run_{uuid4().hex[:8]}"

        async def event_generator():
            try:
                content = await manager.run_agent(namespace=namespace, messages=request.messages)
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

    uvicorn.run("knowledge_agent.main:app", host="0.0.0.0", port=8000, reload=False)
