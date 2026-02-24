# CrewAI Agent Examples

Knowledge-base and Product Hunt agents implemented with [CrewAI](https://docs.crewai.com/) and FastAPI. The services mirror the Agno and Vercel agent examples but use CrewAI crews plus simple SSE streams that work with CometChat or any SSE client.

- `knowledge_agent` - ingest docs, semantic search with Chroma, CrewAI-backed answers, and a CometChat-style `/stream`.
- `product_hunt_agent` - Product Hunt launch assistant with GraphQL, Algolia search tools, and celebratory confetti payloads.

## Prerequisites

- Python 3.10 or newer (3.11 recommended)
- `OPENAI_API_KEY` with access to GPT-4o or a compatible model
- Optional: `PRODUCTHUNT_API_TOKEN` for live Product Hunt data (the agent still replies with empty datasets if missing)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Environment variables can live in a project-level `.env` or service-specific `.env` inside `knowledge_agent/` or `product_hunt_agent/`.

```env
# Shared
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1
# KNOWLEDGE_OPENAI_MODEL=gpt-4o-mini
# KNOWLEDGE_EMBEDDING_MODEL=text-embedding-3-small

# Product Hunt agent
PRODUCTHUNT_API_TOKEN=phc-...
# PRODUCTHUNT_DEFAULT_TIMEZONE=America/New_York
```

## Knowledge Agent

```bash
uvicorn knowledge_agent.main:app --host 0.0.0.0 --port 8000 --reload
```

Key endpoints:
- `POST /api/tools/ingest` - ingest text/markdown/URLs/uploads into a namespace (defaults to `default`).
- `POST /api/tools/searchDocs` - semantic search over the namespace (Chroma + OpenAI embeddings).
- `POST /api/agents/knowledge/generate` - non-streaming answer via a CrewAI agent.
- `POST /stream` - newline-delimited SSE (`text_delta`, `text_done`, `done`) compatible with CometChat BYOA.

Example ingestion:

```bash
curl -X POST http://localhost:8000/api/tools/ingest \
  -H "Content-Type: application/json" \
  -d '{
        "namespace": "default",
        "sources": [
          { "type": "url", "value": "https://docs.crewai.com/" },
          { "type": "markdown", "title": "Notes", "value": "# CrewAI Rocks" }
        ]
      }'
```

Streaming example:

```bash
curl -N http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{
        "thread_id": "thread_1",
        "run_id": "run_001",
        "messages": [
          { "role": "user", "content": "Summarize the CrewAI agent lifecycle." }
        ]
      }'
```

Streaming format (NDJSON):
- `text_start` -> `text_delta` chunks -> `text_end` -> `done`
- Tool events: `tool_call_start`, `tool_call_args`, `tool_call_end`, `tool_result`
- Payloads include `message_id`, `thread_id`, `run_id`, and `content` for deltas.

## Product Hunt Agent

```bash
uvicorn product_hunt_agent.main:app --host 0.0.0.0 --port 8001 --reload
```

Key endpoints:
- `GET /api/top`, `/api/top-week`, `/api/top-range` - Product Hunt GraphQL lookups.
- `GET /api/search` - Product Hunt Algolia search.
- `POST /api/chat` - non-streaming chat with the CrewAI agent and tool calls.
- `POST /stream` - SSE stream (`text_delta`, `text_done`, `done`) chunked from the agent response.

Streaming example:

```bash
curl -N http://localhost:8001/stream \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
          { "role": "user", "content": "What were the top launches last week?" }
        ]
      }'
```

Streaming format matches the knowledge agent (`text_start` -> `text_delta` -> `text_end` -> `done` with tool call events and message/run/thread IDs).

## Streaming Notes

Both `/stream` routes emit newline-delimited JSON (`application/x-ndjson`) with `thread_id`, `run_id`, `type`, and `content`. Tool call events are emitted when the CrewAI agent invokes tools.
