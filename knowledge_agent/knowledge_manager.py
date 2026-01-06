from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import KnowledgeAgentSettings, get_settings
from .ingestion import IngestedDocument, collect_documents, hash_content, normalize_namespace
from .schemas import MessagePayload


@dataclass
class NamespaceContext:
    name: str
    vector_store: Chroma
    directory: Path
    chroma_path: Path
    document_hashes: set[str]
    lock: asyncio.Lock


class KnowledgeManager:
    """Orchestrates ingestion, retrieval, and CrewAI agent creation per namespace."""

    def __init__(self, settings: Optional[KnowledgeAgentSettings] = None) -> None:
        self.settings = settings or get_settings()
        self._namespaces: Dict[str, NamespaceContext] = {}
        self._namespaces_lock: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None

        if current and current is not self._loop:
            self._loop = current
        elif self._loop is None or self._loop.is_closed():
            self._loop = current or asyncio.new_event_loop()
        return self._loop

    def _namespaces_lock_obj(self) -> asyncio.Lock:
        if self._namespaces_lock is None:
            self._namespaces_lock = asyncio.Lock()
        return self._namespaces_lock

    async def ingest(
        self,
        *,
        namespace: Optional[str],
        raw_sources: Optional[Sequence[Any]],
        uploads: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        """Parse and persist sources into the namespace knowledge base."""

        normalised = normalize_namespace(namespace)
        ctx = await self._get_namespace(normalised)

        documents, parse_errors = await collect_documents(raw_sources, uploads, self.settings, normalised)
        saved: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = list(parse_errors)

        async with ctx.lock:
            for document in documents:
                document.metadata.setdefault("namespace", normalised)
                document.metadata.setdefault("hash", document.hash)

                if document.hash in ctx.document_hashes:
                    skipped.append(
                        {
                            "title": document.title,
                            "reason": "duplicate-content",
                            "hash": document.hash,
                        }
                    )
                    continue

                persisted, reason = self._write_document(ctx.directory, document)
                if not persisted:
                    skipped.append(
                        {
                            "title": document.title,
                            "reason": reason or "duplicate-file",
                            "hash": document.hash,
                        }
                    )
                    continue

                try:
                    await self._add_to_vector_store(ctx, document)
                    ctx.document_hashes.add(document.hash)
                    saved.append(
                        {
                            "title": document.title,
                            "hash": document.hash,
                            "path": str(persisted),
                            "source": document.source,
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "title": document.title,
                            "error": f"Failed to index document: {exc}",
                            "hash": document.hash,
                        }
                    )
                    try:
                        persisted.unlink()
                    except OSError:
                        pass

        return {
            "namespace": normalised,
            "saved": saved,
            "skipped": skipped,
            "errors": errors,
        }

    async def search(
        self,
        *,
        namespace: Optional[str],
        query: str,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search the namespace knowledge base and return serialisable documents."""

        normalised = normalize_namespace(namespace)
        ctx = await self._get_namespace(normalised)

        query_text = query.strip()
        if not query_text:
            return {
                "namespace": normalised,
                "query": query,
                "results": [],
                "error": "Provide a non-empty query.",
            }

        try:
            results = await asyncio.to_thread(
                ctx.vector_store.similarity_search_with_score,
                query_text,
                k=max_results or self.settings.max_tool_results,
            )
        except Exception as exc:
            return {
                "namespace": normalised,
                "query": query_text,
                "results": [],
                "error": f"Search failed: {exc}",
            }

        mapped: List[Dict[str, Any]] = []
        for doc, distance in results:
            metadata = dict(doc.metadata or {})
            hash_value = metadata.get("hash") or hash_content(doc.page_content)
            mapped.append(
                {
                    "title": metadata.get("title") or doc.metadata.get("source") or doc.metadata.get("filename") or "Untitled",
                    "content": doc.page_content,
                    "hash": hash_value,
                    "metadata": metadata,
                    "score": self._distance_to_score(distance),
                }
            )

        return {
            "namespace": normalised,
            "query": query_text,
            "results": mapped,
        }

    async def create_crew(self, *, namespace: Optional[str]) -> Crew:
        """Build a CrewAI crew with a search tool bound to the namespace."""

        self._ensure_loop()
        normalised = normalize_namespace(namespace)
        await self._get_namespace(normalised)

        search_tool = self._create_search_tool(normalised)
        model = ChatOpenAI(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.base_url,
            temperature=self.settings.temperature,
        )

        agent = Agent(
            role="Knowledge Librarian",
            goal="Answer user questions with relevant citations from the knowledge base.",
            backstory=f"You operate inside the '{normalised}' namespace. Use the search tool before answering.",
            tools=[search_tool],
            llm=model,
            allow_delegation=False,
            verbose=False,
        )

        task = Task(
            description=(
                "Use the `search_knowledge_base` tool to gather snippets before replying.\n"
                "Conversation so far:\n{conversation}\n\nLatest question: {question}\n"
                "Answer concisely, ground responses in retrieved documents, and cite sources as "
                "'Sources: <file1>, <file2>'."
            ),
            expected_output="A concise, cited answer grounded in ingested docs.",
            agent=agent,
        )

        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

    async def run_agent(
        self,
        *,
        namespace: Optional[str],
        messages: Sequence[MessagePayload],
    ) -> str:
        """Execute the CrewAI agent against the provided conversation."""

        crew = await self.create_crew(namespace=namespace)
        conversation = self._format_conversation(messages)
        latest = messages[-1].content if messages else ""

        result = await asyncio.to_thread(
            crew.kickoff,
            inputs={"conversation": conversation, "question": latest},
        )
        return self._stringify_result(result)

    async def _get_namespace(self, namespace: str) -> NamespaceContext:
        self._ensure_loop()
        lock = self._namespaces_lock_obj()

        async with lock:
            if namespace in self._namespaces:
                return self._namespaces[namespace]

            directory = self.settings.knowledge_root / namespace
            directory.mkdir(parents=True, exist_ok=True)

            chroma_path = self.settings.chroma_path / namespace
            chroma_path.mkdir(parents=True, exist_ok=True)

            embedder = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.base_url,
            )

            vector_store = Chroma(
                collection_name=f"{namespace}-documents",
                embedding_function=embedder,
                persist_directory=str(chroma_path),
            )

            context = NamespaceContext(
                name=namespace,
                vector_store=vector_store,
                directory=directory,
                chroma_path=chroma_path,
                document_hashes=self._load_hashes(directory),
                lock=asyncio.Lock(),
            )

            self._namespaces[namespace] = context
            return context

    async def _add_to_vector_store(self, ctx: NamespaceContext, document: IngestedDocument) -> None:
        chunks = self._splitter.split_text(document.content)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "source": document.source,
                    "chunk": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        await asyncio.to_thread(ctx.vector_store.add_documents, docs)
        await asyncio.to_thread(ctx.vector_store.persist)

    @staticmethod
    def _distance_to_score(distance: Optional[float]) -> Optional[float]:
        if distance is None:
            return None
        try:
            return round(1.0 / (1.0 + float(distance)), 4)
        except (TypeError, ValueError):
            return None

    def _write_document(self, directory: Path, document: IngestedDocument) -> tuple[Optional[Path], Optional[str]]:
        filename = f"{document.slug}-{document.hash[:12]}.md"
        path = directory / filename
        if path.exists():
            return None, "already-ingested"
        try:
            path.write_text(document.to_markdown(), encoding="utf-8")
            return path, None
        except OSError as exc:
            return None, f"filesystem-error: {exc}"

    def _load_hashes(self, directory: Path) -> set[str]:
        hashes: set[str] = set()
        if not directory.exists():
            return hashes
        for file_path in directory.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                hashes.add(hash_content(content))
            except OSError:
                continue
        return hashes

    def _create_search_tool(self, namespace: str):
        manager = self

        @tool("search_knowledge_base")
        def _search_tool(query: str, max_results: int = 4) -> str:
            """Search the knowledge base for relevant documents matching the query.

            Args:
                query: The search query to find relevant documents.
                max_results: Maximum number of results to return (default: 4).

            Returns:
                JSON string containing matching documents with title, content, hash, and relevance score.
            """
            loop = manager._loop
            try:
                if loop and loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        manager.search(namespace=namespace, query=query, max_results=max_results),
                        loop,
                    )
                    payload = future.result()
                else:
                    payload = asyncio.run(manager.search(namespace=namespace, query=query, max_results=max_results))
            except Exception as exc:
                payload = {"namespace": namespace, "results": [], "error": str(exc)}
            results = payload.get("results", []) if isinstance(payload, dict) else []
            condensed = [
                {
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "hash": item.get("hash"),
                    "score": item.get("score"),
                }
                for item in results
                if isinstance(item, dict)
            ]
            return json.dumps({"namespace": namespace, "results": condensed})

        return _search_tool

    @staticmethod
    def _format_conversation(messages: Sequence[MessagePayload]) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.role.upper()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    @staticmethod
    def _stringify_result(result: Any) -> str:
        if result is None:
            return ""
        for attr in ("raw", "output", "result"):
            if hasattr(result, attr):
                try:
                    return str(getattr(result, attr))
                except Exception:
                    continue
        return str(result)


__all__ = ["KnowledgeManager"]
