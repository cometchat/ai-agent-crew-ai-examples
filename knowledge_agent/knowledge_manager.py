from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from openai import OpenAI

from .config import KnowledgeAgentSettings, get_settings
from .ingestion import IngestedDocument, collect_documents, hash_content, normalize_namespace
from .schemas import MessagePayload


@dataclass
class StoredDocument:
    page_content: str
    metadata: Dict[str, Any]


class OpenAIEmbedder:
    """Lightweight embedder using the official OpenAI client."""

    def __init__(self, *, model: str, api_key: str, base_url: Optional[str]) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class ChromaVectorStore:
    """Minimal Chroma wrapper that stores documents plus metadata."""

    def __init__(self, *, path: Path, namespace: str, embedder: OpenAIEmbedder) -> None:
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=f"{namespace}-documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: List[StoredDocument]) -> None:
        if not docs:
            return
        texts = [doc.page_content for doc in docs]
        embeddings = self.embedder.embed_documents(texts)
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for index, doc in enumerate(docs):
            meta = dict(doc.metadata or {})
            metadatas.append(meta)
            base_hash = meta.get("hash") or hash_content(doc.page_content)
            chunk_index = meta.get("chunk", index)
            ids.append(f"{base_hash}-{chunk_index}")
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        if hasattr(self.client, "persist"):
            self.client.persist()

    def similarity_search_with_score(self, query: str, k: int) -> List[tuple[StoredDocument, Optional[float]]]:
        if not query.strip():
            return []
        embedding = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        mapped: List[tuple[StoredDocument, Optional[float]]] = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            mapped.append((StoredDocument(page_content=content, metadata=metadata or {}), distance))
        return mapped


@dataclass
class NamespaceContext:
    name: str
    vector_store: ChromaVectorStore
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
        self._chunk_size = 1200
        self._chunk_overlap = 150

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

    async def create_crew(self, *, namespace: Optional[str], stream: bool = False) -> Crew:
        """Build a CrewAI crew with a search tool bound to the namespace."""

        self._ensure_loop()
        normalised = normalize_namespace(namespace)
        await self._get_namespace(normalised)

        search_tool = self._create_search_tool(normalised)
        model = LLM(
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
            memory=False,
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
            stream=stream,
        )

    async def run_agent(
        self,
        *,
        namespace: Optional[str],
        messages: Sequence[MessagePayload],
    ) -> str:
        """Execute the CrewAI agent against the provided conversation."""

        crew = await self.create_crew(namespace=namespace, stream=False)
        conversation = self._format_conversation(messages)
        latest = messages[-1].content if messages else ""

        result = await asyncio.to_thread(
            crew.kickoff,
            inputs={"conversation": conversation, "question": latest},
        )
        return self._stringify_result(result)

    async def run_agent_stream(
        self,
        *,
        namespace: Optional[str],
        messages: Sequence[MessagePayload],
    ):
        """Execute the CrewAI agent and return a streaming output."""

        crew = await self.create_crew(namespace=namespace, stream=True)
        conversation = self._format_conversation(messages)
        latest = messages[-1].content if messages else ""

        streaming = await asyncio.to_thread(
            crew.kickoff,
            inputs={"conversation": conversation, "question": latest},
        )
        return streaming

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

            embedder = OpenAIEmbedder(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.base_url,
            )

            vector_store = ChromaVectorStore(
                path=chroma_path,
                namespace=namespace,
                embedder=embedder,
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
        chunks = self._split_text(document.content)
        docs = [
            StoredDocument(
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

    @staticmethod
    def _distance_to_score(distance: Optional[float]) -> Optional[float]:
        if distance is None:
            return None
        try:
            return round(1.0 / (1.0 + float(distance)), 4)
        except (TypeError, ValueError):
            return None

    def _split_text(self, content: str) -> List[str]:
        text = (content or "").strip()
        if not text:
            return []
        step = max(self._chunk_size - self._chunk_overlap, 1)
        chunks: List[str] = []
        for start in range(0, len(text), step):
            end = min(start + self._chunk_size, len(text))
            chunks.append(text[start:end])
        return chunks

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
