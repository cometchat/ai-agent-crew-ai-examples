from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import httpx
from fastapi import UploadFile
from pypdf import PdfReader

from .config import KnowledgeAgentSettings

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_TEXT_CHARS = 500_000  # Increased from 200,000 to handle larger docs
MAX_SOURCES = 30


class IngestionError(Exception):
    """Raised when a source cannot be parsed or converted into a document."""


def slugify(value: str, fallback: str = "document") -> str:
    ascii_value = value.encode("ascii", errors="ignore").decode("ascii") if value else ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return cleaned or fallback


def normalize_namespace(value: Optional[str]) -> str:
    if not value:
        return "default"
    ascii_value = value.encode("ascii", errors="ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", ascii_value).strip("-").lower()
    return normalized or "default"


def ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def limit_text_length(text: str, label: str) -> str:
    stripped = text.strip()
    if len(stripped) > MAX_TEXT_CHARS:
        raise IngestionError(
            f'Text source "{label}" exceeds the maximum length of {MAX_TEXT_CHARS:,} characters.'
        )
    return stripped


def strip_html(raw_html: str) -> Tuple[str, str]:
    title_match = re.search(r"<title>([^<]{2,})</title>", raw_html, flags=re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Imported Web Page"
    without_scripts = re.sub(r"<script[\\s\\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    without_styles = re.sub(r"<style[\\s\\S]*?</style>", " ", without_scripts, flags=re.IGNORECASE)
    without_comments = re.sub(r"<!--[\\s\\S]*?-->", " ", without_styles)
    text = re.sub(r"<[^>]+>", " ", without_comments)
    text = re.sub(r"\\s{2,}", " ", text)
    return title, text.strip()


def convert_pdf_bytes(data: bytes, label: str) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError(
            f'PDF "{label}" exceeds the maximum upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.'
        )
    reader = PdfReader(io.BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted:
            parts.append(extracted)
    combined = "\n\n".join(parts).strip()
    return combined or "(No extractable text)"


@dataclass
class IngestedDocument:
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]

    @property
    def hash(self) -> str:
        return hash_content(self.content)

    @property
    def slug(self) -> str:
        return slugify(self.title)

    def to_markdown(self) -> str:
        header_lines = [
            f"# {self.title}".strip(),
            "",
            f"Source: {self.metadata.get('source', self.source)}",
            f"Ingested At: {self.metadata.get('ingested_at')}",
        ]
        namespace = self.metadata.get("namespace")
        if namespace:
            header_lines.append(f"Namespace: {namespace}")
        if self.metadata.get("content_type"):
            header_lines.append(f"Content-Type: {self.metadata['content_type']}")
        header_lines.append("")

        body = ensure_trailing_newline(self.content)
        return ensure_trailing_newline("\n".join(header_lines) + body)


async def _read_upload_file(file: UploadFile, settings: KnowledgeAgentSettings) -> IngestedDocument:
    data = await file.read()
    if not data:
        raise IngestionError(f'Uploaded file "{file.filename or file.content_type}" is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError(
            f'Uploaded file "{file.filename or file.content_type}" exceeds the maximum size of '
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    title = Path(filename).stem.replace("_", " ") or "Uploaded Document"

    if "pdf" in content_type.lower() or filename.lower().endswith(".pdf"):
        text = convert_pdf_bytes(data, filename)
        content_type_label = "application/pdf"
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        text = limit_text_length(text, filename)
        content_type_label = content_type

    metadata = {
        "filename": filename,
        "source": f"upload:{filename}",
        "content_type": content_type_label,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }

    return IngestedDocument(title=title, content=text, source=metadata["source"], metadata=metadata)


async def _fetch_remote_url(url: str, settings: KnowledgeAgentSettings) -> IngestedDocument:
    timeout = httpx.Timeout(settings.remote_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        raw_bytes = response.content
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise IngestionError(
                f'Remote resource "{url}" exceeds the maximum size of '
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )

        final_url = str(response.url)

        if "pdf" in content_type or final_url.lower().endswith(".pdf"):
            text = convert_pdf_bytes(raw_bytes, final_url)
            content_type_label = "application/pdf"
            title = Path(final_url).name or "Remote PDF"
        elif "html" in content_type:
            decoded = response.text
            title, text = strip_html(decoded)
            text = limit_text_length(text, final_url)
            content_type_label = "text/html"
        else:
            try:
                decoded = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_bytes.decode("latin-1", errors="ignore")
            text = limit_text_length(decoded, final_url)
            title = Path(final_url).name or "Remote Document"
            content_type_label = content_type or "text/plain"

    metadata = {
        "url": final_url,
        "source": f"url:{final_url}",
        "content_type": content_type_label,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }

    return IngestedDocument(title=title, content=text, source=metadata["source"], metadata=metadata)


def _parse_text_payload(
    raw: Union[str, Dict[str, Any]],
    index: int,
    namespace: str,
) -> IngestedDocument:
    if isinstance(raw, str):
        title = f"Document {index + 1}"
        if URL_PATTERN.match(raw.strip()):
            raise IngestionError("URL payload should be handled asynchronously.")
        text = limit_text_length(raw, title)
        metadata = {
            "source": "text:payload",
            "content_type": "text/plain",
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "namespace": namespace,
        }
        return IngestedDocument(title=title, content=text, source=metadata["source"], metadata=metadata)

    payload = dict(raw)
    type_hint = str(payload.get("type", payload.get("kind", "text"))).lower()
    title = str(payload.get("title") or payload.get("name") or f"Document {index + 1}")
    metadata = payload.get("metadata") or {}
    metadata = {**metadata}
    metadata.setdefault("namespace", namespace)
    metadata.setdefault("ingested_at", datetime.utcnow().isoformat() + "Z")

    if type_hint in {"text", "markdown", "md"}:
        value = payload.get("value") or payload.get("content")
        if value is None:
            raise IngestionError(f'Text source "{title}" is missing a "value" field.')
        text = limit_text_length(str(value), title)
        metadata.setdefault("content_type", "text/markdown" if type_hint != "text" else "text/plain")
        metadata.setdefault("source", f"text:{slugify(title)}")
        return IngestedDocument(title=title, content=text, source=metadata["source"], metadata=metadata)

    if type_hint == "url":
        url = payload.get("value") or payload.get("url")
        if not isinstance(url, str) or not URL_PATTERN.match(url):
            raise IngestionError(f'URL source "{title}" is missing a valid "url" or "value".')
        raise IngestionError("URL payload should be handled asynchronously.")

    raise IngestionError(f'Unsupported source type "{type_hint}".')


async def collect_documents(
    raw_sources: Optional[Sequence[Union[str, Dict[str, Any]]]],
    uploads: Optional[Sequence[UploadFile]],
    settings: KnowledgeAgentSettings,
    namespace: str,
) -> Tuple[List[IngestedDocument], List[Dict[str, Any]]]:
    documents: List[IngestedDocument] = []
    errors: List[Dict[str, Any]] = []

    sources = list(raw_sources or [])
    if len(sources) > MAX_SOURCES:
        errors.append(
            {
                "source": "payload",
                "error": f"Too many sources provided ({len(sources)}). Maximum allowed is {MAX_SOURCES}.",
            }
        )
        sources = sources[:MAX_SOURCES]

    pending_remote_tasks: List[asyncio.Task[IngestedDocument]] = []

    for index, entry in enumerate(sources):
        try:
            if isinstance(entry, str) and URL_PATTERN.match(entry.strip()):
                if not settings.allow_remote_http:
                    raise IngestionError("Remote URL ingestion is disabled by server configuration.")
                pending_remote_tasks.append(asyncio.create_task(_fetch_remote_url(entry.strip(), settings)))
                continue

            if isinstance(entry, dict):
                type_hint = str(entry.get("type", entry.get("kind", "text"))).lower()
                if type_hint == "url":
                    url_value = entry.get("value") or entry.get("url")
                    if not isinstance(url_value, str) or not URL_PATTERN.match(url_value):
                        raise IngestionError(f'Source #{index + 1} missing a valid URL.')
                    if not settings.allow_remote_http:
                        raise IngestionError("Remote URL ingestion is disabled by server configuration.")
                    pending_remote_tasks.append(asyncio.create_task(_fetch_remote_url(url_value.strip(), settings)))
                    continue

            document = _parse_text_payload(entry, index, namespace)
            documents.append(document)
        except IngestionError as exc:
            errors.append({"source": entry, "error": str(exc)})
        except Exception as exc:
            errors.append({"source": entry, "error": f"Unexpected failure: {exc}"})

    for task in pending_remote_tasks:
        try:
            document = await task
            document.metadata.setdefault("namespace", namespace)
            documents.append(document)
        except IngestionError as exc:
            errors.append({"source": "url", "error": str(exc)})
        except httpx.HTTPError as exc:
            errors.append({"source": "url", "error": f"HTTP error: {exc}"})
        except Exception as exc:
            errors.append({"source": "url", "error": f"Unexpected failure while fetching URL: {exc}"})

    for upload in uploads or []:
        try:
            document = await _read_upload_file(upload, settings)
            document.metadata.setdefault("namespace", namespace)
            documents.append(document)
        except IngestionError as exc:
            errors.append({"source": upload.filename or "upload", "error": str(exc)})
        except Exception as exc:
            errors.append({"source": upload.filename or "upload", "error": f"Unexpected failure: {exc}"})

    return documents, errors


__all__ = [
    "IngestedDocument",
    "IngestionError",
    "collect_documents",
    "normalize_namespace",
    "slugify",
    "hash_content",
    "MAX_UPLOAD_BYTES",
]
