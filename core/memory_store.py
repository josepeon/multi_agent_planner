"""
Project + User Memory

Two persistent stores layered on top of a pluggable vector backend:

- ``UserMemory``    — cross-project preferences ("I prefer FastAPI over Flask",
                      "always include pytest", "avoid heavyweight ORMs").
- ``ProjectMemory`` — per-project decisions ("chose SQLite for storage",
                      "split into models/services/main").

Both expose ``remember(text, kind=..., tags=...)`` and ``recall(query, k=...)``.

Why this exists: ``core/memory.py`` and ``core/cache.py`` are LLM-response
caches. They dedupe API calls but they do not *learn* — there's no notion of
"the user prefers X" or "last time we built a CLI tool the architect picked
``click``." Architect and Planner now consult these stores before producing
their artifact, so successful patterns and explicit user guidance carry
forward across runs.

Cross-repo note: ``self-improving-agent`` ships a ChromaDB-backed
``VectorStore`` and sentence-transformer ``Embeddings`` that this module is
deliberately compatible with. The default backend here is JSON + hash-based
similarity so the project has no hard dep on chromadb / torch. When the SIA
package is installable (it's local-only today), swap ``DefaultVectorBackend``
for the SIA store and everything else continues to work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


# ===========================================
# Backend protocol
# ===========================================

@dataclass
class MemoryEntry:
    """One stored memory."""

    id: str
    text: str
    kind: str  # e.g. "preference", "decision", "lesson_learned"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecallResult:
    """One hit from a recall() query."""

    entry: MemoryEntry
    score: float  # 0..1; higher = more relevant


class VectorBackend(Protocol):
    """Storage + similarity protocol. JSON default; ChromaDB optional."""

    def add(self, entry: MemoryEntry) -> None: ...
    def query(self, text: str, k: int, kind: str | None = None) -> list[RecallResult]: ...
    def all(self) -> list[MemoryEntry]: ...
    def delete(self, entry_id: str) -> None: ...
    def clear(self) -> None: ...


# ===========================================
# Default backend: JSON + bag-of-words similarity
# ===========================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _term_freq(tokens: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DefaultVectorBackend:
    """JSON-on-disk + bag-of-words cosine similarity.

    Zero external deps. Quality is obviously worse than a real embedding
    model, but adequate for the memory sizes this project sees (dozens to
    a few hundred entries per user/project) and trivially upgrades to a
    sentence-transformer backend when one is available.
    """

    def __init__(self, filepath: str | None = None) -> None:
        self.filepath = filepath
        self._entries: dict[str, MemoryEntry] = {}
        if filepath and os.path.exists(filepath):
            self._load()

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry
        self._save()

    def query(
        self, text: str, k: int, kind: str | None = None
    ) -> list[RecallResult]:
        if not self._entries:
            return []
        target = _term_freq(_tokenize(text))
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self._entries.values():
            if kind is not None and e.kind != kind:
                continue
            score = _cosine(target, _term_freq(_tokenize(e.text)))
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda kv: -kv[0])
        return [RecallResult(entry=e, score=s) for s, e in scored[:k]]

    def all(self) -> list[MemoryEntry]:
        return sorted(self._entries.values(), key=lambda e: -e.timestamp)

    def delete(self, entry_id: str) -> None:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    def _save(self) -> None:
        if not self.filepath:
            return
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(
                {"entries": [asdict(e) for e in self._entries.values()]},
                f,
                indent=2,
            )

    def _load(self) -> None:
        try:
            with open(self.filepath) as f:
                raw = json.load(f)
        except Exception:
            return
        for item in raw.get("entries", []):
            entry = MemoryEntry(**item)
            self._entries[entry.id] = entry


# ===========================================
# Optional ChromaDB backend (lazy import)
# ===========================================

class ChromaVectorBackend:
    """Optional backend — only imported if chromadb is installed.

    Same surface as DefaultVectorBackend; uses real embeddings if a
    sentence-transformer is available, else falls back to Chroma's default
    embedding function.

    Construct with ``ChromaVectorBackend(collection_name=..., persist_dir=...)``.
    Raises ImportError lazily on first use if chromadb isn't installed.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure(self):
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorBackend requires the chromadb package. "
                "Install with: pip install chromadb"
            ) from exc

        if self.persist_dir:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(self.collection_name)

    def add(self, entry: MemoryEntry) -> None:
        self._ensure()
        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.text],
            metadatas=[
                {
                    "kind": entry.kind,
                    "tags": ",".join(entry.tags),
                    "timestamp": entry.timestamp,
                    **entry.metadata,
                }
            ],
        )

    def query(
        self, text: str, k: int, kind: str | None = None
    ) -> list[RecallResult]:
        self._ensure()
        where = {"kind": kind} if kind else None
        result = self._collection.query(
            query_texts=[text], n_results=k, where=where
        )
        out: list[RecallResult] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        # Chroma returns distance; convert to a similarity score
        dists = result.get("distances", [[1.0] * len(ids)])[0]
        for eid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = max(0.0, 1.0 - float(dist))
            tags = meta.pop("tags", "")
            entry = MemoryEntry(
                id=eid,
                text=doc,
                kind=meta.pop("kind", "unknown"),
                tags=[t for t in tags.split(",") if t],
                timestamp=float(meta.pop("timestamp", time.time())),
                metadata=dict(meta),
            )
            out.append(RecallResult(entry=entry, score=score))
        return out

    def all(self) -> list[MemoryEntry]:
        self._ensure()
        raw = self._collection.get()
        entries: list[MemoryEntry] = []
        for eid, doc, meta in zip(
            raw.get("ids", []),
            raw.get("documents", []),
            raw.get("metadatas", []),
        ):
            tags = meta.pop("tags", "")
            entries.append(
                MemoryEntry(
                    id=eid,
                    text=doc,
                    kind=meta.pop("kind", "unknown"),
                    tags=[t for t in tags.split(",") if t],
                    timestamp=float(meta.pop("timestamp", 0.0)),
                    metadata=dict(meta),
                )
            )
        return sorted(entries, key=lambda e: -e.timestamp)

    def delete(self, entry_id: str) -> None:
        self._ensure()
        self._collection.delete(ids=[entry_id])

    def clear(self) -> None:
        self._ensure()
        # Recreate to fully wipe
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(self.collection_name)


def make_backend(
    path: str | None,
    collection_name: str,
) -> VectorBackend:
    """Pick a backend based on env.

    ``MEMORY_BACKEND=chroma`` (and chromadb installed) → ChromaVectorBackend.
    Anything else (default) → DefaultVectorBackend with JSON file backing.
    """
    mode = os.environ.get("MEMORY_BACKEND", "json").strip().lower()
    if mode == "chroma":
        return ChromaVectorBackend(collection_name=collection_name, persist_dir=path)
    return DefaultVectorBackend(filepath=path)


# ===========================================
# Memory facades
# ===========================================

def _make_id(text: str, kind: str) -> str:
    h = hashlib.sha1(f"{kind}::{text}".encode()).hexdigest()
    return h[:16]


class _MemoryBase:
    def __init__(self, backend: VectorBackend) -> None:
        self._backend = backend

    def remember(
        self,
        text: str,
        kind: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=_make_id(text, kind),
            text=text.strip(),
            kind=kind,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._backend.add(entry)
        return entry

    def recall(
        self,
        query: str,
        k: int = 5,
        kind: str | None = None,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        results = self._backend.query(query, k=k, kind=kind)
        return [r for r in results if r.score >= min_score]

    def all(self) -> list[MemoryEntry]:
        return self._backend.all()

    def delete(self, entry_id: str) -> None:
        self._backend.delete(entry_id)

    def clear(self) -> None:
        self._backend.clear()

    def render(self, query: str, k: int = 5, kind: str | None = None) -> str:
        """One-shot render of recalled memories for injection into a prompt."""
        results = self.recall(query, k=k, kind=kind)
        if not results:
            return ""
        lines = ["[remembered]"]
        for r in results:
            tag_str = f" ({', '.join(r.entry.tags)})" if r.entry.tags else ""
            lines.append(f"- {r.entry.text}{tag_str}")
        return "\n".join(lines)


class UserMemory(_MemoryBase):
    """Across-projects: preferences, blacklists, recurring lessons."""

    def __init__(self, path: str = "memory/user_memory.json") -> None:
        super().__init__(make_backend(path, "user_memory"))


class ProjectMemory(_MemoryBase):
    """Per-project: decisions made, libs chosen, files written.

    The project identifier is part of the file path so different projects
    don't share their memory.
    """

    def __init__(self, project_id: str, dir: str = "memory/projects") -> None:
        path = os.path.join(dir, f"{project_id}.json")
        super().__init__(make_backend(path, f"project_{project_id}"))
        self.project_id = project_id


def project_id_for(prompt: str) -> str:
    """Stable, short id derived from the user prompt."""
    return hashlib.sha1(prompt.strip().encode()).hexdigest()[:12]
