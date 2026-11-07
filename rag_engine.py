"""Source-aware retrieval for VentureIQ.

* Documents: ``.txt`` / ``.md`` files with optional front matter (title, source_type, date, url).
* Chunking: word windows (default 300 words, 60 overlap) that remember the nearest heading.
* Lexical retrieval: BM25 (pure Python, always available).
* Semantic retrieval (optional): Sentence Transformers embeddings, searched with FAISS when
  installed or with NumPy otherwise. If unavailable the engine degrades to lexical only.
* Hybrid ranking: reciprocal-rank fusion of the lexical and semantic rankings.
* Index artifacts (JSON + ``.npy``; no pickle) are validated and rebuilt automatically when the
  knowledge base, chunking parameters or embedding model change.

Every returned :class:`Evidence` carries a ``source_id`` (``<doc_id>#c<n>``) that maps to a real
chunk of a real file; nothing is generated.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from utils import sha256_file, sha256_text, utc_now_iso

log = logging.getLogger("ventureiq.rag")

SUPPORTED_SUFFIXES = (".md", ".txt")
MAX_FILE_BYTES = 2_000_000
INDEX_SCHEMA = 1
RRF_K = 60
MIN_SEMANTIC_COSINE = 0.25
MIN_LEXICAL_RATIO = 0.25   # lexical-only hits below 25% of the best BM25 score are dropped

_STOP = frozenset("""a an and are as at be by for from has have how i in is it its of on or that the this to was
what when where which who why will with you your we our can do does should would could into than then there
their they them these those about over under more most such not no""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens with stop-word removal and a light plural stemmer."""
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


# ------------------------------------------------------------------ data classes
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    date: str | None = None
    url: str | None = None
    source_type: str | None = None
    path: str = ""


@dataclass
class Evidence:
    source_id: str
    doc_title: str
    section: str
    excerpt: str
    score: float                  # relevance relative to the best hit (0-1); NOT a correctness probability
    lexical_score: float | None
    semantic_score: float | None
    date: str | None
    url: str | None
    source_type: str | None
    doc_path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self._m = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


# ------------------------------------------------------------------ document processing
def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", raw, flags=re.S)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta, raw[m.end():]


def discover_documents(kb_dir: Path) -> tuple[list[Path], list[str]]:
    """Return supported files and human-readable skip warnings."""
    warnings: list[str] = []
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        return [], [f"Knowledge-base folder not found: {kb_dir.name}/"]
    files = []
    for p in sorted(kb_dir.rglob("*")):
        if not p.is_file() or p.name.lower() == "readme.md" or p.name.startswith("."):
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            warnings.append(f"Skipped unsupported file type: {p.name}")
        elif p.stat().st_size > MAX_FILE_BYTES:
            warnings.append(f"Skipped {p.name}: larger than {MAX_FILE_BYTES // 1_000_000} MB")
        else:
            files.append(p)
    return files, warnings


def _doc_id(kb_dir: Path, path: Path) -> str:
    rel = path.relative_to(kb_dir).with_suffix("")
    return re.sub(r"[^a-z0-9]+", "-", rel.as_posix().lower()).strip("-")


def chunk_document(doc_id: str, body: str, meta: dict[str, str], path: str,
                   words: int = 300, overlap: int = 60) -> list[Chunk]:
    if words <= 0 or overlap < 0 or overlap >= words:
        raise ValueError("chunk_words must be > 0 and 0 <= overlap < chunk_words")
    title = meta.get("title") or doc_id
    section = title
    tokens: list[tuple[str, str]] = []
    for line in body.splitlines():
        h = re.match(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", line)
        if h:
            section = h.group(1).strip()
            line = section
        for w in line.split():
            if re.fullmatch(r"[-*•|>]+", w):
                continue
            tokens.append((w, section))
    chunks, start, step, n = [], 0, words - overlap, len(tokens)
    while start < n:
        end = min(start + words, n)
        window = tokens[start:end]
        chunks.append(Chunk(f"{doc_id}#c{len(chunks) + 1}", doc_id, title, window[0][1],
                            " ".join(w for w, _ in window), meta.get("date"), meta.get("url"),
                            meta.get("source_type"), path))
        if end == n:
            break
        start += step
    return chunks


def kb_fingerprint(kb_dir: Path, files: list[Path], words: int, overlap: int) -> str:
    parts = [f"{p.relative_to(kb_dir).as_posix()}:{sha256_file(p)}" for p in files]
    return sha256_text("|".join(parts) + f"|{words}|{overlap}")


# ------------------------------------------------------------------ BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / self.n) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post[term].append((i, tf))

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for term in set(query):
            plist = self.post.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1.0))
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out


# ------------------------------------------------------------------ engine
class RagEngine:
    def __init__(self, kb_dir: Path, index_dir: Path, chunk_words: int = 300, chunk_overlap: int = 60,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 semantic_mode: str = "auto", embedder: Embedder | None = None,
                 top_k: int = 5, max_chunks_per_doc: int = 2):
        self.kb_dir, self.index_dir = Path(kb_dir), Path(index_dir)
        self.chunk_words, self.chunk_overlap = chunk_words, chunk_overlap
        self.embedding_model, self.semantic_mode = embedding_model, semantic_mode
        self.top_k, self.max_per_doc = top_k, max_chunks_per_doc
        self._embedder = embedder
        self._embedder_tried = embedder is not None
        self.semantic_error: str | None = None
        self.chunks: list[Chunk] = []
        self._bm25: BM25 | None = None
        self._vectors: np.ndarray | None = None
        self._faiss = None
        self._by_id: dict[str, Chunk] = {}
        self.build_warnings: list[str] = []
        self.last_action: str = "not loaded"
        self._loaded = False

    # ---- semantic helpers
    def _get_embedder(self) -> Embedder | None:
        if self._embedder is not None or self._embedder_tried:
            return self._embedder
        self._embedder_tried = True
        if self.semantic_mode == "off":
            self.semantic_error = "Semantic retrieval disabled by configuration."
            return None
        try:
            self._embedder = _SentenceTransformerEmbedder(self.embedding_model)
        except Exception as exc:  # noqa: BLE001 - optional dependency or offline model download
            self.semantic_error = f"Semantic retrieval unavailable ({type(exc).__name__}); using lexical search only."
            log.info(self.semantic_error)
        return self._embedder

    def semantic_dependencies_present(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- paths
    @property
    def _manifest_path(self) -> Path: return self.index_dir / "manifest.json"
    @property
    def _chunks_path(self) -> Path: return self.index_dir / "chunks.json"
    @property
    def _vectors_path(self) -> Path: return self.index_dir / "vectors.npy"

    # ---- state inspection (does not build)
    def inspect(self) -> dict[str, Any]:
        files, warns = discover_documents(self.kb_dir)
        info: dict[str, Any] = {"kb_dir": str(self.kb_dir.name), "n_docs": len(files), "warnings": warns,
                                "state": "ready", "problem": None, "n_chunks": len(self.chunks),
                                "semantic_active": self._vectors is not None, "semantic_error": self.semantic_error,
                                "last_action": self.last_action}
        if not self.kb_dir.is_dir():
            info.update(state="missing", problem=f"Knowledge-base folder '{self.kb_dir.name}/' was not found. Create it and add .md or .txt documents.")
        elif not files:
            info.update(state="empty", problem="The knowledge base contains no supported documents (.md or .txt).")
        else:
            reason = self._invalid_reason(files)
            if reason:
                info.update(state="stale" if self._manifest_path.exists() else "missing",
                            problem=f"Index needs (re)building: {reason}. Run: python scripts/build_index.py")
        return info

    def _invalid_reason(self, files: list[Path]) -> str | None:
        try:
            man = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "no index has been built yet"
        except (OSError, ValueError):
            return "index manifest is unreadable"
        if man.get("schema") != INDEX_SCHEMA:
            return "index format version changed"
        if man.get("fingerprint") != kb_fingerprint(self.kb_dir, files, self.chunk_words, self.chunk_overlap):
            return "documents or chunking settings changed since the index was built"
        try:
            chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "chunk file is missing or corrupted"
        if not isinstance(chunks, list) or len(chunks) != man.get("n_chunks"):
            return "chunk count does not match the manifest"
        if man.get("has_vectors"):
            try:
                vec = np.load(self._vectors_path, allow_pickle=False)
            except (OSError, ValueError):
                return "vector file is missing or corrupted"
            if vec.ndim != 2 or vec.shape[0] != len(chunks):
                return "vector file shape does not match chunks"
        return None

    # ---- build / load
    def build(self) -> dict[str, Any]:
        """(Re)build the index from the knowledge-base folder."""
        files, warns = discover_documents(self.kb_dir)
        self.build_warnings = warns
        chunks: list[Chunk] = []
        for p in files:
            try:
                raw = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                self.build_warnings.append(f"Skipped {p.name}: could not be read as UTF-8 text")
                continue
            meta, body = parse_front_matter(raw)
            if not body.strip():
                self.build_warnings.append(f"Skipped {p.name}: no text content")
                continue
            chunks.extend(chunk_document(_doc_id(self.kb_dir, p), body, meta, p.relative_to(self.kb_dir).as_posix(),
                                         self.chunk_words, self.chunk_overlap))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = None
        emb = self._get_embedder() if chunks else None
        if emb is not None:
            try:
                vectors = _normalise(np.asarray(emb.encode([c.text for c in chunks]), dtype=np.float32))
            except Exception as exc:  # noqa: BLE001
                self.semantic_error = f"Embedding failed ({type(exc).__name__}); using lexical search only."
                vectors = None
        manifest = {"schema": INDEX_SCHEMA, "built_at": utc_now_iso(), "n_chunks": len(chunks),
                    "n_docs": len({c.doc_id for c in chunks}), "has_vectors": vectors is not None,
                    "embedding_model": self.embedding_model if vectors is not None else None,
                    "fingerprint": kb_fingerprint(self.kb_dir, files, self.chunk_words, self.chunk_overlap),
                    "chunk_words": self.chunk_words, "chunk_overlap": self.chunk_overlap}
        self._chunks_path.write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
        if vectors is not None:
            np.save(self._vectors_path, vectors, allow_pickle=False)
        elif self._vectors_path.exists():
            self._vectors_path.unlink()
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._activate(chunks, vectors)
        self.last_action = f"built index ({len(chunks)} chunks)"
        return self.inspect()

    def _activate(self, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25([tokenize(f"{c.title} {c.section} {c.text}") for c in chunks])
        self._vectors, self._faiss = vectors, None
        if vectors is not None:
            try:
                import faiss  # optional
                idx = faiss.IndexFlatIP(vectors.shape[1])
                idx.add(vectors)
                self._faiss = idx
            except Exception:  # noqa: BLE001 - NumPy fallback is used
                self._faiss = None
        self._loaded = True

    def ensure_index(self, force: bool = False) -> dict[str, Any]:
        """Load a valid index, or rebuild when missing, stale, or corrupted."""
        files, _ = discover_documents(self.kb_dir)
        if not files:
            self.chunks, self._bm25, self._vectors, self._by_id = [], None, None, {}
            self._loaded = True
            self.last_action = "no documents"
            return self.inspect()
        reason = "forced rebuild" if force else self._invalid_reason(files)
        if reason is None and self._wants_vectors_but_missing():
            reason = "semantic retrieval is now available but the index has no embeddings"
        if reason is None and not self._loaded:
            chunks = [Chunk(**c) for c in json.loads(self._chunks_path.read_text(encoding="utf-8"))]
            man = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            vec = np.load(self._vectors_path, allow_pickle=False) if man.get("has_vectors") else None
            self._activate(chunks, vec)
            self.last_action = "loaded existing index"
        elif reason is not None:
            log.info("Rebuilding RAG index: %s", reason)
            self.build()
            self.last_action = f"rebuilt index ({reason})"
        return self.inspect()

    def _wants_vectors_but_missing(self) -> bool:
        try:
            man = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if man.get("has_vectors") and man.get("embedding_model") != self.embedding_model and self._embedder is None:
            return self.semantic_mode != "off" and self.semantic_dependencies_present()
        if man.get("has_vectors"):
            return False
        if self.semantic_mode == "off":
            return False
        if self._embedder is not None:
            return True
        return not self._embedder_tried and self.semantic_dependencies_present()

    # ---- retrieval
    def search(self, query: str, k: int | None = None) -> list[Evidence]:
        k = k or self.top_k
        if not (query or "").strip():
            return []
        self.ensure_index()
        if not self.chunks or self._bm25 is None:
            return []
        lex = self._bm25.scores(tokenize(query))
        lex_rank = [i for i in sorted(range(len(lex)), key=lambda i: -lex[i]) if lex[i] > 0][:50]
        sem_scores: dict[int, float] = {}
        sem_rank: list[int] = []
        if self._vectors is not None and self._get_embedder() is not None:
            try:
                q = _normalise(np.asarray(self._get_embedder().encode([query]), dtype=np.float32))
                sims = self._vectors @ q[0]
                sem_rank = [int(i) for i in np.argsort(-sims)[:50] if sims[i] >= MIN_SEMANTIC_COSINE]
                sem_scores = {i: float(sims[i]) for i in sem_rank}
            except Exception as exc:  # noqa: BLE001
                self.semantic_error = f"Semantic search failed ({type(exc).__name__}); lexical only."
        fused: dict[int, float] = defaultdict(float)
        for ranking in (lex_rank, sem_rank):
            for r, i in enumerate(ranking):
                fused[i] += 1.0 / (RRF_K + r + 1)
        if not fused:
            return []
        lexical_only = not sem_rank
        max_lex = max((lex[i] for i in fused), default=0.0) or 1.0
        top = max(fused.values())
        picked, per_doc = [], Counter()
        for i in sorted(fused, key=lambda i: -fused[i]):
            if lexical_only and lex[i] < MIN_LEXICAL_RATIO * max_lex:
                continue  # weak partial matches add noise
            c = self.chunks[i]
            if per_doc[c.doc_id] >= self.max_per_doc:
                continue
            per_doc[c.doc_id] += 1
            rel = lex[i] / max_lex if lexical_only else fused[i] / top
            picked.append(Evidence(c.chunk_id, c.title, c.section, c.text, round(rel, 4),
                                   round(lex[i], 4) if lex[i] > 0 else None, sem_scores.get(i),
                                   c.date, c.url, c.source_type, c.path))
            if len(picked) >= k:
                break
        return picked

    def get_source(self, source_id: str) -> Chunk | None:
        self.ensure_index()
        return self._by_id.get(source_id)


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype(np.float32)


def engine_from_settings(settings=None, embedder: Embedder | None = None) -> RagEngine:
    from config import get_settings
    s = settings or get_settings()
    return RagEngine(s.kb_dir, s.index_dir, s.chunk_words, s.chunk_overlap, s.embedding_model,
                     s.semantic_mode, embedder, s.top_k, s.max_chunks_per_doc)
