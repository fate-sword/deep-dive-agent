"""Passage chunking, BM25 ranking, and uploaded-document indexing (no embedding cost)."""
import io
import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_text(text: str, size: int = 800) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 40]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p[: size * 2]
    if cur:
        chunks.append(cur)
    return chunks


def rank_passages(passages: list[str], query: str, k: int = 3) -> list[str]:
    if not passages:
        return []
    bm = BM25Okapi([tokenize(p) or ["_"] for p in passages])
    scores = bm.get_scores(tokenize(query) or ["_"])
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    return [passages[i] for i in order[:k]]


def read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((pg.extract_text() or "") for pg in reader.pages)
    return data.decode("utf-8", errors="ignore")


class DocIndex:
    """Chunks of user-uploaded documents, searchable with BM25."""

    def __init__(self, files: list[tuple[str, bytes]]):
        self.items: list[tuple[str, str]] = []  # (filename, passage)
        for name, data in files:
            for ch in chunk_text(read_upload(name, data), 900):
                self.items.append((name, ch))
        self._bm = BM25Okapi([tokenize(p) or ["_"] for _, p in self.items]) if self.items else None

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        if not self._bm:
            return []
        scores = self._bm.get_scores(tokenize(query) or ["_"])
        order = sorted(range(len(self.items)), key=lambda i: scores[i], reverse=True)
        return [self.items[i] for i in order[:k] if scores[i] > 0]
