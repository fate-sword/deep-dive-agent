"""Passage chunking, BM25 ranking, and uploaded-document indexing (no embedding cost)."""
import io
import re
from urllib.parse import urlparse

from rank_bm25 import BM25Okapi, BM25Plus

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


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "").lower()
    except Exception:
        return url[:30]


_STOP = {"what", "how", "why", "which", "are", "is", "the", "do", "does", "have", "for", "and", "with", "that", "this", "from", "were", "been"}


def filter_and_rank_passages(
    evidence: list[dict],
    subq: dict,
    k: int = 8,
    max_per_url: int = 3,
    min_distinct_domains: int = 2,
) -> list[dict]:
    """Topical relevance filter for evidence passages before the drafter.

    - Ranks passages by topical relevance to sub-question + answer_spec FIRST (BM25Plus + overlap).
    - Source quality is a bounded multiplier (max ~1.3x), never the primary key.
    - Drops passages below relevance threshold (zero keyword overlap).
    - Caps passages per URL (e.g. 3).
    - Ensures passages from at least 2 distinct domains when available.
    """
    if not evidence:
        return []

    query_parts = [
        subq.get("question", ""),
        subq.get("answer_spec", ""),
        " ".join(subq.get("queries", [])),
    ]
    query_text = " ".join(p for p in query_parts if p).strip()
    raw_q_tokens = tokenize(query_text)
    q_tokens = [w for w in raw_q_tokens if w not in _STOP and len(w) > 2]
    if not q_tokens:
        q_tokens = raw_q_tokens or ["_"]

    tokenized_passages = [tokenize(e.get("text", "")) or ["_"] for e in evidence]
    bm = BM25Plus(tokenized_passages)
    raw_scores = bm.get_scores(q_tokens)
    q_set = set(q_tokens)

    scored = []
    for i, e in enumerate(evidence):
        p_tokens = tokenized_passages[i]
        overlap = len(q_set & set(p_tokens))
        s_bm = float(raw_scores[i]) if len(raw_scores) > i else 0.0

        # Drop passages with zero term overlap with query
        if overlap == 0:
            continue

        # Bounded quality multiplier (max 1.3x)
        qual = float(e.get("quality", 0.7))
        q_norm = max(0.0, min(1.0, (qual - 0.5) / 0.5))
        quality_mult = 1.0 + 0.3 * q_norm
        final_score = (s_bm + overlap * 1.5) * quality_mult

        scored.append({
            "evidence": e,
            "domain": _extract_domain(e.get("url", "")),
            "url": e.get("url", ""),
            "overlap": overlap,
            "final_score": final_score,
        })

    if not scored:
        # If no passage had strict non-stopword overlap, score raw token overlap or return evidence
        for i, e in enumerate(evidence):
            raw_overlap = len(set(raw_q_tokens) & set(tokenized_passages[i]))
            if raw_overlap > 0:
                scored.append({
                    "evidence": e,
                    "domain": _extract_domain(e.get("url", "")),
                    "url": e.get("url", ""),
                    "overlap": raw_overlap,
                    "final_score": float(raw_overlap),
                })
        if not scored:
            return evidence[:k]

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Cap passages per URL
    url_counts: dict[str, int] = {}
    capped: list[dict] = []
    for item in scored:
        u = item["url"]
        count = url_counts.get(u, 0)
        if count < max_per_url:
            url_counts[u] = count + 1
            capped.append(item)

    if not capped:
        return [item["evidence"] for item in scored[:k]]

    available_domains = {item["domain"] for item in capped if item["domain"]}
    if len(available_domains) >= min_distinct_domains:
        chosen: list[dict] = []
        chosen_domains: set[str] = set()
        for item in capped:
            if len(chosen) >= k:
                break
            d = item["domain"]
            if len(chosen_domains) < min_distinct_domains and d not in chosen_domains:
                chosen.append(item)
                chosen_domains.add(d)
        for item in capped:
            if len(chosen) >= k:
                break
            if item not in chosen:
                chosen.append(item)
                chosen_domains.add(item["domain"])
        chosen.sort(key=lambda x: x["final_score"], reverse=True)
        return [c["evidence"] for c in chosen[:k]]

    return [c["evidence"] for c in capped[:k]]


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
