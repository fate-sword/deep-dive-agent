"""Turns claim-citations into numbered source citations and enforces 'no uncited sentence survives'."""
import re

from .source_quality import tier

CITE_RE = re.compile(r"\[(C\d+(?:\s*[,;]\s*C\d+)*)\]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while ensuring citations stay attached to their preceding text."""
    if not text:
        return []
    # Normalize: if a citation appears right after a period like '. [C1]', merge it into ' [C1].' so periods don't split it
    normalized = re.sub(r"\.\s*(\[(?:C\d+(?:\s*[,;]\s*C\d+)*)\])", r" \1.", text)
    # Split on sentence boundaries: period, question mark, or exclamation followed by space and capital letter or newline
    raw_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", normalized.strip()) if s.strip()]
    merged: list[str] = []
    for part in raw_parts:
        # Check if this part contains ONLY citations or punctuation with no real words
        words = re.sub(CITE_RE, "", part).strip()
        is_only_citations = not bool(re.search(r"[a-zA-Z]{2,}", words)) and bool(CITE_RE.search(part))
        if is_only_citations and merged:
            # Reattach orphaned citations to the preceding sentence
            merged[-1] = merged[-1].rstrip(".!? ") + " " + part.strip()
            if not merged[-1].endswith((".", "!", "?")):
                merged[-1] += "."
        else:
            merged.append(part)
    return merged


class SourceRegistry:
    """Assigns [1], [2]... numbers to unique sources in order of first citation."""

    def __init__(self, evidence: list[dict]):
        self.by_eid = {e["eid"]: e for e in evidence}
        self.numbers: dict[str, int] = {}
        self.meta: dict[str, dict] = {}

    def numbers_for_claim(self, claim: dict) -> list[int]:
        nums = []
        for eid in claim["evidence_ids"]:
            e = self.by_eid.get(eid)
            if not e:
                continue
            if e["url"] not in self.numbers:
                self.numbers[e["url"]] = len(self.numbers) + 1
                self.meta[e["url"]] = {"title": e["title"], "url": e["url"], "quality": e["quality"],
                                       "tier": tier(e["quality"]), "kind": e["kind"]}
            n = self.numbers[e["url"]]
            if n not in nums:
                nums.append(n)
        return sorted(nums)

    def bibliography(self) -> list[dict]:
        return [{"n": n, **self.meta[u]} for u, n in sorted(self.numbers.items(), key=lambda x: x[1])]


def ground_text(text: str, claims_by_id: dict[str, dict], registry: SourceRegistry) -> tuple[str, list[str]]:
    """Keep only sentences that cite verified claims; rewrite [C3] into source numbers [1][2]."""
    kept, dropped = [], []
    for sent in split_sentences(text or ""):
        # Guard: delete any sentence mentioning being unverified or flagged as unverified
        if re.search(r"\bunverified\b", sent, re.I):
            dropped.append(sent)
            continue
        cids = [c.strip() for m in CITE_RE.findall(sent) for c in re.split(r"[,;]", m)]
        valid = [c for c in cids if c in claims_by_id]
        if not valid:
            dropped.append(sent)
            continue
        # Guard: delete any sentence citing a claim whose status is not "verified"
        if any(claims_by_id[c].get("status") != "verified" for c in valid):
            dropped.append(sent)
            continue
        body = re.sub(r"\s+([.!?,;])", r"\1", CITE_RE.sub("", sent)).strip()
        # Enforce that the sentence actually contains substantive words, never bare numbers
        if not re.search(r"[a-zA-Z]{2,}", body):
            dropped.append(sent)
            continue
        end = re.search(r"[.!?]+$", body)
        punct = end.group(0) if end else "."
        body = body[: len(body) - len(punct)] if end else body
        nums = sorted({n for c in valid for n in registry.numbers_for_claim(claims_by_id[c])})
        marker = "".join(f"[{n}]" for n in nums)
        kept.append(f"{body} {marker}{punct}")
    return " ".join(kept), dropped
