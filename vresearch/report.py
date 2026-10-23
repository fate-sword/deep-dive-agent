"""Turns claim-citations into numbered source citations and enforces 'no uncited sentence survives'."""
import re

from .source_quality import tier

CITE_RE = re.compile(r"\[(C\d+(?:\s*[,;]\s*C\d+)*)\]")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z\[])", text.strip()) if s.strip()]


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
    """Keep only sentences that cite real claims; rewrite [C3] into source numbers [1][2]."""
    kept, dropped = [], []
    for sent in split_sentences(text or ""):
        cids = [c.strip() for m in CITE_RE.findall(sent) for c in re.split(r"[,;]", m)]
        valid = [c for c in cids if c in claims_by_id]
        if not valid:
            dropped.append(sent)
            continue
        body = re.sub(r"\s+([.!?,;])", r"\1", CITE_RE.sub("", sent)).strip()
        end = re.search(r"[.!?]+$", body)
        punct = end.group(0) if end else "."
        body = body[: len(body) - len(punct)] if end else body
        nums = sorted({n for c in valid for n in registry.numbers_for_claim(claims_by_id[c])})
        marker = "".join(f"[{n}]" for n in nums)
        flag = " (unverified)" if all(claims_by_id[c]["status"] != "verified" for c in valid) else ""
        kept.append(f"{body} {marker}{flag}{punct}")
    return " ".join(kept), dropped
