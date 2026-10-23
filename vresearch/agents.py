"""The five distinct agents. Each has its own prompt, its own inputs, and its own job:
  PLANNER      question -> sub-questions + search queries
  RESEARCHER   sub-question -> stored evidence passages (web + uploaded docs)
  DRAFTER      evidence -> atomic claims, each citing evidence ids
  VERIFIER     (claim, cited evidence) -> supported / partial / unsupported (+ programmatic checks)
  CONTRADICTOR verified claims -> pairs that disagree
  SYNTHESIZER  verified claims -> structured report text citing claim ids
"""
import datetime
import re

PLANNER_SYS = """You are the PLANNER in a research system. Break the user's question into 4-6 specific,
non-overlapping sub-questions that together cover it (for example: market size, key players, known risks,
recent events, regulation). For each sub-question give 2 distinct web-search queries. Today is {date}.
Return JSON: {"sub_questions":[{"id":"Q1","question":"...","queries":["...","..."]}]}"""

REFINE_SYS = """You are the RESEARCHER. A sub-question is thinly supported by evidence so far. Propose 2 NEW web-search
queries that use different wording or angles than the ones already tried, aiming for primary or authoritative
sources (regulators, company filings, research institutes, reputable press). Today is {date}.
Return JSON: {"queries":["...","..."]}"""

DRAFTER_SYS = """You are the DRAFTER. Write 3-6 atomic factual claims that answer the sub-question, using ONLY the
evidence passages provided. Hard rules:
- Every claim must be directly stated by, or a close paraphrase of, the passages it cites.
- Cite 1-3 evidence ids per claim (like "E4"). Never cite an id that was not provided.
- Any number, date or name in a claim must appear in the cited passage.
- Do NOT add background knowledge. If the evidence cannot support a claim, leave it out; fewer claims is fine.
- Do not repeat claims listed under ALREADY ESTABLISHED.
Return JSON: {"claims":[{"text":"...","evidence_ids":["E1"]}]}"""

VERIFIER_SYS = """You are the VERIFIER, an adversarial fact-checker. You see one CLAIM and the passages it cites.
Decide whether the passages themselves support the claim. Ignore your own knowledge: judge only from the text shown.
- "supported": the passages state the claim (paraphrase is fine), including every number, date and name in it.
- "partial": the passages support only part of the claim, or support it only loosely or by inference.
- "unsupported": the passages do not say this, or contradict it.
Also copy the shortest VERBATIM excerpt (max 200 chars) from a passage that supports the claim; leave it empty if none.
Return JSON: {"verdict":"supported|partial|unsupported","quote":"...","reason":"one sentence"}"""

CONTRA_SYS = """You are the CONTRADICTION DETECTOR. Given verified claims (with ids), find pairs that genuinely
disagree (conflicting numbers, opposing conclusions, incompatible dates). Do not flag claims that are merely
different or complementary. Return JSON:
{"contradictions":[{"claim_ids":["C1","C4"],"explanation":"what exactly conflicts"}]}
Return {"contradictions":[]} if there are none."""

SYNTH_SYS = """You are the SYNTHESIZER. Write a comprehensive, cited research report answering the user's question using ONLY the claims
provided (each has an id like C3). Rules:
- The "summary" MUST be an informative Executive Summary of 3-5 sentences summarizing the core verified findings across the sub-questions.
- In both "summary" and "sections", EVERY sentence MUST end with its claim citations like [C3] or [C3][C7]. Place citations directly before the period (e.g. "... increases cellular energy [C3].").
- Prioritize VERIFIED claims. Do not add facts, numbers or opinions that are not in the claims.
- Where claims are flagged as contradicting, state the disagreement openly.
- For sub-questions marked THIN, say briefly that evidence was limited rather than filling the gap.

Example format:
{
  "summary": "Regular exercise boosts mitochondrial efficiency and daily energy levels [C10]. Consistent sleep habits and limiting blue light prevent circadian disruptions [C1][C4]. Proper hydration and unrefined diets maintain cognitive focus throughout the day [C7][C8].",
  "sections": [
    {"id": "Q1", "text": "High-protein snacks elevate glucose sustainably without causing energy crashes [C1][C2]. Drinking water immediately counteracts sluggishness caused by mild dehydration [C3]."}
  ]
}"""

FOLLOWUP_SYS = """You answer a follow-up question using ONLY the earlier verified claims provided (ids like C3).
Every sentence must end with claim citations like [C3]. If the claims do not cover the question, say so in one
sentence WITHOUT inventing anything, and suggest a new research run. Return JSON: {"answer":"..."}"""


def today() -> str:
    return datetime.date.today().isoformat()


def plan(llm, question: str) -> list[dict]:
    out = llm.json(PLANNER_SYS.replace("{date}", today()), f"Question: {question}", tag="planner")
    subs = out.get("sub_questions", []) if isinstance(out, dict) else out
    result = []
    for i, s in enumerate(subs[:6], 1):
        q = (s.get("question") or "").strip()
        if not q:
            continue
        result.append({"id": f"Q{i}", "question": q, "queries": [x for x in s.get("queries", []) if x][:3] or [q],
                       "tried": []})
    if not result:
        result = [{"id": "Q1", "question": question, "queries": [question], "tried": []}]
    return result


def refine_queries(llm, subq: dict, evidence_summary: str) -> list[str]:
    user = (f"Sub-question: {subq['question']}\nQueries already tried: {subq['tried']}\n"
            f"What we have so far: {evidence_summary or 'nothing useful'}")
    out = llm.json(REFINE_SYS.replace("{date}", today()), user, tag="researcher_refine")
    qs = [q for q in out.get("queries", []) if q and q not in subq["tried"]]
    return qs[:2] or [subq["question"] + " report analysis"]


def draft_claims(llm, subq: dict, evidence: list[dict], existing: list[str]) -> list[dict]:
    block = "\n\n".join(f"[{e['eid']}] (source quality {e['quality']:.2f}, {e['title'][:60]})\n{e['text'][:1000]}"
                        for e in evidence)
    user = (f"Sub-question: {subq['question']}\n\nALREADY ESTABLISHED:\n"
            + ("\n".join(f"- {t}" for t in existing) or "(none)") + f"\n\nEVIDENCE:\n{block}")
    out = llm.json(DRAFTER_SYS, user, tag="drafter", max_tokens=1800)
    allowed = {e["eid"] for e in evidence}
    claims = []
    for c in out.get("claims", []):
        text = (c.get("text") or "").strip()
        ids = [i for i in c.get("evidence_ids", []) if i in allowed]
        if text:
            claims.append({"text": text, "evidence_ids": ids})
    return claims


def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[^a-z0-9%$.,' ]+", " ", re.sub(r"\s+", " ", s)).strip()


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)}


def verify_claim(llm, claim: dict, evidence_by_id: dict[str, dict], model: str | None = None) -> dict:
    """Returns the claim fields: status (verified/unverified/rejected), verdict, reason, quote."""
    cited = [evidence_by_id[i] for i in claim["evidence_ids"] if i in evidence_by_id]
    if not cited:
        return {"status": "rejected", "verdict": "unsupported", "reason": "cites no retrievable source", "quote": ""}
    block = "\n\n".join(f"[{e['eid']}] {e['text'][:1400]}" for e in cited)
    out = llm.json(VERIFIER_SYS, f"CLAIM: {claim['text']}\n\nCITED PASSAGES:\n{block}",
                   model=model, tag="verifier", max_tokens=500)
    verdict = str(out.get("verdict", "unsupported")).lower()
    quote, reason = (out.get("quote") or "").strip(), (out.get("reason") or "").strip()
    if verdict == "unsupported":
        return {"status": "rejected", "verdict": verdict, "reason": reason or "source does not support claim",
                "quote": ""}
    if verdict != "supported":
        return {"status": "unverified", "verdict": "partial", "reason": reason or "only partly supported",
                "quote": quote}
    # Programmatic grounding checks: the judge's quote must really be in a cited passage,
    # and every number in the claim must appear in the cited text.
    joined = _norm(" ".join(e["text"] for e in cited))
    if not quote or _norm(quote) not in joined:
        return {"status": "unverified", "verdict": "supported",
                "reason": "judge said supported but its quoted excerpt is not verbatim in the cited source",
                "quote": quote}
    missing = [n for n in _numbers(claim["text"]) if n not in _numbers(" ".join(e["text"] for e in cited))]
    if missing:
        return {"status": "unverified", "verdict": "supported",
                "reason": f"numbers not found in cited source: {', '.join(sorted(missing))}", "quote": quote}
    return {"status": "verified", "verdict": "supported", "reason": reason, "quote": quote}


def find_contradictions(llm, claims: list[dict]) -> list[dict]:
    if len(claims) < 2:
        return []
    listing = "\n".join(f"{c['cid']}: {c['text']}" for c in claims[:60])
    out = llm.json(CONTRA_SYS, listing, tag="contradictor", max_tokens=1200)
    valid = {c["cid"] for c in claims}
    res = []
    for item in out.get("contradictions", []):
        ids = [i for i in item.get("claim_ids", []) if i in valid]
        if len(set(ids)) >= 2:
            res.append({"claim_ids": list(dict.fromkeys(ids)), "explanation": item.get("explanation", "")})
    return res


def synthesize(llm, question: str, subqs: list[dict], claims: list[dict], contradictions: list[dict],
               thin_ids: set[str]) -> dict:
    lines = []
    for q in subqs:
        flag = " [THIN: limited verified evidence]" if q["id"] in thin_ids else ""
        lines.append(f"\n{q['id']}: {q['question']}{flag}")
        for c in claims:
            if c["subq_id"] == q["id"]:
                tag = "" if c["status"] == "verified" else " (UNVERIFIED)"
                lines.append(f"  {c['cid']}{tag}: {c['text']}")
    if contradictions:
        lines.append("\nCONTRADICTIONS: " + "; ".join(
            f"{' vs '.join(x['claim_ids'])} ({x['explanation']})" for x in contradictions))
    return llm.json(SYNTH_SYS, f"Question: {question}\n" + "\n".join(lines), tag="synthesizer", max_tokens=2500)


def followup(llm, question: str, claims: list[dict]) -> str:
    listing = "\n".join(f"{c['cid']}: {c['text']}" for c in claims)
    out = llm.json(FOLLOWUP_SYS, f"Follow-up question: {question}\n\nVERIFIED CLAIMS:\n{listing}",
                   tag="followup", max_tokens=900)
    return out.get("answer", "") if isinstance(out, dict) else str(out)
