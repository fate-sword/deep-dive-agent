"""Findings Board: notegrid data structure builder.

Transforms a verified research report into a lane-based notegrid board:
- Verbatim accepted claims only on the board
- Rejected and unverified claims quarantined in corner tag
- Sub-question lanes with Caveat headers and thin-evidence notices
"""
import re
from urllib.parse import urlparse


def _extract_domain(url_or_title: str) -> str:
    if not url_or_title:
        return ""
    try:
        netloc = urlparse(url_or_title).netloc
        if netloc:
            return netloc.replace("www.", "")
    except Exception:
        pass
    return url_or_title[:24]


def _clean_lane_title(q_id: str, question: str) -> str:
    """Generate a concise, human-friendly Caveat lane header like 'Q2 · regulatory risks'."""
    q_clean = re.sub(r"^(what|how|why|which|are|is|describe|explain)\s+(is|are|the|do|does|have)?\s*", "", question, flags=re.IGNORECASE).strip()
    q_clean = re.sub(r"[?!.]+$", "", q_clean).strip()
    words = q_clean.split()
    short = " ".join(words[:4]).lower() if words else question[:25].lower()
    return f"{q_id} · {short}"


def build_findings_board(report: dict) -> dict:
    """Build the Findings Board structure strictly honoring the grounding rule:

    - Every note's text exactly equals an accepted claim's verbatim text.
    - No unverified or rejected claims are EVER rendered as notes on the board.
    - Rejected and unverified claims appear exclusively in the corner tag with reasons.
    - Lanes with fewer than 2 verified claims show 'not enough proof found'.
    """
    accepted = report.get("accepted", [])
    unverified = report.get("unverified", [])
    rejected = report.get("rejected", [])
    subqs = report.get("sub_questions", [])
    if not subqs and report.get("sections"):
        subqs = [{"id": s.get("id"), "question": s.get("question", "")} for s in report.get("sections", [])]

    # Map sources by number and url for domain chips
    source_map = {}
    for s in report.get("sources", []):
        n = s.get("n")
        domain = _extract_domain(s.get("url") or s.get("title", ""))
        if n:
            source_map[n] = domain
        if s.get("url"):
            source_map[s["url"]] = domain

    # Group accepted claims by sub-question ID
    accepted_by_subq: dict[str, list[dict]] = {}
    for c in accepted:
        sq = c.get("subq_id", "Q1")
        accepted_by_subq.setdefault(sq, []).append(c)

    # 1. Headline Finding: one serif sentence from summary or lead verified claim
    headline_text = ""
    highlight_phrase = ""
    summary = (report.get("summary") or "").strip()
    if summary:
        # Pick the first sentence
        first_sent = re.split(r"(?<=[.!?])\s+", summary)[0].strip()
        headline_text = first_sent
    elif accepted:
        headline_text = accepted[0].get("text", "")

    # Pick 2-4 words in headline for the hand-drawn marker highlight
    h_words = [w.strip(".,;:?!\"'()") for w in headline_text.split() if len(w) > 3]
    if len(h_words) >= 2:
        highlight_phrase = f"{h_words[0]} {h_words[1]}"
    elif h_words:
        highlight_phrase = h_words[0]

    # 2. Build lanes (one per sub-question)
    lanes = []
    for q in subqs:
        qid = q.get("id", "Q1")
        q_question = q.get("question", "")
        q_claims = accepted_by_subq.get(qid, [])
        v_count = len(q_claims)
        is_thin = v_count < 2

        notes = []
        for c in q_claims:
            # Source domain chips
            nums = c.get("source_numbers", [])
            domains = []
            for n in nums:
                d = source_map.get(n)
                if d and d not in domains:
                    domains.append(d)
            if not domains and c.get("sources"):
                domains = list(c["sources"])[:2]
            if not domains:
                domains = ["retrieved source"]

            notes.append({
                "cid": c.get("cid", ""),
                "text": str(c.get("text", "")),  # VERBATIM text
                "domains": domains[:2],
                "source_numbers": nums,
                "status": "verified",
                "status_label": "Verified",
            })

        lanes.append({
            "subq_id": qid,
            "title": _clean_lane_title(qid, q_question),
            "full_question": q_question,
            "verified_count": v_count,
            "is_thin": is_thin,
            "thin_label": "not enough proof found" if is_thin else "",
            "notes": notes,
        })

    # 3. Corner Tag: quarantined rejected & unverified claims
    rejected_list = []
    for r_c in rejected:
        rejected_list.append({
            "cid": r_c.get("cid", ""),
            "text": str(r_c.get("text", "")),
            "reason": r_c.get("reason", "unsupported by cited source"),
            "status": "rejected",
        })

    unverified_list = []
    for u_c in unverified:
        unverified_list.append({
            "cid": u_c.get("cid", ""),
            "text": str(u_c.get("text", "")),
            "reason": u_c.get("reason", "partially supported or unverified quote"),
            "status": "unverified",
        })

    corner_tag = {
        "rejected_count": len(rejected_list),
        "unverified_count": len(unverified_list),
        "label": f"{len(rejected_list)} rejected, {len(unverified_list)} unverified",
        "rejected": rejected_list,
        "unverified": unverified_list,
    }

    # 4. Footer row
    grounded_val = report.get("metrics", {}).get("groundedness", 0.0)
    grounded_pct = int(round(grounded_val * 100)) if grounded_val <= 1.0 else int(grounded_val)

    return {
        "mono_label": "KEY FINDINGS · PINNED FROM VERIFIED CLAIMS",
        "headline": {
            "text": headline_text,
            "highlight": highlight_phrase,
        },
        "lanes": lanes,
        "corner_tag": corner_tag,
        "footer": {
            "grounded_pct": grounded_pct,
            "prose_summary": summary,
            "sources_count": len(report.get("sources", [])),
        },
    }
