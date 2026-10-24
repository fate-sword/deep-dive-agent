"""Diagnostic trace tool for Deeptrace research runs.

Usage:
    python -m deeptrace.diagnose --run-id <id> [--subq Q1]
    python -m vresearch.diagnose --run-id <id> [--subq Q1]
"""
import argparse
import json
import os
import re
import sys
from urllib.parse import urlparse

from vresearch.config import get_settings
from vresearch.ingest import rank_passages
from vresearch.source_quality import score_source
from vresearch.store import RunStore


def _domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "")
    except Exception:
        return url[:30]


def run_diagnose(run_id: str, subq_filter: str | None = None, db_path: str | None = None):
    settings = get_settings()
    db = db_path or settings.db_path or "data/runs.db"
    if not os.path.exists(db):
        print(f"Error: Database file not found at {db}")
        sys.exit(1)

    store = RunStore(db)
    runs = {r["id"]: r for r in store.list_runs(100)}
    if run_id not in runs:
        print(f"Error: Run ID '{run_id}' not found in {db}.")
        print(f"Available runs: {list(runs.keys())[:10]}")
        sys.exit(1)

    run_meta = runs[run_id]
    report = store.load_report(run_id) or {}
    all_events = store.load_events(run_id)
    all_evidence = store.load_evidence(run_id)
    all_claims = store.load_claims(run_id)

    print("=" * 80)
    print(f"DIAGNOSTIC TRACE FOR RUN: {run_id}")
    print(f"Main Question: {run_meta.get('question')}")
    print(f"Status: {run_meta.get('status')} | Total Evidence: {len(all_evidence)} | Total Claims: {len(all_claims)}")
    print("=" * 80)

    # Sub-questions from report or plan event
    subqs = report.get("sub_questions", [])
    if not subqs:
        for ev in all_events:
            if ev.get("step") == "plan" and ev.get("kind") == "subquestions":
                subqs = ev.get("payload", {}).get("sub_questions", [])
                break

    if not subqs:
        # Fallback to subqs extracted from evidence
        subq_ids = sorted(list({e["subq_id"] for e in all_evidence if e.get("subq_id")}))
        subqs = [{"id": sid, "question": f"Sub-question {sid}"} for sid in subq_ids]

    if subq_filter:
        subqs = [q for q in subqs if q.get("id") == subq_filter]
        if not subqs:
            print(f"No sub-question found matching '{subq_filter}'.")
            return

    for q in subqs:
        qid = q.get("id")
        q_text = q.get("question")
        print("\n" + "#" * 80)
        print(f"SUB-QUESTION [{qid}]: {q_text}")
        print("#" * 80)

        # 1. Queries and search provider
        search_events = [e for e in all_events if e.get("step") == "research" and e.get("kind") == "search" and e.get("payload", {}).get("subq") == qid]
        print("\n1. SEARCH QUERIES & PROVIDERS:")
        if not search_events:
            print("  (No search events logged for this sub-question)")
        for se in search_events:
            p = se.get("payload", {})
            print(f"  - Query: \"{p.get('query')}\"")
            print(f"    Provider: {p.get('provider', 'unknown')} | Results returned: {p.get('results', 0)}")
            print(f"    URLs returned ({len(p.get('urls', []))}):")
            for u in p.get("urls", []):
                print(f"      • {u} ({_domain(u)})")

        # 2. URLs returned vs fetched & Passages per page
        q_evidence = [e for e in all_evidence if e.get("subq_id") == qid]
        fetched_urls = {}
        for ev in q_evidence:
            u = ev.get("url", "")
            fetched_urls.setdefault(u, []).append(ev)

        print(f"\n2. FETCHED URLS & DOMAINS ({len(fetched_urls)} unique URLs fetched, {len(q_evidence)} passages stored):")
        domain_counts = {}
        for u, ev_list in fetched_urls.items():
            d = _domain(u)
            domain_counts[d] = domain_counts.get(d, 0) + len(ev_list)
            q_score = ev_list[0].get("quality", 0.0) if ev_list else score_source(u)
            print(f"  - [{d}] {u}")
            print(f"    Passages extracted: {len(ev_list)} | Quality Multiplier: {q_score:.2f}")

        print(f"\n   Domain Distribution:")
        for d, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     • {d}: {count} passages")

        # 3. Top 10 passages after ranking WITH scores and quality multiplier
        print(f"\n3. TOP 10 PASSAGES AFTER RANKING:")
        # Calculate BM25 / lexical relevance score against sub-question text
        q_words = set(re.findall(r"\w{3,}", q_text.lower()))
        scored_passages = []
        for ev in q_evidence:
            p_text = ev.get("text", "")
            p_words = set(re.findall(r"\w{3,}", p_text.lower()))
            overlap = len(q_words & p_words)
            rel_score = overlap / max(1, len(q_words))
            q_mult = ev.get("quality", score_source(ev.get("url", "")))
            combined = rel_score * q_mult
            scored_passages.append({
                "eid": ev.get("eid"),
                "url": ev.get("url"),
                "domain": _domain(ev.get("url", "")),
                "quality": q_mult,
                "relevance": rel_score,
                "combined": combined,
                "text": p_text
            })

        # Rank by current pipeline logic (quality only) vs combined
        sorted_by_quality = sorted(scored_passages, key=lambda x: x["quality"], reverse=True)
        for idx, sp in enumerate(sorted_by_quality[:10], 1):
            print(f"\n  Rank #{idx} | {sp['eid']} | Domain: {sp['domain']} | Quality Multiplier: {sp['quality']:.2f} | Topical Rel: {sp['relevance']:.3f} | Combined: {sp['combined']:.3f}")
            print(f"  URL: {sp['url']}")
            snippet = sp['text'].replace('\n', ' ')[:220]
            print(f"  Snippet: \"{snippet}...\"")

        # 4. Exact text given to the drafter
        # In current pipeline, top 8 by quality are given to the drafter
        drafter_passages = sorted([e for e in q_evidence], key=lambda e: e.get("quality", 0), reverse=True)[:8]
        print(f"\n4. EXACT EVIDENCE GIVEN TO DRAFTER ({len(drafter_passages)} passages):")
        for dp in drafter_passages:
            print(f"  [{dp['eid']}] ({_domain(dp.get('url', ''))}, quality={dp.get('quality', 0.0):.2f}):")
            print(f"    \"{dp.get('text', '').replace(chr(10), ' ')[:180]}...\"")

        # 5. Drafted claims
        q_claims = [c for c in all_claims if c.get("subq_id") == qid]
        print(f"\n5. DRAFTED CLAIMS FOR [{qid}] ({len(q_claims)} claims):")
        for c in q_claims:
            print(f"  • {c.get('cid')}: \"{c.get('text')}\"")
            print(f"    Status: {c.get('status')} | Verdict: {c.get('verdict')} | Evidence IDs: {c.get('evidence_ids')}")

    print("\n" + "=" * 80)
    print("STAGE FAILURE DIAGNOSTIC ASSESSMENT:")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Deeptrace Diagnostic Trace")
    parser.add_argument("--run-id", required=True, help="Run ID to diagnose")
    parser.add_argument("--subq", default=None, help="Optional sub-question ID (e.g. Q1)")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    args = parser.parse_args()

    run_diagnose(args.run_id, args.subq, args.db)


if __name__ == "__main__":
    main()
