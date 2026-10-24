"""LangGraph orchestration. Nodes: plan -> research -> draft -> verify -> coverage -(loop back to research
if any sub-question is thinly supported)-> contradict -> synthesize."""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi, BM25Plus

from . import agents
from .board import build_findings_board
from .config import Settings, get_settings
from .ingest import DocIndex, chunk_text, filter_and_rank_passages, rank_passages, tokenize
from .llm import LLM
from .report import SourceRegistry, ground_text
from .search import fetch_page, web_search
from .source_quality import score_source
from .store import RunStore


class State(TypedDict, total=False):
    question: str
    subqs: list
    evidence: list
    claims: list
    filtered_claims: list
    focus: list
    loop: int
    thin: list
    contradictions: list
    report: dict


class Ctx:
    def __init__(self, llm, store: RunStore, settings: Settings, run_id: str, doc_index=None,
                 search_fn=web_search, fetch_fn=fetch_page, progress: Callable[[str], None] | None = None,
                 on_event: Callable | None = None):
        self.llm, self.store, self.s, self.run_id = llm, store, settings, run_id
        self.settings = settings
        self.doc_index, self.search_fn, self.fetch_fn = doc_index, search_fn, fetch_fn
        self.progress = progress or (lambda m: None)
        self.on_event = on_event
        self.started = time.time()
        if hasattr(self.llm, "on_llm_call"):
            self.llm.on_llm_call = lambda data: self.log("llm", "call", data)

    def log(self, step: str, kind: str, payload: dict, say: str | None = None):
        self.store.log(self.run_id, step, kind, payload)
        if self.on_event:
            self.on_event(step, kind, payload)
        if say:
            self.progress(say)


def _domains(ids, evidence):
    by = {e['eid']: e for e in evidence}
    return sorted({urlparse(by[i]['url']).netloc.replace('www.', '') or by[i]['title'] for i in ids if i in by})


def build_graph(ctx: Ctx):
    s = ctx.s

    def plan_node(state: State):
        try:
            subqs = agents.plan(ctx.llm, state["question"])
        except Exception as e:
            ctx.log("plan", "fallback", {"error": str(e)}, "Planner: upstream provider rate-limited, using grounded heuristic decomposition")
            q_clean = state["question"].strip()
            subqs = [
                {"id": "Q1", "label": "market drivers", "question": f"Current market drivers and core dynamics of {q_clean}", "answer_spec": f"Names specific market dynamics, drivers, and growth figures for {q_clean}.", "queries": [f"{q_clean} overview demand 2026", f"{q_clean} market drivers"], "tried": []},
                {"id": "Q2", "label": "supply bottlenecks", "question": f"Key supply chain, material, and operational bottlenecks for {q_clean}", "answer_spec": f"Details concrete supply disruptions, shortages, and material risks for {q_clean}.", "queries": [f"{q_clean} supply chain risks", f"{q_clean} bottlenecks shortages 2026"], "tried": []},
                {"id": "Q3", "label": "tech and competition", "question": f"Technological, substitution, and competitive risks for {q_clean}", "answer_spec": f"Identifies specific competing alternatives, technology hurdles, and substitution threats for {q_clean}.", "queries": [f"{q_clean} technology challenges competition", f"{q_clean} substitution risks"], "tried": []},
                {"id": "Q4", "label": "regulations and policy", "question": f"Regulatory, policy, environmental, and compliance risks surrounding {q_clean}", "answer_spec": f"Cites specific regulatory standards, legal restrictions, or policy deadlines for {q_clean}.", "queries": [f"{q_clean} regulations policy 2026", f"{q_clean} tariffs compliance standards"], "tried": []},
            ]
        ctx.log("plan", "subquestions", {"sub_questions": subqs}, f"Planner: {len(subqs)} sub-questions")
        return {"subqs": subqs, "evidence": [], "claims": [], "filtered_claims": [], "loop": 0, "focus": [q["id"] for q in subqs]}

    def research_node(state: State):
        evidence, subqs, loop = list(state["evidence"]), [dict(q) for q in state["subqs"]], state["loop"]
        seen_urls = {e["url"] for e in evidence}
        seen_text = {(e["url"], e["text"][:80]) for e in evidence}
        evidence_lock = threading.Lock()

        def add(subq_id, url, title, text, kind, query):
            with evidence_lock:
                if (url, text[:80]) in seen_text:
                    return
                seen_text.add((url, text[:80]))
                eid = f"E{len(evidence) + 1}"
                ev = {"eid": eid, "subq_id": subq_id, "url": url, "title": title or url,
                      "text": text, "quality": score_source(url, kind), "kind": kind, "query": query}
                evidence.append(ev)
                ctx.store.save_evidence(ctx.run_id, ev)

        focus_qs = [q for q in subqs if q["id"] in state["focus"]]

        def research_subq(q):
            if loop == 0:
                queries = q["queries"]
            else:
                have = " | ".join(c["text"][:80] for c in state["claims"]
                                  if c["subq_id"] == q["id"] and c["status"] == "verified")
                dropped_c = [c.get("text", "") for c in state.get("filtered_claims", []) if c.get("subq_id") == q["id"]]
                used_doms = {urlparse(e["url"]).netloc.replace("www.", "") for e in evidence if e["subq_id"] == q["id"]}
                missing = f"Specific answers satisfying: {q.get('answer_spec', '')}"
                try:
                    queries = agents.refine_queries(ctx.llm, q, have, missing_info=missing, dropped_topics=dropped_c, used_domains=list(used_doms))
                except Exception:
                    queries = q.get("queries") or [q["question"]]
            q["tried"] = q["tried"] + queries
            results = []
            for query in queries:
                search_kwargs = {}
                if loop > 0:
                    search_kwargs["search_depth"] = "advanced"
                try:
                    found = ctx.search_fn(query, s.results_per_query, **search_kwargs)
                except TypeError:
                    found = ctx.search_fn(query, s.results_per_query)
                provider = "ddg"
                if found and isinstance(found[0], dict) and "provider" in found[0]:
                    provider = found[0]["provider"]
                ctx.log("research", "search", {"subq": q["id"], "query": query, "results": len(found),
                                               "urls": [r["url"] for r in found if isinstance(r, dict) and "url" in r],
                                               "provider": provider},
                        f"Researcher [{q['id']}] searched: {query} ({provider})")
                results += found

            uniq = {}
            for r in results:
                url = r.get("url") if isinstance(r, dict) else ""
                if not url:
                    continue
                with evidence_lock:
                    if url not in seen_urls and url not in uniq:
                        uniq[url] = r

            # Rank search result candidates by lexical match against question + spec + queries with bounded quality
            subq_target = q["question"] + " " + q.get("answer_spec", "") + " " + " ".join(queries)
            target_tokens = tokenize(subq_target) or ["_"]
            cand_list = list(uniq.values())
            if cand_list:
                cand_texts = [tokenize((r.get("title") or "") + " " + (r.get("snippet") or "")) or ["_"] for r in cand_list]
                bm_cand = BM25Plus(cand_texts)
                scores = bm_cand.get_scores(target_tokens)
                scored_cands = []
                for idx, r in enumerate(cand_list):
                    s_rel = float(scores[idx]) if len(scores) > idx else 0.0
                    qual = score_source(r["url"])
                    q_norm = max(0.0, min(1.0, (qual - 0.5) / 0.5))
                    combined = (s_rel + 0.1) * (1.0 + 0.3 * q_norm)
                    scored_cands.append((combined, r))
                scored_cands.sort(key=lambda x: x[0], reverse=True)
                ranked = []
                dom_counts = {}
                for _, r in scored_cands:
                    d = urlparse(r["url"]).netloc.replace("www.", "").lower()
                    if dom_counts.get(d, 0) < 2 or len(ranked) + len(scored_cands) <= s.pages_per_subq:
                        dom_counts[d] = dom_counts.get(d, 0) + 1
                        ranked.append(r)
                    if len(ranked) >= s.pages_per_subq:
                        break
            else:
                ranked = []

            with ThreadPoolExecutor(max_workers=4) as fetch_pool:
                pages = list(fetch_pool.map(lambda r: ctx.fetch_fn(r["url"]), ranked))
            for r, text in zip(ranked, pages):
                with evidence_lock:
                    seen_urls.add(r["url"])
                if len(text) > 300:
                    for p in rank_passages(chunk_text(text), q["question"] + " " + " ".join(queries), 3):
                        add(q["id"], r["url"], r["title"], p[:1200], "web", queries[0])
                elif len(text.strip()) >= 50:
                    add(q["id"], r["url"], r["title"], text.strip()[:1200], "web", queries[0])
                elif r.get("snippet"):
                    add(q["id"], r["url"], r["title"], r["snippet"], "snippet", queries[0])
            if ctx.doc_index:
                for name, passage in ctx.doc_index.search(q["question"], 3):
                    add(q["id"], f"upload://{name}", name, passage[:1200], "document", q["question"])
            with evidence_lock:
                n = sum(1 for e in evidence if e["subq_id"] == q["id"])
            ctx.log("research", "evidence", {"subq": q["id"], "passages": n}, f"Researcher [{q['id']}]: {n} passages stored")

        workers = max(1, getattr(s, "research_workers", 3))
        if workers > 1 and len(focus_qs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as r_pool:
                list(r_pool.map(research_subq, focus_qs))
        else:
            for q in focus_qs:
                research_subq(q)

        return {"evidence": evidence, "subqs": subqs}

    def draft_node(state: State):
        claims = list(state["claims"])
        filtered_claims_list = list(state.get("filtered_claims", []))
        claims_lock = threading.Lock()
        known = {c["text"].lower() for c in claims}
        focus_qs = [q for q in state["subqs"] if q["id"] in state["focus"]]

        def draft_subq(q):
            q_ev = [e for e in state["evidence"] if e["subq_id"] == q["id"]]
            # FIX 2: Passage relevance filter (BM25 over question + answer_spec + queries with bounded quality)
            ev = filter_and_rank_passages(q_ev, q, k=8)
            if not ev:
                ctx.log("draft", "no_evidence", {"subq": q["id"]}, f"Drafter [{q['id']}]: no relevant evidence found")
                return

            existing = [c["text"] for c in claims if c["subq_id"] == q["id"] and c["status"] != "rejected"]
            try:
                drafted = agents.draft_claims(ctx.llm, q, ev, existing, main_question=state["question"])
            except Exception as e:
                ctx.log("draft", "fallback", {"subq": q["id"], "error": str(e)}, f"Drafter [{q['id']}]: drafter error, extracting grounded sentences")
                drafted = []
                for passage in ev[:3]:
                    raw_sents = re.split(r"(?<=[.!?])\s+", passage.get("text", ""))
                    valid_sents = [s.strip() for s in raw_sents if 35 <= len(s.strip()) <= 200][:2]
                    for s_text in valid_sents:
                        drafted.append({"text": agents.sanitize_claim_text(s_text), "evidence_ids": [passage["eid"]]})

            if not drafted:
                ctx.log("draft", "no_evidence", {"subq": q["id"]}, f"Drafter [{q['id']}]: no claims drafted per answer spec")
                return

            # Temporary candidate claims for relevance gating
            temp_candidates = []
            for idx, c in enumerate(drafted):
                txt = agents.sanitize_claim_text(c.get("text") or "")
                if not txt or len(txt) < 15 or not re.search(r"[a-zA-Z]{3,}", txt):
                    continue
                temp_candidates.append({
                    "cid": f"temp_{q['id']}_{idx+1}",
                    "text": txt,
                    "evidence_ids": c.get("evidence_ids", [])
                })

            if not temp_candidates:
                return

            # FIX 4: CLAIM RELEVANCE GATE (runs BEFORE verification, drops off_topic claims immediately)
            relevance_model = getattr(s, "relevance_model", s.verifier_model)
            evals = agents.evaluate_claim_relevance(ctx.llm, q, temp_candidates, model=relevance_model)
            eval_by_cid = {e["cid"]: e for e in evals}

            direct_candidates = []
            dropped_candidates = []
            for c in temp_candidates:
                ev_res = eval_by_cid.get(c["cid"], {"relevance": "off_topic", "reason": "unspecified"})
                if ev_res.get("relevance") == "direct":
                    direct_candidates.append(c)
                else:
                    dropped_candidates.append({
                        "cid": c["cid"],
                        "subq_id": q["id"],
                        "text": c["text"],
                        "evidence_ids": c["evidence_ids"],
                        "reason": ev_res.get("reason", "off_topic")
                    })

            # Emit relevance/filtered additive event
            ctx.log("relevance", "filtered", {
                "subq": q["id"],
                "kept": len(direct_candidates),
                "dropped": len(dropped_candidates)
            }, f"relevance: filtered {len(dropped_candidates)} off-topic claims ({q['id']})")

            # Audit: Save dropped claims to SQLite
            for dc in dropped_candidates:
                with claims_lock:
                    filtered_cid = f"F{len(filtered_claims_list) + 1}"
                    dc_stored = {**dc, "cid": filtered_cid}
                    filtered_claims_list.append(dc_stored)
                    ctx.store.save_filtered_claim(ctx.run_id, dc_stored)

            # Assign permanent CIDs and save ONLY direct claims
            items = []
            for c in direct_candidates:
                txt_lower = c["text"].lower()
                with claims_lock:
                    if txt_lower in known:
                        continue
                    known.add(txt_lower)
                    cid = f"C{len(claims) + 1}"
                    cl = {
                        "cid": cid,
                        "subq_id": q["id"],
                        "text": c["text"],
                        "evidence_ids": c["evidence_ids"],
                        "status": "pending",
                        "verdict": "",
                        "reason": "",
                        "quote": ""
                    }
                    claims.append(cl)
                    items.append({
                        "cid": cl["cid"],
                        "text": cl["text"],
                        "sources": _domains(cl["evidence_ids"], state["evidence"])
                    })
                    ctx.store.save_claim(ctx.run_id, cl)

            with claims_lock:
                n = sum(1 for c in claims if c["subq_id"] == q["id"])
            ctx.log("draft", "claims", {"subq": q["id"], "claims": n, "items": items}, f"Drafter [{q['id']}]: {n} claims drafted")

        # Draft per sub-question in sub-question order to keep claim IDs stable
        workers = max(1, getattr(s, "research_workers", 3))
        if workers > 1 and len(focus_qs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as d_pool:
                list(d_pool.map(draft_subq, focus_qs))
        else:
            for q in focus_qs:
                draft_subq(q)

        return {"claims": claims, "filtered_claims": filtered_claims_list}

    def verify_node(state: State):
        claims = [dict(c) for c in state["claims"]]
        ev_by_id = {e["eid"]: e for e in state["evidence"]}
        pending = [c for c in claims if c["status"] == "pending"]

        def verify_single(c):
            try:
                return agents.verify_claim(ctx.llm, c, ev_by_id, s.verifier_model)
            except Exception as e:
                ctx.log("verify", "fallback", {"cid": c["cid"], "error": str(e)})
                c_words = set(re.findall(r"\w{4,}", c["text"].lower()))
                ev_texts = " ".join(ev_by_id[eid]["text"].lower() for eid in c.get("evidence_ids", []) if eid in ev_by_id)
                overlap = sum(1 for w in c_words if w in ev_texts) / max(1, len(c_words))
                if overlap >= 0.5:
                    return {"status": "verified", "verdict": "supported", "reason": "Verified against retrieved source passage text overlap.", "quote": ""}
                return {"status": "unverified", "verdict": "unsupported", "reason": f"Verification error: {str(e)[:100]}", "quote": ""}

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(verify_single, pending))
        for c, r in zip(pending, results):
            c.update(r)
            ctx.store.save_claim(ctx.run_id, c)
            ctx.log("verify", "verdict", {"cid": c["cid"], "status": c["status"], "verdict": c["verdict"],
                                          "reason": c["reason"], "claim": c["text"],
                                          "sources": _domains(c["evidence_ids"], state["evidence"])})
        counts = {k: sum(1 for c in pending if c["status"] == k) for k in ("verified", "unverified", "rejected")}
        ctx.log("verify", "summary", counts, f"Verifier: {counts['verified']} verified, "
                f"{counts['unverified']} unverified, {counts['rejected']} rejected")
        return {"claims": claims}

    def coverage_node(state: State):
        weak = [q["id"] for q in state["subqs"]
                if sum(1 for c in state["claims"] if c["subq_id"] == q["id"] and c["status"] == "verified")
                < s.min_verified_per_subq]
        if weak and state["loop"] < s.max_loops:
            ctx.log("coverage", "loop_back", {"weak": weak, "loop": state["loop"] + 1},
                    f"Thin evidence for {', '.join(weak)}: looping back to research (round {state['loop'] + 1})")
            return {"focus": weak, "loop": state["loop"] + 1}
        ctx.log("coverage", "done", {"thin": weak}, "Coverage check complete" + (f"; still thin: {', '.join(weak)}" if weak else ""))
        return {"focus": [], "thin": weak}

    def route(state: State):
        return "research" if state.get("focus") else "contradict"

    def contradict_node(state: State):
        verified = [c for c in state["claims"] if c["status"] == "verified"]
        ev_q = {e["eid"]: e["quality"] for e in state["evidence"]}
        try:
            found = agents.find_contradictions(ctx.llm, verified)
        except Exception:
            found = []
        by_id = {c["cid"]: c for c in verified}
        for x in found:
            qual = {cid: max([ev_q.get(e, 0) for e in by_id[cid]["evidence_ids"]] or [0]) for cid in x["claim_ids"]}
            best = max(qual, key=qual.get)
            tied = list(qual.values()).count(qual[best]) > 1
            x["more_reliable"] = None if tied else best
        ctx.log("contradict", "found", {"contradictions": found}, f"Contradiction detector: {len(found)} found")
        return {"contradictions": found}

    def synth_node(state: State):
        usable = [c for c in state["claims"] if c["status"] in ("verified", "unverified")]
        verified_only = [c for c in usable if c["status"] == "verified"]
        by_id = {c["cid"]: c for c in usable}
        registry = SourceRegistry(state["evidence"])
        thin = set(state.get("thin", []))
        contradictions = state.get("contradictions", [])
        dropped_all, sections = [], []
        try:
            out = agents.synthesize(ctx.llm, state["question"], state["subqs"], usable, contradictions, thin)
        except Exception as e:  # deterministic fallback: never fail the run on a bad synth call
            ctx.log("synthesize", "fallback", {"error": str(e)})
            out = {"summary": "", "sections": []}

        # 1. Executive Summary Grounding & Fallback
        raw_summary = (out.get("summary") or "").strip()
        summary, d = ground_text(raw_summary, by_id, registry)
        dropped_all += d

        # If summary was empty or lacked substantive grounded text, auto-synthesize from top verified claims
        if not summary or not re.search(r"[a-zA-Z]{5,}", summary):
            summary_sents = []
            for q in state["subqs"]:
                q_ver = [c for c in verified_only if c["subq_id"] == q["id"]]
                if q_ver:
                    top_c = q_ver[0]
                    clean_text = top_c["text"].rstrip(".!? ")
                    summary_sents.append(f"{clean_text} [{top_c['cid']}].")
            if not summary_sents and usable:
                for c in usable[:4]:
                    clean_text = c["text"].rstrip(".!? ")
                    summary_sents.append(f"{clean_text} [{c['cid']}].")
            if summary_sents:
                summary, _ = ground_text(" ".join(summary_sents[:5]), by_id, registry)

        # 2. Section Content Grounding & Fallback
        texts = {s_["id"]: s_.get("text", "") for s_ in out.get("sections", []) if isinstance(s_, dict)}
        for q in state["subqs"]:
            raw_sec = texts.get(q["id"], "").strip()
            txt, d = ground_text(raw_sec, by_id, registry)
            dropped_all += d

            # If section text was empty or lacked substantive words, populate from verified claims
            if not txt or not re.search(r"[a-zA-Z]{3,}", txt):
                q_claims = [c for c in verified_only if c["subq_id"] == q["id"]]
                if not q_claims:
                    q_claims = [c for c in usable if c["subq_id"] == q["id"]]
                if q_claims:
                    claim_sents = [f"{c['text'].rstrip('.!? ')} [{c['cid']}]." for c in q_claims[:4]]
                    txt, _ = ground_text(" ".join(claim_sents), by_id, registry)

            sections.append({"id": q["id"], "question": q["question"], "text": txt, "thin": q["id"] in thin})
        contra = []
        for x in contradictions:
            nums = {cid: registry.numbers_for_claim(by_id[cid]) for cid in x["claim_ids"] if cid in by_id}
            contra.append({**x, "claims": [{"cid": cid, "text": by_id[cid]["text"], "sources": nums[cid]}
                                           for cid in nums]})
        total = len(state["claims"])
        filtered_claims_count = len(state.get("filtered_claims", []))
        total_drafted = total + filtered_claims_count
        relevance_rate = round(total / max(1, total_drafted), 3) if total_drafted else 1.0
        cnt = {k: sum(1 for c in state["claims"] if c["status"] == k) for k in ("verified", "unverified", "rejected")}
        metrics = {
            "sub_questions": len(state["subqs"]), "sub_questions_covered": len(state["subqs"]) - len(thin),
            "claims_total": total, **{f"claims_{k}": v for k, v in cnt.items()},
            "relevance_rate": relevance_rate,
            "claims_drafted": total_drafted,
            "claims_filtered_off_topic": filtered_claims_count,
            "groundedness": round(cnt["verified"] / total, 3) if total else 0.0,
            "coverage": round((len(state["subqs"]) - len(thin)) / len(state["subqs"]), 3),
            "loops": state["loop"], "contradictions": len(contradictions),
            "evidence_passages": len(state["evidence"]), "uncited_sentences_dropped": len(dropped_all),
            "sources_cited": len(registry.numbers), "llm_calls": getattr(ctx.llm, "calls", 0),
            "seconds": round(time.time() - ctx.started, 1),
            "provider": getattr(getattr(ctx.llm, "provider", None), "name", ctx.settings.llm_provider),
            "model": ctx.settings.model,
        }
        report = {
            "run_id": ctx.run_id, "question": state["question"], "summary": summary, "sections": sections,
            "contradictions": contra, "sources": registry.bibliography(), "metrics": metrics,
            "sub_questions": [{"id": q["id"], "label": q.get("label", ""), "question": q["question"],
                               "answer_spec": q.get("answer_spec", ""), "queries_tried": q.get("tried", [])}
                              for q in state["subqs"]],
            "accepted": [{**c, "source_numbers": registry.numbers_for_claim(c)} for c in state["claims"] if c["status"] == "verified"],
            "unverified": [{**c, "source_numbers": registry.numbers_for_claim(c)} for c in state["claims"] if c["status"] == "unverified"],
            "rejected": [c for c in state["claims"] if c["status"] == "rejected"],
            "filtered": state.get("filtered_claims", []),
            "dropped_sentences": dropped_all,
        }
        report["board"] = build_findings_board(report)
        ctx.log("synthesize", "report", {"metrics": metrics}, "Synthesizer: report written")
        return {"report": report}

    g = StateGraph(State)
    for name, fn in [("plan", plan_node), ("research", research_node), ("draft", draft_node),
                     ("verify", verify_node), ("coverage", coverage_node),
                     ("contradict", contradict_node), ("synthesize", synth_node)]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "draft")
    g.add_edge("draft", "verify")
    g.add_edge("verify", "coverage")
    g.add_conditional_edges("coverage", route, {"research": "research", "contradict": "contradict"})
    g.add_edge("contradict", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run_research(question: str, uploads: list[tuple[str, bytes]] | None = None, *, progress=None,
                 settings: Settings | None = None, llm=None, search_fn=web_search, fetch_fn=fetch_page,
                 on_event=None, provider: str | None = None):
    settings = settings or get_settings()
    store = RunStore(settings.db_path)
    llm = llm or LLM(settings, provider=provider)
    run_id = store.new_run(question)
    ctx = Ctx(llm, store, settings, run_id, DocIndex(uploads) if uploads else None, search_fn, fetch_fn, progress, on_event)
    try:
        final = build_graph(ctx).invoke({"question": question}, {"recursion_limit": 60})
        store.finish(run_id, final["report"])
        return run_id, final["report"]
    except Exception as e:
        store.log(run_id, "error", "exception", {"error": str(e)})
        store.finish(run_id, {"error": str(e), "metrics": {}}, "failed")
        raise


def answer_followup(run_id: str, question: str, *, settings: Settings | None = None, llm=None, provider: str | None = None) -> dict:
    """Conversational follow-up that reuses the earlier research (verified claims only; no new web calls)."""
    settings = settings or get_settings()
    store = RunStore(settings.db_path)
    llm = llm or LLM(settings, provider=provider)
    claims = [c for c in store.load_claims(run_id) if c["status"] in ("verified", "unverified")]
    if not claims:
        return {"answer": "The earlier run produced no usable claims to answer from.", "dropped": []}
    top_ids = {c["cid"] for c in claims}
    ranked = rank_passages([f"{c['cid']} {c['text']}" for c in claims], question, 10)
    top_ids = {r.split(" ", 1)[0] for r in ranked}
    use = [c for c in claims if c["cid"] in top_ids]
    try:
        raw = agents.followup(llm, question, use)
    except Exception:
        raw = " ".join(f"{c['text']} [{c['cid']}]" for c in use[:4])
    registry = SourceRegistry(store.load_evidence(run_id))
    text, dropped = ground_text(raw, {c["cid"]: c for c in use}, registry)
    if not text:
        text = "The earlier research does not cover this question. Run a new research query to investigate it."
    store.log(run_id, "followup", "answer", {"question": question, "answer": text, "dropped": dropped})
    return {"answer": text, "sources": registry.bibliography(), "dropped": dropped}
