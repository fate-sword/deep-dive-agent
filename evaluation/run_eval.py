"""Self-evaluation. Usage:  python -m evaluation.run_eval [--limit N] [--judge-model openai/gpt-4o]
Runs the real pipeline on evaluation/questions.json and writes results/eval_results.{json,md}.

Metrics per question
  groundedness       verified claims / drafted claims (the verifier's pass rate)
  quote_grounded     verified claims whose supporting quote is verbatim in the cited source (programmatic)
  coverage           sub-questions with >= MIN verified claims after loop-backs
  topic_recall       fraction of expected_topics keywords present in the report text (completeness proxy)
  independent_agree  (optional) agreement of a DIFFERENT model re-judging a sample of accepted claims
The 'thin-evidence' question is an adversarial case: a good system should produce few/no verified claims
and flag thin sections rather than invent an answer."""
import argparse
import json
import os
import time

from vresearch import agents
from vresearch.config import get_settings
from vresearch.llm import LLM
from vresearch.pipeline import run_research
from vresearch.store import RunStore


def independent_agreement(llm, settings, run_id, judge_model, sample=8):
    store = RunStore(settings.db_path)
    ev = {e["eid"]: e for e in store.load_evidence(run_id)}
    claims = [c for c in store.load_claims(run_id) if c["status"] == "verified"][:sample]
    if not claims:
        return None
    agree = sum(1 for c in claims if agents.verify_claim(llm, c, ev, judge_model)["verdict"] == "supported")
    return round(agree / len(claims), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--allow-dev-provider", action="store_true", default=False,
                    help="Allow evaluation using dev provider")
    args = ap.parse_args()
    settings = get_settings()

    if settings.llm_provider != "mesh" and not args.allow_dev_provider:
        raise SystemExit(
            f"Evaluation refused: active provider is '{settings.llm_provider}'. "
            "Hackathon submission evaluation must run with production 'mesh' provider. "
            "Pass --allow-dev-provider to run dev evaluations."
        )

    is_dev_run = settings.llm_provider != "mesh"
    llm = LLM(settings)
    here = os.path.dirname(__file__)
    questions = json.load(open(os.path.join(here, "questions.json")))[: args.limit]
    rows = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        t0 = time.time()
        try:
            run_id, rep = run_research(q["question"], settings=settings, llm=llm)
        except Exception as e:
            print("  FAILED:", e)
            rows.append({"id": q["id"], "error": str(e)})
            continue
        m = rep["metrics"]
        verified = rep["accepted"]
        ev = {e["eid"]: e for e in RunStore(settings.db_path).load_evidence(run_id)}
        quote_ok = sum(1 for c in verified if c["quote"] and agents._norm(c["quote"]) in agents._norm(
            " ".join(ev[i]["text"] for i in c["evidence_ids"] if i in ev)))
        text = (rep["summary"] + " " + " ".join(s["text"] for s in rep["sections"])).lower()
        topics = q.get("expected_topics", [])
        row = {"id": q["id"], "groundedness": m["groundedness"], "claims_total": m["claims_total"],
               "claims_verified": m["claims_verified"], "claims_unverified": m["claims_unverified"],
               "claims_rejected": m["claims_rejected"],
               "quote_grounded": round(quote_ok / len(verified), 3) if verified else None,
               "coverage": m["coverage"], "loops": m["loops"], "contradictions": m["contradictions"],
               "sources_cited": m["sources_cited"], "uncited_dropped": m["uncited_sentences_dropped"],
               "topic_recall": round(sum(t in text for t in topics) / len(topics), 3) if topics else None,
               "independent_agree": independent_agreement(llm, settings, run_id, args.judge_model) if args.judge_model else None,
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print("  ", {k: v for k, v in row.items() if k != "id"})

    ok = [r for r in rows if "error" not in r]

    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {k: avg(k) for k in ("groundedness", "quote_grounded", "coverage", "topic_recall", "independent_agree", "loops")}
    out_dir = os.path.join(os.path.dirname(here), "results")
    os.makedirs(out_dir, exist_ok=True)
    json_payload = {
        "provider": settings.llm_provider,
        "model": settings.model,
        "verifier_model": settings.verifier_model,
        "judge_model": args.judge_model,
        "dev_run": is_dev_run,
        "date": time.strftime("%Y-%m-%d"),
        "summary": summary,
        "rows": rows,
    }
    if is_dev_run:
        json_payload["banner"] = "DEV RUN, not valid for submission"

    json.dump(json_payload, open(os.path.join(out_dir, "eval_results.json"), "w"), indent=2)

    cols = ["id", "claims_total", "claims_verified", "claims_unverified", "claims_rejected", "groundedness",
            "quote_grounded", "coverage", "topic_recall", "independent_agree", "loops", "contradictions", "seconds"]

    lines = []
    if is_dev_run:
        lines += [
            "> [!WARNING]",
            "> **DEV RUN, not valid for submission**",
            f"> This evaluation was run with development provider `{settings.llm_provider}`. Final hackathon submission results must be produced with the Mesh API.",
            "",
        ]
    lines += [
        f"# Evaluation results ({time.strftime('%Y-%m-%d')})", "",
        f"Provider: `{settings.llm_provider}` · Model: `{settings.model}` · Verifier: `{settings.verifier_model}` · Independent judge: `{args.judge_model}`", "",
        "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) if "error" not in r else (r["id"] if c == "id" else "error") for c in cols) + " |")
    lines += ["", "**Averages:** " + ", ".join(f"{k} = {v}" for k, v in summary.items())]
    open(os.path.join(out_dir, "eval_results.md"), "w").write("\n".join(lines))
    print("\nSummary:", summary, "\nWrote results/eval_results.json and results/eval_results.md")


if __name__ == "__main__":
    main()
