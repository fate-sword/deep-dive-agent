"""Shareable exports: Markdown, standalone HTML page, and PDF (reportlab, no external services)."""
import html
import io
from xml.sax.saxutils import escape


def _m(r: dict) -> dict:
    return r.get("metrics", {})


def to_markdown(r: dict) -> str:
    m, out = _m(r), [f"# {r['question']}", ""]
    out += [f"*Groundedness: {m.get('groundedness', 0):.0%} of claims verified "
            f"({m.get('claims_verified', 0)}/{m.get('claims_total', 0)}); "
            f"sub-question coverage {m.get('coverage', 0):.0%}; research loops: {m.get('loops', 0)}*", ""]
    if r.get("summary"):
        out += ["## Summary", r["summary"], ""]
    for s in r["sections"]:
        out += [f"## {s['question']}", s["text"] or "_No verified evidence was found for this sub-question._"]
        if s["thin"]:
            out.append("> Evidence was limited here; treat conclusions with caution.")
        out.append("")
    out.append("## Sources")
    out += [f"{b['n']}. [{b['title']}]({b['url']}) ({b['tier']} source)" for b in r["sources"]]
    out += ["", "## Audit trail: accepted claims"]
    out += [f"- **{c['cid']}** {c['text']} " + "".join(f"[{n}]" for n in c["source_numbers"]) for c in r["accepted"]]
    out += ["", "## Audit trail: unverified claims"]
    out += [f"- **{c['cid']}** {c['text']} (reason: {c['reason']})" for c in r["unverified"]] or ["- none"]
    out += ["", "## Audit trail: rejected claims"]
    out += [f"- **{c['cid']}** {c['text']} (reason: {c['reason']})" for c in r["rejected"]] or ["- none"]
    return "\n".join(out)


def to_html(r: dict) -> str:
    e, m = html.escape, _m(r)
    p = [f"<h1>{e(r['question'])}</h1>",
         f"<p class='meta'>Groundedness {m.get('groundedness', 0):.0%} "
         f"({m.get('claims_verified', 0)}/{m.get('claims_total', 0)} claims verified) &middot; "
         f"coverage {m.get('coverage', 0):.0%} &middot; {m.get('loops', 0)} research loop(s)</p>"]
    if r.get("summary"):
        p.append(f"<h2>Summary</h2><p>{e(r['summary'])}</p>")
    for s in r["sections"]:
        p.append(f"<h2>{e(s['question'])}</h2><p>{e(s['text']) or '<em>No verified evidence found.</em>'}</p>")
        if s["thin"]:
            p.append("<p class='warn'>Evidence was limited here.</p>")
    p.append("<h2>Sources</h2><ol>" + "".join(
        f"<li><a href='{e(b['url'])}'>{e(b['title'])}</a> <small>({b['tier']})</small></li>" for b in r["sources"]) + "</ol>")
    for title, key in (("Accepted claims", "accepted"), ("Unverified claims", "unverified"), ("Rejected claims", "rejected")):
        rows = "".join(f"<li><b>{e(c['cid'])}</b> {e(c['text'])}"
                       + (f" <small>({e(c['reason'])})</small>" if key != "accepted" else "") + "</li>" for c in r[key])
        p.append(f"<h2>Audit: {title}</h2><ul>{rows or '<li>none</li>'}</ul>")
    css = ("body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a}"
           ".meta{color:#555}.warn{color:#a15c00}small{color:#666}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(r['question'])}</title><style>{css}</style></head><body>{''.join(p)}</body></html>"


def _pdf_text(s: str) -> str:
    return escape((s or "").encode("cp1252", "replace").decode("cp1252"))


def to_pdf(r: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    st, m = getSampleStyleSheet(), _m(r)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=r["question"][:80])
    f = [Paragraph(_pdf_text(r["question"]), st["Title"]),
         Paragraph(_pdf_text(f"Groundedness {m.get('groundedness', 0):.0%} ({m.get('claims_verified', 0)}/"
                             f"{m.get('claims_total', 0)} claims verified), coverage {m.get('coverage', 0):.0%}, "
                             f"{m.get('loops', 0)} research loop(s)"), st["Italic"]), Spacer(1, 10)]

    def h(t):
        f.append(Paragraph(_pdf_text(t), st["Heading2"]))

    def para(t, style="BodyText"):
        f.append(Paragraph(_pdf_text(t), st[style]))
        f.append(Spacer(1, 4))

    if r.get("summary"):
        h("Summary")
        para(r["summary"])
    for s in r["sections"]:
        h(s["question"])
        para(s["text"] or "No verified evidence was found for this sub-question.")
        if s["thin"]:
            para("Evidence was limited here; treat conclusions with caution.", "Italic")
    h("Sources")
    for b in r["sources"]:
        para(f"[{b['n']}] {b['title']} - {b['url']} ({b['tier']})")
    h("Audit trail: accepted claims")
    for c in r["accepted"]:
        para(f"{c['cid']}: {c['text']}")
    h("Audit trail: unverified claims")
    for c in r["unverified"] or [{"cid": "-", "text": "none", "reason": ""}]:
        para(f"{c['cid']}: {c['text']} {('(' + c['reason'] + ')') if c['reason'] else ''}")
    h("Audit trail: rejected claims")
    for c in r["rejected"] or [{"cid": "-", "text": "none", "reason": ""}]:
        para(f"{c['cid']}: {c['text']} {('(' + c['reason'] + ')') if c['reason'] else ''}")
    doc.build(f)
    return buf.getvalue()
