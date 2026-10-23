"""FastAPI site: serves the Deeptrace research UI, streams agent steps over SSE,
handles document uploads, decision trails, and follow-up QA."""
import asyncio
import json
import os
import queue
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from vresearch import answer_followup, run_research
from vresearch.config import get_settings
from vresearch.export import to_html, to_markdown, to_pdf
from vresearch.store import RunStore

WEB = os.path.join(os.path.dirname(__file__), "web")
app = FastAPI(title="Deeptrace", description="AI Research Agent")
settings = get_settings()

SAMPLE_PATH = os.path.join(WEB, "sample_run.json")
SAMPLE = json.load(open(SAMPLE_PATH, encoding="utf-8")) if os.path.exists(SAMPLE_PATH) else {"events": [], "report": {}}

slots = threading.Semaphore(2)
last_by_ip: dict[str, float] = {}
ip_lock = threading.Lock()

# Active runs store: run_id -> run metadata
active_runs: dict[str, dict] = {}
runs_lock = threading.Lock()


def sse_format(data) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


def sample_events():
    for e in SAMPLE.get("events", []):
        time.sleep(e.get("delay", 0.4))
        yield sse_format({k: v for k, v in e.items() if k != "delay"})
    yield sse_format({"kind": "report", "report": SAMPLE.get("report", {}), "sample": True})


def live_events(question: str, uploads: list[tuple[str, bytes]] | None = None, cancel_event: threading.Event | None = None, provider: str | None = None):
    q: queue.Queue = queue.Queue()
    worker_started = threading.Event()

    def work():
        worker_started.set()
        try:
            with slots:
                if cancel_event and cancel_event.is_set():
                    return
                run_id, rep = run_research(
                    question,
                    uploads=uploads,
                    settings=settings,
                    on_event=lambda s, k, p: q.put({"step": s, "kind": k, "payload": p}),
                    provider=provider
                )
            q.put({"kind": "report", "report": rep})
        except Exception as ex:
            q.put({"kind": "error", "message": str(ex)[:300]})
        finally:
            q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    worker_started.wait(timeout=2.0)

    last_heartbeat = time.time()
    while True:
        if cancel_event and cancel_event.is_set():
            break
        try:
            item = q.get(timeout=1.0)
            if item is None:
                break
            yield sse_format(item)
            last_heartbeat = time.time()
        except queue.Empty:
            # Send keepalive comment every ~15 seconds to keep connection open through proxies
            if time.time() - last_heartbeat >= 15.0:
                yield ": keep-alive\n\n"
                last_heartbeat = time.time()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/status")
def status():
    return {
        "name": "Deeptrace",
        "live": bool(settings.api_key or settings.openai_api_key),
        "mesh": bool(settings.api_key),
        "openai": bool(settings.openai_api_key),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "default_provider": settings.llm_provider,
        "model": settings.model,
        "verifier_model": settings.verifier_model,
    }


@app.post("/api/runs")
async def create_run(
    request: Request,
    question: str = Form(""),
    sample: int = Form(0),
    provider: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    q = question.strip()[:400]
    client_ip = request.client.host if request.client else "unknown"
    selected_provider = (provider or settings.llm_provider or "mesh").lower()

    # Check upload sizes and extensions
    uploads: list[tuple[str, bytes]] = []
    total_upload_size = 0
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename.lower())[1]
        if ext not in (".pdf", ".txt", ".md"):
            continue
        content = await f.read()
        total_upload_size += len(content)
        if total_upload_size > 10 * 1024 * 1024:  # 10 MB total limit
            raise HTTPException(400, "Uploaded documents exceed 10 MB total limit.")
        uploads.append((f.filename, content))

    has_key = bool(settings.openai_api_key if selected_provider == "openai" else settings.api_key)
    is_sample = bool(sample or not has_key or not q)
    if not is_sample:
        with ip_lock:
            now = time.time()
            if now - last_by_ip.get(client_ip, 0) < 30:
                raise HTTPException(429, "Please wait 30 seconds between research runs.")
            last_by_ip[client_ip] = now

    run_id = f"sample_{uuid.uuid4().hex[:8]}" if is_sample else uuid.uuid4().hex[:10]
    with runs_lock:
        active_runs[run_id] = {
            "run_id": run_id,
            "question": q,
            "sample": is_sample,
            "provider": selected_provider,
            "uploads": uploads,
            "created": time.time(),
            "cancel": threading.Event(),
        }

    return {"run_id": run_id, "sample": is_sample, "question": q, "provider": selected_provider}


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(request: Request, run_id: str):
    with runs_lock:
        run_data = active_runs.get(run_id)

    if not run_data:
        # Fall back to sample if run_id starts with sample
        if run_id.startswith("sample") or run_id == "sample":
            run_data = {"sample": True, "question": "", "uploads": [], "cancel": threading.Event()}
        else:
            raise HTTPException(404, "Run not found or expired.")

    cancel_event = run_data.get("cancel")

    async def event_generator():
        try:
            if run_data.get("sample"):
                for chunk in sample_events():
                    if await request.is_disconnected():
                        break
                    yield chunk
            else:
                for chunk in live_events(run_data["question"], run_data.get("uploads"), cancel_event, provider=run_data.get("provider")):
                    if await request.is_disconnected():
                        if cancel_event:
                            cancel_event.set()
                        break
                    yield chunk
        finally:
            if cancel_event:
                cancel_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Backward-compatible endpoint: GET /api/run
@app.get("/api/run")
def run_legacy(request: Request, q: str = "", sample: int = 0):
    q = q.strip()[:400]
    if sample or not settings.api_key or not q:
        return StreamingResponse(sample_events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    ip = request.client.host if request.client else "?"
    with ip_lock:
        now = time.time()
        if now - last_by_ip.get(ip, 0) < 30:
            return StreamingResponse(
                iter([sse_format({"kind": "error", "message": "Please wait 30 seconds between runs."})]),
                media_type="text/event-stream"
            )
        last_by_ip[ip] = now

    return StreamingResponse(
        live_events(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


@app.post("/api/runs/{run_id}/followup")
def followup_run(run_id: str, payload: dict):
    question = (payload.get("question") or "").strip()[:400]
    provider = payload.get("provider")
    if not question:
        raise HTTPException(400, "Question is required")

    if run_id.startswith("sample") or run_id == "sample":
        # Deterministic sample follow-up
        return {
            "answer": "Earlier research indicates strong battery demand growth driven by falling cell costs [1][2], although regulatory compliance and penalties remain primary investment risks [3].",
            "sources": SAMPLE.get("report", {}).get("sources", []),
            "dropped": []
        }

    try:
        res = answer_followup(run_id, question, settings=settings, provider=provider)
        return res
    except Exception as e:
        raise HTTPException(500, f"Followup error: {str(e)}")


@app.get("/api/runs/{run_id}/trail")
def decision_trail(run_id: str):
    if run_id.startswith("sample") or run_id == "sample":
        return {
            "run_id": "sample",
            "events": SAMPLE.get("events", []),
            "accepted": SAMPLE.get("report", {}).get("accepted", []),
            "unverified": SAMPLE.get("report", {}).get("unverified", []),
            "rejected": SAMPLE.get("report", {}).get("rejected", []),
            "dropped_sentences": SAMPLE.get("report", {}).get("dropped_sentences", []),
            "contradictions": SAMPLE.get("report", {}).get("contradictions", []),
        }

    store = RunStore(settings.db_path)
    events = store.load_events(run_id)
    report = store.load_report(run_id)
    if not events and not report:
        raise HTTPException(404, "Decision trail not found")

    return {
        "run_id": run_id,
        "events": events,
        "accepted": report.get("accepted", []) if report else [],
        "unverified": report.get("unverified", []) if report else [],
        "rejected": report.get("rejected", []) if report else [],
        "dropped_sentences": report.get("dropped_sentences", []) if report else [],
        "contradictions": report.get("contradictions", []) if report else [],
    }


@app.get("/api/report/{run_id}.{fmt}")
def report(run_id: str, fmt: str):
    rep = SAMPLE.get("report") if (run_id == "sample" or run_id.startswith("sample")) else RunStore(settings.db_path).load_report(run_id)
    if not rep or "error" in rep:
        raise HTTPException(404, "Report not found")
    if fmt == "pdf":
        return Response(to_pdf(rep), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=deeptrace-{run_id}.pdf"})
    if fmt == "html":
        return Response(to_html(rep), media_type="text/html")
    if fmt == "md":
        return Response(to_markdown(rep), media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=deeptrace-{run_id}.md"})
    raise HTTPException(404, "Unknown format")
