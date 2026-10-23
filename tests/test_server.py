"""FastAPI server tests: endpoints, SSE streaming of sample runs, uploads, exports, follow-up, and trail."""
import json
from fastapi.testclient import TestClient

from server import app


def test_status_endpoint():
    client = TestClient(app)
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Deeptrace"
    assert "live" in data


def test_create_run_and_stream_sample():
    client = TestClient(app)
    # 1. Create run in sample mode
    r = client.post("/api/runs", data={"question": "Test EV battery market?", "sample": 1})
    assert r.status_code == 200
    run_info = r.json()
    assert "run_id" in run_info
    assert run_info["sample"] is True
    run_id = run_info["run_id"]

    # 2. Stream events
    r_stream = client.get(f"/api/runs/{run_id}/events")
    assert r_stream.status_code == 200
    assert "text/event-stream" in r_stream.headers["content-type"]

    events = []
    for line in r_stream.text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            events.append(payload)

    assert len(events) >= 5
    # First event should be plan/subquestions
    assert events[0]["step"] == "plan"
    assert events[0]["kind"] == "subquestions"
    # Terminal event should be report
    assert events[-1]["kind"] == "report"
    assert "metrics" in events[-1]["report"]


def test_create_run_with_provider():
    client = TestClient(app)
    r = client.post("/api/runs", data={"question": "Test EV battery market?", "sample": 1, "provider": "openai"})
    assert r.status_code == 200
    run_info = r.json()
    assert run_info["provider"] == "openai"


def test_file_upload_handling():
    client = TestClient(app)
    fake_txt = ("Battery pack recycling requires strict hydrometallurgical processing standards.\n" * 5).encode()
    files = [("files", ("notes.txt", fake_txt, "text/plain"))]
    r = client.post("/api/runs", data={"question": "What are recycling requirements?", "sample": 1}, files=files)
    assert r.status_code == 200
    assert "run_id" in r.json()


def test_followup_and_trail():
    client = TestClient(app)
    # Followup on sample
    r = client.post("/api/runs/sample/followup", json={"question": "What about costs?"})
    assert r.status_code == 200
    assert "answer" in r.json()
    assert len(r.json()["answer"]) > 10

    # Decision trail
    r_trail = client.get("/api/runs/sample/trail")
    assert r_trail.status_code == 200
    trail = r_trail.json()
    assert "events" in trail
    assert len(trail["events"]) > 0


def test_export_formats():
    client = TestClient(app)
    for fmt in ("html", "md", "pdf"):
        r = client.get(f"/api/report/sample.{fmt}")
        assert r.status_code == 200
        if fmt == "html":
            assert "<!doctype html>" in r.text
        elif fmt == "md":
            assert "# " in r.text
        elif fmt == "pdf":
            assert r.content.startswith(b"%PDF")


def test_legacy_api_run():
    client = TestClient(app)
    r = client.get("/api/run?sample=1")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: " in r.text
