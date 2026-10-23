"""SQLite persistence: runs, evidence (with source), claims (with verdicts) and the full event trail."""
import json
import os
import sqlite3
import threading
import time
import uuid


class RunStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock, self._c() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, question TEXT, created REAL,
                    status TEXT, report TEXT, metrics TEXT);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                    ts REAL, step TEXT, kind TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS evidence(run_id TEXT, eid TEXT, subq_id TEXT, url TEXT,
                    title TEXT, text TEXT, quality REAL, kind TEXT, PRIMARY KEY(run_id, eid));
                CREATE TABLE IF NOT EXISTS claims(run_id TEXT, cid TEXT, subq_id TEXT, text TEXT,
                    evidence_ids TEXT, status TEXT, verdict TEXT, reason TEXT, quote TEXT,
                    PRIMARY KEY(run_id, cid));
                """
            )

    def _c(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def new_run(self, question: str) -> str:
        rid = uuid.uuid4().hex[:10]
        with self._lock, self._c() as c:
            c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (rid, question, time.time(), "running", "", "{}"))
        return rid

    def log(self, run_id: str, step: str, kind: str, payload: dict) -> None:
        with self._lock, self._c() as c:
            c.execute("INSERT INTO events(run_id,ts,step,kind,payload) VALUES(?,?,?,?,?)",
                      (run_id, time.time(), step, kind, json.dumps(payload, default=str)))

    def save_evidence(self, run_id: str, ev: dict) -> None:
        with self._lock, self._c() as c:
            c.execute("INSERT OR REPLACE INTO evidence VALUES(?,?,?,?,?,?,?,?)",
                      (run_id, ev["eid"], ev["subq_id"], ev["url"], ev["title"], ev["text"], ev["quality"], ev["kind"]))

    def save_claim(self, run_id: str, cl: dict) -> None:
        with self._lock, self._c() as c:
            c.execute("INSERT OR REPLACE INTO claims VALUES(?,?,?,?,?,?,?,?,?)",
                      (run_id, cl["cid"], cl["subq_id"], cl["text"], json.dumps(cl["evidence_ids"]),
                       cl["status"], cl.get("verdict", ""), cl.get("reason", ""), cl.get("quote", "")))

    def finish(self, run_id: str, report: dict, status: str = "done") -> None:
        with self._lock, self._c() as c:
            c.execute("UPDATE runs SET status=?, report=?, metrics=? WHERE id=?",
                      (status, json.dumps(report, default=str), json.dumps(report.get("metrics", {})), run_id))

    def load_report(self, run_id: str) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT report FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(r["report"]) if r and r["report"] else None

    def load_claims(self, run_id: str) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT * FROM claims WHERE run_id=? ORDER BY CAST(SUBSTR(cid,2) AS INTEGER)",
                             (run_id,)).fetchall()
        return [{**dict(r), "evidence_ids": json.loads(r["evidence_ids"])} for r in rows]

    def load_evidence(self, run_id: str) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT * FROM evidence WHERE run_id=? ORDER BY CAST(SUBSTR(eid,2) AS INTEGER)",
                             (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def load_events(self, run_id: str) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [{"ts": r["ts"], "step": r["step"], "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows]

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT id, question, created, status, metrics FROM runs ORDER BY created DESC LIMIT ?",
                             (limit,)).fetchall()
        return [{**dict(r), "metrics": json.loads(r["metrics"] or "{}")} for r in rows]
