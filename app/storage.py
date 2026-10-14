"""Tiny JSON-file storage for research sessions.

Sessions are written one-per-file under ``data/`` so the folder stays easy to
inspect and back up. Thread safety is handled with a simple module-level lock;
swap in a database by re-implementing these functions when needed.

INTEGRATION POINT: the web layer can expose CRUD endpoints over these helpers.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from app.schemas import ResearchSession

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_lock = threading.Lock()


def _path_for(session_id: str) -> Path:
    return DATA_DIR / f"session_{session_id}.json"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session: ResearchSession) -> Path:
    """Persist a session; returns the file it was written to."""
    _ensure_dir()
    path = _path_for(session.id)
    with _lock:
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return path


def load_session(session_id: str) -> Optional[ResearchSession]:
    """Load a session by id, or ``None`` when it does not exist."""
    path = _path_for(session_id)
    if not path.is_file():
        return None
    with _lock:
        data = json.loads(path.read_text(encoding="utf-8"))
    return ResearchSession.from_dict(data)


def list_session_ids() -> list[str]:
    """Return all stored session ids (newest file first)."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        (p.name[len("session_") : -len(".json")] for p in DATA_DIR.glob("session_*.json")),
        reverse=True,
    )


def delete_session(session_id: str) -> bool:
    """Delete a stored session. Returns True when something was removed."""
    path = _path_for(session_id)
    with _lock:
        if path.is_file():
            path.unlink()
            return True
    return False