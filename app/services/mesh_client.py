"""Client for the (future) distributed research mesh service.

All network calls use the standard library so this module has no hard
dependencies and stays importable in any environment.

INTEGRATION POINT: point ``MESH_SERVICE_URL`` (see ``.env.example``) at a
running mesh node exposing::

    POST /jobs        {"type": "research", "payload": {...}} -> {"job_id": ...}
    GET  /jobs/{id}   -> {"status": "...", "result": {...}}

Until then, ``MeshClient.available()`` is False and callers should treat the
mesh as disabled.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_TIMEOUT = 3.0  # seconds — the mesh is optional, never block on it


class MeshError(RuntimeError):
    """Raised when a mesh call fails."""


class MeshClient:
    """Minimal HTTP client for mesh job submission and polling."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = (base_url or os.getenv("MESH_SERVICE_URL", "") or "").rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        """True when a mesh URL is configured."""
        return bool(self.base_url)

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a job payload, returns the JSON response (contains job_id)."""
        return self._request("POST", "/jobs", body=payload)

    def job_status(self, job_id: str) -> dict[str, Any]:
        """GET the status/result of a previously submitted job."""
        return self._request("GET", f"/jobs/{urllib.request.quote(job_id)}")

    # -- internals ---------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict[str, Any]:
        if not self.base_url:
            raise MeshError("mesh service URL is not configured (set MESH_SERVICE_URL)")

        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MeshError(f"mesh request failed: {exc}") from exc