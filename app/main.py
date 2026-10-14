"""ProofLens web entry point.

Serves the DEEP DIVE interactive landing screen and the workspace hand-off:

    /                -> app/static/index.html     (DEEP DIVE landing screen)
    /workspace.html  -> app/static/workspace.html (workspace integration point)
    /api/health      -> JSON health check
    /css /js /assets -> static files from app/static/

Run from the project root:

    python -m uvicorn app.main:app --reload      (requires requirements.txt)
    python -m app.main                           (works with stdlib only)
"""
from __future__ import annotations

import os
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

try:
    # --- FastAPI application (preferred) ---------------------------------
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="ProofLens",
        version="0.1.0",
        description="Research agents that go beyond the surface.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/api/health")
    def health() -> dict:
        """Liveness probe used by tooling and tests."""
        return {"status": "ok", "app": "prooflens"}

    # Static-first mount (registered last so /api/* takes precedence):
    # serves app/static/ at the site root with index.html as the default page.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

except ImportError:  # pragma: no cover - exercised only without requirements
    # --- Standard-library fallback server (zero dependencies) ------------
    import json
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    app = None  # type: ignore[assignment]

    class _LandingHandler(SimpleHTTPRequestHandler):
        """Serves app/static/ plus the /api/health JSON endpoint."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path.split("?", 1)[0].rstrip("/") in ("/api/health", "/api/health/"):
                body = json.dumps({"status": "ok", "app": "prooflens"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *args) -> None:  # keep the console calm
            pass


def run() -> None:
    """Start the web server (FastAPI/uvicorn when available, stdlib otherwise)."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    if app is not None:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    else:
        with ThreadingHTTPServer((host, port), _LandingHandler) as httpd:
            print(f"ProofLens (stdlib fallback) serving on http://{host}:{port}/")
            httpd.serve_forever()


if __name__ == "__main__":
    run()