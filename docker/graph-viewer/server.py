"""Tiny static graph viewer server."""

from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
DATA_PATH = Path(os.environ.get("GRAPH_VIEWER_DATA", ROOT / "data" / "graph.json"))


class Handler(SimpleHTTPRequestHandler):
    """Serve static assets and graph JSON."""

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.path in {"/", "/index.html"}:
            return str(STATIC_ROOT / "index.html")
        return str(STATIC_ROOT / parsed.path.lstrip("/"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/graph":
            self._send_graph()
            return
        super().do_GET()

    def _send_graph(self) -> None:
        try:
            payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = os.environ.get("GRAPH_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPH_VIEWER_PORT", "3000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"LLMASM graph viewer serving http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
