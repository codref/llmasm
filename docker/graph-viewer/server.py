"""LLMASM graph viewer — DB-backed diagnostic server."""

from __future__ import annotations

import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
DATA_PATH = Path(os.environ.get("GRAPH_VIEWER_DATA", ROOT / "data" / "graph.json"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# DB connection (lazy singleton, psycopg3)
# ---------------------------------------------------------------------------

_conn = None


def _db_conn():
    global _conn
    if _conn is None or _conn.closed:
        try:
            import psycopg
            from psycopg.rows import dict_row

            _conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to DB: {exc}") from exc
    return _conn


def _query(sql: str, params: tuple = ()) -> list[dict]:
    conn = _db_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = _query(sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------


def _api_workspaces() -> list:
    return _query(
        """
        SELECT wg.id, wg.name, wg.status, wg.created_at,
               COUNT(DISTINCT tg.id) AS task_graph_count,
               COUNT(DISTINCT r.id)  AS run_count
        FROM workspace_graphs wg
        LEFT JOIN task_graphs tg ON tg.workspace_graph_id = wg.id
        LEFT JOIN runs r ON r.workspace_graph_id = wg.id
        GROUP BY wg.id
        ORDER BY wg.created_at DESC
        """
    )


def _api_workspace_task_graphs(ws_id: str) -> list:
    return _query(
        """
        SELECT tg.id, tg.status, tg.parent_task_graph_id, tg.created_at,
               COUNT(n.id) AS node_count,
               r.id AS run_id, r.status AS run_status
        FROM task_graphs tg
        LEFT JOIN nodes n ON n.task_graph_id = tg.id
        LEFT JOIN LATERAL (
            SELECT id, status FROM runs
            WHERE task_graph_id = tg.id
            ORDER BY started_at DESC NULLS LAST
            LIMIT 1
        ) r ON true
        WHERE tg.workspace_graph_id = %s
        GROUP BY tg.id, tg.status, tg.parent_task_graph_id, tg.created_at, r.id, r.status
        ORDER BY tg.created_at DESC
        """,
        (ws_id,),
    )


def _api_workspace_graph(ws_id: str, task_graph_id: str | None) -> dict:
    # ── workspace meta ────────────────────────────────────────────────────
    ws = _query_one("SELECT id, name, status FROM workspace_graphs WHERE id = %s", (ws_id,))
    if ws is None:
        raise KeyError(f"Workspace not found: {ws_id!r}")

    # ── subgraph tree for sidebar ─────────────────────────────────────────
    task_graphs = _api_workspace_task_graphs(ws_id)

    # ── nodes (with latest run status) ────────────────────────────────────
    if task_graph_id:
        node_rows = _query(
            """
            SELECT n.id, n.kind, n.name, n.task_graph_id,
                   n.input_schema, n.output_schema, n.execution_json, n.metadata,
                   rns.status AS run_status, rns.attempts,
                   rns.last_error_json, latest.run_id
            FROM nodes n
            LEFT JOIN LATERAL (
                SELECT r.id AS run_id FROM runs r
                WHERE r.task_graph_id = n.task_graph_id
                ORDER BY r.started_at DESC NULLS LAST
                LIMIT 1
            ) latest ON true
            LEFT JOIN run_node_states rns
                   ON rns.node_id = n.id AND rns.run_id = latest.run_id
            WHERE n.workspace_graph_id = %s AND n.task_graph_id = %s
            ORDER BY n.id
            """,
            (ws_id, task_graph_id),
        )
        edge_rows = _query(
            "SELECT * FROM task_edges WHERE workspace_graph_id = %s AND task_graph_id = %s",
            (ws_id, task_graph_id),
        )
    else:
        node_rows = _query(
            """
            SELECT n.id, n.kind, n.name, n.task_graph_id,
                   n.input_schema, n.output_schema, n.execution_json, n.metadata,
                   rns.status AS run_status, rns.attempts,
                   rns.last_error_json, latest.run_id
            FROM nodes n
            LEFT JOIN LATERAL (
                SELECT r.id AS run_id FROM runs r
                WHERE r.task_graph_id = n.task_graph_id
                ORDER BY r.started_at DESC NULLS LAST
                LIMIT 1
            ) latest ON true
            LEFT JOIN run_node_states rns
                   ON rns.node_id = n.id AND rns.run_id = latest.run_id
            WHERE n.workspace_graph_id = %s
            ORDER BY n.id
            """,
            (ws_id,),
        )
        edge_rows = _query(
            "SELECT * FROM task_edges WHERE workspace_graph_id = %s",
            (ws_id,),
        )

    nodes = [
        {
            "id": row["id"],
            "label": row["name"],
            "kind": row["kind"],
            "task_graph_id": row["task_graph_id"],
            "status": row["run_status"] or "pending",
            "run_id": row.get("run_id"),
            "attempts": row.get("attempts"),
            "last_error": row.get("last_error_json"),
            "schema": {
                "input": row["input_schema"],
                "output": row["output_schema"],
            },
            "metrics": {},
            "metadata": row["metadata"] or {},
            "execution": row["execution_json"] or {},
        }
        for row in node_rows
    ]

    edges = [
        {
            "id": row["id"],
            "source": row["from_node_id"],
            "target": row["to_node_id"],
            "type": "dataflow",
            "label": f"{row['from_port']} → {row['to_port']}",
            "metadata": row["metadata"] or {},
        }
        for row in edge_rows
    ]
    # When showing all subgraphs, add synthetic cross-task-graph edges so the
    # layered layout renders a conversational chain instead of N independent
    # rows.  task_graphs.parent_task_graph_id is not always populated, so we
    # fall back to chronological ordering (created_at ASC).
    if not task_graph_id:
        tg_node_map: dict[str, list] = {}
        for n in nodes:
            tg_node_map.setdefault(n["task_graph_id"], []).append(n)

        node_successors: dict[str, set] = {}
        node_predecessors: dict[str, set] = {}
        for e in edges:
            node_successors.setdefault(e["source"], set()).add(e["target"])
            node_predecessors.setdefault(e["target"], set()).add(e["source"])

        # Build ordered chain: prefer parent_task_graph_id links, otherwise
        # fall back to sorted-by-created_at pairs.
        tg_parent_map = {tg["id"]: tg.get("parent_task_graph_id") for tg in task_graphs}
        has_parent_links = any(v for v in tg_parent_map.values())

        if has_parent_links:
            ordered_pairs = [
                (tg["parent_task_graph_id"], tg["id"])
                for tg in task_graphs
                if tg.get("parent_task_graph_id")
            ]
        else:
            # Sort task_graphs by created_at ascending and chain them
            sorted_tgs = sorted(
                [tg for tg in task_graphs if tg["id"] in tg_node_map],
                key=lambda t: t["created_at"],
            )
            ordered_pairs = [
                (sorted_tgs[i]["id"], sorted_tgs[i + 1]["id"])
                for i in range(len(sorted_tgs) - 1)
            ]

        for parent_tg_id, child_tg_id in ordered_pairs:
            parent_nodes = tg_node_map.get(parent_tg_id, [])
            child_nodes = tg_node_map.get(child_tg_id, [])
            if not parent_nodes or not child_nodes:
                continue
            parent_ids = {n["id"] for n in parent_nodes}
            child_ids = {n["id"] for n in child_nodes}
            # Leaf = node with no successors inside the parent subgraph
            leaves = [
                n for n in parent_nodes
                if not node_successors.get(n["id"], set()) & parent_ids
            ]
            # Root = node with no predecessors inside the child subgraph
            roots = [
                n for n in child_nodes
                if not node_predecessors.get(n["id"], set()) & child_ids
            ]
            if leaves and roots:
                edges.append({
                    "id": f"cross__{parent_tg_id}__{child_tg_id}",
                    "source": leaves[0]["id"],
                    "target": roots[0]["id"],
                    "type": "cross",
                    "label": "",
                    "metadata": {},
                })
    # overall run status: take the "worst" status across all latest runs
    run_statuses = {row["run_status"] for row in node_rows if row.get("run_status")}
    STATUS_ORDER = ["failed", "running", "pending", "succeeded"]
    overall = next((s for s in STATUS_ORDER if s in run_statuses), "unknown")

    return {
        "metadata": {
            "workspace_id": ws["id"],
            "workspace_name": ws["name"],
            "task_graph_id": task_graph_id or "all",
            "run_status": overall,
        },
        "task_graphs": [_jsonable(tg) for tg in task_graphs],
        "nodes": nodes,
        "edges": edges,
    }


def _api_node_detail(node_id: str, run_id: str | None) -> dict:
    node = _query_one("SELECT * FROM nodes WHERE id = %s", (node_id,))
    if node is None:
        raise KeyError(f"Node not found: {node_id!r}")

    if not run_id:
        latest = _query_one(
            """
            SELECT id FROM runs WHERE task_graph_id = %s
            ORDER BY started_at DESC NULLS LAST LIMIT 1
            """,
            (node["task_graph_id"],),
        )
        run_id = latest["id"] if latest else None

    run_state = None
    artifacts = []
    model_calls = []
    tool_calls = []

    if run_id:
        run_state = _query_one(
            "SELECT * FROM run_node_states WHERE node_id = %s AND run_id = %s",
            (node_id, run_id),
        )
        artifacts = _query(
            "SELECT id, port, content_type, content_json, token_count, created_at "
            "FROM artifacts WHERE node_id = %s AND run_id = %s ORDER BY created_at",
            (node_id, run_id),
        )
        model_calls = _query(
            "SELECT id, model, provider, status, token_json, created_at "
            "FROM model_calls WHERE node_id = %s AND run_id = %s ORDER BY created_at",
            (node_id, run_id),
        )
        tool_calls = _query(
            "SELECT id, tool_name, status, latency_ms, input_json, created_at "
            "FROM tool_calls WHERE node_id = %s AND run_id = %s ORDER BY created_at",
            (node_id, run_id),
        )

    return {
        "node": _jsonable(node),
        "run_id": run_id,
        "run_state": _jsonable(run_state) if run_state else None,
        "artifacts": [_jsonable(a) for a in artifacts],
        "model_calls": [_jsonable(m) for m in model_calls],
        "tool_calls": [_jsonable(t) for t in tool_calls],
    }


def _api_search(q: str, workspace_id: str | None, limit: int) -> dict:
    like = f"%{q}%"
    ws_filter = (workspace_id,) if workspace_id else ()

    node_sql = (
        "SELECT n.id, n.name AS label, n.kind, n.task_graph_id, n.workspace_graph_id "
        "FROM nodes n WHERE n.name ILIKE %s"
        + (" AND n.workspace_graph_id = %s" if workspace_id else "")
        + f" LIMIT {limit}"
    )
    memory_sql = (
        "SELECT id, kind, text, confidence FROM memory_items WHERE text ILIKE %s"
        + (" AND workspace_graph_id = %s" if workspace_id else "")
        + f" LIMIT {limit}"
    )
    goal_sql = (
        "SELECT id, text, status FROM goals WHERE text ILIKE %s"
        + (" AND workspace_graph_id = %s" if workspace_id else "")
        + f" LIMIT {limit}"
    )
    artifact_sql = (
        "SELECT a.id, a.node_id, a.run_id, a.port, "
        "LEFT(a.content_json::text, 300) AS excerpt "
        "FROM artifacts a WHERE a.content_json::text ILIKE %s"
        + (
            " AND EXISTS (SELECT 1 FROM runs r WHERE r.id = a.run_id AND r.workspace_graph_id = %s)"
            if workspace_id
            else ""
        )
        + f" LIMIT {limit}"
    )

    params = (like, *ws_filter)
    return {
        "nodes": [_jsonable(r) for r in _query(node_sql, params)],
        "memory_items": [_jsonable(r) for r in _query(memory_sql, params)],
        "goals": [_jsonable(r) for r in _query(goal_sql, params)],
        "artifacts": [_jsonable(r) for r in _query(artifact_sql, params)],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(obj):
    """Recursively convert psycopg dict rows (with datetime, Decimal, etc.) to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    try:
        # datetime, date, Decimal, UUID …
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    except Exception:
        return str(obj)


def _json_response(handler: "Handler", payload, status: int = 200) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_WS_TASK_GRAPHS_RE = re.compile(r"^/api/workspaces/([^/]+)/task-graphs$")
_WS_GRAPH_RE = re.compile(r"^/api/workspaces/([^/]+)/graph$")
_NODE_DETAIL_RE = re.compile(r"^/api/nodes/([^/]+)/detail$")


class Handler(SimpleHTTPRequestHandler):
    """Serve static assets and the diagnostic REST API."""

    def log_message(self, fmt, *args):  # reduce noise
        pass

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.path in {"/", "/index.html"}:
            return str(STATIC_ROOT / "index.html")
        return str(STATIC_ROOT / parsed.path.lstrip("/"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        p = parsed.path
        qs = parse_qs(parsed.query)

        def qs1(key, default=None):
            vals = qs.get(key)
            return vals[0] if vals else default

        try:
            if p == "/api/workspaces":
                _json_response(self, _api_workspaces())
            elif m := _WS_TASK_GRAPHS_RE.match(p):
                _json_response(self, _api_workspace_task_graphs(m.group(1)))
            elif m := _WS_GRAPH_RE.match(p):
                _json_response(self, _api_workspace_graph(m.group(1), qs1("task_graph_id")))
            elif m := _NODE_DETAIL_RE.match(p):
                _json_response(self, _api_node_detail(m.group(1), qs1("run_id")))
            elif p == "/api/search":
                q = qs1("q", "")
                limit = int(qs1("limit", "20"))
                _json_response(self, _api_search(q, qs1("workspace_id"), max(1, min(limit, 100))))
            elif p == "/api/graph":
                self._send_static_graph()
            else:
                super().do_GET()
        except KeyError as exc:
            _json_response(self, {"error": str(exc)}, 404)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, 500)

    def _send_static_graph(self) -> None:
        try:
            payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            _json_response(self, payload)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    host = os.environ.get("GRAPH_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPH_VIEWER_PORT", "3000"))
    db_info = f" db={DATABASE_URL.split('@')[-1]}" if DATABASE_URL else " (no DB — static mode)"
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"LLMASM graph viewer http://{host}:{port}{db_info}")
    server.serve_forever()


if __name__ == "__main__":
    main()
