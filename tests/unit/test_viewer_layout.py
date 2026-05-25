from __future__ import annotations

import json
from pathlib import Path


def test_seeded_viewer_graph_layout_has_no_overlapping_nodes() -> None:
    graph = json.loads(Path("docker/graph-viewer/data/graph.json").read_text())
    nodes = graph["nodes"]
    edges = graph["edges"]
    by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        edge["sourceNode"] = by_id[edge["source"]]
        edge["targetNode"] = by_id[edge["target"]]
    apply_layered_layout(nodes, edges)

    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            assert not overlaps(left, right), f"{left['label']} overlaps {right['label']}"


def apply_layered_layout(nodes: list[dict], edges: list[dict]) -> None:
    layers: dict[int, list[dict]] = {}
    for node in nodes:
        node["layer"] = compute_layer(node["id"], edges)
        node["width"] = measure_node_width(node)
        node["height"] = 64
        layers.setdefault(node["layer"], []).append(node)

    sorted_layers = sorted(layers)
    layer_widths = {
        layer: max(node["width"] for node in layers[layer])
        for layer in sorted_layers
    }
    gap_x = 96
    gap_y = 36
    x = 80
    for layer in sorted_layers:
        column = sorted(layers[layer], key=lambda node: node["label"])
        max_width = layer_widths[layer]
        total_height = sum(node["height"] for node in column) + max(0, len(column) - 1) * gap_y
        y = max(80, 260 - total_height / 2)
        for node in column:
            node["x"] = x + (max_width - node["width"]) / 2
            node["y"] = y
            y += node["height"] + gap_y
        x += max_width + gap_x


def compute_layer(node_id: str, edges: list[dict]) -> int:
    layer = 0
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge["target"] == node_id:
                next_layer = compute_layer(edge["source"], edges) + 1
                if next_layer > layer:
                    layer = next_layer
                    changed = True
    return layer


def measure_node_width(node: dict) -> int:
    label_length = len(str(node.get("label") or ""))
    subtitle_length = len(f"{node.get('kind')} / {node.get('status')}")
    return max(176, min(280, max(label_length, subtitle_length) * 8 + 34))


def overlaps(left: dict, right: dict) -> bool:
    padding = 12
    return not (
        left["x"] + left["width"] + padding <= right["x"]
        or right["x"] + right["width"] + padding <= left["x"]
        or left["y"] + left["height"] + padding <= right["y"]
        or right["y"] + right["height"] + padding <= left["y"]
    )
