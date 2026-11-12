"""
Data-center topology: the single source of truth for component IDs, 3D positions,
links, service placement and ground-truth fault impact.

IDs defined here MUST match the object names used by Members 2 and 3 (graph, API, Unity).
All distances are in metres (Unity: 1 unit = 1 m).
"""
import json
from pathlib import Path

import networkx as nx

# ---- Real-world dimensions (metres) ----------------------------------------
RACK_W, RACK_D, RACK_H = 0.60, 1.07, 2.00   # standard 42U rack
U = 0.04445                                  # 1U height
RACK_GAP = 1.20                              # aisle between racks
RACK_X = {"rack-1": 0.0, "rack-2": RACK_W + RACK_GAP}

GATEWAY = "wan"  # external node; "down" = loses connectivity to it


def slot_y(u: int) -> float:
    """Height (m) of rack unit `u` (1-based) above the floor."""
    return round(0.10 + (u - 1) * U, 4)


def _node(nid, ntype, rack=None, slot=None, role=""):
    if rack is None:  # external node, placed outside the racks
        pos = {"x": -1.5, "y": 1.0, "z": 0.0}
    else:
        pos = {"x": RACK_X[rack], "y": slot_y(slot), "z": 0.0}
    return {"id": nid, "type": ntype, "rack": rack, "slot": slot, "role": role, "pos": pos}


NODES = [
    _node("wan", "external", role="gateway"),
    _node("sw-core", "switch", "rack-1", 38, "core"),
    _node("sw-tor-1", "switch", "rack-1", 40, "tor"),
    _node("sw-tor-2", "switch", "rack-2", 40, "tor"),
    _node("srv-1-1", "server", "rack-1", 10), _node("srv-1-2", "server", "rack-1", 12),
    _node("srv-1-3", "server", "rack-1", 14), _node("srv-2-1", "server", "rack-2", 10),
    _node("srv-2-2", "server", "rack-2", 12), _node("srv-2-3", "server", "rack-2", 14),
]

# (link id, endpoint a, endpoint b, monitored?)
LINKS = [
    ("lnk-wan-core", "wan", "sw-core", False),
    ("lnk-core-tor1", "sw-core", "sw-tor-1", True),
    ("lnk-core-tor2", "sw-core", "sw-tor-2", True),
    ("lnk-tor1-tor2", "sw-tor-1", "sw-tor-2", True),   # redundant cross-link
    ("lnk-srv-1-1", "sw-tor-1", "srv-1-1", True), ("lnk-srv-1-2", "sw-tor-1", "srv-1-2", True),
    ("lnk-srv-1-3", "sw-tor-1", "srv-1-3", True), ("lnk-srv-2-1", "sw-tor-2", "srv-2-1", True),
    ("lnk-srv-2-2", "sw-tor-2", "srv-2-2", True), ("lnk-srv-2-3", "sw-tor-2", "srv-2-3", True),
]

# service -> servers hosting it
SERVICES = {
    "svc-web": ["srv-1-1", "srv-2-1"],
    "svc-db": ["srv-1-2"],
    "svc-cache": ["srv-2-2"],
    "svc-batch": ["srv-1-3", "srv-2-3"],
}


def build_graph() -> nx.Graph:
    g = nx.Graph()
    for n in NODES:
        g.add_node(n["id"], **{k: n[k] for k in ("type", "rack", "slot", "role")})
    for lid, a, b, monitored in LINKS:
        g.add_edge(a, b, id=lid, monitored=monitored)
    return g


def monitored_components() -> list:
    """Components that produce telemetry: servers, switches and monitored links."""
    ids = [n["id"] for n in NODES if n["type"] in ("server", "switch")]
    ids += [lid for lid, _, _, monitored in LINKS if monitored]
    return ids


def component_type(cid: str) -> str:
    if cid.startswith("srv-"):
        return "server"
    if cid.startswith("sw-"):
        return "switch"
    if cid.startswith("lnk-"):
        return "link"
    if cid.startswith("svc-"):
        return "service"
    raise KeyError(cid)


def compute_impact(source_id: str) -> dict:
    """
    Ground-truth impact IF `source_id` fails (node or link removed from the graph).
      "down"     = no longer reachable from the gateway (or its link is unusable)
      "degraded" = still reachable but on a longer / rerouted path
    Services are "down" if all hosts are lost, "degraded" if some are.
    """
    g = build_graph()
    base = nx.single_source_shortest_path_length(g, GATEWAY)
    h = g.copy()
    if source_id in h.nodes:
        h.remove_node(source_id)
    else:
        edge = next((u, v) for u, v, d in g.edges(data=True) if d["id"] == source_id)
        h.remove_edge(*edge)
    after = nx.single_source_shortest_path_length(h, GATEWAY)

    impact = {}
    for n, d in g.nodes(data=True):
        if n == source_id or d["type"] == "external":
            continue
        if n not in after:
            impact[n] = "down"
        elif after[n] > base[n]:
            impact[n] = "degraded"

    # links: down if an endpoint is gone/down; degraded if on a rerouted path
    rerouted = set()
    for n, lvl in list(impact.items()):
        if lvl == "degraded":
            path = nx.shortest_path(h, n, GATEWAY)
            rerouted.update(h[u][v]["id"] for u, v in zip(path, path[1:]))
    down_nodes = {n for n, l in impact.items() if l == "down"}
    for u, v, d in g.edges(data=True):
        lid = d["id"]
        if lid == source_id or not d["monitored"]:
            continue
        if {u, v} & (down_nodes | {source_id}):
            impact[lid] = "down"
        elif lid in rerouted:
            impact[lid] = "degraded"

    for svc, hosts in SERVICES.items():
        lost = [x for x in hosts if x == source_id or impact.get(x) == "down"]
        if len(lost) == len(hosts):
            impact[svc] = "down"
        elif lost:
            impact[svc] = "degraded"
    return impact


def export_json(path="data/topology.json") -> None:
    """Share this file with Members 2 and 3 (GET /topology and Unity scene layout)."""
    out = {
        "units": "metres",
        "rack": {"width": RACK_W, "depth": RACK_D, "height": RACK_H},
        "nodes": NODES,
        "links": [{"id": i, "a": a, "b": b, "monitored": m} for i, a, b, m in LINKS],
        "services": SERVICES,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    export_json()
    print("components:", monitored_components())
    for src in ("srv-1-2", "sw-tor-1", "lnk-core-tor1", "sw-core"):
        print(f"\nimpact if {src} fails:", compute_impact(src))
