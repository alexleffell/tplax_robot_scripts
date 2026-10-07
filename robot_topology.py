"""Reference geometries for the supported robot topologies.

Shared by format_tracks.py (body-angle Procrustes template) and analyze_modes.py (the
Hessian reference configuration) so the two always use the *same* geometry.

Because the robot is flexible, a data-derived mean shape is not repeatable across
experiments, so we use idealized regular templates selected per topology. Add new
topologies here as needed (dispatch is by node count via `resolve_topology`).

The template radius is a circumradius, set from the measured rest node-to-node distance
(DEFAULT_REST_SPACING, 153 mm for the 6-node ring; overridable with --baseline). It does NOT
affect the fitted body angle (Kabsch is scale-invariant) nor the relaxed central-force mode
shapes; it sets the spring rest lengths (spring PE), and with bending (κ > 0) the ratio
κ/(k R²), i.e. the eigenvalues.
"""

import numpy as np

# Node-count -> topology name. Extend as new lattices are added.
TOPOLOGY_BY_NODE_COUNT = {
    7: "hub_spoke",   # central hub + 6-node ring (the original robot)
    6: "ring",        # 6-node hexagonal ring, no center node
}


# Measured rest node-to-node (spring) distance per topology, in metres. This is the default
# "baseline" for every stage (format_tracks body-angle template, analyze_modes reference
# geometry, spring rest lengths, bending reference angles). Override with --baseline.
DEFAULT_REST_SPACING = {
    "ring": 0.153,    # 6-node hexagonal ring: node-to-node distance at rest = 153 mm
}


def default_rest_spacing(topology, n_nodes):
    """Measured rest node-to-node distance (m) for the topology, or None if unknown."""
    return DEFAULT_REST_SPACING.get(resolve_topology(topology, n_nodes))


def circumradius_from_spacing(spacing, topology, n_nodes):
    """Template circumradius for a given node-to-node (bond) length.
    ring: side of a regular N-gon → R = s / (2 sin(π/N))  (R = s for a hexagon);
    hub_spoke: the spoke is the radius → R = s."""
    topo = resolve_topology(topology, n_nodes)
    if topo == "ring":
        return spacing / (2 * np.sin(np.pi / n_nodes))
    return spacing


def resolve_topology(topology, n_nodes):
    """Map 'auto'/None to a topology name by node count; otherwise pass the name through."""
    if topology and topology != "auto":
        return topology
    return TOPOLOGY_BY_NODE_COUNT.get(n_nodes)


def ring_cycle(nodes, connections):
    """Node ids in cycle order if the connections form one simple cycle through every node,
    starting at the smallest id and stepping to its smaller-id neighbour (so the default
    [(1,2),...,(6,1)] gives 1,2,...,6). Otherwise None."""
    nodes = list(nodes)
    adj = {n: [] for n in nodes}
    for a, b in connections:
        if a not in adj or b not in adj:
            return None
        adj[a].append(b)
        adj[b].append(a)
    if len(nodes) < 3 or any(len(v) != 2 for v in adj.values()):
        return None
    start = min(nodes)
    order, prev, cur = [start], None, start
    nxt = min(adj[start])
    while nxt != start:
        order.append(nxt)
        prev, cur = cur, nxt
        a, b = adj[cur]
        nxt = b if a == prev else a
        if len(order) > len(nodes):
            return None
    return order if len(order) == len(nodes) else None


def ring_direction(xy, cycle_idx):
    """Rotational sense of the observed ring in DATA coordinates.

    xy: (T, N, 2) node positions; cycle_idx: column indices of the nodes in connection-cycle
    order. Per frame, the polar angle of each node about the centroid is taken in cycle order;
    the frame is 'monotonic' if every step to the next node has the same sign (+ = counter-
    clockwise in the data's x-y axes, − = clockwise). Returns a dict:
      direction       +1 / −1 (majority of monotonic frames; +1 if none)
      frac_ccw, frac_cw, frac_monotonic   fractions of valid frames
      bad_step_frac   per cycle step (k → k+1): fraction of frames whose step disagrees with
                      `direction` (large values point at swapped / mislabelled nodes)
      n_frames        frames with all nodes finite
    Note: camera coordinates have y pointing down, so 'counter-clockwise in the data' is
    clockwise as seen in the image."""
    P = np.asarray(xy, dtype=float)[:, list(cycle_idx), :]
    ok = np.isfinite(P).all(axis=(1, 2))
    P = P[ok]
    n = len(P)
    if n == 0:
        return dict(direction=1, frac_ccw=np.nan, frac_cw=np.nan, frac_monotonic=np.nan,
                    bad_step_frac=np.full(len(cycle_idx), np.nan), n_frames=0)
    c = P.mean(axis=1, keepdims=True)
    a = np.arctan2(P[..., 1] - c[..., 1], P[..., 0] - c[..., 0])
    steps = (np.roll(a, -1, axis=1) - a + np.pi) % (2 * np.pi) - np.pi
    ccw, cw = (steps > 0).all(axis=1), (steps < 0).all(axis=1)
    direction = 1 if ccw.sum() >= cw.sum() else -1
    return dict(direction=direction, frac_ccw=float(ccw.mean()), frac_cw=float(cw.mean()),
                frac_monotonic=float((ccw | cw).mean()),
                bad_step_frac=(np.sign(steps) != direction).mean(axis=0), n_frames=n)


def infer_ring_order(xy, node_ids, max_frames=3000):
    """Physical cyclic order of the nodes, from their polar angles about the centroid.

    xy: (T, N, 2) positions with columns in `node_ids` order. Each (sub-sampled) frame gives a
    cyclic order; orders are compared as undirected edge sets, so direction and starting node do
    not matter. Returns (cycle, support): `cycle` = node ids in cycle order (starting at the
    smallest id, stepping to its smaller-id neighbour, as ring_cycle does), `support` = fraction
    of frames whose order is that cycle. (None, 0.0) if no frame has every node."""
    node_ids = list(node_ids)
    P = np.asarray(xy, dtype=float)
    ok = np.isfinite(P).all(axis=(1, 2))
    P = P[ok]
    if len(P) == 0:
        return None, 0.0
    P = P[np.linspace(0, len(P) - 1, min(max_frames, len(P))).astype(int)]
    c = P.mean(axis=1, keepdims=True)
    ang = np.arctan2(P[..., 1] - c[..., 1], P[..., 0] - c[..., 0])
    n = len(node_ids)
    counts = {}
    for a in ang:
        o = [node_ids[i] for i in np.argsort(a)]
        key = frozenset(frozenset((o[i], o[(i + 1) % n])) for i in range(n))
        counts[key] = counts.get(key, 0) + 1
    best = max(counts, key=counts.get)
    edges = [tuple(sorted(e)) for e in best]
    return ring_cycle(node_ids, edges), counts[best] / len(ang)


def report_ring_direction(info, cycle_nodes, log, min_monotonic=0.95):
    """Log the detected ring direction and warn if the node order is not cleanly monotonic."""
    if info["n_frames"] == 0:
        log("WARNING: ring direction could not be determined (no frame with all nodes); "
            "assuming counter-clockwise.")
        return
    sense = "counter-clockwise" if info["direction"] > 0 else "clockwise"
    log(f"Ring direction: nodes {cycle_nodes} run {sense} in the data's x-y axes "
        f"({100 * info['frac_ccw']:.1f}% of frames CCW, {100 * info['frac_cw']:.1f}% CW, "
        f"{info['n_frames']} frames).")
    if info["frac_monotonic"] < min_monotonic:
        bad = [(f"{cycle_nodes[k]}->{cycle_nodes[(k + 1) % len(cycle_nodes)]}", f)
               for k, f in enumerate(info["bad_step_frac"]) if f > 0.05]
        log(f"WARNING: node order is monotonic around the ring in only "
            f"{100 * info['frac_monotonic']:.1f}% of frames (< {100 * min_monotonic:.0f}%). "
            "Either the connections do not follow the physical ring order (swapped or "
            "mislabelled tags) or the ring folds strongly. Steps that most often go the wrong "
            "way: " + (", ".join(f"{s} ({100 * f:.0f}%)" for s, f in bad) or "none > 5%"))
    minority = min(info["frac_ccw"], info["frac_cw"])
    if minority > 0.01:
        log(f"WARNING: the ring appears in BOTH senses ({100 * minority:.1f}% of frames in the "
            "minority sense); check for tracking swaps between nodes.")


def reference_template(nodes, connections, topology="auto", radius=1.0, direction=1):
    """Return (template, resolved_topology).

    template = {node_id: (x, y)} idealized reference positions, or None if the topology is
    unknown (caller then falls back to a mean shape). resolved_topology is the name used.

    - hub_spoke: max-degree node at the origin; the remaining nodes evenly on a circle of the
      given radius, in ascending id order.
    - ring: all nodes evenly on a circle of the given radius in the order of the connection
      cycle (ascending id order if the connections do not form one cycle), counter-clockwise
      for direction=+1 or clockwise for direction=−1, so the template has the same rotational
      sense as the observed ring (see ring_direction). A mirror-image template cannot be
      aligned by any rotation.
    """
    nodes = list(nodes)
    topo = resolve_topology(topology, len(nodes))

    if topo == "hub_spoke":
        degree = {n: 0 for n in nodes}
        for a, b in connections:
            degree[a] += 1
            degree[b] += 1
        hub = min(nodes, key=lambda n: (-degree[n], n))
        ring = [n for n in nodes if n != hub]
        template = {hub: (0.0, 0.0)}
        m = len(ring)
        for k, n in enumerate(ring):
            template[n] = (radius * np.cos(2 * np.pi * k / m),
                           radius * np.sin(2 * np.pi * k / m))
        return template, topo

    if topo == "ring":
        ring = ring_cycle(nodes, connections) or sorted(nodes)
        m = len(ring)
        sgn = 1.0 if direction >= 0 else -1.0
        template = {n: (radius * np.cos(sgn * 2 * np.pi * k / m),
                        radius * np.sin(sgn * 2 * np.pi * k / m))
                    for k, n in enumerate(ring)}
        return template, topo

    return None, topo
