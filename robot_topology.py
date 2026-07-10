"""Reference geometries for the supported robot topologies.

Shared by format_tracks.py (body-angle Procrustes template) and analyze_modes.py (the
Hessian reference configuration) so the two always use the *same* geometry.

Because the robot is flexible, a data-derived mean shape is not repeatable across
experiments, so we use idealized regular templates selected per topology. Add new
topologies here as needed (dispatch is by node count via `resolve_topology`).

The template radius is a circumradius. It does NOT affect the fitted body angle (Kabsch is
scale-invariant) nor the relaxed-lattice normal modes (eigenvectors/eigenvalues depend only
on bond directions); it only sets the scale for the harmonic-PE cross-check and any
pre-tensioned potential energy.
"""

import numpy as np

# Node-count -> topology name. Extend as new lattices are added.
TOPOLOGY_BY_NODE_COUNT = {
    7: "hub_spoke",   # central hub + 6-node ring (the original robot)
    6: "ring",        # 6-node hexagonal ring, no center node
}


def resolve_topology(topology, n_nodes):
    """Map 'auto'/None to a topology name by node count; otherwise pass the name through."""
    if topology and topology != "auto":
        return topology
    return TOPOLOGY_BY_NODE_COUNT.get(n_nodes)


def reference_template(nodes, connections, topology="auto", radius=1.0):
    """Return (template, resolved_topology).

    template = {node_id: (x, y)} idealized reference positions, or None if the topology is
    unknown (caller then falls back to a mean shape). resolved_topology is the name used.

    - hub_spoke: max-degree node at the origin; the remaining nodes evenly on a circle of the
      given radius, in ascending id order.
    - ring: all nodes evenly on a circle of the given radius, in ascending id order.
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
        ring = sorted(nodes)
        m = len(ring)
        template = {n: (radius * np.cos(2 * np.pi * k / m),
                        radius * np.sin(2 * np.pi * k / m))
                    for k, n in enumerate(ring)}
        return template, topo

    return None, topo
