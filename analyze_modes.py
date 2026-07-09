#!/usr/bin/env python3
"""
Modal / energetic analysis of a formatted robot track.

Consumes the wide per-frame CSV from format_tracks.py and computes:
  - center-of-mass trajectory;
  - elastic normal modes of the spring lattice (unit-mass spring Hessian at a
    regular-polygon reference config);
  - node-velocity projection onto the normal modes (rigid-body motion removed
    for the deformation spectrum; retained for the zero-mode ratio);
  - caster-angle projection onto (a) the network graph-Laplacian modes and
    (b) the elastic normal modes (via the polarity embedding p=(cos,sin));
  - spring potential energy (per spring and total), kinetic energy, total energy;
  - ratio of modal velocity energy in the zero (rigid-body) modes to the total;
  - ratio of caster-angle energy in the zero modes to all modes (both bases);
  - orientation (polar) order parameter of the caster field;
  - caster-angle rotational diffusion coefficient;
  - 2D histogram/PDF of the CoM in the arena;
  - PSDs of the modal energies and the order parameter.

All results are stored in a single ``.npz`` bundle; a companion ``.txt`` summary
lists the parameters, sanity-check residuals, and scalar results.

Conventions
-----------
- Unit mass: KE = 0.5 * sum |v|^2.
- "Zero modes" = the 3 rigid-body modes (2 translations + 1 rotation). Their KE
  fraction measures how rigid-body-like the motion is; the deformation spectrum
  is computed after removing rigid-body motion.

Example
-------
    python analyze_modes.py ../Data/020525/020525_clipped_2_robot.csv --k 1.0 --l0 0.1
"""

import argparse
import ast
import hashlib
import os
from io import StringIO

import numpy as np
import pandas as pd
from scipy.signal import welch, savgol_filter

DEFAULT_MODES_DIR = "/Users/alexleffell/Documents/PhD/tplax/tplax_paper"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_csv_comments(path):
    metadata, data_lines = {}, []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                key, value = line[2:].strip().split(":", 1)
                key, value = key.strip(), value.strip()
                try:
                    metadata[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    metadata[key] = value
            else:
                data_lines.append(line)
    df = pd.read_csv(StringIO("".join(data_lines)))
    df.attrs = metadata
    return df


# --------------------------------------------------------------------------- #
# Reference configuration (regular-polygon template)
# --------------------------------------------------------------------------- #
def build_reference(nodes, connections, baseline, X, Y, log):
    """Regular-polygon reference positions {node: (x, y)} in template coordinates.

    Hub (max-degree node) at the origin; ring nodes on a circle of radius `baseline`
    in ascending id order. Falls back to the time-averaged shape for non-star graphs.
    X, Y are (T, N) arrays whose columns follow `nodes` order.
    """
    col = {n: i for i, n in enumerate(nodes)}
    degree = {n: 0 for n in nodes}
    for a, b in connections:
        degree[a] += 1
        degree[b] += 1
    hub = min(nodes, key=lambda n: (-degree[n], n))
    ring = [n for n in nodes if n != hub]
    conn_set = {frozenset(c) for c in connections}
    is_star = all(frozenset((hub, n)) in conn_set for n in ring)

    if not is_star:
        log("WARNING: non-hub-and-spoke topology; using time-averaged shape as the "
            "reference configuration.")
        return {n: (float(np.nanmean(X[:, col[n]])), float(np.nanmean(Y[:, col[n]]))) for n in nodes}

    r = baseline
    if r is None:
        dists = [np.nanmean(np.hypot(X[:, col[n]] - X[:, col[hub]],
                                     Y[:, col[n]] - Y[:, col[hub]])) for n in ring]
        r = float(np.nanmean(dists))
        log(f"Derived baseline (mean hub->ring distance): {r:.5f}")
    ref = {hub: (0.0, 0.0)}
    m = len(ring)
    for k, n in enumerate(ring):
        ref[n] = (r * np.cos(2 * np.pi * k / m), r * np.sin(2 * np.pi * k / m))
    log(f"Reference: hub={hub}, ring={ring}, radius={r:.5f}")
    return ref


# --------------------------------------------------------------------------- #
# Spring-network Hessian (2N x 2N), unit mass
# --------------------------------------------------------------------------- #
def build_hessian(nodes, connections, ref, k, l0):
    """Small-oscillation stiffness matrix for central-force springs.

    Each bond contributes k * (n n^T) longitudinally plus (t/L)(I - n n^T)
    transversely, where t = k (L - l0) is the equilibrium tension and L the
    equilibrium bond length. With l0 == L the bond is relaxed (longitudinal only).
    """
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    K = np.zeros((2 * N, 2 * N))
    for a, b in connections:
        ia, ib = idx[a], idx[b]
        pa, pb = np.array(ref[a]), np.array(ref[b])
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        n = d / L
        nn = np.outer(n, n)
        rest = L if l0 is None else l0     # l0 is None -> relaxed bond (zero tension)
        t = k * (L - rest)
        kb = k * nn + (t / L) * (np.eye(2) - nn)
        for (p, q, s) in [(ia, ia, +1), (ib, ib, +1), (ia, ib, -1), (ib, ia, -1)]:
            K[2 * p:2 * p + 2, 2 * q:2 * q + 2] += s * kb
    K = 0.5 * (K + K.T)  # symmetrize against round-off
    return K


def graph_laplacian(nodes, connections):
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N))
    for a, b in connections:
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
    L = np.diag(A.sum(axis=1)) - A
    return 0.5 * (L + L.T)


# --------------------------------------------------------------------------- #
# Normal-mode cache (fixed ordering/sign per lattice, shared across experiments)
# --------------------------------------------------------------------------- #
def _canon_connections(connections):
    return sorted(tuple(sorted(c)) for c in connections)


def lattice_signature(nodes, connections, k, l0, radius):
    """Stable hash of the lattice. Eigenvectors of a relaxed regular lattice are
    radius-independent, so radius enters only when a rest length (tension) is set."""
    parts = {
        "nodes": list(nodes),
        "connections": _canon_connections(connections),
        "k": round(float(k), 8),
        "l0": None if l0 is None else round(float(l0), 8),
    }
    if l0 is not None:
        parts["radius"] = round(float(radius), 4)
    return hashlib.md5(repr(parts).encode()).hexdigest()[:12]


def load_or_build_modes(K, nodes, connections, k, l0, radius, ref_arr, modes_dir, recompute, log):
    """Return (eigenvalues, eigenvectors, source). Cached in modes_dir keyed by lattice."""
    os.makedirs(modes_dir, exist_ok=True)
    sig = lattice_signature(nodes, connections, k, l0, radius)
    path = os.path.join(modes_dir, f"modes_{sig}.npz")
    canon = _canon_connections(connections)

    if os.path.exists(path) and not recompute:
        cache = np.load(path, allow_pickle=True)
        match = (list(cache["nodes"]) == list(nodes)
                 and [tuple(x) for x in cache["connections_canon"].tolist()] == canon
                 and cache["eigenvectors"].shape == (2 * len(nodes), 2 * len(nodes)))
        if match:
            log(f"Loaded cached normal modes: {path}")
            return cache["eigenvalues"], cache["eigenvectors"], "cache"
        log(f"WARNING: mode cache at {path} does not match this lattice; recomputing.")

    evals, evecs = np.linalg.eigh(K)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    # Canonical sign: make the largest-magnitude component of each vector positive.
    for j in range(evecs.shape[1]):
        p = int(np.argmax(np.abs(evecs[:, j])))
        if evecs[p, j] < 0:
            evecs[:, j] *= -1
    np.savez(path, nodes=np.array(nodes),
             connections_canon=np.array(canon),
             eigenvalues=evals, eigenvectors=evecs,
             k=k, l0=(np.nan if l0 is None else l0), radius=radius, ref=ref_arr)
    log(f"Computed and cached normal modes -> {path}")
    return evals, evecs, "computed"


# --------------------------------------------------------------------------- #
# Rigid-body removal
# --------------------------------------------------------------------------- #
def remove_rigid(pos, vel):
    """Return (vel_def, v_cm, omega): velocities with CoM translation and best-fit
    rigid rotation removed. pos, vel: (T, N, 2)."""
    com = pos.mean(axis=1, keepdims=True)
    r = pos - com
    v_cm = vel.mean(axis=1, keepdims=True)
    rel = vel - v_cm
    cross = r[..., 0] * rel[..., 1] - r[..., 1] * rel[..., 0]      # (T, N)
    inertia = (r ** 2).sum(axis=(1, 2))                            # (T,)
    omega = cross.sum(axis=1) / np.where(inertia > 0, inertia, np.nan)
    vrot = np.stack([-omega[:, None] * r[..., 1], omega[:, None] * r[..., 0]], axis=-1)
    vel_def = vel - v_cm - vrot
    return vel_def, v_cm[:, 0, :], omega


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Diffusion
# --------------------------------------------------------------------------- #
def angular_msd_diffusion(theta_t, dt, max_lag_frac=0.25, fit_frac=0.5):
    """Rotational diffusion coefficient from the unwrapped-angle MSD.

    theta_t: (T,) one caster's angle time series. Returns (D_r, lags, msd)."""
    u = np.unwrap(theta_t)
    T = len(u)
    max_lag = max(2, int(T * max_lag_frac))
    lags = np.arange(1, max_lag)
    msd = np.array([np.mean((u[lag:] - u[:-lag]) ** 2) for lag in lags])
    t = lags * dt
    nfit = max(2, int(len(lags) * fit_frac))
    slope = np.polyfit(t[:nfit], msd[:nfit], 1)[0]
    return slope / 2.0, t, msd


def persistence_time(acf, dt):
    """Integral correlation time: integral of C(tau) from 0 to its first zero crossing.

    Far more stable than a 1/e-crossing threshold, which is fragile when C(tau) hovers
    near 1/e (it can jump by many seconds for tiny changes in the curve)."""
    zc = np.where(acf < 0)[0]
    end = int(zc[0]) if len(zc) else len(acf)
    if end < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", np.trapz)   # np.trapz deprecated in NumPy 2.0
    return float(trapezoid(acf[:end], dx=dt))


def hub_and_ring(nodes, connections):
    """Return (hub, ring_ids_in_ascending_order) for a hub-and-spoke lattice, else (None, None)."""
    degree = {n: 0 for n in nodes}
    for a, b in connections:
        degree[a] += 1
        degree[b] += 1
    hub = min(nodes, key=lambda n: (-degree[n], n))
    ring = [n for n in nodes if n != hub]
    conn_set = {frozenset(c) for c in connections}
    if all(frozenset((hub, n)) in conn_set for n in ring):
        return hub, ring
    return None, None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Modal/energetic analysis of formatted robot tracks")
    p.add_argument("robot_csv", type=str, help="Formatted CSV from format_tracks.py")
    p.add_argument("--k", type=float, default=1.0, help="Spring constant (uniform). Default: 1.0")
    p.add_argument("--l0", type=float, default=None,
                   help="Spring rest length (uniform). Default: each bond's equilibrium length "
                        "(relaxed network, no pre-tension).")
    p.add_argument("--baseline", type=float, default=None,
                   help="Template radius. Default: from CSV header, else derived from data.")
    p.add_argument("--angle-frame", choices=["lab", "body"], default="lab",
                   help="Caster-angle frame for order parameter, projections, and diffusion: "
                        "'lab' uses {n}_theta (default), 'body' uses {n}_angle (rigid rotation removed).")
    p.add_argument("--nematic", action="store_true",
                   help="Use nematic order parameter |<e^{2i theta}>| instead of polar (default).")
    p.add_argument("--modes-dir", type=str, default=DEFAULT_MODES_DIR,
                   help=f"Directory for the shared normal-mode cache. Default: {DEFAULT_MODES_DIR}")
    p.add_argument("--recompute-modes", action="store_true",
                   help="Recompute and overwrite cached normal modes for this lattice.")
    p.add_argument("--vel-smooth-window", type=int, default=0,
                   help="Savitzky-Golay window (frames, odd) for the velocity estimate: velocity is "
                        "the SG analytic derivative (deriv=1) over this window. 0 = off (plain "
                        "central difference). Reduces the noise floor that inflates deformation KE.")
    p.add_argument("--bins", type=int, default=50, help="Bins per axis for the CoM 2D histogram. Default: 50")
    p.add_argument("--zero-mode-tol", type=float, default=1e-6,
                   help="Eigenvalue tolerance for reporting how many modes are numerically zero.")
    p.add_argument("--output", type=str, default=None, help="Output .npz. Default: <robot>_analysis.npz")
    p.add_argument("--summary", type=str, default=None, help="Summary .txt. Default: <robot>_analysis.txt")
    return p.parse_args()


def main():
    args = parse_args()
    core = args.robot_csv[:-4]
    if core.endswith("_robot"):
        core = core[:-6]
    out_npz = args.output or (core + "_analysis.npz")
    summary_path = args.summary or (core + "_analysis.txt")

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    df = read_csv_comments(args.robot_csv)
    meta = df.attrs
    fps = float(meta.get("fps", 30))
    dt = 1.0 / fps
    connections = [tuple(c) for c in meta["connections"]]
    nodes = list(meta.get("nodes", sorted({n for c in connections for n in c})))
    baseline = args.baseline if args.baseline is not None else meta.get("baseline", None)
    N = len(nodes)
    T = len(df)

    log("=" * 64)
    log("Modal / energetic analysis")
    log("=" * 64)
    log(f"Input        : {args.robot_csv}")
    log(f"Frames       : {T}   fps: {fps}   dt: {dt:.5f}")
    log(f"Nodes (N)    : {N} -> {nodes}")
    log(f"Connections  : {len(connections)} springs")
    log(f"Frame        : {meta.get('frame', '?')}")
    log(f"k, l0, base  : k={args.k}, l0={args.l0}, baseline={baseline}")

    # Node position/angle arrays (T, N), interpolate any residual gaps in time.
    def col(nm):
        return pd.to_numeric(df[nm], errors="coerce").interpolate(limit_direction="both").to_numpy()

    X = np.stack([col(f"{n}_x") for n in nodes], axis=1)   # (T, N)
    Y = np.stack([col(f"{n}_y") for n in nodes], axis=1)
    angle_col = "theta" if args.angle_frame == "lab" else "angle"
    TH = np.stack([col(f"{n}_{angle_col}") for n in nodes], axis=1)
    log(f"Caster-angle frame: {args.angle_frame} (column {{n}}_{angle_col}); "
        f"heading source: {meta.get('heading_source', 'tag')}")
    # A node with no heading at all (e.g. absent from a sensor-only run) is all-NaN after
    # interpolation; zero it so the linear-algebra steps don't propagate NaN, and warn.
    allnan = [nodes[j] for j in range(N) if not np.isfinite(TH[:, j]).any()]
    if allnan:
        log(f"WARNING: no heading for node(s) {allnan}; their angle set to 0 for the analysis.")
        TH = np.nan_to_num(TH, nan=0.0)
    pos = np.stack([X, Y], axis=-1)                        # (T, N, 2)

    com = np.stack([df["centroid_x"].to_numpy(), df["centroid_y"].to_numpy()], axis=-1)

    # Velocities, unit mass. With --vel-smooth-window, use a Savitzky-Golay analytic
    # derivative (fit a local polynomial, evaluate its derivative) -- the standard smooth
    # differentiator for noisy uniformly-sampled data; it suppresses the per-frame
    # tracking-noise floor that otherwise inflates deformation KE at low speed. Otherwise
    # fall back to a plain 2nd-order central difference (np.gradient).
    if args.vel_smooth_window and args.vel_smooth_window > 1 and T >= 5:
        w = int(args.vel_smooth_window)
        if w % 2 == 0:
            w += 1
        w = min(w, (T // 2) * 2 - 1)      # keep odd and < T
        poly = min(3, w - 1)
        vel = savgol_filter(pos, w, poly, deriv=1, delta=dt, axis=0)   # (T, N, 2)
        log(f"Velocity: Savitzky-Golay derivative (window={w}, poly={poly}).")
    else:
        vel = np.gradient(pos, dt, axis=0)                 # 2nd-order central difference
    vel_def, v_cm, omega = remove_rigid(pos, vel)

    # --- Elastic normal modes ------------------------------------------- #
    ref = build_reference(nodes, connections, baseline, X, Y, log)
    ref_arr = np.array([ref[n] for n in nodes])
    radius = float(np.mean(np.linalg.norm(ref_arr - ref_arr.mean(axis=0), axis=1)))
    l0 = args.l0
    # l0 is None -> each bond relaxed at its own equilibrium length (no pre-tension).
    K = build_hessian(nodes, connections, ref, args.k, l0)
    evals, evecs, modes_source = load_or_build_modes(
        K, nodes, connections, args.k, l0, radius, ref_arr,
        args.modes_dir, args.recompute_modes, log)
    n_zero_numeric = int(np.sum(np.abs(evals) < args.zero_mode_tol))
    rigid_idx = np.arange(3)                               # 2 translations + 1 rotation
    deform_idx = np.arange(3, 2 * N)
    log(f"Elastic spectrum: {2 * N} modes ({modes_source}); "
        f"{n_zero_numeric} within |lambda|<{args.zero_mode_tol:g}")
    log(f"  lowest eigenvalues: {np.array2string(evals[:min(6, 2*N)], precision=4)}")

    # Flatten velocities to (T, 2N) in [x0,y0,x1,y1,...] order.
    # Modal projection of node velocities is done with rigid-body motion removed
    # (deformation spectrum). The zero-mode (rigid-body) KE is kept for the ratio
    # via the orthogonal split |v|^2 = |v_rigid|^2 + |v_def|^2.
    V = vel.reshape(T, 2 * N)
    Vdef = vel_def.reshape(T, 2 * N)
    A_def = Vdef @ evecs                                   # (T, 2N) rigid-removed modal amplitudes
    modal_KE = 0.5 * A_def ** 2                            # (T, 2N) deformation modal energies
    KE_total = 0.5 * (V ** 2).sum(axis=1)                  # (T,)
    KE_deform = 0.5 * (Vdef ** 2).sum(axis=1)
    KE_zero = KE_total - KE_deform                         # rigid-body kinetic energy
    zero_mode_KE_ratio = np.divide(KE_zero, KE_total, out=np.zeros_like(KE_zero),
                                   where=KE_total > 0)

    # --- Spring potential energy ---------------------------------------- #
    idx = {n: i for i, n in enumerate(nodes)}
    spring_len = np.zeros((T, len(connections)))
    spring_PE = np.zeros((T, len(connections)))
    for s, (a, b) in enumerate(connections):
        dx = X[:, idx[b]] - X[:, idx[a]]
        dy = Y[:, idx[b]] - Y[:, idx[a]]
        L = np.hypot(dx, dy)
        rest = l0 if l0 is not None else np.linalg.norm(np.array(ref[b]) - np.array(ref[a]))
        spring_len[:, s] = L
        spring_PE[:, s] = 0.5 * args.k * (L - rest) ** 2
    PE_total = spring_PE.sum(axis=1)
    E_total = KE_total + PE_total

    # --- Modal displacement amplitudes (body-frame deformation) --------- #
    # Align the reference (centroid + body-angle rotation) to each frame, then the
    # residual q = actual - aligned_reference is the pure deformation; project onto modes.
    ba = pd.to_numeric(df["body_angle"], errors="coerce").interpolate(limit_direction="both").to_numpy()
    cba, sba = np.cos(ba), np.sin(ba)
    rx = cba[:, None] * ref_arr[:, 0][None, :] - sba[:, None] * ref_arr[:, 1][None, :]
    ry = sba[:, None] * ref_arr[:, 0][None, :] + cba[:, None] * ref_arr[:, 1][None, :]
    aligned_ref = np.stack([rx + com[:, 0:1], ry + com[:, 1:2]], axis=-1)   # (T, N, 2)
    q = (pos - aligned_ref).reshape(T, 2 * N)
    Q = q @ evecs                                          # (T, 2N) modal displacement amplitudes

    # --- Caster-angle projections --------------------------------------- #
    # (a) graph-Laplacian modes (scalar field).
    Lg = graph_laplacian(nodes, connections)
    lg_evals, lg_evecs = np.linalg.eigh(Lg)
    lg_order = np.argsort(lg_evals)
    lg_evals, lg_evecs = lg_evals[lg_order], lg_evecs[:, lg_order]
    lap_zero_idx = np.where(np.abs(lg_evals) < 1e-9)[0]     # uniform / disconnected-component modes
    B = TH @ lg_evecs                                       # (T, N) projections of the angle field
    ang_norm2 = (TH ** 2).sum(axis=1)
    lap_zero_ratio = np.divide((B[:, lap_zero_idx] ** 2).sum(axis=1), ang_norm2,
                               out=np.zeros(T), where=ang_norm2 > 0)

    # (b) elastic modes via polarity embedding p=(cos,sin).
    P = np.zeros((T, 2 * N))
    P[:, 0::2] = np.cos(TH)
    P[:, 1::2] = np.sin(TH)
    C = P @ evecs                                           # (T, 2N)
    P_norm2 = (P ** 2).sum(axis=1)                          # == N (unit polarity)
    elastic_zero_ratio = np.divide((C[:, rigid_idx] ** 2).sum(axis=1), P_norm2,
                                   out=np.zeros(T), where=P_norm2 > 0)

    # --- Orientation order parameter ------------------------------------ #
    mult = 2 if args.nematic else 1
    order_param = np.abs(np.mean(np.exp(1j * mult * TH), axis=1))

    # --- Caster-angle diffusion ----------------------------------------- #
    D_list, msd_stack = [], []
    for i in range(N):
        D_i, lags_t, msd_i = angular_msd_diffusion(TH[:, i], dt)
        D_list.append(D_i)
        msd_stack.append(msd_i)
    D_per_node = np.array(D_list)
    D_mean = float(np.mean(D_per_node))
    msd_stack = np.array(msd_stack)

    # --- CoM 2D histogram / PDF ----------------------------------------- #
    finite = np.isfinite(com[:, 0]) & np.isfinite(com[:, 1])
    hist, xedges, yedges = np.histogram2d(com[finite, 0], com[finite, 1],
                                          bins=args.bins, density=True)

    # ===================================================================== #
    # Active-solid diagnostics
    # ===================================================================== #
    Vnorm = np.linalg.norm(V, axis=1)
    Vdefnorm = np.linalg.norm(Vdef, axis=1)
    Pnorm = np.sqrt(P_norm2)

    def cos_sim(a, b, na, nb):
        denom = na * nb
        return np.divide((a * b).sum(axis=1), denom, out=np.zeros(T), where=denom > 0)

    # (1) Polarity <-> velocity coupling (collective-actuation order parameter).
    coupling_pv = cos_sim(P, V, Pnorm, Vnorm)                  # polarity . full velocity
    coupling_pvdef = cos_sim(P, Vdef, Pnorm, Vdefnorm)         # polarity . deformation velocity
    # Per-mode actuation: temporal correlation between the polarity projection C_i and the
    # velocity projection A_def_i (are polarity and motion condensing on the SAME mode?).
    actuation_spectrum = np.zeros(2 * N)
    for i in range(2 * N):
        if np.std(C[:, i]) > 1e-12 and np.std(A_def[:, i]) > 1e-12:
            actuation_spectrum[i] = np.corrcoef(C[:, i], A_def[:, i])[0, 1]
    actuation_coproj = (np.abs(C) * np.abs(A_def)).mean(axis=0)   # (2N,) co-projection magnitude

    # (2) Participation ratio / spectral entropy of the deformation modal energy.
    Etot_modal = modal_KE.sum(axis=1)
    pmode = np.divide(modal_KE, Etot_modal[:, None], out=np.zeros_like(modal_KE),
                      where=Etot_modal[:, None] > 0)
    participation_ratio = np.divide(1.0, (pmode ** 2).sum(axis=1),
                                    out=np.full(T, np.nan), where=Etot_modal > 0)
    # Entropy with the 0*log0 = 0 convention; take the log only on positive entries.
    plogp = np.zeros_like(pmode)
    pos = pmode > 0
    plogp[pos] = pmode[pos] * np.log(pmode[pos])
    spectral_entropy = -plogp.sum(axis=1)

    # (3) Dominant deformation modes (by mean modal displacement variance) for phase portraits.
    disp_var = (Q[:, deform_idx] ** 2).mean(axis=0)
    dominant_modes = deform_idx[np.argsort(disp_var)[::-1][:2]]
    first_two_nonzero = deform_idx[:2]                         # explicit mode-4/mode-5 request

    # (4) Equipartition / per-mode effective temperature (unit mass).
    Teff_kin = 2.0 * modal_KE.mean(axis=0)                    # flat across modes == equipartition
    Teff_pot = evals * (Q ** 2).mean(axis=0)                  # lambda_i <Q_i^2>

    # (5) Chirality / net rotation.
    mean_omega = float(np.nanmean(omega))
    std_omega = float(np.nanstd(omega))
    Zpol = np.mean(np.exp(1j * TH), axis=1)
    pol_angle = np.unwrap(np.angle(Zpol))
    time = df["time"].to_numpy() if "time" in df.columns else np.arange(T) * dt
    pol_rot_rate = float(np.polyfit(time, pol_angle, 1)[0]) if T > 2 else np.nan
    qa, qb = Q[:, dominant_modes[0]], Q[:, dominant_modes[1]]
    qa_dot, qb_dot = np.gradient(qa, dt), np.gradient(qb, dt)
    orbit_chirality = float(np.mean(qa * qb_dot - qb * qa_dot))   # signed orbit area rate

    # (6) Orientational autocorrelation, persistence time, velocity autocorrelation.
    max_lag = max(2, int(T * 0.25))
    corr_lags = np.arange(max_lag)
    orient_acf = np.array([1.0 if lag == 0 else np.mean(np.cos(TH[lag:] - TH[:-lag]))
                           for lag in corr_lags])
    tau_persist = persistence_time(orient_acf, dt)
    vv0 = np.mean(np.sum(vel * vel, axis=2))
    vacf = np.array([1.0 if lag == 0 else np.mean(np.sum(vel[lag:] * vel[:-lag], axis=2)) / vv0
                     for lag in corr_lags])
    corr_time = corr_lags * dt

    # (7) Net active force vs CoM motion.
    active_force = np.stack([np.cos(TH).sum(axis=1), np.sin(TH).sum(axis=1)], axis=-1)  # (T, 2)
    v_com = np.gradient(com, dt, axis=0)
    afn, vcn = np.linalg.norm(active_force, axis=1), np.linalg.norm(v_com, axis=1)
    force_vel_cos = np.divide((active_force * v_com).sum(axis=1), afn * vcn,
                              out=np.zeros(T), where=(afn * vcn) > 0)
    mean_force_vel_alignment = float(np.nanmean(force_vel_cos))

    # (8) Spatial polarity structure: bond alignment and ring winding number.
    bond_align = np.zeros(T)
    for (a, b) in connections:
        bond_align += np.cos(TH[:, idx[a]] - TH[:, idx[b]])
    bond_align /= len(connections)
    hub, ring = hub_and_ring(nodes, connections)
    if ring is not None and len(ring) >= 3:
        ring_cols = [idx[n] for n in ring]
        ring_th = TH[:, ring_cols]
        dth = np.diff(np.concatenate([ring_th, ring_th[:, :1]], axis=1), axis=1)
        winding = ((dth + np.pi) % (2 * np.pi) - np.pi).sum(axis=1) / (2 * np.pi)
    else:
        winding = np.full(T, np.nan)

    # --- PSDs ----------------------------------------------------------- #
    nper = min(256, T)

    def psd(sig):
        f, pxx = welch(sig - np.nanmean(sig), fs=fps, nperseg=nper)
        return f, pxx

    psd_freq, psd_order = psd(order_param)
    _, psd_KE = psd(KE_total)
    _, psd_PE = psd(PE_total)
    modal_energy_psd = np.array([psd(modal_KE[:, i])[1] for i in range(2 * N)])  # (2N, nf)

    # --- Per-node angular velocity: PSD + pairwise correlation ---------- #
    # Angular velocity of each node's caster heading (in the selected angle frame).
    node_omega = np.gradient(np.unwrap(TH, axis=0), dt, axis=0)          # (T, N)
    node_omega_psd = np.array([psd(node_omega[:, i])[1] for i in range(N)])  # (N, nf)

    # Pairwise node velocity correlation: time-mean normalized dot product of the
    # (full) 2D node velocities. Cij in [-1, 1], diagonal 1.
    vel_corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            vi, vj = vel[:, i, :], vel[:, j, :]
            m = np.isfinite(vi).all(axis=1) & np.isfinite(vj).all(axis=1)
            if m.sum() < 2:
                continue
            num = np.mean(np.sum(vi[m] * vj[m], axis=1))
            di = np.mean(np.sum(vi[m] * vi[m], axis=1))
            dj = np.mean(np.sum(vj[m] * vj[m], axis=1))
            if di > 0 and dj > 0:
                vel_corr[i, j] = num / np.sqrt(di * dj)

    # Pairwise angular-velocity correlation between node headings (Pearson).
    omega_corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            a, b = node_omega[:, i], node_omega[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 2 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
                continue
            omega_corr[i, j] = np.corrcoef(a[m], b[m])[0, 1]

    # --- Sanity checks -------------------------------------------------- #
    log("\n" + "=" * 64)
    log("Sanity checks (max abs residual over frames)")
    log("=" * 64)
    res_vel = np.max(np.abs(modal_KE.sum(axis=1) - KE_deform))
    log(f"[velocity]  sum(modal KE) == KE_deform        : {res_vel:.3e}")
    Vrigid = V - Vdef
    res_orth = np.max(np.abs((Vrigid * Vdef).sum(axis=1)))
    log(f"[velocity]  v_rigid . v_def == 0 (orthogonal) : {res_orth:.3e}")
    res_split = np.max(np.abs((KE_zero + KE_deform) - KE_total))
    log(f"[velocity]  KE_zero + KE_deform == KE_total   : {res_split:.3e}")
    res_lap = np.max(np.abs((B ** 2).sum(axis=1) - ang_norm2))
    log(f"[angle-Lap] sum(proj^2)   == |theta|^2        : {res_lap:.3e}")
    res_ela = np.max(np.abs((C ** 2).sum(axis=1) - P_norm2))
    log(f"[angle-ela] sum(proj^2)   == |p|^2            : {res_ela:.3e}")
    res_etot = np.max(np.abs((KE_total + PE_total) - E_total))
    log(f"[energy]    KE + PE == E_total                : {res_etot:.3e}")
    # Harmonic PE approximation (small-deformation cross-check), using the body-frame
    # modal displacement amplitudes Q computed above.
    PE_harm = 0.5 * (evals[None, :] * Q ** 2).sum(axis=1)
    denom = np.where(PE_total > 1e-12, PE_total, np.nan)
    pe_rel = np.nanmedian(np.abs(PE_harm - PE_total) / denom)
    log(f"[energy]    harmonic PE vs spring PE (median rel, small-deformation): {pe_rel:.3e}")
    # Active-solid diagnostic bounds.
    log(f"[active]    participation ratio in [1,{2 * N}]        : "
        f"min {np.nanmin(participation_ratio):.3f}, max {np.nanmax(participation_ratio):.3f}")
    log(f"[active]    max |polarity-velocity coupling| <= 1: {np.nanmax(np.abs(coupling_pv)):.4f}")

    log("\n" + "=" * 64)
    log("Scalar results")
    log("=" * 64)
    log(f"Mean zero-mode (rigid) KE fraction     : {np.nanmean(zero_mode_KE_ratio):.4f}")
    log(f"Mean caster zero-mode fraction (Lap)   : {np.nanmean(lap_zero_ratio):.4f}")
    log(f"Mean caster zero-mode fraction (elastic): {np.nanmean(elastic_zero_ratio):.4f}")
    log(f"Mean orientation order parameter        : {np.nanmean(order_param):.4f} "
        f"({'nematic' if args.nematic else 'polar'})")
    log(f"Caster diffusion D_r (mean over nodes)  : {D_mean:.5e}  [rad^2/time]")
    log(f"  per-node D_r: {np.array2string(D_per_node, precision=4)}")
    log("")
    log(f"Mean polarity-velocity coupling         : {np.nanmean(coupling_pv):+.4f} (full), "
        f"{np.nanmean(coupling_pvdef):+.4f} (deformation)")
    log(f"Mean participation ratio / entropy      : {np.nanmean(participation_ratio):.3f} / "
        f"{np.nanmean(spectral_entropy):.3f}   (of {len(deform_idx)} deformation modes)")
    log(f"Dominant deformation modes              : {dominant_modes.tolist()}")
    log(f"First two non-zero modes (phase portrait): {first_two_nonzero.tolist()}")
    log(f"Mean body angular velocity <omega>      : {mean_omega:+.5e} +/- {std_omega:.3e}")
    log(f"Net polarization rotation rate          : {pol_rot_rate:+.5e}  [rad/time]")
    log(f"Dominant-pair orbit chirality (signed)  : {orbit_chirality:+.5e}")
    log(f"Orientational integral corr. time tau   : {tau_persist:.4f}  [time]")
    log(f"Mean absolute internal (deformation) KE : {np.nanmean(KE_deform):.5e}")
    log(f"Mean absolute rigid-body KE             : {np.nanmean(KE_zero):.5e}")
    log(f"Mean bond alignment <cos dtheta>        : {np.nanmean(bond_align):+.4f}")
    log(f"Mean ring winding number                : {np.nanmean(winding):+.4f}")
    log(f"Mean active-force / CoM-velocity align  : {mean_force_vel_alignment:+.4f}")

    # --- Save ----------------------------------------------------------- #
    np.savez(
        out_npz,
        time=time, fps=fps, nodes=np.array(nodes), connections=np.array(connections),
        # geometry / modes
        ref=ref_arr, hessian=K, eigenvalues=evals, eigenvectors=evecs,
        rigid_idx=rigid_idx, deform_idx=deform_idx,
        laplacian=Lg, lap_eigenvalues=lg_evals, lap_eigenvectors=lg_evecs, lap_zero_idx=lap_zero_idx,
        # trajectories / kinematics
        com=com, pos=pos, vel=vel, vel_def=vel_def, v_cm=v_cm, omega=omega,
        # modal projections (rigid-body motion removed)
        modal_amp=A_def, modal_KE=modal_KE, modal_disp=Q,
        caster_lap_proj=B, caster_elastic_proj=C,
        angle_frame=args.angle_frame, modes_source=modes_source,
        heading_source=str(meta.get("heading_source", "tag")),
        # active-solid diagnostics
        coupling_pv=coupling_pv, coupling_pvdef=coupling_pvdef,
        actuation_spectrum=actuation_spectrum, actuation_coproj=actuation_coproj,
        participation_ratio=participation_ratio, spectral_entropy=spectral_entropy,
        dominant_modes=dominant_modes, first_two_nonzero=first_two_nonzero,
        Teff_kin=Teff_kin, Teff_pot=Teff_pot,
        mean_omega=mean_omega, std_omega=std_omega,
        pol_angle=pol_angle, pol_rot_rate=pol_rot_rate, orbit_chirality=orbit_chirality,
        corr_time=corr_time, orient_acf=orient_acf, vacf=vacf, tau_persist=tau_persist,
        active_force=active_force, v_com=v_com, force_vel_cos=force_vel_cos,
        mean_force_vel_alignment=mean_force_vel_alignment,
        bond_align=bond_align, winding=winding,
        # energies
        spring_len=spring_len, spring_PE=spring_PE,
        KE_total=KE_total, PE_total=PE_total, E_total=E_total,
        KE_zero=KE_zero, KE_deform=KE_deform,
        # ratios / order / diffusion
        zero_mode_KE_ratio=zero_mode_KE_ratio,
        lap_zero_ratio=lap_zero_ratio, elastic_zero_ratio=elastic_zero_ratio,
        order_param=order_param, is_nematic=args.nematic,
        D_per_node=D_per_node, D_mean=D_mean, ang_msd_lags=lags_t, ang_msd=msd_stack,
        # histogram
        com_hist=hist, com_xedges=xedges, com_yedges=yedges,
        # PSDs
        psd_freq=psd_freq, psd_order=psd_order, psd_KE=psd_KE, psd_PE=psd_PE,
        modal_energy_psd=modal_energy_psd,
        node_omega_psd=node_omega_psd, vel_corr=vel_corr, omega_corr=omega_corr,
        # params
        k=args.k, l0=(l0 if l0 is not None else np.nan), baseline=(baseline if baseline else np.nan),
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\nSaved analysis bundle -> {out_npz}")
    log(f"Saved summary         -> {summary_path}")


if __name__ == "__main__":
    main()
