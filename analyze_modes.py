#!/usr/bin/env python3
"""
Modal / kinematic analysis of a formatted ring-robot track (format_tracks.py output).

Model: N nodes on a regular ring, relaxed central-force springs (no pre-stress) plus an
optional harmonic bond-bending stiffness --kappa about the ideal interior angle. Unit mass.

Frames. Normal modes live in the BODY frame of the template (node k at angle 2πk/N). Every
modal projection therefore uses body-frame vectors: per frame the best-fit rotation β(t)
(closed-form 2-D Kabsch, template → observed) is removed from
  - displacements   q_body = R(−β)(x − x_cm) − x_ref            → Q   (modal_disp)
  - velocities      v_body = R(−β)(v − rigid part)               → A   (modal_amp, modal_KE)
  - polarities      p_body = (cos(γ−β), sin(γ−β))                → C   (caster_elastic_proj)
Lab-frame quantities (order parameter, kymographs, CoM, winding, correlations) are unchanged.

Symmetry-adapted modes and the strain-wave order parameter. The ring is invariant under the
one-node rotation S ((S q)_{k+1} = R(2π/N) q_k), which commutes with the Hessian. Within every
degenerate eigenvalue band the eigenvectors are re-based (real Schur form of S restricted to
the band) into SECTORS:
  - 2-D sectors: a pair (u1, u2) on which S acts as a rotation by φ = 2πm/N, 0 < φ < π, with
    the pair oriented so a pattern travelling counter-clockwise (in the body frame) makes the
    complex amplitude z = Q_u1 + i Q_u2 turn counter-clockwise. m is the angular wavenumber.
  - 1-D sectors: S = +1 (m = 0, e.g. breathing) or −1 (m = N/2).
Sectors are basis-independent (unlike individual modes in a degenerate band) and do not depend
on k or κ for their identity. For each 2-D sector j:
  share_j(t)  = |z_j|² / Σ_deform Q²          fraction of the deformation in sector j
  circ_j(t)   = Im(z̄ ż) / (|z| |ż|)  ∈ [−1,1]  +1 CCW travelling wave, 0 standing / noise
  Λ_j         = ⟨Im(z̄ ż)⟩ / ⟨|z||ż|⟩           time-averaged circulation
  Ω_j         = ⟨Im(z̄ ż)⟩ / ⟨|z|²⟩            mean phase speed (rad/s; pattern turns at Ω_j/m)
  W_j(t)      = share_j · circ_j                strain-wave order parameter (windowed version
                                                 W_win from --wave-window averages)
Heading wave (heading_wave_stats): the same idea for the caster headings. The body-frame heading
field around the ring is split into twist numbers q (DFT of e^{iγ_k}); q = 0 is flocking, q = ±1
the vortex counted by the winding number. H = share_q* · Λ_q* is ±1 for a clean travelling twist
(the stripes in the heading kymograph) and 0 for flocking, disorder or a frozen twist.

A strain-wave limit cycle (a travelling shear wave = rotation inside one degenerate shear pair)
gives |W| → 1; a standing oscillation or a spread over sectors gives W → 0.
Sector condensation (participation ratio over sectors by displacement share) is
λ-independent and replaces the per-mode participation ratio as the condensation measure.

Other outputs: CoM trajectory/PDF; rigid vs deformation KE; spring + bending PE; polar order
parameter with its finite-N random-heading baseline; heading MSD after removing each node's
mean spin (rotational diffusion about the deterministic rotation); autocorrelations;
polarity/velocity diagnostics; bond alignment; ring winding; kymographs; PSDs; correlations.

Notes on interpretation
  - Unit mass and k = 1 are arbitrary: KE and PE are in different units and are not summed.
  - With no-slip wheels the node velocity is slaved to the heading, so polarity–velocity
    coupling and the per-mode actuation correlation are close to 1 by kinematics; departures
    measure caster swing (l·γ̇) and slip, not elastic selection.

Example
-------
    python analyze_modes.py ../Data/090726/chiral_1_trim_robot.csv --kappa 0.005 --vel-smooth-window 7
"""

import argparse
import ast
import hashlib
import os
from io import StringIO

import numpy as np
import pandas as pd
from scipy.linalg import schur
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch, savgol_filter

from robot_topology import (circumradius_from_spacing, default_rest_spacing,
                            reference_template, report_ring_direction, resolve_topology,
                            ring_cycle, ring_direction)

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


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_series(s):
    """Linear interpolation of residual gaps (constant extrapolation at the ends)."""
    return pd.to_numeric(s, errors="coerce").interpolate(limit_direction="both").to_numpy()


def interp_angle(s):
    """Gap interpolation of a wrapped angle: unwrap the finite samples, interpolate, re-wrap."""
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return a
    out = np.full_like(a, np.nan)
    out[ok] = np.unwrap(a[ok])
    out = pd.Series(out).interpolate(limit_direction="both").to_numpy()
    return wrap(out)


# --------------------------------------------------------------------------- #
# Reference configuration
# --------------------------------------------------------------------------- #
def build_reference(nodes, connections, baseline, topology, X, Y, log, direction=1):
    """Idealized reference positions {node: (x, y)} (see robot_topology.py) and the rest node
    spacing used.

    Rest node-to-node distance: --baseline / CSV header if set, else the measured default for
    the topology (0.153 m for the 6-node ring), else (unknown topology only) the mean observed
    spring length. Converted to the template circumradius (equal for a hexagon).
    """
    col = {n: i for i, n in enumerate(nodes)}
    spacing, src = baseline, "--baseline / CSV header"
    if spacing is None:
        spacing, src = default_rest_spacing(topology, len(nodes)), "measured default"
    if spacing is None:
        edge = [np.nanmean(np.hypot(X[:, col[a]] - X[:, col[b]], Y[:, col[a]] - Y[:, col[b]]))
                for a, b in connections]
        edge = [e for e in edge if np.isfinite(e)]
        spacing = float(np.nanmean(edge)) if edge else 1.0
        src = "mean observed spring length (no measured default for this topology)"
    r = circumradius_from_spacing(spacing, topology, len(nodes))
    log(f"Rest node spacing {spacing:.4f} m ({src}) -> template circumradius {r:.4f} m")
    edge_obs = [np.nanmean(np.hypot(X[:, col[a]] - X[:, col[b]], Y[:, col[a]] - Y[:, col[b]]))
                for a, b in connections]
    if np.isfinite(edge_obs).any():
        log(f"  observed mean spring length {np.nanmean(edge_obs):.4f} m "
            f"({100 * (np.nanmean(edge_obs) / spacing - 1):+.1f}% vs rest)")
    template, topo = reference_template(nodes, connections, topology, radius=r,
                                        direction=direction)
    if template is None:
        log(f"WARNING: no built-in template for topology '{topo}' ({len(nodes)} nodes); "
            f"using the time-averaged shape (not repeatable).")
        return {n: (float(np.nanmean(X[:, col[n]])), float(np.nanmean(Y[:, col[n]])))
                for n in nodes}, topo, spacing
    log(f"Reference: topology={topo}, circumradius={r:.5f}")
    return template, topo, spacing


# --------------------------------------------------------------------------- #
# Hessian (2N x 2N), unit mass, relaxed springs + bond bending
# --------------------------------------------------------------------------- #
def _hinges(nodes, connections, ref_arr):
    """Bending hinges (iA, iB, iC): at every node B, angularly adjacent pairs of its bonds
    whose sector is < π (for a ring node this is exactly its interior angle). Straight or
    reflex sectors are skipped: their angle has no well-defined gradient."""
    idx = {n: i for i, n in enumerate(nodes)}
    adj = {i: set() for i in range(len(nodes))}
    for a, b in connections:
        adj[idx[a]].add(idx[b]); adj[idx[b]].add(idx[a])
    hinges = []
    for b, nbrs in adj.items():
        if len(nbrs) < 2:
            continue
        nb = list(nbrs)
        ang = [np.arctan2(*(ref_arr[n] - ref_arr[b])[::-1]) for n in nb]
        order = [nb[i] for i in np.argsort(ang)]
        srt = sorted(ang)
        for i in range(len(order)):
            j = (i + 1) % len(order)
            sector = (srt[j] - srt[i]) % (2 * np.pi)
            if 1e-6 < sector < np.pi - 1e-6:
                hinges.append((order[i], b, order[j]))
    return hinges


def _angle(x, iA, iB, iC):
    """Angle /_ABC (rad, in [0, π]) from flat coords x=[x0,y0,x1,y1,...]."""
    u = x[2 * iA:2 * iA + 2] - x[2 * iB:2 * iB + 2]
    v = x[2 * iC:2 * iC + 2] - x[2 * iB:2 * iB + 2]
    return np.arctan2(abs(u[0] * v[1] - u[1] * v[0]), u[0] * v[0] + u[1] * v[1])


def bending_hessian(ref_arr, hinges, kappa, hstep=1e-6):
    """κ Σ_h ∇θ_h ∇θ_hᵀ: exact Hessian of ½κ Σ (θ_h − θ_h0)² at the reference (θ_h = θ_h0)."""
    two_n = 2 * len(ref_arr)
    x0 = ref_arr.reshape(-1)
    H = np.zeros((two_n, two_n))
    for (iA, iB, iC) in hinges:
        g = np.zeros(two_n)
        for c in (2 * iA, 2 * iA + 1, 2 * iB, 2 * iB + 1, 2 * iC, 2 * iC + 1):
            xp = x0.copy(); xp[c] += hstep
            xm = x0.copy(); xm[c] -= hstep
            g[c] = (_angle(xp, iA, iB, iC) - _angle(xm, iA, iB, iC)) / (2 * hstep)
        H += kappa * np.outer(g, g)
    return H


def build_hessian(nodes, connections, ref, k, kappa=0.0):
    """Stiffness matrix of relaxed central-force springs (k n nᵀ per bond, no tension term)
    plus optional bond bending. The reference is an equilibrium, so this is the exact
    small-deformation operator; rigid-body modes are exact zero modes."""
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    K = np.zeros((2 * N, 2 * N))
    for a, b in connections:
        ia, ib = idx[a], idx[b]
        d = np.array(ref[b]) - np.array(ref[a])
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        n = d / L
        kb = k * np.outer(n, n)
        for (p, q, s) in [(ia, ia, +1), (ib, ib, +1), (ia, ib, -1), (ib, ia, -1)]:
            K[2 * p:2 * p + 2, 2 * q:2 * q + 2] += s * kb
    ref_arr = np.array([ref[n] for n in nodes], dtype=float)
    hinges = _hinges(nodes, connections, ref_arr)
    if kappa and kappa > 0:
        K = K + bending_hessian(ref_arr, hinges, kappa)
    return 0.5 * (K + K.T), hinges


def rigid_body_modes(ref_arr):
    """Orthonormal 2N x 3 basis of the rigid-body subspace at the reference."""
    N = len(ref_arr)
    r = ref_arr - ref_arr.mean(axis=0)
    tx = np.zeros(2 * N); tx[0::2] = 1.0
    ty = np.zeros(2 * N); ty[1::2] = 1.0
    rot = np.zeros(2 * N); rot[0::2] = -r[:, 1]; rot[1::2] = r[:, 0]
    Rb, _ = np.linalg.qr(np.stack([tx, ty, rot], axis=1))
    return Rb[:, :3]


def separate_rigid_mechanism(evals, evecs, ref_arr, tol):
    """Re-base the λ≈0 block into [rigid (3), mechanism (rest)]. Returns
    (evecs, rigid_idx, mechanism_idx, deform_idx)."""
    two_n = evecs.shape[0]
    null_idx = np.where(np.abs(evals) < tol)[0]
    if len(null_idx) < 3:
        raise SystemExit(f"Only {len(null_idx)} modes with |λ| < {tol:g}; a relaxed ring must "
                         "have 3 rigid-body zero modes. Check --zero-mode-tol / the lattice.")
    Rb = rigid_body_modes(ref_arr)
    U0 = evecs[:, null_idx]
    rigid_b, _ = np.linalg.qr(U0 @ (U0.T @ Rb))
    rigid_b = rigid_b[:, :3]
    Mproj = U0 @ U0.T - rigid_b @ rigid_b.T
    w, Vv = np.linalg.eigh(Mproj)
    mech_b = Vv[:, w > 0.5]
    evecs = evecs.copy()
    evecs[:, null_idx] = np.concatenate([rigid_b, mech_b], axis=1)
    rigid_idx = null_idx[:3]
    mechanism_idx = null_idx[3:3 + mech_b.shape[1]]
    deform_idx = np.array([i for i in range(two_n) if i not in set(rigid_idx.tolist())])
    return evecs, rigid_idx, mechanism_idx, deform_idx


def graph_laplacian(nodes, connections):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, b in connections:
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
    return np.diag(A.sum(axis=1)) - A


def group_bands(members, vals, atol=1e-6, rtol=1e-3):
    """Group mode indices with (near-)equal eigenvalues."""
    order = sorted(members, key=lambda i: vals[i])
    groups = [[int(order[0])]]
    for k in order[1:]:
        prev = vals[groups[-1][-1]]
        if abs(vals[k] - prev) <= atol + rtol * abs(prev):
            groups[-1].append(int(k))
        else:
            groups.append([int(k)])
    return groups


# --------------------------------------------------------------------------- #
# Ring symmetry: shift operator and symmetry-adapted sectors
# --------------------------------------------------------------------------- #
def ring_shift_operator(ref_arr):
    """S (2N x 2N): rotate the displacement field one node counter-clockwise about the ring
    centre, (S q)_{next(i)} = R(2π/N) q_i, with 'next' = next node by template polar angle."""
    N = len(ref_arr)
    c = ref_arr - ref_arr.mean(axis=0)
    order = np.argsort(np.arctan2(c[:, 1], c[:, 0]) % (2 * np.pi))
    a = 2 * np.pi / N
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    S = np.zeros((2 * N, 2 * N))
    for p in range(N):
        i, j = order[p], order[(p + 1) % N]
        S[2 * j:2 * j + 2, 2 * i:2 * i + 2] = R
    return S


def symmetry_adapt(evecs, bands, S, N, log):
    """Re-base each band into S-sectors (see module docstring). Returns (evecs, sectors) with
    sectors = list of dicts {modes: [i] or [i1, i2], m, dim}."""
    evecs = evecs.copy()
    sectors = []
    for b in bands:
        U = evecs[:, b]
        Sb = U.T @ S @ U
        if np.linalg.norm(Sb.T @ Sb - np.eye(len(b))) > 1e-6:
            log(f"  WARNING: band {b} is not closed under the ring rotation (accidental "
                "near-degeneracy?); its modes are left unadapted.")
            sectors += [dict(modes=[i], m=np.nan, dim=1) for i in b]
            continue
        T, Z = schur(Sb, output="real")
        Un = U @ Z
        p = 0
        while p < len(b):
            if p + 1 < len(b) and abs(T[p + 1, p]) > 1e-8:
                if T[p + 1, p] < 0:                       # orient: S rotates (u1,u2) by +φ
                    Un[:, p + 1] *= -1
                phi = np.arctan2(abs(T[p + 1, p]), T[p, p])
                sectors.append(dict(modes=[b[p], b[p + 1]], m=phi * N / (2 * np.pi), dim=2))
                p += 2
            else:
                sectors.append(dict(modes=[b[p]], m=0.0 if T[p, p] > 0 else N / 2, dim=1))
                p += 1
        evecs[:, b] = Un
    return evecs, sectors


# --------------------------------------------------------------------------- #
# Normal-mode cache (raw eigh output; adaptation is done after loading)
# --------------------------------------------------------------------------- #
def _canon_connections(connections):
    return sorted(tuple(sorted(c)) for c in connections)


def lattice_signature(nodes, connections, k, radius, kappa=0.0, direction=1):
    parts = {"nodes": list(nodes), "connections": _canon_connections(connections),
             "k": round(float(k), 8), "l0": None, "kappa": round(float(kappa), 8)}
    if kappa and kappa > 0:
        parts["radius"] = round(float(radius), 4)
    if direction < 0:
        parts["direction"] = -1                      # CCW keys stay identical to older caches
    return hashlib.md5(repr(parts).encode()).hexdigest()[:12]


def load_or_build_modes(K, nodes, connections, k, radius, ref_arr, modes_dir, recompute, log,
                        kappa=0.0, direction=1):
    os.makedirs(modes_dir, exist_ok=True)
    sig = lattice_signature(nodes, connections, k, radius, kappa, direction)
    path = os.path.join(modes_dir, f"modes_{sig}.npz")
    canon = _canon_connections(connections)
    if os.path.exists(path) and not recompute:
        cache = np.load(path, allow_pickle=True)
        match = (list(cache["nodes"]) == list(nodes)
                 and [tuple(x) for x in cache["connections_canon"].tolist()] == canon
                 and cache["eigenvectors"].shape == (2 * len(nodes), 2 * len(nodes))
                 and ("ref" not in cache.files or np.allclose(
                     cache["ref"] / np.abs(cache["ref"]).max(), ref_arr / np.abs(ref_arr).max(),
                     atol=1e-6)))
        if match:
            log(f"Loaded cached normal modes: {path}")
            return cache["eigenvalues"], cache["eigenvectors"], "cache"
        log(f"WARNING: mode cache at {path} does not match this lattice; recomputing.")
    evals, evecs = np.linalg.eigh(K)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    for j in range(evecs.shape[1]):
        p = int(np.argmax(np.abs(evecs[:, j])))
        if evecs[p, j] < 0:
            evecs[:, j] *= -1
    np.savez(path, nodes=np.array(nodes), connections_canon=np.array(canon),
             eigenvalues=evals, eigenvectors=evecs, k=k, l0=np.nan, radius=radius, ref=ref_arr)
    log(f"Computed and cached normal modes -> {path}")
    return evals, evecs, "computed"


# --------------------------------------------------------------------------- #
# Kinematics helpers
# --------------------------------------------------------------------------- #
def remove_rigid(pos, vel):
    """(vel_def, v_cm, omega): velocities with CoM translation and the least-squares rigid
    rotation about the instantaneous centroid removed. pos, vel: (T, N, 2)."""
    com = pos.mean(axis=1, keepdims=True)
    r = pos - com
    v_cm = vel.mean(axis=1, keepdims=True)
    rel = vel - v_cm
    cross = r[..., 0] * rel[..., 1] - r[..., 1] * rel[..., 0]
    inertia = (r ** 2).sum(axis=(1, 2))
    omega = cross.sum(axis=1) / np.where(inertia > 0, inertia, np.nan)
    vrot = np.stack([-omega[:, None] * r[..., 1], omega[:, None] * r[..., 0]], axis=-1)
    return vel - v_cm - vrot, v_cm[:, 0, :], omega


def kabsch_angle_series(ref_c, rel):
    """Closed-form 2-D Kabsch angle per frame mapping the centred template ref_c (N,2) onto the
    centred observed positions rel (T,N,2)."""
    s = (ref_c[None, :, 0] * rel[..., 1] - ref_c[None, :, 1] * rel[..., 0]).sum(axis=1)
    c = (ref_c[None, :, 0] * rel[..., 0] + ref_c[None, :, 1] * rel[..., 1]).sum(axis=1)
    return np.arctan2(s, c)


def rotate(vecs, ang):
    """Rotate (T, N, 2) vectors by per-frame angle ang (T,)."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return np.stack([c * vecs[..., 0] - s * vecs[..., 1],
                     s * vecs[..., 0] + c * vecs[..., 1]], axis=-1)


def time_derivative(x, dt, window):
    """Savitzky–Golay derivative along axis 0 if window > 1, else central differences."""
    T = x.shape[0]
    if window and window > 1 and T >= 5:
        w = int(window) + (1 - int(window) % 2)
        w = min(w, (T // 2) * 2 - 1)
        return savgol_filter(x, w, min(3, w - 1), deriv=1, delta=dt, axis=0)
    return np.gradient(x, dt, axis=0)


def heading_wave_stats(H, dt, sg_window, win):
    """Twisted-state (heading-wave) decomposition of the caster heading field on the ring.

    H: (T, m) headings, nodes in counter-clockwise ring order k = 0..m−1, BODY frame (rigid
    rotation removed, like the strain wave). For each twist number q (q = 0, ±1, …, from the
    DFT) ψ_q(t) = (1/m) Σ_k e^{iγ_k} e^{−i2πqk/m}; p_q = |ψ_q|² sums to 1 over q (p_0 = polar
    order² = flocking). A q-twist (each heading offset by 2πq/m from its neighbour; q = ±1 is
    the vortex counted by the winding number) whose casters all spin at rate Ω has ψ_q of fixed
    size rotating at Ω, and the heading pattern travels around the ring at −Ω/q (CCW
    positive). Per q: share ⟨p_q⟩, circulation Λ_q = ⟨Im ψ̄ψ̇⟩/⟨|ψ||ψ̇|⟩, spin Ω_q =
    ⟨Im ψ̄ψ̇⟩/⟨|ψ|²⟩, travel speed −Ω_q/q (NaN for q = 0 and the alias-ambiguous q = m/2).
    The dominant twist q* is the q ≠ 0 with the largest share; the heading-wave order parameter
    is H = share_q* · Λ_q* (±1 for a clean travelling twist, 0 for flocking / disorder /
    a frozen twist), with windowed H_win(t) as for the strain wave."""
    T, m = H.shape
    q = np.rint(np.fft.fftfreq(m) * m).astype(int)
    psi = np.fft.fft(np.exp(1j * H), axis=1) / m
    p = np.abs(psi) ** 2
    psid = (time_derivative(psi.real, dt, sg_window)
            + 1j * time_derivative(psi.imag, dt, sg_window))
    L = np.imag(np.conj(psi) * psid)
    den = np.abs(psi) * np.abs(psid)
    share = p.mean(axis=0)
    lam = np.divide(L.mean(axis=0), den.mean(axis=0), out=np.zeros(m), where=den.mean(axis=0) > 0)
    spin = np.divide(L.mean(axis=0), share, out=np.zeros(m), where=share > 0)
    speed = np.full(m, np.nan)
    ok = (q != 0) & (2 * np.abs(q) != m)
    speed[ok] = -spin[ok] / q[ok]
    sm = lambda x: uniform_filter1d(x, size=win, axis=0, mode="nearest")
    sden = sm(den)
    H_w = sm(p) * np.divide(sm(L), sden, out=np.zeros_like(L), where=sden > 0)
    cand = np.flatnonzero(q != 0)
    j = int(cand[np.argmax(share[cand])])
    circ_t = np.divide(L[:, j], den[:, j], out=np.zeros(T), where=den[:, j] > 0)
    return dict(q=q, share=share, circulation=lam, spin=spin, travel_speed=speed, share_t=p,
                q_star=int(q[j]), j_star=j, H=float(share[j] * lam[j]), H_t=p[:, j] * circ_t,
                H_win=H_w[:, j], abs_win=float(np.mean(np.abs(H_w[:, j]))),
                flock_share=float(share[q == 0][0]))


def angular_msd_detrended(theta_t, t):
    """Rotational diffusion about the deterministic spin: unwrap, remove the mean rotation rate
    (least-squares slope), MSD of the residual; D_r = slope/2 over the first half of lags up
    to T/4. Returns (D_r, spin_rate, lags_s, msd)."""
    u = np.unwrap(theta_t)
    spin = float(np.polyfit(t, u, 1)[0]) if len(t) > 2 else 0.0
    ud = u - spin * (t - t[0])
    T = len(ud)
    max_lag = max(3, int(T * 0.25))
    lags = np.arange(1, max_lag)
    msd = np.array([np.mean((ud[lag:] - ud[:-lag]) ** 2) for lag in lags])
    dt = float(np.median(np.diff(t)))
    nfit = max(2, int(len(lags) * 0.5))
    D = float(np.polyfit(lags[:nfit] * dt, msd[:nfit], 1)[0]) / 2.0
    return D, spin, lags * dt, msd


def persistence_time(acf, dt):
    """Integral of C(τ) to its first zero crossing (for spinning headings this is ~1/ω, a
    decorrelation-by-rotation time rather than a persistence time)."""
    zc = np.where(acf < 0)[0]
    end = int(zc[0]) if len(zc) else len(acf)
    if end < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(acf[:end], dx=dt))


def order_param_null(N, n_samples=20000, seed=0):
    """E|⟨e^{iθ}⟩| for N independent uniform headings (finite-N baseline)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, size=(n_samples, N))
    return float(np.abs(np.exp(1j * th).mean(axis=1)).mean())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Modal/kinematic analysis of a ring-robot track")
    p.add_argument("robot_csv", type=str, help="Formatted CSV from format_tracks.py")
    p.add_argument("--k", type=float, default=1.0, help="Spring constant (uniform). Default: 1.0")
    p.add_argument("--kappa", type=float, default=0.0,
                   help="Bond-bending stiffness about the ideal interior angle. Default 0. "
                        "Sector identities (m) do not depend on it; eigenvalues and the "
                        "radial/tangential mix inside a sector do.")
    p.add_argument("--baseline", type=float, default=None,
                   help="Rest node-to-node (spring) distance (m). Default: CSV header, else the "
                        "measured value for the topology (0.153 m for the 6-node ring). Sets the "
                        "template, spring rest lengths and the bending reference.")
    p.add_argument("--topology", type=str, default="auto",
                   help="Reference topology ('auto' = CSV header / node count). The wave "
                        "analysis requires 'ring'.")
    p.add_argument("--angle-frame", choices=["lab", "body"], default="lab",
                   help="Heading frame for the order parameter, kymographs, autocorrelations "
                        "and diffusion. Modal projections always use the body frame.")
    p.add_argument("--nematic", action="store_true",
                   help="Nematic |⟨e^{2iγ}⟩| instead of polar order parameter.")
    p.add_argument("--modes-dir", type=str, default=DEFAULT_MODES_DIR,
                   help=f"Normal-mode cache directory. Default: {DEFAULT_MODES_DIR}")
    p.add_argument("--recompute-modes", action="store_true",
                   help="Recompute and overwrite cached normal modes for this lattice.")
    p.add_argument("--vel-smooth-window", type=int, default=0,
                   help="Savitzky–Golay window (frames, odd) for velocities and modal-amplitude "
                        "derivatives. 0 = central differences.")
    p.add_argument("--wave-window", type=float, default=1.0,
                   help="Averaging window (s) for the windowed strain-wave order parameter "
                        "W_win(t). Default: 1.0")
    p.add_argument("--bins", type=int, default=50, help="Bins per axis for the CoM histogram.")
    p.add_argument("--zero-mode-tol", type=float, default=1e-6,
                   help="|λ| below which a mode is a zero mode. Default: 1e-6")
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
    N, T = len(nodes), len(df)
    idx = {n: i for i, n in enumerate(nodes)}
    time = df["time"].to_numpy() if "time" in df.columns else np.arange(T) * dt

    log("=" * 64)
    log("Modal / kinematic analysis (ring, relaxed springs)")
    log("=" * 64)
    log(f"Input        : {args.robot_csv}")
    log(f"Frames       : {T}   fps: {fps}   dt: {dt:.5f}")
    log(f"Nodes (N)    : {N} -> {nodes}")
    log(f"Connections  : {connections}")
    log(f"Frame        : {meta.get('frame', '?')}")
    log(f"k, kappa     : k={args.k}, kappa={args.kappa}, baseline={baseline}")
    if ring_cycle(nodes, connections) is None:
        log("WARNING: connections do not form a single ring; the wave-sector analysis needs one.")

    X = np.stack([interp_series(df[f"{n}_x"]) for n in nodes], axis=1)
    Y = np.stack([interp_series(df[f"{n}_y"]) for n in nodes], axis=1)
    TH_lab = np.stack([interp_angle(df[f"{n}_theta"]) for n in nodes], axis=1)
    angle_col = "theta" if args.angle_frame == "lab" else "angle"
    TH = TH_lab if angle_col == "theta" else \
        np.stack([interp_angle(df[f"{n}_angle"]) for n in nodes], axis=1)
    log(f"Heading frame for order/kymographs: {args.angle_frame} (column {{n}}_{angle_col}); "
        f"heading source: {meta.get('heading_source', 'tag')}")
    allnan = [nodes[j] for j in range(N) if not np.isfinite(TH_lab[:, j]).any()]
    if allnan:
        log(f"WARNING: no heading for node(s) {allnan}; set to 0 (orientation results biased).")
        TH_lab = np.nan_to_num(TH_lab, nan=0.0)
        TH = np.nan_to_num(TH, nan=0.0)
    pos = np.stack([X, Y], axis=-1)                          # (T, N, 2) lab
    com = pos.mean(axis=1)                                   # (T, 2)

    vel = time_derivative(pos, dt, args.vel_smooth_window)   # (T, N, 2) lab
    if args.vel_smooth_window and args.vel_smooth_window > 1:
        log(f"Velocity: Savitzky–Golay derivative (window={args.vel_smooth_window}).")
    vel_def, v_cm, omega = remove_rigid(pos, vel)

    # --- Reference, Hessian, modes ---------------------------------------- #
    topology = args.topology if args.topology != "auto" else meta.get("topology", "auto")
    ring_dir, ring_info, cyc = 1, None, ring_cycle(nodes, connections)
    if resolve_topology(topology, N) == "ring" and cyc is not None:
        ring_info = ring_direction(pos, [idx[n] for n in cyc])
        report_ring_direction(ring_info, cyc, log)
        ring_dir = ring_info["direction"]
        hdr = meta.get("ring_direction")
        if hdr is not None and int(hdr) != ring_dir:
            log(f"WARNING: ring direction in the CSV header ({int(hdr):+d}) differs from the one "
                f"detected here ({ring_dir:+d}); the body angle in the CSV may be unreliable. "
                "Using the detected direction (analysis recomputes its own body angle).")
    ref, topo, rest_spacing = build_reference(nodes, connections, baseline, topology, X, Y, log,
                                              direction=ring_dir)
    ref_arr = np.array([ref[n] for n in nodes], dtype=float)
    ref_c = ref_arr - ref_arr.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(ref_c, axis=1)))
    K, hinges = build_hessian(nodes, connections, ref, args.k, args.kappa)
    evals, evecs, modes_source = load_or_build_modes(
        K, nodes, connections, args.k, radius, ref_arr, args.modes_dir,
        args.recompute_modes, log, kappa=args.kappa, direction=ring_dir)
    n_zero_numeric = int(np.sum(np.abs(evals) < args.zero_mode_tol))
    evecs, rigid_idx, mechanism_idx, deform_idx = separate_rigid_mechanism(
        evals, evecs, ref_arr, args.zero_mode_tol)
    bands = group_bands(list(deform_idx), evals)
    soft_idx = np.array(bands[0]) if bands else np.array([], dtype=int)

    sectors = []
    if topo == "ring":
        S = ring_shift_operator(ref_arr)
        comm = np.linalg.norm(S @ K - K @ S) / max(np.linalg.norm(K), 1e-12)
        log(f"Ring rotation symmetry: ||SK − KS|| / ||K|| = {comm:.2e}")
        evecs, sectors = symmetry_adapt(evecs, bands, S, N, log)
    else:
        log("Non-ring topology: wave sectors not computed (each mode its own sector).")
        sectors = [dict(modes=[i], m=np.nan, dim=1) for i in deform_idx]
    for s in sectors:
        s["lam"] = float(np.mean(evals[s["modes"]]))
    log(f"Elastic spectrum: {2 * N} modes ({modes_source}); {n_zero_numeric} with "
        f"|λ|<{args.zero_mode_tol:g} -> {len(rigid_idx)} rigid + {len(mechanism_idx)} mechanism.")
    log(f"  eigenvalues: {np.array2string(evals, precision=4)}")
    log("  sectors (non-rigid): " + ", ".join(
        f"[{'/'.join(map(str, s['modes']))}] m={s['m']:.3g} λ={s['lam']:.3g}" for s in sectors))

    # --- Body frame -------------------------------------------------------- #
    beta = kabsch_angle_series(ref_c, pos - com[:, None, :])          # (T,)
    q_body = (rotate(pos - com[:, None, :], -beta) - ref_c[None]).reshape(T, 2 * N)
    vel_body = rotate(vel, -beta)
    vdef_body = rotate(vel_def, -beta).reshape(T, 2 * N)
    th_body = TH_lab - beta[:, None]
    P_body = np.zeros((T, 2 * N))
    P_body[:, 0::2] = np.cos(th_body)
    P_body[:, 1::2] = np.sin(th_body)

    Q = q_body @ evecs                                       # (T, 2N) modal displacements
    A_def = vdef_body @ evecs                                # (T, 2N) modal velocities
    C = P_body @ evecs                                       # (T, 2N) polarity overlaps
    modal_KE = 0.5 * A_def ** 2
    V = vel.reshape(T, 2 * N)
    KE_total = 0.5 * (V ** 2).sum(axis=1)
    KE_deform = 0.5 * (vdef_body ** 2).sum(axis=1)
    KE_zero = KE_total - KE_deform
    zero_mode_KE_ratio = np.divide(KE_zero, KE_total, out=np.zeros(T), where=KE_total > 0)

    # --- Potential energy: springs + bending ------------------------------- #
    spring_len = np.zeros((T, len(connections)))
    spring_PE = np.zeros((T, len(connections)))
    for s_i, (a, b) in enumerate(connections):
        L = np.hypot(X[:, idx[b]] - X[:, idx[a]], Y[:, idx[b]] - Y[:, idx[a]])
        rest = np.linalg.norm(ref_arr[idx[b]] - ref_arr[idx[a]])
        spring_len[:, s_i] = L
        spring_PE[:, s_i] = 0.5 * args.k * (L - rest) ** 2
    PE_spring = spring_PE.sum(axis=1)
    PE_bend = np.zeros(T)
    if args.kappa and args.kappa > 0:
        for (iA, iB, iC) in hinges:
            u, v = pos[:, iA] - pos[:, iB], pos[:, iC] - pos[:, iB]
            th = np.arctan2(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]),
                            (u * v).sum(axis=1))
            th0 = _angle(ref_arr.reshape(-1), iA, iB, iC)
            PE_bend += 0.5 * args.kappa * (th - th0) ** 2
    PE_total = PE_spring + PE_bend

    # --- Caster-angle projections ----------------------------------------- #
    Lg = graph_laplacian(nodes, connections)
    lg_evals, lg_evecs = np.linalg.eigh(Lg)
    lap_zero_idx = np.where(np.abs(lg_evals) < 1e-9)[0]
    Zh = np.exp(1j * TH)                                     # complex heading field
    B = Zh @ lg_evecs                                        # (T, N) complex
    lap_zero_ratio = (np.abs(B[:, lap_zero_idx]) ** 2).sum(axis=1) / N     # = R² (polar)
    P_norm2 = (P_body ** 2).sum(axis=1)
    elastic_zero_ratio = (C[:, soft_idx] ** 2).sum(axis=1) / P_norm2 if len(soft_idx) \
        else np.zeros(T)
    propulsion_ratio = (C[:, rigid_idx] ** 2).sum(axis=1) / P_norm2
    mean_polarity = np.stack([np.nanmean(np.cos(TH), axis=0),
                              np.nanmean(np.sin(TH), axis=0)], axis=1)
    Pc = P_body - P_body.mean(axis=0, keepdims=True)
    _, S_pod, Vt_pod = np.linalg.svd(Pc, full_matrices=False)
    n_pod = min(3, Vt_pod.shape[0])
    pod_polarity = Vt_pod[:n_pod]
    pod_variance = (S_pod[:n_pod] ** 2) / max((S_pod ** 2).sum(), 1e-30)

    # --- Order parameter, diffusion, autocorrelation ---------------------- #
    mult = 2 if args.nematic else 1
    order_param = np.abs(np.mean(np.exp(1j * mult * TH), axis=1))
    op_null = order_param_null(N)
    D_list, spin_list, msd_stack = [], [], []
    for i in range(N):
        D_i, spin_i, lags_t, msd_i = angular_msd_detrended(TH[:, i], time)
        D_list.append(D_i); spin_list.append(spin_i); msd_stack.append(msd_i)
    D_per_node, spin_per_node = np.array(D_list), np.array(spin_list)
    D_mean = float(np.mean(D_per_node))
    msd_stack = np.array(msd_stack)

    finite = np.isfinite(com).all(axis=1)
    hist, xedges, yedges = np.histogram2d(com[finite, 0], com[finite, 1],
                                          bins=args.bins, density=True)

    # ===================================================================== #
    # Strain-wave sectors (body-frame modal displacements)
    # ===================================================================== #
    Qdot = time_derivative(Q, dt, args.vel_smooth_window)
    Qd2 = (Q[:, deform_idx] ** 2).sum(axis=1)                # total deformation |q|²
    rms_def = float(np.sqrt(Qd2.mean() / N) / radius)
    log(f"RMS per-node deformation / ring radius: {rms_def:.3f}")
    if rms_def > 0.5:
        log("WARNING: deformation exceeds half the ring radius on average — the template probably "
            "does not match the robot (node order / ring direction / rest spacing). Modal and "
            "wave results are unreliable.")
    # Sector shares and the wave use deviations from the run's mean shape: a static offset
    # (e.g. the ring sitting a few % smaller than the rest spacing → constant m=0 breathing, or a
    # permanent distortion) is not dynamics and would otherwise dilute the share and W.
    Q_mean = Q.mean(axis=0)
    Qf = Q - Q_mean[None, :]
    Qf2 = (Qf[:, deform_idx] ** 2).sum(axis=1)
    static_frac = float((Q_mean[deform_idx] ** 2).sum() / Qd2.mean()) if Qd2.mean() > 0 else np.nan
    log(f"Static (time-mean) part of the deformation: {100 * static_frac:.1f}% of ⟨|q|²⟩ "
        "(excluded from sector shares and W)")
    win = max(3, int(round(args.wave_window * fps)))

    def smooth(x):
        return uniform_filter1d(x, size=win, axis=0, mode="nearest")

    sec_share_t, sec_dim, sec_m, sec_lam, sec_modes = [], [], [], [], []
    wave_rows = []                                            # per 2-D sector time series
    for s in sectors:
        i = s["modes"]
        share = np.divide((Qf[:, i] ** 2).sum(axis=1), Qf2, out=np.zeros(T), where=Qf2 > 0)
        sec_share_t.append(share)
        sec_dim.append(s["dim"]); sec_m.append(s["m"]); sec_lam.append(s["lam"])
        sec_modes.append(i + [-1] * (2 - len(i)))
        if s["dim"] == 2:
            z = Qf[:, i[0]] + 1j * Qf[:, i[1]]
            zd = Qdot[:, i[0]] + 1j * Qdot[:, i[1]]
            L_t = np.imag(np.conj(z) * zd)                    # angular momentum in the pair
            denom = np.abs(z) * np.abs(zd)
            circ = np.divide(L_t, denom, out=np.zeros(T), where=denom > 0)
            Lam = float(L_t.mean() / denom.mean()) if denom.mean() > 0 else 0.0
            Om = float(L_t.mean() / (np.abs(z) ** 2).mean()) if (np.abs(z) ** 2).mean() > 0 else 0.0
            eta_w = np.divide(smooth((Qf[:, i] ** 2).sum(axis=1)), smooth(Qf2),
                              out=np.zeros(T), where=smooth(Qf2) > 0)
            sd = smooth(denom)
            circ_w = np.divide(smooth(L_t), sd, out=np.zeros(T), where=sd > 0)
            amp = np.abs(z)
            wave_rows.append(dict(sector=len(sec_share_t) - 1, share=share, circ=circ,
                                  W=share * circ, W_win=eta_w * circ_w, Lam=Lam, Om=Om,
                                  amp_cv=float(amp.std() / amp.mean()) if amp.mean() > 0 else np.nan))
    sec_share_t = np.array(sec_share_t).T                     # (T, n_sec)
    sector_Q2 = np.array([(Qf[:, s["modes"]] ** 2).sum(axis=1).mean() for s in sectors])
    sector_share = sector_Q2 / Qf2.mean() if Qf2.mean() > 0 else np.zeros(len(sectors))
    sector_KE = np.array([modal_KE[:, s["modes"]].sum(axis=1).mean() for s in sectors])
    sector_drive = np.array([(C[:, s["modes"]] ** 2).sum(axis=1).mean() for s in sectors])
    psec = sec_share_t
    sector_participation = np.divide(1.0, (psec ** 2).sum(axis=1), out=np.full(T, np.nan),
                                     where=(psec ** 2).sum(axis=1) > 0)
    sector_circulation = np.full(len(sectors), np.nan)
    sector_phase_speed = np.full(len(sectors), np.nan)
    sector_amp_cv = np.full(len(sectors), np.nan)
    for w in wave_rows:
        sector_circulation[w["sector"]] = w["Lam"]
        sector_phase_speed[w["sector"]] = w["Om"]
        sector_amp_cv[w["sector"]] = w["amp_cv"]

    two_d = [w for w in wave_rows]
    if two_d:
        dom = max(two_d, key=lambda w: sector_share[w["sector"]])
        soft2 = min(two_d, key=lambda w: sectors[w["sector"]]["lam"])
        dom_sec = dom["sector"]
        wave_order_t, wave_order_win = dom["W"], dom["W_win"]
        wave_share = float(sector_share[dom_sec])
        wave_circulation = float(dom["Lam"])
        wave_order = wave_share * wave_circulation
        wave_order_abs = float(np.nanmean(np.abs(dom["W_win"])))
        wave_m = float(sectors[dom_sec]["m"])
        dominant_modes = np.array(sectors[dom_sec]["modes"])
        first_two_nonzero = np.array(sectors[soft2["sector"]]["modes"])
        wave_all_win = np.array([w["W_win"] for w in two_d]).T            # (T, n_2d)
        wave_sectors_2d = np.array([w["sector"] for w in two_d])
    else:
        dom_sec, wave_order_t, wave_order_win = -1, np.zeros(T), np.zeros(T)
        wave_share = wave_circulation = wave_order = wave_order_abs = wave_m = np.nan
        disp_var = (Q[:, deform_idx] ** 2).mean(axis=0)
        dominant_modes = deform_idx[np.argsort(disp_var)[::-1][:2]]
        first_two_nonzero = deform_idx[:2]
        wave_all_win = np.zeros((T, 0))
        wave_sectors_2d = np.array([], dtype=int)
    orbit_chirality = wave_circulation

    # Per-mode / per-band condensation (kept for comparison; basis-dependent within bands).
    Kd = modal_KE[:, deform_idx]
    Etot_modal = Kd.sum(axis=1)
    pmode = np.divide(Kd, Etot_modal[:, None], out=np.zeros_like(Kd),
                      where=Etot_modal[:, None] > 0)
    participation_ratio = np.divide(1.0, (pmode ** 2).sum(axis=1), out=np.full(T, np.nan),
                                    where=Etot_modal > 0)
    plogp = np.zeros_like(pmode)
    nz = pmode > 0
    plogp[nz] = pmode[nz] * np.log(pmode[nz])
    spectral_entropy = -plogp.sum(axis=1)
    band_lambdas = np.array([float(np.mean(evals[b])) for b in bands])
    band_sizes = np.array([len(b) for b in bands])
    band_KE = np.array([float(modal_KE[:, b].sum(axis=1).mean()) for b in bands])
    band_Q2 = np.array([float((Q[:, b] ** 2).sum(axis=1).mean()) for b in bands])
    band_drive = np.array([float((C[:, b] ** 2).sum(axis=1).mean()) for b in bands])
    band_energy_t = np.stack([modal_KE[:, b].sum(axis=1) for b in bands], axis=1)
    Eb = band_energy_t.sum(axis=1)
    pb = np.divide(band_energy_t, Eb[:, None], out=np.zeros_like(band_energy_t),
                   where=Eb[:, None] > 0)
    band_participation = np.divide(1.0, (pb ** 2).sum(axis=1), out=np.full(T, np.nan),
                                   where=Eb > 0)

    # ===================================================================== #
    # Polarity / velocity / rotation diagnostics
    # ===================================================================== #
    P_lab = np.zeros((T, 2 * N))
    P_lab[:, 0::2] = np.cos(TH_lab)
    P_lab[:, 1::2] = np.sin(TH_lab)
    Vdef_lab = vel_def.reshape(T, 2 * N)

    def cos_sim(a, b):
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.divide((a * b).sum(axis=1), denom, out=np.zeros(T), where=denom > 0)

    coupling_pv = cos_sim(P_lab, V)
    coupling_pvdef = cos_sim(P_lab, Vdef_lab)
    actuation_spectrum = np.zeros(2 * N)
    for i in range(2 * N):
        if np.std(C[:, i]) > 1e-12 and np.std(A_def[:, i]) > 1e-12:
            actuation_spectrum[i] = np.corrcoef(C[:, i], A_def[:, i])[0, 1]
    actuation_coproj = (np.abs(C) * np.abs(A_def)).mean(axis=0)

    mean_omega, std_omega = float(np.nanmean(omega)), float(np.nanstd(omega))
    Zpol = np.mean(np.exp(1j * TH), axis=1)
    pol_angle = np.unwrap(np.angle(Zpol))
    pol_rot_rate = float(np.polyfit(time, pol_angle, 1)[0]) if T > 2 else np.nan

    max_lag = max(2, int(T * 0.25))
    corr_lags = np.arange(max_lag)
    orient_acf = np.array([1.0 if lag == 0 else np.mean(np.cos(TH[lag:] - TH[:-lag]))
                           for lag in corr_lags])
    tau_persist = persistence_time(orient_acf, dt)
    vv0 = np.mean(np.sum(vel * vel, axis=2))
    vacf = np.array([1.0 if lag == 0 else np.mean(np.sum(vel[lag:] * vel[:-lag], axis=2)) / vv0
                     for lag in corr_lags])
    corr_time = corr_lags * dt

    active_force = np.stack([np.cos(TH_lab).sum(axis=1), np.sin(TH_lab).sum(axis=1)], axis=-1)
    v_com = v_cm
    afn, vcn = np.linalg.norm(active_force, axis=1), np.linalg.norm(v_com, axis=1)
    force_vel_cos = np.divide((active_force * v_com).sum(axis=1), afn * vcn,
                              out=np.zeros(T), where=(afn * vcn) > 0)
    mean_force_vel_alignment = float(np.nanmean(force_vel_cos))

    bond_align = np.zeros(T)
    for (a, b) in connections:
        bond_align += np.cos(TH[:, idx[a]] - TH[:, idx[b]])
    bond_align /= len(connections)

    # Ring order for winding number / kymographs: counter-clockwise in the data's x-y axes,
    # whatever the labelling direction, so their signs mean the same thing in every dataset.
    ring = None
    if topo == "ring" and cyc is not None:
        ring = list(cyc) if ring_dir > 0 else [cyc[0]] + list(cyc[:0:-1])
    if ring is not None:
        rcols = [idx[n] for n in ring]
        m_r = len(ring)
        ring_th = TH[:, rcols]
        dth = np.diff(np.concatenate([ring_th, ring_th[:, :1]], axis=1), axis=1)
        winding = wrap(dth).sum(axis=1) / (2 * np.pi)
        ring_nodes = np.array(ring)
        ring_headings = wrap(ring_th)
        bond_angle_baseline = np.pi * (m_r - 2) / m_r
        bond_angle_dev = np.zeros((T, m_r))
        for kk in range(m_r):
            iB, iA, iC = rcols[kk], rcols[(kk - 1) % m_r], rcols[(kk + 1) % m_r]
            u, v = pos[:, iA] - pos[:, iB], pos[:, iC] - pos[:, iB]
            interior = np.arctan2(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]),
                                  (u * v).sum(axis=1))
            bond_angle_dev[:, kk] = interior - bond_angle_baseline
    # Heading wave (twisted states of the caster field), body frame, CCW ring order.
    hw = None
    if ring is not None:
        ring_headings_body = wrap(th_body[:, rcols])
        hw = heading_wave_stats(ring_headings_body, dt, args.vel_smooth_window,
                                max(3, int(round(args.wave_window * fps))))
        log("Heading twist decomposition (body frame; share p_q, circulation Λ_q, spin Ω_q, "
            "pattern travel speed −Ω_q/q):")
        for qq, sh, la, sp, tv in zip(hw["q"], hw["share"], hw["circulation"], hw["spin"],
                                      hw["travel_speed"]):
            log(f"  q={qq:+d}: share {sh:.3f}  Λ {la:+.3f}  Ω {sp:+.3f} rad/s  "
                f"travel {tv:+.3f} rad/s" + ("  (flocking)" if qq == 0 else ""))
        log(f"HEADING-WAVE ORDER PARAMETER (dominant twist q={hw['q_star']:+d}): "
            f"H = share·Λ = {hw['H']:+.3f};  ⟨|H_win|⟩ = {hw['abs_win']:.3f}")
    else:
        ring_headings_body = np.zeros((T, 0))
    if ring is None:
        winding = np.full(T, np.nan)
        ring_nodes, ring_headings = np.array([]), np.zeros((T, 0))
        bond_angle_dev, bond_angle_baseline = np.zeros((T, 0)), np.nan

    # --- PSDs ----------------------------------------------------------- #
    nper = min(256, T)

    def psd(sig):
        return welch(sig - np.nanmean(sig), fs=fps, nperseg=nper)

    psd_freq, psd_order = psd(order_param)
    _, psd_KE = psd(KE_total)
    _, psd_PE = psd(PE_total)
    modal_amp_psd = np.array([psd(Q[:, i])[1] for i in range(2 * N)])        # amplitude, not energy
    node_omega = time_derivative(np.unwrap(TH, axis=0), dt, args.vel_smooth_window)
    node_omega_psd = np.array([psd(node_omega[:, i])[1] for i in range(N)])

    vel_corr = np.full((N, N), np.nan)
    omega_corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            vi, vj = vel[:, i, :], vel[:, j, :]
            di, dj = np.mean((vi * vi).sum(axis=1)), np.mean((vj * vj).sum(axis=1))
            if di > 0 and dj > 0:
                vel_corr[i, j] = np.mean((vi * vj).sum(axis=1)) / np.sqrt(di * dj)
            a, b = node_omega[:, i], node_omega[:, j]
            if np.std(a) > 0 and np.std(b) > 0:
                omega_corr[i, j] = np.corrcoef(a, b)[0, 1]

    # --- Checks --------------------------------------------------------- #
    log("\n" + "=" * 64)
    log("Checks")
    log("=" * 64)
    PE_harm = 0.5 * (evals[None, :] * Q ** 2).sum(axis=1)
    pe_scale = float(np.mean(PE_total)) if np.mean(PE_total) > 0 else np.nan
    pe_rel = float(np.median(np.abs(PE_harm - PE_total)) / pe_scale)
    log(f"[physics] harmonic PE ½Σλ_iQ_i² vs spring+bending PE (median |Δ| / mean PE): "
        f"{pe_rel:.3e}   (≪ 1 when deformation is in stiff modes; with κ=0 a large-amplitude "
        "mechanism stretches springs at 2nd order, which the harmonic model misses)")
    strain = np.nanmedian(np.abs(spring_len - np.linalg.norm(
        ref_arr[[idx[b] for a, b in connections]] - ref_arr[[idx[a] for a, b in connections]],
        axis=1)[None, :])) / radius
    log(f"[physics] median |spring strain| (rel. to radius): {strain:.3e}")
    rigid_leak = float((A_def[:, rigid_idx] ** 2).sum(axis=1).mean() / max(
        (A_def ** 2).sum(axis=1).mean(), 1e-30))
    log(f"[frame]   deformation KE leaking into rigid modes: {rigid_leak:.3e}   (small; grows with "
        "deformation amplitude, since rigid motion is removed in the current shape)")
    log(f"[algebra] Σ modal KE − KE_deform: {np.max(np.abs(modal_KE.sum(axis=1) - KE_deform)):.2e}")

    log("\n" + "=" * 64)
    log("Scalar results")
    log("=" * 64)
    log(f"Rigid-body KE fraction (mean)          : {np.nanmean(zero_mode_KE_ratio):.4f}")
    log(f"Order parameter (mean) / random null   : {np.nanmean(order_param):.4f} / {op_null:.4f} "
        f"({'nematic' if args.nematic else 'polar'})")
    log(f"Polarity overlap with soft band (mean) : {np.nanmean(elastic_zero_ratio):.4f}  "
        f"(soft band λ={evals[soft_idx].mean() if len(soft_idx) else np.nan:.3g})")
    log(f"Body angular velocity <ω>              : {mean_omega:+.4f} ± {std_omega:.4f} rad/s")
    log(f"Mean heading spin (per node)           : {np.array2string(spin_per_node, precision=3)} rad/s")
    log(f"Rotational diffusion about mean spin   : {D_mean:.4e} rad²/s")
    log(f"Banded participation ratio (mean)      : {np.nanmean(band_participation):.3f} of {len(bands)} bands")
    log(f"Sector participation ratio (mean)      : {np.nanmean(sector_participation):.3f} of {len(sectors)} sectors")
    log("Sectors (displacement share, circulation Λ, phase speed Ω, amplitude CV):")
    for j, s in enumerate(sectors):
        extra = (f"  Λ={sector_circulation[j]:+.3f}  Ω={sector_phase_speed[j]:+.3f} rad/s  "
                 f"CV|z|={sector_amp_cv[j]:.2f}") if s["dim"] == 2 else ""
        log(f"  [{'/'.join(map(str, s['modes']))}] m={s['m']:.3g} λ={s['lam']:.3g}  "
            f"share={sector_share[j]:.3f}{extra}")
    log(f"STRAIN-WAVE ORDER PARAMETER (dominant 2-D sector, m={wave_m:.3g}): "
        f"W = share·Λ = {wave_order:+.3f}   (share {wave_share:.3f}, Λ {wave_circulation:+.3f}); "
        f"⟨|W_win|⟩ = {wave_order_abs:.3f} ({args.wave_window:g} s window)")
    log(f"Bond alignment / ring winding (mean)   : {np.nanmean(bond_align):+.4f} / "
        f"{np.nanmean(winding) if np.isfinite(winding).any() else np.nan:+.4f}")
    log(f"Active-force / CoM-velocity alignment  : {mean_force_vel_alignment:+.4f}")

    np.savez(
        out_npz,
        time=time, fps=fps, nodes=np.array(nodes), connections=np.array(connections),
        ref=ref_arr, hessian=K, eigenvalues=evals, eigenvectors=evecs,
        rigid_idx=rigid_idx, deform_idx=deform_idx, mechanism_idx=mechanism_idx, soft_idx=soft_idx,
        laplacian=Lg, lap_eigenvalues=lg_evals, lap_eigenvectors=lg_evecs, lap_zero_idx=lap_zero_idx,
        com=com, pos=pos, vel=vel, vel_def=vel_def, vel_body=vel_body, v_cm=v_cm, omega=omega,
        body_angle=beta,
        modal_amp=A_def, modal_KE=modal_KE, modal_disp=Q, modal_disp_dot=Qdot,
        caster_lap_proj=B, caster_elastic_proj=C,
        angle_frame=args.angle_frame, modes_source=modes_source,
        heading_source=str(meta.get("heading_source", "tag")),
        # sectors / strain wave
        sector_modes=np.array(sec_modes), sector_dim=np.array(sec_dim), sector_m=np.array(sec_m),
        sector_lambda=np.array(sec_lam), sector_share=sector_share, sector_share_t=sec_share_t,
        sector_KE=sector_KE, sector_drive=sector_drive, sector_circulation=sector_circulation,
        sector_phase_speed=sector_phase_speed, sector_amp_cv=sector_amp_cv,
        sector_participation=sector_participation,
        wave_sectors_2d=wave_sectors_2d, wave_order_win_all=wave_all_win,
        wave_dominant_sector=dom_sec, wave_order_t=wave_order_t, wave_order_win=wave_order_win,
        wave_order=wave_order, wave_order_abs=wave_order_abs, wave_share=wave_share,
        wave_circulation=wave_circulation, wave_m=wave_m, wave_window=args.wave_window,
        # condensation (per mode / per band)
        coupling_pv=coupling_pv, coupling_pvdef=coupling_pvdef,
        actuation_spectrum=actuation_spectrum, actuation_coproj=actuation_coproj,
        participation_ratio=participation_ratio, spectral_entropy=spectral_entropy,
        dominant_modes=dominant_modes, first_two_nonzero=first_two_nonzero,
        band_lambdas=band_lambdas, band_sizes=band_sizes, band_KE=band_KE, band_Q2=band_Q2,
        band_drive=band_drive, band_participation=band_participation,
        # rotation / orientation
        mean_omega=mean_omega, std_omega=std_omega, pol_angle=pol_angle, pol_rot_rate=pol_rot_rate,
        orbit_chirality=orbit_chirality,
        corr_time=corr_time, orient_acf=orient_acf, vacf=vacf, tau_persist=tau_persist,
        active_force=active_force, v_com=v_com, force_vel_cos=force_vel_cos,
        mean_force_vel_alignment=mean_force_vel_alignment,
        bond_align=bond_align, winding=winding,
        ring_nodes=ring_nodes, ring_headings=ring_headings,
        bond_angle_dev=bond_angle_dev, bond_angle_baseline=bond_angle_baseline,
        # energies (KE: unit mass; PE: units of k — not commensurate, not summed)
        spring_len=spring_len, spring_PE=spring_PE, PE_spring=PE_spring, PE_bend=PE_bend,
        KE_total=KE_total, PE_total=PE_total, KE_zero=KE_zero, KE_deform=KE_deform,
        PE_harmonic=PE_harm, pe_check=pe_rel,
        zero_mode_KE_ratio=zero_mode_KE_ratio, lap_zero_ratio=lap_zero_ratio,
        elastic_zero_ratio=elastic_zero_ratio, propulsion_ratio=propulsion_ratio,
        order_param=order_param, order_param_null=op_null, is_nematic=args.nematic,
        D_per_node=D_per_node, D_mean=D_mean, spin_per_node=spin_per_node,
        ang_msd_lags=lags_t, ang_msd=msd_stack,
        com_hist=hist, com_xedges=xedges, com_yedges=yedges,
        psd_freq=psd_freq, psd_order=psd_order, psd_KE=psd_KE, psd_PE=psd_PE,
        modal_amp_psd=modal_amp_psd, node_omega_psd=node_omega_psd,
        vel_corr=vel_corr, omega_corr=omega_corr,
        mean_polarity=mean_polarity, pod_polarity=pod_polarity, pod_variance=pod_variance,
        k=args.k, kappa=args.kappa, baseline=rest_spacing, rest_spacing=rest_spacing,
        ring_headings_body=ring_headings_body,
        heading_q=(hw["q"] if hw else np.array([])),
        heading_share=(hw["share"] if hw else np.array([])),
        heading_circulation=(hw["circulation"] if hw else np.array([])),
        heading_spin=(hw["spin"] if hw else np.array([])),
        heading_travel_speed=(hw["travel_speed"] if hw else np.array([])),
        heading_share_t=(hw["share_t"] if hw else np.zeros((T, 0))),
        heading_wave_q=(hw["q_star"] if hw else 0), heading_wave_H=(hw["H"] if hw else np.nan),
        heading_wave_H_t=(hw["H_t"] if hw else np.zeros(T)),
        heading_wave_H_win=(hw["H_win"] if hw else np.zeros(T)),
        heading_wave_abs=(hw["abs_win"] if hw else np.nan),
        heading_flock_share=(hw["flock_share"] if hw else np.nan),
        ring_direction=ring_dir, rms_deformation=rms_def, static_deformation_frac=static_frac,
        modal_disp_mean=Q_mean,
        ring_monotonic_frac=(ring_info['frac_monotonic'] if ring_info else np.nan),
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nSaved analysis bundle -> {out_npz}")
    log(f"Saved summary         -> {summary_path}")


if __name__ == "__main__":
    main()
