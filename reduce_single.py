#!/usr/bin/env python3
"""
SO(2) reduction for a single motorized-caster node in an elastic well.

Trials are never concatenated. Unwrapping, differentiation, and pairing happen
strictly within a trial; pooling is only of already-reduced samples.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from format_tracks import wrap_angle, circ_mean


def wrap(a):
    return wrap_angle(a)


def circ_std(angles):
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan
    R = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
    R = min(max(R, 1e-15), 1.0)
    return float(np.sqrt(-2.0 * np.log(R)))


def odd_window(n, lo=5):
    n = max(int(n), lo)
    if n % 2 == 0:
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Circle / well-center
# --------------------------------------------------------------------------- #
def kasa_circle(x, y):
    """Algebraic circle fit. Returns (xc, yc, R)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = 0.5 * sol[0], 0.5 * sol[1]
    R = float(np.sqrt(max(xc ** 2 + yc ** 2 + sol[2], 0.0)))
    return float(xc), float(yc), R


def geometric_circle(x, y, init=None):
    """Geometric circle fit (LM). Returns (xc, yc, R, residual_rms)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if init is None:
        init = kasa_circle(x, y)

    def resid(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    res = least_squares(resid, np.asarray(init, float), method="lm")
    xc, yc, R = (float(v) for v in res.x)
    rms = float(np.sqrt(np.mean(resid(res.x) ** 2)))
    return xc, yc, R, rms


def fit_center_joint(regime_xy):
    """One center, per-regime radius. regime_xy: list of (x, y) arrays.

    Returns (xc, yc, list[R_k], residual_rms).
    """
    chunks = [(np.asarray(x, float), np.asarray(y, float)) for x, y in regime_xy]
    chunks = [(x, y) for x, y in chunks if len(x) >= 3]
    if not chunks:
        return np.nan, np.nan, [], np.nan
    allx = np.concatenate([x for x, _ in chunks])
    ally = np.concatenate([y for _, y in chunks])
    xc0, yc0, R0 = kasa_circle(allx, ally)
    nR = len(chunks)
    p0 = np.array([xc0, yc0] + [kasa_circle(x, y)[2] if len(x) >= 3 else R0
                                for x, y in chunks], dtype=float)

    def resid(p):
        xc, yc = p[0], p[1]
        out = []
        for k, (x, y) in enumerate(chunks):
            out.append(np.hypot(x - xc, y - yc) - p[2 + k])
        return np.concatenate(out)

    res = least_squares(resid, p0, method="lm")
    xc, yc = float(res.x[0]), float(res.x[1])
    Rs = [float(v) for v in res.x[2:]]
    rms = float(np.sqrt(np.mean(resid(res.x) ** 2)))
    # pad Rs if some regimes were empty
    return xc, yc, Rs, rms


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    track: int
    regime: str
    dt: float = np.nan
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    psi: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_u: np.ndarray = field(default_factory=lambda: np.array([]))
    psi_u: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_u: np.ndarray = field(default_factory=lambda: np.array([]))
    rdot: np.ndarray = field(default_factory=lambda: np.array([]))
    psidot: np.ndarray = field(default_factory=lambda: np.array([]))
    phidot: np.ndarray = field(default_factory=lambda: np.array([]))
    valid: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    asymptotic: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    T_settle: float = 0.0


def split_on_gaps(t, *series, max_gap_frames=2):
    """Split a trial at gaps larger than max_gap_frames * median dt. Returns list of index slices."""
    t = np.asarray(t, float)
    if t.size < 3:
        return [np.arange(t.size)]
    dt = np.diff(t)
    med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    if med <= 0:
        return [np.arange(t.size)]
    breaks = np.where(dt > max_gap_frames * med)[0]
    out, start = [], 0
    for b in breaks:
        if b + 1 - start >= 3:
            out.append(np.arange(start, b + 1))
        start = b + 1
    if t.size - start >= 3:
        out.append(np.arange(start, t.size))
    return out or [np.arange(t.size)]


def savgol_deriv(sig, dt, window_s, polyorder=3):
    n = len(sig)
    if n < 5 or not np.isfinite(dt) or dt <= 0:
        return np.full(n, np.nan)
    W = odd_window(round(window_s / dt), lo=5)
    W = min(W, n if n % 2 == 1 else n - 1)
    W = max(W, polyorder + 2 + (1 - (polyorder + 2) % 2))  # odd, > polyorder
    if W > n:
        return np.full(n, np.nan)
    return savgol_filter(sig, window_length=W, polyorder=polyorder, deriv=1, delta=dt,
                         mode="interp")


def reduce_trial(tr, xc, yc, A, delta, r_min, savgol_s, polyorder, T_settle=None):
    """Fill r, phi, psi and SG derivatives on a Trial. Mutates tr."""
    x = tr.x - A * np.cos(tr.theta + delta)
    y = tr.y - A * np.sin(tr.theta + delta)
    dx, dy = x - xc, y - yc
    tr.r = np.hypot(dx, dy)
    tr.phi = np.arctan2(dy, dx)
    tr.theta_u = np.unwrap(tr.theta)
    tr.phi_u = np.unwrap(tr.phi)
    # Plotting convention: rotate ψ by π so ψ = 0 is radially inward.
    tr.psi_u = tr.theta_u - tr.phi_u + np.pi
    tr.psi = wrap(tr.psi_u)

    dt = float(np.median(np.diff(tr.t))) if tr.t.size > 1 else np.nan
    tr.dt = dt
    tr.rdot = savgol_deriv(tr.r, dt, savgol_s, polyorder)
    tr.psidot = savgol_deriv(tr.psi_u, dt, savgol_s, polyorder)
    tr.phidot = savgol_deriv(tr.phi_u, dt, savgol_s, polyorder)

    tr.valid = np.isfinite(tr.r) & np.isfinite(tr.psi) & (tr.r >= r_min)
    # T_settle from radial relaxation on valid samples.
    if T_settle is not None:
        tr.T_settle = float(T_settle)
    else:
        tr.T_settle = auto_settle(tr.t, tr.r, tr.valid)
    tr.asymptotic = tr.valid & (tr.t >= tr.T_settle)
    return tr


def auto_settle(t, r, valid, frac_tail=0.30, n_sig=3.0, min_asym_frac=0.5):
    """First time after which r stays in the tail band, keeping >= min_asym_frac of the trial."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    ok = valid & np.isfinite(r) & np.isfinite(t)
    if ok.sum() < 10:
        return float(t[0]) if t.size else 0.0
    tt, rr = t[ok], r[ok]
    n_tail = max(5, int(frac_tail * len(rr)))
    tail = rr[-n_tail:]
    r_as = float(np.median(tail))
    mad = float(np.median(np.abs(tail - r_as))) * 1.4826
    p10, p90 = np.percentile(tail, [10, 90])
    tol = max(n_sig * mad, 0.5 * (p90 - p10), 0.1 * abs(r_as), 1e-4)
    inside = np.abs(rr - r_as) <= tol
    frac_in = np.array([np.mean(inside[i:]) for i in range(len(inside))])
    hit = np.where(frac_in >= 0.90)[0]
    settle_i = int(hit[0]) if hit.size else 0
    settle_i = min(settle_i, int((1.0 - min_asym_frac) * len(tt)))
    return float(tt[settle_i])


# --------------------------------------------------------------------------- #
# Binning on the (psi, r) cylinder
# --------------------------------------------------------------------------- #
@dataclass
class BinnedField:
    psi_edges: np.ndarray
    r_edges: np.ndarray
    psi_c: np.ndarray
    r_c: np.ndarray
    rdot: np.ndarray
    psidot: np.ndarray
    phidot: np.ndarray
    count: np.ndarray
    mask: np.ndarray          # True = keep (count >= n_min)


def bin_field(trials, n_psi=50, n_r=35, n_min=20, r_lim=None):
    psi, r, rd, pd, hd = [], [], [], [], []
    for tr in trials:
        m = tr.asymptotic
        if m.sum() == 0:
            continue
        psi.append(tr.psi[m]); r.append(tr.r[m])
        rd.append(tr.rdot[m]); pd.append(tr.psidot[m]); hd.append(tr.phidot[m])
    if not psi:
        empty = np.full((n_r, n_psi), np.nan)
        pe = np.linspace(-np.pi, np.pi, n_psi + 1)
        re = np.linspace(0, 1, n_r + 1)
        return BinnedField(pe, re, 0.5 * (pe[:-1] + pe[1:]), 0.5 * (re[:-1] + re[1:]),
                           empty, empty.copy(), empty.copy(), np.zeros_like(empty),
                           np.zeros_like(empty, dtype=bool))
    psi = np.concatenate(psi)
    r = np.concatenate(r)
    rd = np.concatenate(rd)
    pd = np.concatenate(pd)
    hd = np.concatenate(hd)
    if r_lim is None:
        rlo = float(np.nanpercentile(r, 1))
        rhi = float(np.nanpercentile(r, 99))
        if rhi <= rlo:
            rhi = rlo + 1e-3
    else:
        rlo, rhi = r_lim
    psi_edges = np.linspace(-np.pi, np.pi, n_psi + 1)
    r_edges = np.linspace(rlo, rhi, n_r + 1)
    # psi index with wrap: last edge is +pi, identical to -pi
    ip = np.digitize(wrap(psi), psi_edges) - 1
    ip = np.clip(ip, 0, n_psi - 1)
    ir = np.digitize(r, r_edges) - 1
    ir = np.clip(ir, 0, n_r - 1)
    shape = (n_r, n_psi)
    count = np.zeros(shape, dtype=float)
    srd = np.zeros(shape); spd = np.zeros(shape); shd = np.zeros(shape)
    for i, j, a, b, c in zip(ir, ip, rd, pd, hd):
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        count[i, j] += 1
        srd[i, j] += a; spd[i, j] += b; shd[i, j] += c
    with np.errstate(invalid="ignore", divide="ignore"):
        rdot = srd / count
        psidot = spd / count
        phidot = shd / count
    mask = count >= n_min
    rdot = np.where(mask, rdot, np.nan)
    psidot = np.where(mask, psidot, np.nan)
    phidot = np.where(mask, phidot, np.nan)
    psi_c = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    return BinnedField(psi_edges, r_edges, psi_c, r_c, rdot, psidot, phidot, count, mask)


def wrap_grid_for_contour(field):
    """Pad psi by wrapping first/last columns so contours see the seam."""
    def pad(M):
        return np.concatenate([M[:, -1:], M, M[:, :1]], axis=1)
    psi = np.concatenate([[field.psi_c[0] - (field.psi_c[1] - field.psi_c[0])],
                          field.psi_c,
                          [field.psi_c[-1] + (field.psi_c[-1] - field.psi_c[-2])]])
    return psi, field.r_c, pad(field.rdot), pad(field.psidot), pad(field.phidot)


# --------------------------------------------------------------------------- #
# Fixed points
# --------------------------------------------------------------------------- #
@dataclass
class FixedPoint:
    r: float
    psi: float
    Omega: float
    evals: tuple
    kind: str
    n_nbhd: int


def classify_evals(evals):
    ev = np.asarray(evals, dtype=complex)
    re, im = ev.real, ev.imag
    if np.any(re > 1e-6) and np.any(re < -1e-6):
        return "saddle"
    if np.all(re < -1e-6):
        return "stable spiral" if np.any(np.abs(im) > 1e-6) else "stable node"
    if np.all(re > 1e-6):
        return "unstable spiral" if np.any(np.abs(im) > 1e-6) else "unstable node"
    return "marginal"


def extract_fixed_points(field, trials, nbhd_frac=0.08, min_sep_psi=0.25, min_sep_r=None):
    """Coarse nullcline intersections, then local linear refine."""
    rdot, psidot = field.rdot, field.psidot
    psi_c, r_c = field.psi_c, field.r_c
    # sign-change in both fields on the grid (4-neighbor)
    seeds = []
    nr, np_ = rdot.shape
    for i in range(nr - 1):
        for j in range(np_):
            j2 = (j + 1) % np_
            block_r = rdot[i:i + 2, :][:, [j, j2]]
            block_p = psidot[i:i + 2, :][:, [j, j2]]
            if np.any(~np.isfinite(block_r)) or np.any(~np.isfinite(block_p)):
                continue
            if np.nanmin(block_r) * np.nanmax(block_r) <= 0 and np.nanmin(block_p) * np.nanmax(block_p) <= 0:
                seeds.append((0.5 * (r_c[i] + r_c[i + 1]), wrap(0.5 * (psi_c[j] + psi_c[j2]
                             if abs(psi_c[j] - psi_c[j2]) < np.pi
                             else psi_c[j]))))
    if min_sep_r is None:
        min_sep_r = 0.15 * (r_c[-1] - r_c[0] or 1.0)
    kept = []
    for r0, p0 in seeds:
        if any(abs(r0 - r1) < min_sep_r and abs(wrap(p0 - p1)) < min_sep_psi for r1, p1 in kept):
            continue
        kept.append((r0, p0))
    if not kept:
        # Relative equilibrium: rates ~0 in a blob, no clean sign-change. Use the
        # peak of the asymptotic (r, psi) histogram as a seed.
        pop = field.count
        if np.nanmax(pop) >= 5:
            k = int(np.nanargmax(pop))
            ir, ip = np.unravel_index(k, pop.shape)
            kept.append((float(r_c[ir]), float(psi_c[ip])))

    # pool asymptotic samples
    rs, ps, rds, pds, hds = [], [], [], [], []
    for tr in trials:
        m = tr.asymptotic
        rs.append(tr.r[m]); ps.append(tr.psi[m])
        rds.append(tr.rdot[m]); pds.append(tr.psidot[m]); hds.append(tr.phidot[m])
    if not rs:
        return []
    rs, ps = np.concatenate(rs), np.concatenate(ps)
    rds, pds, hds = np.concatenate(rds), np.concatenate(pds), np.concatenate(hds)
    r_span = float(np.nanmax(rs) - np.nanmin(rs)) or 1.0
    dr = max(nbhd_frac * r_span, 1e-4)
    dpsi = max(nbhd_frac * 2 * np.pi, 0.15)

    fps = []
    for r0, p0 in kept:
        dpsi_i = wrap(ps - p0)
        nb = np.isfinite(rds) & np.isfinite(pds) & (np.abs(rs - r0) < dr) & (np.abs(dpsi_i) < dpsi)
        if nb.sum() < 15:
            continue
        # [rdot, psidot] = J [r-r0, psi-p0] + b
        X = np.column_stack([rs[nb] - r0, dpsi_i[nb], np.ones(nb.sum())])
        try:
            cr, *_ = np.linalg.lstsq(X, rds[nb], rcond=None)
            cp, *_ = np.linalg.lstsq(X, pds[nb], rcond=None)
        except np.linalg.LinAlgError:
            continue
        # J = [[cr0, cr1], [cp0, cp1]], b = [cr2, cp2]
        J = np.array([[cr[0], cr[1]], [cp[0], cp[1]]], dtype=float)
        b = np.array([cr[2], cp[2]], dtype=float)
        try:
            delta = -np.linalg.solve(J, b)
        except np.linalg.LinAlgError:
            continue
        r_star = r0 + float(delta[0])
        psi_star = wrap(p0 + float(delta[1]))
        if r_star <= 0:
            continue
        dpsi2 = wrap(ps - psi_star)
        nb2 = np.isfinite(hds) & (np.abs(rs - r_star) < dr) & (np.abs(dpsi2) < dpsi)
        Omega = float(np.nanmean(hds[nb2])) if nb2.sum() else np.nan
        evals = tuple(np.linalg.eigvals(J))
        fps.append(FixedPoint(r=r_star, psi=psi_star, Omega=Omega, evals=evals,
                              kind=classify_evals(evals), n_nbhd=int(nb.sum())))
    return fps


def symmetry_residual(field):
    """S_G, S_H, S_F as specified. Returns dict."""
    # interpolate G(r, -psi) onto the same grid by reversing psi columns
    # psi_c is increasing from -pi to pi; -psi is reverse order (approx)
    G = field.psidot
    H = field.phidot
    F = field.rdot
    # map column j (psi) to column of -psi
    psi = field.psi_c
    idx = []
    for p in psi:
        k = int(np.argmin(np.abs(wrap(psi - (-p)))))
        idx.append(k)
    idx = np.asarray(idx)
    Gm = G[:, idx]
    Hm = H[:, idx]
    Fm = F[:, idx]
    pop = field.mask & field.mask[:, idx]
    def rel(A, B, even=False):
        # even: F(r,-psi)=F(r,psi) -> A-B; odd: A+B
        D = (A - B) if even else (A + B)
        num = np.nansum(np.where(pop, D ** 2, 0.0))
        den = np.nansum(np.where(pop, A ** 2, 0.0))
        return float(np.sqrt(num / den)) if den > 0 else np.nan
    return {"S_G": rel(G, Gm, even=False), "S_H": rel(H, Hm, even=False),
            "S_F": rel(F, Fm, even=True)}


def mean_phidot_trial(tr):
    """Asymptotic winding rate from unwrapped phi, not a mean of wrapped rates."""
    m = tr.asymptotic
    if m.sum() < 2:
        return np.nan
    t, pu = tr.t[m], tr.phi_u[m]
    dt = t[-1] - t[0]
    if dt <= 0:
        return np.nan
    return float((pu[-1] - pu[0]) / dt)


def terminal_state(tr, frac=0.08):
    """Mean r and circular-mean psi of the last `frac` of asymptotic samples."""
    m = tr.asymptotic
    if m.sum() < 3:
        return np.nan, np.nan
    r, psi = tr.r[m], tr.psi[m]
    n = max(3, int(frac * len(r)))
    return float(np.mean(r[-n:])), float(circ_mean(psi[-n:]))
