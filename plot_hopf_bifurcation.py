#!/usr/bin/env python3
"""
Hopf-bifurcation diagnostics from format_tracks_single.py output.

Does not replace the scalar sweep figures in plot_hopf_metrics.py. Adds:

  1. Representative x(t), y(t), x–y, r(t), unwrapped polar angle, and x–ẋ
     portraits below / near / above a fitted Dc (several trials each).
  2. Bifurcation diagrams of R = ⟨r²⟩^{1/2} and R² vs motor drive, with
     trial-level scatter, a shared Hopf threshold Dc, and linear / jump
     alternatives.
  3. Onset frequency: signed and absolute orbital angular velocity, signed
     circulation L = ⟨xẏ − yẋ⟩, CW/CCW fractions, and x(t) spectra.

The rest point used for r and φ is, for each length l, the mean (x, y) of
trials that do not complete --n-rev heading turns (fallback: lowest drive).

Example
-------
    python plot_hopf_bifurcation.py ../Data/250926
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares
from scipy.signal import welch

from reduce_single import geometric_circle, savgol_deriv
from plot_hopf_metrics import (
    crop_settle, load_robot, n_rev_window, parse_ml,
)


@dataclass
class Trial:
    m: float
    length: int
    track: int
    src: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    reached_nrev: bool = False
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_u: np.ndarray = field(default_factory=lambda: np.array([]))
    xdot: np.ndarray = field(default_factory=lambda: np.array([]))
    ydot: np.ndarray = field(default_factory=lambda: np.array([]))
    R_rms: float = np.nan
    R2: float = np.nan
    R_fit: float = np.nan
    L: float = np.nan
    omega_L: float = np.nan
    omega_abs: float = np.nan
    omega_phi: float = np.nan
    dphi: float = np.nan
    peak_freq: float = np.nan
    spec_f: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_p: np.ndarray = field(default_factory=lambda: np.array([]))


def dt_of(t):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.nan
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.nan


def rest_point_for_length(trials, length):
    """Mean (x, y) of non-rotating trials; else lowest-drive trials."""
    group = [tr for tr in trials if tr.length == length]
    quiet = [tr for tr in group if not tr.reached_nrev]
    if not quiet:
        m0 = min(tr.m for tr in group)
        quiet = [tr for tr in group if tr.m == m0]
    xs = np.concatenate([tr.x for tr in quiet if tr.x.size])
    ys = np.concatenate([tr.y for tr in quiet if tr.y.size])
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    if xs.size == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def fill_trial(tr, xc, yc, savgol_s, polyorder, r_min, fmin=0.15):
    dt = dt_of(tr.t)
    x0 = tr.x - xc
    y0 = tr.y - yc
    tr.r = np.hypot(x0, y0)
    tr.R_rms = float(np.sqrt(np.nanmean(tr.r ** 2))) if tr.r.size else np.nan
    tr.R2 = tr.R_rms ** 2 if np.isfinite(tr.R_rms) else np.nan
    tr.phi_u = np.unwrap(np.arctan2(y0, x0))
    tr.dphi = float(tr.phi_u[-1] - tr.phi_u[0]) if tr.phi_u.size else np.nan
    tr.xdot = savgol_deriv(x0, dt, savgol_s, polyorder)
    tr.ydot = savgol_deriv(y0, dt, savgol_s, polyorder)
    c = x0 * tr.ydot - y0 * tr.xdot
    tr.L = float(np.nanmean(c)) if c.size else np.nan
    r2 = float(np.nanmean(x0 ** 2 + y0 ** 2)) if x0.size else np.nan
    tr.omega_L = (tr.L / r2) if (np.isfinite(r2) and r2 > r_min ** 2) else np.nan
    tr.omega_abs = abs(tr.omega_L) if np.isfinite(tr.omega_L) else np.nan
    ok = np.isfinite(tr.r) & np.isfinite(tr.phi_u) & (tr.r >= r_min)
    if ok.sum() >= 5 and np.isfinite(dt):
        phidot = savgol_deriv(tr.phi_u, dt, savgol_s, polyorder)
        tr.omega_phi = float(np.nanmean(phidot[ok]))
    else:
        tr.omega_phi = np.nan
    if np.isfinite(tr.dphi) and abs(tr.dphi) >= 2.0 * np.pi and x0.size >= 8:
        try:
            _, _, Rfit, rms = geometric_circle(x0[np.isfinite(x0)], y0[np.isfinite(y0)])
            tr.R_fit = float(Rfit) if (np.isfinite(Rfit) and rms < max(3.0 * Rfit, 1e-4)) else np.nan
        except Exception:
            tr.R_fit = np.nan
    xfin = x0[np.isfinite(x0)]
    if xfin.size >= 64 and np.isfinite(dt) and dt > 0:
        fs = 1.0 / dt
        nperseg = int(min(512, max(64, 2 * (xfin.size // 4))))
        if nperseg % 2:
            nperseg -= 1
        nperseg = max(nperseg, 32)
        if nperseg < xfin.size:
            f, pxx = welch(xfin - np.mean(xfin), fs=fs, nperseg=nperseg,
                           detrend="constant")
            tr.spec_f, tr.spec_p = f, pxx
            band = (f >= fmin) & (f <= 0.45 * fs)
            if np.any(band) and np.nanmax(pxx[band]) > 0:
                tr.peak_freq = float(f[band][np.argmax(pxx[band])])
    return tr


def load_trials(directory, settle, min_samples, cutoff, butter_order, n_rev):
    files = sorted(glob.glob(os.path.join(directory, "*_robot.csv")))
    if not files:
        raise SystemExit(f"No *_robot.csv files in {directory}")
    trials = []
    for path in files:
        parsed = parse_ml(path)
        if parsed is None:
            print(f"  WARNING: no m/l in {os.path.basename(path)}; skipping")
            continue
        m_drive, length = parsed
        df, nid, th_col, x_col, y_col = load_robot(path)
        src = os.path.basename(path)
        n = 0
        for tr_id, sub in df.groupby("track", sort=True):
            t = sub["time"].to_numpy(dtype=float)
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            th = sub[th_col].to_numpy(dtype=float)
            t, x, y, th = crop_settle(t, x, y, th, settle)
            if t.size < min_samples:
                continue
            dt = dt_of(t)
            t_nrev, _ = n_rev_window(t, th, dt, cutoff, butter_order, n_rev)
            trials.append(Trial(
                m=float(m_drive), length=int(length), track=int(tr_id), src=src,
                t=t, x=x, y=y, heading=th,
                reached_nrev=bool(np.isfinite(t_nrev)),
            ))
            n += 1
        print(f"{src}: {n} SS trial(s)")
    return trials


# --------------------------------------------------------------------------- #
# Onset fits (shared Dc from R² Hopf / relu)
# --------------------------------------------------------------------------- #
def _rmse(y, yhat):
    d = np.asarray(yhat, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan


def fit_relu(D, Y):
    """Y = B + A * max(D - Dc, 0), A>=0, B>=0."""
    D = np.asarray(D, float)
    Y = np.asarray(Y, float)
    ok = np.isfinite(D) & np.isfinite(Y)
    D, Y = D[ok], Y[ok]
    if D.size < 6:
        return dict(Dc=np.nan, A=np.nan, B=np.nan, rmse=np.nan)
    Dmin, Dmax = float(D.min()), float(D.max())
    ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))

    def resid(p):
        Dc, A, B = p
        return Y - (B + A * np.maximum(D - Dc, 0.0))

    best, best_cost = None, np.inf
    for Dc0 in np.linspace(Dmin, Dmax, 9):
        A0 = max(ymax - ymin, 1e-16) / max(Dmax - Dmin, 1.0)
        p0 = np.array([Dc0, A0, max(ymin, 0.0)])
        try:
            res = least_squares(
                resid, p0,
                bounds=([Dmin - 0.5 * (Dmax - Dmin), 0.0, 0.0],
                        [Dmax + 0.5 * (Dmax - Dmin), np.inf, ymax + 1e-9]),
            )
        except ValueError:
            continue
        if res.cost < best_cost:
            best, best_cost = res, res.cost
    if best is None:
        return dict(Dc=np.nan, A=np.nan, B=np.nan, rmse=np.nan)
    Dc, A, B = (float(v) for v in best.x)
    yhat = B + A * np.maximum(D - Dc, 0.0)
    return dict(Dc=Dc, A=A, B=B, rmse=_rmse(Y, yhat))


def fit_jump(D, Y, Dc):
    """Y = B below Dc, C at/above Dc. Dc frozen."""
    D = np.asarray(D, float)
    Y = np.asarray(Y, float)
    ok = np.isfinite(D) & np.isfinite(Y)
    D, Y = D[ok], Y[ok]
    lo, hi = Y[D < Dc], Y[D >= Dc]
    B = float(np.nanmean(lo)) if lo.size else float(np.nanmean(Y))
    C = float(np.nanmean(hi)) if hi.size else B
    yhat = np.where(D < Dc, B, C)
    return dict(B=B, C=C, rmse=_rmse(Y, yhat))


def pick_regimes(ms, Dc, r_by_m=None, r_noise=None, circ_frac=None):
    """Three distinct drives: quiet (below), nearest to Dc, clearly above."""
    ms = np.array(sorted(set(float(m) for m in ms)))
    if ms.size == 0:
        return None
    if ms.size == 1:
        v = float(ms[0])
        return v, v, v
    if ms.size == 2:
        return float(ms[0]), float(ms[0]), float(ms[1])
    if not np.isfinite(Dc):
        return float(ms[0]), float(ms[len(ms) // 2]), float(ms[-1])

    def lookup(d, m, default=np.nan):
        for k, v in (d or {}).items():
            if abs(float(k) - float(m)) < 1e-6:
                return float(v)
        return default

    near = float(ms[np.argmin(np.abs(ms - Dc))])
    below_c = ms[ms < Dc]
    if below_c.size == 0:
        below = float(ms[0])
    elif r_by_m and np.isfinite(r_noise) and r_noise > 0:
        quiet = [float(m) for m in below_c if lookup(r_by_m, m, np.inf) <= 1.5 * r_noise]
        below = max(quiet) if quiet else float(below_c[0])
    else:
        below = float(below_c[0])

    above_c = ms[ms > max(near, Dc)]
    rotating = [float(m) for m in ms if lookup(circ_frac, m, 0.0) >= 0.5 and m > Dc]
    if rotating:
        above = float(sorted(rotating)[len(rotating) // 2])
    elif above_c.size:
        above = float(above_c[min(len(above_c) - 1, max(0, len(above_c) // 2))])
    else:
        above = float(ms[-1])

    if below == near:
        lower = ms[ms < near]
        if lower.size:
            below = float(lower[0])
    if above == near:
        higher = ms[ms > near]
        if higher.size:
            above = float(higher[-1])
    return below, near, above


def trials_at(trials, length, m, n_show):
    group = [tr for tr in trials if tr.length == length and tr.m == m]
    group = sorted(group, key=lambda tr: (tr.track, tr.src))
    if len(group) <= n_show:
        return group
    # Evenly spaced by index so the set is reproducible, not "prettiest".
    idx = np.linspace(0, len(group) - 1, n_show).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [group[i] for i in idx]


def cap_time(tr, tmax):
    t0 = tr.t[0]
    keep = tr.t <= (t0 + tmax)
    if keep.sum() < 5:
        keep = np.ones(tr.t.size, dtype=bool)
    t = tr.t[keep] - t0
    return keep, t


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_representative(trials, length, regimes, Dc, xc, yc, outdir, dpi,
                        n_show, tmax):
    labels = ("below $D_c$", "near $D_c$", r"above $D_c$")
    fig = plt.figure(figsize=(12.5, 14.5))
    gs = GridSpec(5, 3, figure=fig, hspace=0.38, wspace=0.28,
                  left=0.07, right=0.99, top=0.93, bottom=0.04)
    # Shared xy window from the above-threshold column.
    above = trials_at(trials, length, regimes[2], n_show)
    if above:
        span = np.concatenate([np.hypot(tr.x - xc, tr.y - yc) for tr in above])
        lim = float(np.nanpercentile(span, 97)) if span.size else 0.05
    else:
        lim = 0.05
    lim = max(lim, 0.01)

    axes = {}
    for col, (m_val, lab) in enumerate(zip(regimes, labels)):
        chosen = trials_at(trials, length, m_val, n_show)
        ax_xy = fig.add_subplot(gs[0, col])
        ax_xy.set_aspect("equal")
        ax_t = fig.add_subplot(gs[1, col])
        ax_r = fig.add_subplot(gs[2, col])
        ax_phi = fig.add_subplot(gs[3, col])
        ax_ps = fig.add_subplot(gs[4, col])
        axes[col] = (ax_xy, ax_t, ax_r, ax_phi, ax_ps)
        title = fr"$l={length}$, $m={m_val:g}$ ({lab})"
        if np.isfinite(Dc):
            title += fr"  $D_c={Dc:.0f}$"
        ax_xy.set_title(title, fontsize=10)
        for k, tr in enumerate(chosen):
            color = f"C{k % 10}"
            keep, t = cap_time(tr, tmax)
            x0, y0 = tr.x[keep] - xc, tr.y[keep] - yc
            ax_xy.plot(x0, y0, color=color, lw=0.9, alpha=0.75)
            ax_xy.plot(x0[0], y0[0], "s", color=color, ms=3, alpha=0.8)
            ax_t.plot(t, x0, color=color, lw=0.9, alpha=0.7)
            ax_t.plot(t, y0, color=color, lw=0.9, alpha=0.7, ls="--")
            ax_r.plot(t, tr.r[keep], color=color, lw=0.9, alpha=0.75)
            ax_phi.plot(t, tr.phi_u[keep] - tr.phi_u[keep][0],
                        color=color, lw=0.9, alpha=0.75)
            ax_ps.plot(x0, tr.xdot[keep], color=color, lw=0.8, alpha=0.75)
        ax_xy.plot(0, 0, "+", color="k", ms=8)
        ax_xy.set_xlim(-lim, lim)
        ax_xy.set_ylim(-lim, lim)
        ax_xy.set_xlabel("x (m)")
        ax_xy.set_ylabel("y (m)")
        ax_t.set_xlabel("time (s)")
        ax_t.set_ylabel("x, y (m)")
        ax_t.plot([], [], color="0.3", lw=1.2, label="x")
        ax_t.plot([], [], color="0.3", lw=1.2, ls="--", label="y")
        if col == 0:
            ax_t.legend(frameon=False, fontsize=8)
        ax_r.set_xlabel("time (s)")
        ax_r.set_ylabel(r"$r$ (m)")
        ax_phi.set_xlabel("time (s)")
        ax_phi.set_ylabel(r"unwrapped $\phi$ (rad)")
        ax_ps.set_xlabel("x (m)")
        ax_ps.set_ylabel(r"$\dot x$ (m/s)")
        ax_ps.locator_params(axis="x", nbins=4)
        for ax in (ax_xy, ax_t, ax_r, ax_phi, ax_ps):
            ax.grid(alpha=0.25)
        print(f"  repr l={length} m={m_val:g} ({lab}): {len(chosen)} trials")

    path = os.path.join(outdir, f"14_repr_trajectories_l{length}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def scatter_and_mean(ax, trials, length, yattr, color, jitter, rng):
    group = [tr for tr in trials if tr.length == length]
    ys = np.array([getattr(tr, yattr) for tr in group], dtype=float)
    ms = np.array([tr.m for tr in group], dtype=float)
    ok = np.isfinite(ys) & np.isfinite(ms)
    ax.scatter(ms[ok] + rng.uniform(-jitter, jitter, ok.sum()),
               ys[ok], s=10, alpha=0.18, color=color, linewidths=0, zorder=2)
    df = pd.DataFrame({"m": ms[ok], "y": ys[ok]})
    if df.empty:
        return None, None, None
    g = df.groupby("m")["y"].agg(["mean", "std", "count"])
    g["std"] = g["std"].fillna(0.0)
    ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="o-", color=color,
                lw=1.6, ms=6, capsize=3, zorder=3, label=fr"$l={length}$")
    return g.index.to_numpy(), g["mean"].to_numpy(), g["std"].to_numpy()


def plot_bifurcation(trials, fits, outdir, dpi):
    lengths = sorted({tr.length for tr in trials})
    rng = np.random.default_rng(0)
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    for i, length in enumerate(lengths):
        color = f"C{i}"
        scatter_and_mean(ax1, trials, length, "R_rms", color, 1.6, rng)
        scatter_and_mean(ax2, trials, length, "R2", color, 1.6, rng)
        # R_fit as faint squares (circulating trials only)
        circ = [tr for tr in trials
                if tr.length == length and np.isfinite(tr.R_fit)]
        if circ:
            ax1.scatter([tr.m for tr in circ], [tr.R_fit for tr in circ],
                        marker="s", s=11, alpha=0.15, color=color,
                        linewidths=0, zorder=1)
        fit = fits.get(length, {})
        hopf = fit.get("hopf")
        if not hopf or not np.isfinite(hopf.get("Dc", np.nan)):
            continue
        Dc, A, B = hopf["Dc"], hopf["A"], hopf["B"]
        ms = np.array([tr.m for tr in trials if tr.length == length])
        D = np.linspace(ms.min(), ms.max(), 400)
        R2 = B + A * np.maximum(D - Dc, 0.0)
        R = np.sqrt(np.maximum(R2, 0.0))
        ax1.plot(D, R, color=color, lw=2.0, zorder=4)
        ax2.plot(D, R2, color=color, lw=2.0, zorder=4)
        lin = fit.get("linear")
        jmp = fit.get("jump")
        if lin:
            R_lin = lin["B"] + lin["A"] * np.maximum(D - Dc, 0.0)
            ax1.plot(D, R_lin, color=color, lw=1.2, ls="--", alpha=0.85, zorder=4)
            ax2.plot(D, np.maximum(R_lin, 0.0) ** 2, color=color, lw=1.2,
                     ls="--", alpha=0.85, zorder=4)
        if jmp:
            R_j = np.where(D < Dc, jmp["B"], jmp["C"])
            ax1.plot(D, R_j, color=color, lw=1.2, ls=":", alpha=0.9, zorder=4)
            ax2.plot(D, np.maximum(R_j, 0.0) ** 2, color=color, lw=1.2,
                     ls=":", alpha=0.9, zorder=4)
        ax1.axvline(Dc, color=color, lw=0.8, alpha=0.4, zorder=1)
        ax2.axvline(Dc, color=color, lw=0.8, alpha=0.4, zorder=1)

    ax1.set_xlabel("motor drive $m$")
    ax1.set_ylabel(r"$R=\langle r^2\rangle^{1/2}$ (m)")
    ax1.set_title("orbital amplitude")
    ax1.plot([], [], color="0.3", lw=2.0, label="Hopf")
    ax1.plot([], [], color="0.3", lw=1.2, ls="--", label="linear onset")
    ax1.plot([], [], color="0.3", lw=1.2, ls=":", label="discontinuous jump")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("motor drive $m$")
    ax2.set_ylabel(r"$R^2=\langle r^2\rangle$ (m$^2$)")
    ax2.set_title(r"orbital amplitude squared (Hopf: $R^2\propto m-D_c$)")
    ax2.plot([], [], color="0.3", lw=2.0, label="Hopf")
    ax2.plot([], [], color="0.3", lw=1.2, ls="--", label="linear onset")
    ax2.plot([], [], color="0.3", lw=1.2, ls=":", label="discontinuous jump")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    fig2.tight_layout()
    p1 = os.path.join(outdir, "15_bifurcation_R.png")
    p2 = os.path.join(outdir, "16_bifurcation_R2.png")
    fig1.savefig(p1, dpi=dpi)
    fig2.savefig(p2, dpi=dpi)
    plt.close(fig1)
    plt.close(fig2)
    return p1, p2


def plot_frequency(trials, fits, outdir, dpi):
    lengths = sorted({tr.length for tr in trials})
    rng = np.random.default_rng(1)
    fig_s, ax_s = plt.subplots(figsize=(7.2, 4.6))
    fig_a, ax_a = plt.subplots(figsize=(7.2, 4.6))
    fig_L, ax_L = plt.subplots(figsize=(7.2, 4.6))
    fig_f, ax_f = plt.subplots(figsize=(7.2, 4.6))
    for i, length in enumerate(lengths):
        color = f"C{i}"
        scatter_and_mean(ax_s, trials, length, "omega_L", color, 1.6, rng)
        scatter_and_mean(ax_a, trials, length, "omega_abs", color, 1.6, rng)
        scatter_and_mean(ax_L, trials, length, "L", color, 1.6, rng)
        scatter_and_mean(ax_f, trials, length, "peak_freq", color, 1.6, rng)
        Dc = fits.get(length, {}).get("hopf", {}).get("Dc", np.nan)
        if np.isfinite(Dc):
            for ax in (ax_s, ax_a, ax_L, ax_f):
                ax.axvline(Dc, color=color, lw=0.8, alpha=0.4)
    ax_s.set_xlabel("motor drive $m$")
    ax_s.set_ylabel(r"$\omega_L=\langle x\dot y-y\dot x\rangle/\langle r^2\rangle$ (rad/s)")
    ax_s.set_title("signed orbital angular velocity")
    ax_s.legend(frameon=False)
    ax_s.grid(alpha=0.3)
    ax_a.set_xlabel("motor drive $m$")
    ax_a.set_ylabel(r"$|\omega_L|$ (rad/s)")
    ax_a.set_title("absolute orbital angular velocity")
    ax_a.legend(frameon=False)
    ax_a.grid(alpha=0.3)
    ax_L.set_xlabel("motor drive $m$")
    ax_L.set_ylabel(r"$L=\langle x\dot y-y\dot x\rangle$ (m$^2$/s)")
    ax_L.set_title("signed circulation")
    ax_L.legend(frameon=False)
    ax_L.grid(alpha=0.3)
    ax_f.set_xlabel("motor drive $m$")
    ax_f.set_ylabel("peak frequency of $x(t)$ (Hz)")
    ax_f.set_title("spectral peak of $x(t)$")
    ax_f.legend(frameon=False)
    ax_f.grid(alpha=0.3)
    paths = []
    for fig, name in (
        (fig_s, "17_omega_signed.png"),
        (fig_a, "18_omega_abs.png"),
        (fig_L, "19_circulation_L.png"),
        (fig_f, "22_peak_freq.png"),
    ):
        fig.tight_layout()
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_chirality(trials, fits, outdir, dpi, n_rev):
    """CW / CCW / neither fractions. Sign from circulation L; 'neither' if
    polar angle does not complete one turn (avoids noisy sign at the origin)."""
    lengths = sorted({tr.length for tr in trials})
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.2 * len(lengths) + 1, 4.2),
                             sharey=True, squeeze=False)
    for ax, length in zip(axes[0], lengths):
        group = [tr for tr in trials if tr.length == length]
        rows = []
        for m, sub in pd.DataFrame({
            "m": [tr.m for tr in group],
            "L": [tr.L for tr in group],
            "dphi": [tr.dphi for tr in group],
        }).groupby("m"):
            n = len(sub)
            circulating = np.isfinite(sub["dphi"]) & (np.abs(sub["dphi"]) >= 2 * np.pi)
            ccw = circulating & (sub["L"] > 0)
            cw = circulating & (sub["L"] < 0)
            rows.append((m, ccw.mean(), cw.mean(), 1.0 - circulating.mean(), n))
        rows = sorted(rows)
        m = np.array([r[0] for r in rows])
        ax.plot(m, [r[1] for r in rows], "o-", color="C0", label="CCW ($L>0$)")
        ax.plot(m, [r[2] for r in rows], "s-", color="C3", label="CW ($L<0$)")
        ax.plot(m, [r[3] for r in rows], "^-", color="0.5",
                label="no full polar turn")
        Dc = fits.get(length, {}).get("hopf", {}).get("Dc", np.nan)
        if np.isfinite(Dc):
            ax.axvline(Dc, color="k", lw=0.8, alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("motor drive $m$")
        ax.set_title(fr"$l={length}$")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("trial fraction")
    fig.suptitle("chirality (sign of $L$; polar winding $\\geq 2\\pi$)")
    fig.tight_layout()
    path = os.path.join(outdir, "20_chirality_frac.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_spectra(trials, length, regimes, Dc, outdir, dpi, n_show):
    labels = ("below $D_c$", "near $D_c$", r"above $D_c$")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, m_val, lab in zip(axes, regimes, labels):
        chosen = trials_at(trials, length, m_val, n_show)
        for k, tr in enumerate(chosen):
            if tr.spec_f.size == 0:
                continue
            ax.loglog(tr.spec_f, tr.spec_p, color=f"C{k % 10}",
                      lw=0.9, alpha=0.7)
        title = fr"$l={length}$, $m={m_val:g}$ ({lab})"
        if np.isfinite(Dc):
            title += fr", $D_c={Dc:.0f}$"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"PSD of $x(t)$")
    fig.tight_layout()
    path = os.path.join(outdir, f"21_spectra_l{length}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def run_bifurcation(args):
    directory = args.directory
    outdir = args.outdir or os.path.join(directory, "hopf_metric_plots")
    os.makedirs(outdir, exist_ok=True)
    savgol_s = getattr(args, "savgol_window", 0.35)
    polyorder = getattr(args, "savgol_poly", 3)
    r_min = getattr(args, "r_min", 0.005)
    n_show = getattr(args, "n_show", 6)
    tmax = getattr(args, "tmax", 4.0)

    print("\n--- Hopf bifurcation figures ---")
    trials = load_trials(directory, args.settle, args.min_samples,
                         args.cutoff, args.butter_order, args.n_rev)
    if not trials:
        print("  no trials; skipping bifurcation figures")
        return []

    lengths = sorted({tr.length for tr in trials})
    centers = {length: rest_point_for_length(trials, length) for length in lengths}
    for length, (xc, yc) in centers.items():
        print(f"  rest point l={length}: ({xc:.4f}, {yc:.4f})")
    for tr in trials:
        xc, yc = centers[tr.length]
        fill_trial(tr, xc, yc, savgol_s, polyorder, r_min)

    fits = {}
    for length in lengths:
        group = [tr for tr in trials if tr.length == length]
        dfm = pd.DataFrame({
            "m": [tr.m for tr in group],
            "R": [tr.R_rms for tr in group],
            "circ": [np.isfinite(tr.dphi) and abs(tr.dphi) >= 2 * np.pi
                     for tr in group],
        })
        med = dfm.groupby("m")[["R"]].median()
        D = med.index.to_numpy(dtype=float)
        R = med["R"].to_numpy(dtype=float)
        R2 = R ** 2
        hopf = fit_relu(D, R2)
        lin = jmp = None
        if np.isfinite(hopf["Dc"]):
            ok = np.isfinite(R)
            Dc = hopf["Dc"]
            def resid_lin(p):
                A, B = p
                return R[ok] - (B + A * np.maximum(D[ok] - Dc, 0.0))
            try:
                span = max(float(np.nanmax(R) - np.nanmin(R)), 1e-6)
                res = least_squares(
                    resid_lin,
                    [span / max(float(np.ptp(D[ok])), 1.0), max(float(np.nanmin(R)), 0.0)],
                    bounds=([0.0, 0.0], [np.inf, float(np.nanmax(R)) + 1e-9]),
                )
                A, B = (float(v) for v in res.x)
                lin = dict(Dc=Dc, A=A, B=B,
                           rmse=_rmse(R[ok], B + A * np.maximum(D[ok] - Dc, 0.0)))
            except ValueError:
                lin = dict(Dc=Dc, A=np.nan, B=np.nan, rmse=np.nan)
            jmp = fit_jump(D, R, Dc)
        fits[length] = dict(hopf=hopf, linear=lin, jump=jmp)
        lin_s = "nan" if not lin or not np.isfinite(lin["rmse"]) else f"{lin['rmse']:.3g}"
        jmp_s = "nan" if not jmp or not np.isfinite(jmp["rmse"]) else f"{jmp['rmse']:.3g}"
        print(f"  l={length}: Hopf Dc={hopf['Dc']:.1f}  RMSE_R2={hopf['rmse']:.3g}  "
              f"linear RMSE_R={lin_s}  jump RMSE_R={jmp_s}")

    saved = []
    regimes_by_l = {}
    for length in lengths:
        Dc = fits[length]["hopf"]["Dc"]
        ms = [tr.m for tr in trials if tr.length == length]
        r_by_m = (pd.DataFrame({
            "m": [tr.m for tr in trials if tr.length == length],
            "R": [tr.R_rms for tr in trials if tr.length == length],
        }).groupby("m")["R"].median().to_dict())
        circ_frac = (pd.DataFrame({
            "m": [tr.m for tr in trials if tr.length == length],
            "c": [float(np.isfinite(tr.dphi) and abs(tr.dphi) >= 2 * np.pi)
                  for tr in trials if tr.length == length],
        }).groupby("m")["c"].mean().to_dict())
        B = fits[length]["hopf"].get("B", np.nan)
        r_noise = float(np.sqrt(B)) if np.isfinite(B) and B >= 0 else np.nan
        regimes = pick_regimes(ms, Dc, r_by_m, r_noise, circ_frac)
        regimes_by_l[length] = regimes
        print(f"  l={length} regimes below/near/above: {regimes}")
        saved.append(plot_representative(
            trials, length, regimes, Dc, *centers[length],
            outdir, args.dpi, n_show, tmax))
        saved.append(plot_spectra(
            trials, length, regimes, Dc, outdir, args.dpi, n_show))

    saved.extend(plot_bifurcation(trials, fits, outdir, args.dpi))
    saved.extend(plot_frequency(trials, fits, outdir, args.dpi))
    saved.append(plot_chirality(trials, fits, outdir, args.dpi, args.n_rev))

    # Scalar table
    rows = []
    for tr in trials:
        rows.append(dict(
            file=tr.src, m=tr.m, l=tr.length, track=tr.track,
            reached_nrev=int(tr.reached_nrev),
            R_rms=tr.R_rms, R2=tr.R2, R_fit=tr.R_fit, L=tr.L,
            omega_L=tr.omega_L, omega_phi=tr.omega_phi, dphi=tr.dphi,
            peak_freq=tr.peak_freq,
        ))
    csv_path = os.path.join(outdir, "hopf_bifurcation_trials.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    fit_rows = []
    for length, f in fits.items():
        h = f["hopf"]
        fit_rows.append(dict(
            l=length, Dc=h["Dc"], A_R2=h["A"], B_R2=h["B"], rmse_R2=h["rmse"],
            linear_rmse_R=(f["linear"] or {}).get("rmse"),
            jump_rmse_R=(f["jump"] or {}).get("rmse"),
            xc=centers[length][0], yc=centers[length][1],
        ))
    fit_csv = os.path.join(outdir, "hopf_bifurcation_fits.csv")
    pd.DataFrame(fit_rows).to_csv(fit_csv, index=False)
    print(f"Wrote {len(saved)} Hopf figures + {os.path.basename(csv_path)} "
          f"+ {os.path.basename(fit_csv)}")
    return saved


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Hopf bifurcation figures from *_robot.csv")
    p.add_argument("directory", nargs="?",
                   default="/Users/alexleffell/Documents/PhD/tplax/Data/200826/hopf")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--butter-order", type=int, default=4)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--n-rev", type=int, default=5)
    p.add_argument("--savgol-window", type=float, default=0.35)
    p.add_argument("--savgol-poly", type=int, default=3)
    p.add_argument("--r-min", type=float, default=0.005)
    p.add_argument("--n-show", type=int, default=6,
                   help="Trials overlaid in representative / spectra panels.")
    p.add_argument("--tmax", type=float, default=4.0,
                   help="Seconds of SS shown in representative time series.")
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


if __name__ == "__main__":
    run_bifurcation(parse_args())
