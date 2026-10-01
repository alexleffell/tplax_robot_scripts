#!/usr/bin/env python3
"""
Per-track kinematics from format_tracks_single.py output.

For every track in each ``*_robot.csv``: heading vs time, x(t) and y(t),
phase portraits of heading vs x and heading vs y, and dx vs dθ after
Savitzky–Golay derivatives of x and unwrapped heading. Heading is wrapped
to [−π, π] and polylines are broken at the ±π seam.

Offset summary (kinematics_vs_offset.png)
-----------------------------------------
For a passive caster pushed back and forth on a rail, runs are grouped by caster
offset l, parsed from the file name (``l3_...`` → 3 × --l-step mm). The rail axis
is the principal axis of (x, y); each track is split into strokes at velocity
reversals (the first stroke, which starts from the initial perpendicular pose, is
dropped). Per stroke, ψ = heading − (motion direction + --trail-offset) is the
angle from the trailing-aligned pose, so each flip runs |ψ| ≈ π → 0. The
kinematic (no-slip) caster obeys dψ/ds = −sin ψ / l, i.e.
tan(|ψ|/2) = tan(|ψ₀|/2)·exp(−s/l), so
  - l_eff = −1 / slope of ln tan(|ψ|/2) vs s over 0.1π < |ψ| < 0.9π,
  - s10 / s90 = distance from reversal to 10 % / 90 % of the flip,
  - end lag = −sense·ψ at the stroke end (> 0: still lagging; a tag-mount offset
    cancels only when both flip senses occur),
  - remaining = (stroke length − s90) / l_eff, the relaxation lengths left after the
    flip; the no-slip prediction for the end lag is
    pred_lag = 2 arctan(tan(0.05π)·exp(−remaining)), compared with the measured lag,
  - same-sense fraction: consecutive flips turning the same way (1 = continuous
    spinning, 0 = back-and-forth swinging).
Per-stroke values go to kinematics_strokes.csv.

Paper figure (caster_figure.pdf/.png, γ ≡ ψ): (a) caster schematic with the no-slip
law, (b) |γ| vs distance since reversal per offset (median, IQR band), (c) reorientation
length (10 → 90 % of the flip) vs l with a linear fit.

Example
-------
    python plot_kinematics_tracks.py ../Data/260926/kinematics
    python plot_kinematics_tracks.py a_robot.csv b_robot.csv
    python plot_kinematics_tracks.py ../Data/300926/kinematics --summary-only
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from format_tracks import read_csv_comments, wrap_angle
from reduce_single import savgol_deriv


PSI_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PSI_TICKLABELS = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-track heading, position, and heading–position portraits")
    p.add_argument("inputs", nargs="*",
                   default=["/Users/alexleffell/Documents/PhD/tplax/Data/260926/kinematics"],
                   help="Directory of *_robot.csv files, or CSV paths. "
                        "Default: Data/260926/kinematics")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/kinematics_plots/ "
                        "or <first csv>_kinematics/")
    p.add_argument("--dpi", type=int, default=130, help="Figure DPI. Default: 130")
    p.add_argument("--savgol-window", type=float, default=0.35,
                   help="Savitzky–Golay window (s) for dx and dθ. Default: 0.35.")
    p.add_argument("--savgol-poly", type=int, default=3,
                   help="Savitzky–Golay polyorder. Default: 3.")
    p.add_argument("--l-regex", type=str, default=r"(?:^|_)l(\d+)(?:_|$)",
                   help="Regex whose first group is the offset index in the file prefix. "
                        r"Default: '(?:^|_)l(\d+)(?:_|$)'")
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per index step (mm). Default: 5.0")
    p.add_argument("--trail-offset", type=float, default=np.pi,
                   help="Heading of the trailing-aligned caster relative to the motion "
                        "direction (rad). Default: π")
    p.add_argument("--min-stroke-frac", type=float, default=0.5,
                   help="Keep strokes spanning at least this fraction of the rail range. "
                        "Default: 0.5")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip the per-track figures; write only the offset summary.")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip the offset summary figure, paper figure, and CSV.")
    p.add_argument("--paper-width", type=float, default=7.0,
                   help="Width (in) of the 3-panel caster_figure.pdf. Default: 7.0")
    return p.parse_args()


def collect_csvs(inputs):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, "*_robot.csv"))))
        else:
            files.append(item)
    return files


def file_prefix(path):
    base = os.path.basename(path)
    if base.endswith("_robot.csv"):
        return base[:-len("_robot.csv")]
    stem, _ = os.path.splitext(base)
    return stem[:-len("_robot")] if stem.endswith("_robot") else stem


def load_robot(path):
    df = read_csv_comments(path)
    meta = df.attrs
    nodes = meta.get("nodes")
    if nodes is None:
        nodes = [int(c[:-2]) for c in df.columns
                 if c.endswith("_x") and not c.startswith("extra")]
    nid = int(meta.get("node_id", nodes[0]))
    if "track" not in df.columns:
        df = df.copy()
        df["track"] = 1
    th_col = f"{nid}_theta" if f"{nid}_theta" in df.columns else "body_angle"
    x_col = f"{nid}_x" if f"{nid}_x" in df.columns else "centroid_x"
    y_col = f"{nid}_y" if f"{nid}_y" in df.columns else "centroid_y"
    return df, nid, th_col, x_col, y_col


def seam_segments(u, v):
    """Split (u, v) wherever u jumps by more than π (heading wrap)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.size < 2:
        return [(u, v)] if u.size else []
    cuts = np.where(np.abs(np.diff(u)) > np.pi)[0]
    bounds = np.concatenate([[0], cuts + 1, [len(u)]])
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 2:
            segs.append((u[a:b], v[a:b]))
    return segs


def style_heading_axis(ax, which="x"):
    ticks = PSI_TICKS
    labels = PSI_TICKLABELS
    if which == "x":
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def plot_track(t, x, y, th, title, path, dpi, savgol_window, savgol_poly):
    thw = wrap_angle(th)
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 11.0))
    ax_th, ax_xy = axes[0, 0], axes[0, 1]
    ax_hx, ax_hy = axes[1, 0], axes[1, 1]
    ax_dx, ax_off = axes[2, 0], axes[2, 1]
    ax_off.set_visible(False)

    for u, v in seam_segments(thw, t):
        ax_th.plot(v, u, color="C0", lw=1.0)
    ax_th.plot(t[0], thw[0], "s", color="C0", ms=5)
    ax_th.set_xlabel("time (s)")
    ax_th.set_ylabel("heading (rad)")
    ax_th.set_title("heading vs time")
    style_heading_axis(ax_th, "y")
    ax_th.grid(alpha=0.3)

    ax_xy.plot(t, x, color="C0", lw=1.0, label="x")
    ax_xy.plot(t, y, color="C1", lw=1.0, label="y")
    ax_xy.set_xlabel("time (s)")
    ax_xy.set_ylabel("position (m)")
    ax_xy.set_title("position vs time")
    ax_xy.legend(frameon=False)
    ax_xy.grid(alpha=0.3)

    dt = float(np.median(np.diff(t))) if t.size > 1 else np.nan
    dth = savgol_deriv(np.unwrap(th), dt, savgol_window, savgol_poly)
    dx = savgol_deriv(x, dt, savgol_window, savgol_poly)
    ok = np.isfinite(dth) & np.isfinite(dx)
    ax_dx.plot(dth[ok], dx[ok], color="C0", lw=0.8, alpha=0.7)
    if ok.any():
        i0 = int(np.flatnonzero(ok)[0])
        ax_dx.plot(dth[i0], dx[i0], "s", color="C0", ms=5)
    ax_dx.set_xlabel(r"$d\theta/dt$ (rad/s)")
    ax_dx.set_ylabel(r"$dx/dt$ (m/s)")
    ax_dx.set_title("dx vs dheading")
    ax_dx.grid(alpha=0.3)
    ax_dx.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax_dx.axvline(0.0, color="0.75", lw=0.8, zorder=0)

    for u, v in seam_segments(thw, x):
        ax_hx.plot(u, v, color="C0", lw=1.0)
    ax_hx.plot(thw[0], x[0], "s", color="C0", ms=5)
    ax_hx.set_xlabel("heading (rad)")
    ax_hx.set_ylabel("x (m)")
    ax_hx.set_title("heading vs x")
    style_heading_axis(ax_hx, "x")
    ax_hx.grid(alpha=0.3)

    for u, v in seam_segments(thw, y):
        ax_hy.plot(u, v, color="C0", lw=1.0)
    ax_hy.plot(thw[0], y[0], "s", color="C0", ms=5)
    ax_hy.set_xlabel("heading (rad)")
    ax_hy.set_ylabel("y (m)")
    ax_hy.set_title("heading vs y")
    style_heading_axis(ax_hy, "x")
    ax_hy.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def rail_frame(x, y):
    """Rail angle (principal axis of x, y; +x-ish orientation) and coordinate along it."""
    P = np.c_[x - x.mean(), y - y.mean()]
    _, V = np.linalg.eigh(P.T @ P)
    u = V[:, 1] if V[0, 1] >= 0 else -V[:, 1]
    return float(np.arctan2(u[1], u[0])), P @ u


def split_strokes(s, v, min_frac):
    """(start, stop) index pairs of constant-sign velocity spanning ≥ min_frac of the rail."""
    cuts = np.flatnonzero(np.diff(np.sign(v)) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(s)]])
    span = np.nanmax(s) - np.nanmin(s)
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:])
            if b - a >= 5 and abs(s[b - 1] - s[a]) >= min_frac * span]


def first_reach(dist, apsi, level):
    """Distance at which |ψ| first drops to ≤ level (NaN if never)."""
    hit = np.flatnonzero(apsi <= level)
    return float(dist[hit[0]]) if hit.size else np.nan


def analyze_strokes(t, x, y, th, trail, min_frac, savgol_window, savgol_poly):
    """Per-stroke flip metrics for one track (first stroke dropped). Lengths in m."""
    rail, s = rail_frame(x, y)
    dt = float(np.median(np.diff(t)))
    v = savgol_deriv(s, dt, savgol_window, savgol_poly)
    th_u = np.unwrap(th)
    rows = []
    for k, (a, b) in enumerate(split_strokes(s, v, min_frac)[1:], start=1):
        direction = 1 if s[b - 1] > s[a] else -1
        alpha = rail if direction > 0 else rail + np.pi
        psi = wrap_angle(th[a:b] - alpha - trail)
        apsi = np.abs(psi)
        dist = np.abs(s[a:b] - s[a])
        dtheta = th_u[b - 1] - th_u[a]
        flip = abs(dtheta) > np.pi / 2
        sense = int(np.sign(dtheta)) if flip else 0
        row = dict(stroke=k, direction=direction, flip=flip, sense=sense,
                   dtheta=dtheta, length=float(dist[-1]), dist=dist, apsi=apsi,
                   s10=np.nan, s50=np.nan, s90=np.nan, l_eff=np.nan,
                   end_lag=np.nan, remaining=np.nan, pred_lag=np.nan)
        if flip:
            row["s10"] = first_reach(dist, apsi, 0.9 * np.pi)
            row["s50"] = first_reach(dist, apsi, 0.5 * np.pi)
            row["s90"] = first_reach(dist, apsi, 0.1 * np.pi)
            m = (apsi > 0.1 * np.pi) & (apsi < 0.9 * np.pi)
            if np.isfinite(row["s90"]):
                m &= dist <= row["s90"]
            if m.sum() >= 5:
                slope = np.polyfit(dist[m], np.log(np.tan(apsi[m] / 2)), 1)[0]
                if slope < 0:
                    row["l_eff"] = -1.0 / slope
            tail = dist >= 0.95 * dist[-1]
            row["end_lag"] = float(-sense * np.median(psi[tail]))
            if np.isfinite(row["l_eff"]) and np.isfinite(row["s90"]):
                row["remaining"] = (dist[-1] - row["s90"]) / row["l_eff"]
                row["pred_lag"] = 2 * np.arctan(np.tan(0.05 * np.pi) * np.exp(-row["remaining"]))
        rows.append(row)
    return rows


def binned(xs, ys, edges):
    """Median and IQR of pooled (xs, ys) in bins; NaN where a bin has < 3 samples."""
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    idx = np.digitize(xs, edges) - 1
    n = len(edges) - 1
    med, lo, hi = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        sel = ys[idx == i]
        if sel.size >= 3:
            lo[i], med[i], hi[i] = np.percentile(sel, [25, 50, 75])
    return 0.5 * (edges[:-1] + edges[1:]), med, lo, hi


def pct(vals):
    """(median, lower err, upper err) from the IQR of finite values."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan, np.nan
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return q2, q2 - q1, q3 - q2


def group_summary(strokes):
    """Per-offset scalars from a list of stroke dicts (all tracks/files of that offset)."""
    flips = [r for r in strokes if r["flip"]]
    same = []
    for prev, cur in zip(strokes[:-1], strokes[1:]):
        if (prev["flip"] and cur["flip"] and prev["key"] == cur["key"]
                and cur["stroke"] == prev["stroke"] + 1):
            same.append(prev["sense"] == cur["sense"])
    n_cyc = 0.5 * len(strokes)
    net = sum(r["dtheta"] for r in strokes) / (2 * np.pi) / n_cyc if n_cyc else np.nan
    return dict(n_strokes=len(strokes), n_flips=len(flips),
                same_sense=float(np.mean(same)) if same else np.nan,
                turns_per_cycle=net,
                **{k: pct([r[k] for r in flips])
                   for k in ("s10", "s90", "l_eff", "end_lag", "remaining", "pred_lag")},
                flip_len=pct([r["s90"] - r["s10"] for r in flips]))


def plot_offset_summary(groups, path, dpi):
    """groups: {offset_mm: [stroke dicts]} → 2×3 summary figure."""
    offs = sorted(groups)
    cmap = plt.get_cmap("viridis")
    color = {o: cmap(i / max(len(offs) - 1, 1)) for i, o in enumerate(offs)}
    summ = {o: group_summary(groups[o]) for o in offs}
    L = np.array(offs, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    # A: |ψ| vs distance since reversal.
    dmax = max(r["length"] for o in offs for r in groups[o])
    edges = np.linspace(0, dmax, 80)
    for o in offs:
        rs = groups[o]
        c, med, lo, hi = binned([r["dist"] for r in rs], [r["apsi"] for r in rs], edges)
        ax_a.fill_between(1e3 * c, lo, hi, color=color[o], alpha=0.2, lw=0)
        ax_a.plot(1e3 * c, med, color=color[o], lw=1.6, label=f"l = {o:g} mm")
    ax_a.set_xlabel("distance since reversal (mm)")
    ax_a.set_ylabel(r"$|\psi|$ from trailing-aligned (rad)")
    ax_a.set_title("reorientation after each reversal")
    style_heading_axis(ax_a, "y")
    ax_a.set_ylim(0, np.pi)
    ax_a.legend(frameon=False, fontsize=8)

    # B: collapse with each offset's median fitted l_eff (one scale per offset, not per
    #    stroke); kinematic prediction 2 arctan(e^{-u}).
    u_edges = np.linspace(-6, 8, 90)
    for o in offs:
        le = summ[o]["l_eff"][0]
        rs = [r for r in groups[o] if np.isfinite(r["s50"])]
        if not rs or not np.isfinite(le):
            continue
        c, med, lo, hi = binned([(r["dist"] - r["s50"]) / le for r in rs],
                                [r["apsi"] for r in rs], u_edges)
        ax_b.fill_between(c, lo, hi, color=color[o], alpha=0.15, lw=0)
        ax_b.plot(c, med, color=color[o], lw=1.6)
    uu = np.linspace(-6, 8, 300)
    ax_b.plot(uu, 2 * np.arctan(np.exp(-uu)), "k--", lw=1.2,
              label=r"no-slip: $2\,\arctan e^{-u}$")
    ax_b.set_xlabel(r"$u = (s - s_{50})\,/\,l_{\rm eff}$")
    ax_b.set_ylabel(r"$|\psi|$ (rad)")
    ax_b.set_title(r"scaled by fitted $l_{\rm eff}$")
    style_heading_axis(ax_b, "y")
    ax_b.set_ylim(0, np.pi)
    ax_b.legend(frameon=False, fontsize=8)

    def errplot(ax, key, scale, **kw):
        m = np.array([summ[o][key][0] for o in offs]) * scale
        e = np.array([[summ[o][key][1] for o in offs], [summ[o][key][2] for o in offs]]) * scale
        ax.errorbar(L, m, yerr=e, fmt="o-", capsize=4, ms=6, lw=1.4, **kw)
        return m

    # C: characteristic lengths.
    errplot(ax_c, "s10", 1e3, label="lingering: reversal → 10 %")
    errplot(ax_c, "flip_len", 1e3, label="flip: 10 → 90 %")
    ax_c.set_xlabel("caster offset l (mm)")
    ax_c.set_ylabel("distance (mm)")
    ax_c.set_title("flip lengths (median, IQR over strokes)")
    ax_c.legend(frameon=False, fontsize=8)

    # D: effective offset from the ln tan(|ψ|/2) slope.
    m = errplot(ax_d, "l_eff", 1e3, color="C2", label=r"$l_{\rm eff}$ from fit")
    ok = np.isfinite(m) & (L > 0)
    lim = np.array([0, max(L.max(), np.nanmax(m) if ok.any() else L.max()) * 1.05])
    ax_d.plot(lim, lim, "k--", lw=1.0, label=r"$l_{\rm eff} = l$")
    if ok.sum() >= 2:
        a, b = np.polyfit(L[ok], m[ok], 1)
        ax_d.plot(lim, a * lim + b, color="C2", lw=0.9, alpha=0.6,
                  label=f"fit: {a:.2f} l + {b:.1f} mm")
    ax_d.set_xlabel("caster offset l (mm)")
    ax_d.set_ylabel(r"$l_{\rm eff}$ (mm)")
    ax_d.set_title(r"effective offset: $\tan(|\psi|/2) \propto e^{-s/l_{\rm eff}}$")
    ax_d.legend(frameon=False, fontsize=8)

    # E: flip sense — spinning vs swinging.
    same = np.array([summ[o]["same_sense"] for o in offs])
    turns = np.array([summ[o]["turns_per_cycle"] for o in offs])
    ax_e.plot(L, same, "o-", color="C3", ms=6, lw=1.4)
    ax_e.set_ylim(-0.05, 1.05)
    ax_e.set_xlabel("caster offset l (mm)")
    ax_e.set_ylabel("same-sense fraction", color="C3")
    ax_e.set_title("1 = spins continuously, 0 = swings back and forth")
    ax_e2 = ax_e.twinx()
    ax_e2.plot(L, turns, "s--", color="0.4", ms=5, lw=1.0)
    ax_e2.set_ylabel("net turns per drive cycle", color="0.4")

    # F: alignment at the end of the stroke — measured vs no-slip prediction.
    errplot(ax_f, "end_lag", 180 / np.pi, color="C4", label="measured")
    errplot(ax_f, "pred_lag", 180 / np.pi, color="0.4", ls="--",
            label=r"no-slip: $2\arctan[\tan(0.05\pi)\,e^{-(L-s_{90})/l_{\rm eff}}]$")
    ax_f.set_yscale("log", nonpositive="clip")
    ax_f.set_xlabel("caster offset l (mm)")
    ax_f.set_ylabel("end-of-stroke lag (deg)")
    ax_f.set_title("alignment at reversal")
    ax_f.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("caster reorientation vs offset")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return summ


def draw_caster_schematic(ax):
    """Top view: pivot moving at v along the rail, wheel trailing at offset l, angle γ."""
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
    from matplotlib.transforms import Affine2D

    g = np.radians(35)                      # drawn caster angle
    L = 1.0                                 # drawn offset
    arm = np.array([-np.cos(g), np.sin(g)])  # pivot → wheel contact
    W = L * arm

    ax.plot([-1.9, 1.6], [0, 0], color="0.75", lw=5, solid_capstyle="round", zorder=0)
    ax.text(1.62, -0.13, "rail", ha="right", va="top", color="0.5")
    ax.plot([0, -1.45], [0, 0], color="0.3", lw=0.8, ls="--", zorder=1)
    ax.plot([0, W[0]], [0, W[1]], color="k", lw=1.6, zorder=2)
    wheel = Rectangle((-0.28, -0.09), 0.56, 0.18, facecolor="0.25", edgecolor="k", lw=0.8,
                      zorder=3)
    wheel.set_transform(Affine2D().rotate(np.pi - g).translate(*W) + ax.transData)
    ax.add_patch(wheel)
    ax.add_patch(Circle((0, 0), 0.07, facecolor="white", edgecolor="k", lw=1.2, zorder=4))
    ax.add_patch(FancyArrowPatch((0.12, 0), (0.95, 0), arrowstyle="-|>", mutation_scale=10,
                                 color="C3", lw=1.4, zorder=4))
    ax.text(0.55, 0.08, r"$v$", color="C3", ha="center", va="bottom")
    ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=180 - np.degrees(g), theta2=180,
                     color="C0", lw=1.2))
    ax.text(-0.62, 0.13, r"$\gamma$", color="C0", ha="center", va="center")
    mid = 0.5 * W + 0.13 * np.array([np.sin(g), np.cos(g)])
    ax.text(*mid, r"$l$", ha="center", va="center")
    ax.text(-0.15, -0.5, r"$l\,\dfrac{d\gamma}{ds} = -\sin\gamma$", ha="center", va="center",
            fontsize=9)
    ax.text(-0.15, -1.15, r"$\gamma = 0$: trailing (stable)" "\n"
            r"$\gamma = \pi$: leading (unstable)", ha="center", va="center", fontsize=7,
            color="0.3", linespacing=1.4)
    ax.set_xlim(-2.0, 1.7)
    ax.set_ylim(-1.5, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_paper_figure(groups, outbase, dpi, width):
    """Three panels: schematic, |γ| vs distance since reversal, reorientation length vs l."""
    offs = sorted(groups)
    summ = {o: group_summary(groups[o]) for o in offs}
    cmap = plt.get_cmap("viridis")
    color = {o: cmap(i / max(len(offs) - 1, 1)) for i, o in enumerate(offs)}
    rc = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
          "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, (ax_s, ax_g, ax_r) = plt.subplots(
            1, 3, figsize=(width, 0.34 * width), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
        draw_caster_schematic(ax_s)

        dmax = max(r["length"] for o in offs for r in groups[o])
        edges = np.linspace(0, dmax, 80)
        for o in offs:
            rs = groups[o]
            c, med, lo, hi = binned([r["dist"] for r in rs], [r["apsi"] for r in rs], edges)
            ax_g.fill_between(1e3 * c, lo, hi, color=color[o], alpha=0.2, lw=0)
            ax_g.plot(1e3 * c, med, color=color[o], lw=1.3, label=f"{o:g}")
        ax_g.set_xlim(0, 1e3 * dmax)
        ax_g.set_ylim(0, np.pi * 1.02)
        ax_g.set_yticks([0, np.pi / 2, np.pi])
        ax_g.set_yticklabels(["$0$", r"$\pi/2$", r"$\pi$"])
        ax_g.set_xlabel("distance since reversal (mm)")
        ax_g.set_ylabel(r"$|\gamma|$")
        ax_g.legend(title=r"$l$ (mm)", frameon=False, ncol=2, loc="upper right",
                    handlelength=1.2, columnspacing=0.8, title_fontsize=7)

        L = np.array([o for o in offs if np.isfinite(summ[o]["flip_len"][0])], dtype=float)
        m = np.array([summ[o]["flip_len"][0] for o in L]) * 1e3
        e = np.array([[summ[o]["flip_len"][1] for o in L],
                      [summ[o]["flip_len"][2] for o in L]]) * 1e3
        for o, mi, e0, e1 in zip(L, m, e[0], e[1]):
            ax_r.errorbar(o, mi, yerr=[[e0], [e1]], fmt="o", ms=3, capsize=3, lw=1.0,
                          capthick=1.0, color=color[o], ecolor="k", mec="k", mew=0.4,
                          zorder=3)
        if L.size >= 2:
            a, b = np.polyfit(L, m, 1)
            xx = np.array([L.min(), L.max()])      # fit valid only over measured offsets
            ax_r.plot(xx, a * xx + b, color="0.4", lw=0.9, ls="--", zorder=1,
                      label=f"{a:.1f}" r"$\,l$" f" + {b:.0f} mm")
            ax_r.legend(frameon=False, loc="upper left")
        if any(o not in L for o in offs):
            ax_r.annotate(r"$l = 0$: no reorientation", xy=(0, 0.04), xycoords=("data", "axes fraction"),
                          fontsize=6.5, color="0.35", ha="left", va="bottom")
        ax_r.set_xlim(-1, max(offs) * 1.08)
        ax_r.set_ylim(0, None)
        ax_r.set_xlabel(r"caster offset $l$ (mm)")
        ax_r.set_ylabel("reorientation length (mm)")

        for ax in (ax_g, ax_r):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.95))
        fig.canvas.draw()                   # resolve the schematic's equal-aspect box
        top = ax_g.get_position().y1 + 0.02
        for ax, lab in zip((ax_s, ax_g, ax_r), "abc"):
            x0 = ax.get_position().x0 - (0.0 if ax is ax_s else 0.06)
            fig.text(x0, top, f"({lab})", fontweight="bold", ha="left", va="bottom")
        fig.savefig(outbase + ".pdf")
        fig.savefig(outbase + ".png", dpi=max(dpi, 300))
        plt.close(fig)


def write_strokes_csv(groups, path):
    cols = ["file", "track", "offset_mm", "stroke", "direction", "flip", "sense",
            "length_mm", "s10_mm", "s50_mm", "s90_mm", "l_eff_mm", "end_lag_deg", "remaining",
            "pred_lag_deg"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for o in sorted(groups):
            for r in groups[o]:
                vals = [r["file"], r["track"], o, r["stroke"], r["direction"], int(r["flip"]),
                        r["sense"], 1e3 * r["length"], 1e3 * r["s10"], 1e3 * r["s50"],
                        1e3 * r["s90"], 1e3 * r["l_eff"], np.degrees(r["end_lag"]),
                        r["remaining"], np.degrees(r["pred_lag"])]
                fh.write(",".join(v if isinstance(v, str) else f"{v:g}" for v in vals) + "\n")


def main():
    args = parse_args()
    files = collect_csvs(args.inputs)
    if not files:
        raise SystemExit("No *_robot.csv files found.")
    if args.outdir:
        outdir = args.outdir
    elif len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        outdir = os.path.join(args.inputs[0], "kinematics_plots")
    else:
        outdir = os.path.splitext(files[0])[0] + "_kinematics"
    os.makedirs(outdir, exist_ok=True)

    n_fig = 0
    groups = {}   # offset_mm -> list of stroke dicts
    l_re = re.compile(args.l_regex)
    for path in files:
        prefix = file_prefix(path)
        df, nid, th_col, x_col, y_col = load_robot(path)
        tracks = sorted(df["track"].dropna().unique())
        m = l_re.search(prefix)
        offset = float(m.group(1)) * args.l_step if m else None
        print(f"{os.path.basename(path)}: {len(tracks)} track(s)"
              + (f", offset {offset:g} mm" if offset is not None else ", no offset in name"))
        for tr_id, sub in df.groupby("track", sort=True):
            t = sub["time"].to_numpy(dtype=float)
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            th = sub[th_col].to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(th)
            if ok.sum() < 2:
                print(f"  skip track {int(tr_id)} (too few samples)")
                continue
            t, x, y, th = t[ok], x[ok], y[ok], th[ok]
            if not args.summary_only:
                fname = f"{prefix}_track{int(tr_id):02d}.png"
                out = os.path.join(outdir, fname)
                plot_track(t, x, y, th, f"{prefix}  track {int(tr_id)}", out, args.dpi,
                           args.savgol_window, args.savgol_poly)
                n_fig += 1
                print(f"  wrote {fname}")
            if args.no_summary or offset is None:
                continue
            rows = analyze_strokes(t, x, y, th, args.trail_offset, args.min_stroke_frac,
                                   args.savgol_window, args.savgol_poly)
            for r in rows:
                r.update(file=prefix, track=int(tr_id), key=(prefix, int(tr_id)))
            groups.setdefault(offset, []).extend(rows)

    if groups:
        summ = plot_offset_summary(groups, os.path.join(outdir, "kinematics_vs_offset.png"),
                                   args.dpi)
        write_strokes_csv(groups, os.path.join(outdir, "kinematics_strokes.csv"))
        plot_paper_figure(groups, os.path.join(outdir, "caster_figure"), args.dpi,
                          args.paper_width)
        n_fig += 2
        print("\n  l(mm)  strokes flips  same-sense turns/cyc  s10(mm)  flip(mm)  "
              "l_eff(mm)  end lag(deg)  pred lag(deg)  remaining")
        for o in sorted(summ):
            s = summ[o]
            print(f"  {o:5g}  {s['n_strokes']:7d} {s['n_flips']:5d}  {s['same_sense']:10.2f} "
                  f"{s['turns_per_cycle']:+9.2f}  {1e3 * s['s10'][0]:7.1f}  "
                  f"{1e3 * s['flip_len'][0]:8.1f}  {1e3 * s['l_eff'][0]:9.1f}  "
                  f"{np.degrees(s['end_lag'][0]):12.2f}  {np.degrees(s['pred_lag'][0]):13.3g}  "
                  f"{s['remaining'][0]:9.1f}")
        print("  wrote kinematics_vs_offset.png, kinematics_strokes.csv, "
              "caster_figure.pdf/.png")

    print(f"\nWrote {n_fig} figure(s) to {outdir}/")


if __name__ == "__main__":
    main()
