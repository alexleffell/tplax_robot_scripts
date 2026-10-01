#!/usr/bin/env python3
"""
Plot the single-node table produced by format_tracks_single.py.

Reads one or more formatted ``*_robot.csv`` files (tracks = trials; each file is a
regime). Lab-frame figures 01–03d; reduced (r, psi) figures 04–12.

Notation and style follow the paper figures (plot_kinematics_tracks.py, plot_orbit_vs_offset.py):
γ is the lab-frame caster heading, ψ the heading relative to the radial direction
(reduce_single.py; ψ = 0 radially inward), positions in mm, CW (γ̇ < 0) blue / CCW red.
Titles give the parameters parsed from the file name (``m<pwm>_l<l>_d<d>``): offset
l = index × --l-step mm, perpendicular offset d = index × --d-step mm, motor PWM.

Example
-------
    python plot_analysis_single.py ../Data/300926/phase_portrait/m250_l4_d0_robot.csv
    python plot_analysis_single.py a_robot.csv b_robot.csv --pdf
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.markers import MarkerStyle
from scipy.fft import rfft, rfftfreq

from format_tracks import read_csv_comments, circ_mean
from reduce_single import (
    Trial, bin_field, extract_fixed_points, fit_center_joint, geometric_circle,
    mean_phidot_trial, reduce_trial, savgol_deriv, split_on_gaps, symmetry_residual,
    terminal_state, wrap, wrap_grid_for_contour, circ_std,
)


TRACK_COLOR = "C0"
TRACK_ALPHA = 0.2
MIRROR_COLOR = "C3"
PSI_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PSI_TICKLABELS = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
GAMMA_LABEL = r"heading $\gamma$ (rad)"
GDOT_LABEL = r"$\dot\gamma$ (rad/s)"


def parse_args():
    p = argparse.ArgumentParser(description="Plot format_tracks_single.py output")
    p.add_argument("robot_csv", nargs="+", help="Formatted CSV(s) from format_tracks_single.py")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Regime labels, one per CSV. Default: parameters parsed from the "
                        "file name (l, d in mm, PWM), else the basename.")
    p.add_argument("--outdir", type=str, default=None, help="Output dir. Default: <first csv>_plots/")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    p.add_argument("--pdf", action="store_true", help="Also write a PDF of every figure.")
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per l index (mm). Default: 5")
    p.add_argument("--d-step", type=float, default=6.43,
                   help="Perpendicular offset per d index (mm). Default: 6.43")
    p.add_argument("--provenance", action="store_true",
                   help="Stamp the source file name (top left) and heading source / frame "
                        "(bottom right) on every figure.")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("XC", "YC"),
                   help="Well center (m). Default: geometric circle fit (centroid fallback).")
    p.add_argument("--tag-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("A", "DELTA"),
                   help="Tag-to-pivot radius A (m) and phase delta (rad). Default: 0 0.")
    p.add_argument("--savgol-window", type=float, default=0.35,
                   help="Savitzky–Golay window (s) for derivatives. Default: 0.35.")
    p.add_argument("--savgol-poly", type=int, default=3, help="Savitzky–Golay polyorder. Default: 3.")
    p.add_argument("--r-min", type=float, default=None,
                   help="Mask r below this (m). Default: max(5 mm, 5 * position noise).")
    p.add_argument("--settle", type=float, default=None,
                   help="T_settle (s) applied to every trial. Default: auto from r relaxation.")
    p.add_argument("--n-min", type=int, default=20, help="Min samples per (psi, r) bin. Default: 20.")
    p.add_argument("--n-psi", type=int, default=50, help="psi bins. Default: 50.")
    p.add_argument("--n-r", type=int, default=35, help="r bins. Default: 35.")
    p.add_argument("--arrow-dt", type=float, default=0.5,
                   help="Seconds between reduced-portrait arrowheads. Default: 0.5.")
    return p.parse_args()


def file_prefix(path):
    """Basename with a trailing ``_robot.csv`` (or ``_robot``) stripped."""
    base = os.path.basename(path)
    if base.endswith("_robot.csv"):
        return base[:-len("_robot.csv")]
    stem, _ = os.path.splitext(base)
    if stem.endswith("_robot"):
        return stem[:-len("_robot")]
    return stem


def param_label(prefix, l_step, d_step):
    """'$l$ = 20 mm, $d$ = 0 mm, PWM 250' from tokens l<i>, d<i>, m<pwm>; else the prefix."""
    m = l = d = None
    for tk in prefix.split("_"):
        if re.fullmatch(r"m\d+", tk):
            m = int(tk[1:])
        elif re.fullmatch(r"l\d+", tk):
            l = int(tk[1:])
        elif re.fullmatch(r"d\d+", tk):
            d = int(tk[1:])
    parts = []
    if l is not None:
        parts.append(f"$l$ = {round(l * l_step, 1):g} mm")
    if d is not None:
        parts.append(f"$d$ = {round(d * d_step, 1):g} mm")
    if m is not None:
        parts.append(f"PWM {m}")
    return ", ".join(parts) if parts else prefix


def savefig(fig, outdir, name, dpi, saved, provenance="", tight=True, src="", pdf=False):
    """Save as <name>__<src>.png (and .pdf). The provenance text is drawn only if given."""
    if src:
        tag = "".join(c if (c.isalnum() or c in "-_+.") else "_" for c in src)
        root, ext = os.path.splitext(name)
        name = f"{root}__{tag}{ext}"
    if provenance:
        fig.text(0.005, 0.995, src, ha="left", va="top", fontsize=6,
                 color="0.3", family="monospace")
        fig.text(0.995, 0.005, provenance, ha="right", va="bottom", fontsize=6,
                 color="0.4", family="monospace")
    path = os.path.join(outdir, name)
    if tight:
        fig.tight_layout(rect=(0, 0.02, 1, 0.97) if provenance else (0, 0, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  wrote {path}")


def end_arrow(ax, x, y, color, n_avg=25, alpha=0.4):
    """Arrow-head marker at the last point, oriented along the trajectory in display space."""
    dth = np.abs(np.diff(x))
    ok = np.where(dth < np.pi)[0]
    if ok.size == 0:
        return
    use = ok[-n_avg:] if ok.size >= n_avg else ok
    p0 = ax.transData.transform(np.column_stack([x[use], y[use]]))
    p1 = ax.transData.transform(np.column_stack([x[use + 1], y[use + 1]]))
    v = (p1 - p0).mean(axis=0)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return
    angle = np.degrees(np.arctan2(v[1], v[0]))
    marker = MarkerStyle(">")
    marker._transform.rotate_deg(angle)
    ax.plot(x[-1], y[-1], marker=marker, color=color, alpha=alpha, markersize=7,
            markeredgecolor="k", markeredgewidth=0.4, linestyle="None", zorder=5)


def heading_zero_sample(th, *series):
    """Interpolate each series at wrapped-heading zero crossings (branch-cut jumps excluded)."""
    th = np.asarray(th, dtype=float)
    series = [np.asarray(s, dtype=float) for s in series]
    empty = tuple(np.array([], dtype=float) for _ in series)
    if th.size < 2:
        return empty
    dth = np.diff(th)
    s0, s1 = th[:-1], th[1:]
    crosses = (np.abs(dth) < np.pi) & (s0 * s1 <= 0.0) & ~((s0 == 0.0) & (s1 == 0.0))
    cols = [[] for _ in series]
    for i in np.where(crosses)[0]:
        denom = s1[i] - s0[i]
        if denom == 0.0:
            continue
        frac = (0.0 - s0[i]) / denom
        for k, s in enumerate(series):
            cols[k].append(s[i] + frac * (s[i + 1] - s[i]))
    return tuple(np.asarray(c, dtype=float) for c in cols)


def heading_series(df, nid):
    """Lab heading (rad), preferring {id}_theta then body_angle."""
    col = f"{nid}_theta" if f"{nid}_theta" in df.columns else "body_angle"
    return pd_to_float(df[col])


def pd_to_float(series):
    return np.asarray(series, dtype=float)


def y_series(df, nid):
    col = f"{nid}_y" if f"{nid}_y" in df.columns else "centroid_y"
    return pd_to_float(df[col])


def x_series(df, nid):
    col = f"{nid}_x" if f"{nid}_x" in df.columns else "centroid_x"
    return pd_to_float(df[col])


def load_csv(path):
    df = read_csv_comments(path)
    meta = df.attrs
    nodes = meta.get("nodes")
    if nodes is None:
        nodes = [int(c[:-2]) for c in df.columns if c.endswith("_x") and not c.startswith("extra")]
    nid = int(meta.get("node_id", nodes[0]))
    if "track" not in df.columns:
        df = df.copy()
        df["track"] = 1
    return df, meta, nid


def style_psi_axis(ax):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(PSI_TICKS)
    ax.set_xticklabels(PSI_TICKLABELS)
    ax.axvline(0.0, color="0.75", lw=0.8, zorder=0)


def polyline_seam(psi, r):
    """Split (psi, r) at the ±π seam. Returns list of (psi, r) segments."""
    if len(psi) < 2:
        return [(psi, r)] if len(psi) else []
    cuts = np.where(np.abs(np.diff(psi)) > np.pi)[0]
    segs, start = [], 0
    for c in cuts:
        segs.append((psi[start:c + 1], r[start:c + 1]))
        start = c + 1
    segs.append((psi[start:], r[start:]))
    return [(a, b) for a, b in segs if len(a) >= 2]


def add_time_arrows(ax, t, psi, r_mm, dt_arrow, color, alpha=0.5):
    """Arrowheads along a reduced trajectory at fixed time intervals."""
    if t.size < 2 or dt_arrow <= 0:
        return
    t_grid = np.arange(t[0] + dt_arrow, t[-1], dt_arrow)
    if t_grid.size == 0:
        return
    # unwrap psi for interpolation, then wrap
    psi_u = np.unwrap(psi)
    pu = np.interp(t_grid, t, psi_u)
    ru = np.interp(t_grid, t, r_mm)
    dpu = np.interp(t_grid, t, np.gradient(psi_u, t))
    dru = np.interp(t_grid, t, np.gradient(r_mm, t))
    ax.quiver(wrap(pu), ru, dpu, dru, color=color, alpha=alpha, angles="xy",
              scale_units="xy", scale=8.0, width=0.003, minlength=0.4, zorder=4)


def lookup_field(field, psi, r, which="rdot"):
    M = {"rdot": field.rdot, "psidot": field.psidot, "phidot": field.phidot}[which]
    ip = np.clip(np.digitize(wrap(psi), field.psi_edges) - 1, 0, len(field.psi_c) - 1)
    ir = np.clip(np.digitize(r, field.r_edges) - 1, 0, len(field.r_c) - 1)
    return M[ir, ip]


def trial_list_from_df(df, nid, regime):
    trials = []
    for tr_id in sorted(df["track"].dropna().unique()):
        sub = df[df["track"] == tr_id]
        t = pd_to_float(sub["time"])
        x = x_series(sub, nid)
        y = y_series(sub, nid)
        th = heading_series(sub, nid)
        finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(th)
        t, x, y, th = t[finite], x[finite], y[finite], th[finite]
        for idx in split_on_gaps(t):
            if len(idx) < 5:
                continue
            trials.append(Trial(t=t[idx], x=x[idx], y=y[idx], theta=th[idx],
                                track=int(tr_id), regime=regime))
    return trials


def reduce_all(regimes, args, log=print):
    """Fit one global center, reduce every trial. regimes: list of (label, trials)."""
    A, delta = float(args.tag_offset[0]), float(args.tag_offset[1])
    # Pass 1: centroid center -> settle -> collect asymptotic circulating xy.
    all_xy = []
    for _, trials in regimes:
        for tr in trials:
            all_xy.append((tr.x, tr.y))
    if args.center is not None:
        xc, yc = float(args.center[0]), float(args.center[1])
        center_src = "manual"
        R_list, rms = [], np.nan
    else:
        pooled_x = np.concatenate([x for x, _ in all_xy]) if all_xy else np.array([])
        pooled_y = np.concatenate([y for _, y in all_xy]) if all_xy else np.array([])
        xc, yc = float(np.mean(pooled_x)), float(np.mean(pooled_y))
        center_src = "centroid"
        R_list, rms = [], np.nan

    r_min = args.r_min
    if r_min is None:
        # noise scale from first differences of position
        sigs = []
        for _, trials in regimes:
            for tr in trials:
                if tr.t.size < 3:
                    continue
                step = np.hypot(np.diff(tr.x), np.diff(tr.y))
                sigs.append(np.median(step) / np.sqrt(2))
        sigma = float(np.median(sigs)) if sigs else 0.001
        r_min = max(0.005, 5.0 * sigma)
    log(f"r_min = {r_min:.4g} m")

    def apply(xc, yc):
        for _, trials in regimes:
            for tr in trials:
                reduce_trial(tr, xc, yc, A, delta, r_min, args.savgol_window,
                             args.savgol_poly, T_settle=args.settle)

    apply(xc, yc)

    if args.center is None:
        circ_xy = []
        for label, trials in regimes:
            xs, ys = [], []
            for tr in trials:
                m = tr.asymptotic
                if m.sum() < 20:
                    continue
                pu = tr.phi_u[m]
                net = abs(float(pu[-1] - pu[0]))
                path = float(np.sum(np.abs(np.diff(pu))))
                if net < np.pi or path < 1e-9 or net < 0.5 * path:
                    continue
                xs.append(tr.x[m]); ys.append(tr.y[m])
            if xs:
                circ_xy.append((np.concatenate(xs), np.concatenate(ys)))

        def sane(xc, yc, R, x, y):
            span = float(np.hypot(np.ptp(x), np.ptp(y)))
            if span < 1e-6:
                return False
            if R > 3.0 * span:
                return False
            if np.hypot(xc - np.mean(x), yc - np.mean(y)) > 2.0 * span:
                return False
            return True

        fitted = False
        if len(circ_xy) == 1:
            x, y = circ_xy[0]
            xc_f, yc_f, R, rms = geometric_circle(x, y)
            if sane(xc_f, yc_f, R, x, y):
                xc, yc, R_list, rms = xc_f, yc_f, [R], rms
                fitted = True
        elif len(circ_xy) > 1:
            xc_f, yc_f, R_list, rms = fit_center_joint(circ_xy)
            x = np.concatenate([a for a, _ in circ_xy])
            y = np.concatenate([b for _, b in circ_xy])
            Rmed = float(np.median(R_list)) if R_list else 0.0
            if sane(xc_f, yc_f, Rmed, x, y):
                xc, yc = xc_f, yc_f
                fitted = True
        if fitted:
            center_src = "geometric circle"
            apply(xc, yc)
        else:
            log("WARNING: no circulating asymptotic data (or circle fit not sane); "
                "well center is the pooled centroid.")
            R_list, rms = [], np.nan

    log(f"well center ({center_src}): ({xc:.5f}, {yc:.5f}) m"
        + (f"  residual RMS={rms:.4g} m" if np.isfinite(rms) else "  (centroid fallback)"))
    if R_list:
        log("  per-regime R: " + ", ".join(f"{R:.4f}" for R in R_list) + " m")
    log(f"Savitzky–Golay: window={args.savgol_window}s  polyorder={args.savgol_poly}")
    for label, trials in regimes:
        for tr in trials:
            log(f"  {label} track {tr.track}: T_settle={tr.T_settle:.3f}s  "
                f"n_asym={int(tr.asymptotic.sum())}/{len(tr.t)}")
    return xc, yc, r_min, center_src, rms


def plot_reduced(regimes, xc, yc, args, save, r_min, src=""):
    """Figures 04–12. Shared (psi, r) limits across regimes. `src` is the parameter title
    (single regime) or the joined regime labels."""
    def st(title):
        # one regime: its parameters already head the figure; several: panel titles carry them
        return f"{src}\n{title}" if src and len(regimes) == 1 else title
    fields, fps_per = {}, {}
    r_all = []
    for label, trials in regimes:
        for tr in trials:
            r_all.append(tr.r[tr.valid])
        fields[label] = bin_field(trials, n_psi=args.n_psi, n_r=args.n_r, n_min=args.n_min)
        fps_per[label] = extract_fixed_points(fields[label], trials)
        print(f"  {label}: {len(fps_per[label])} fixed point(s)")
        for fp in fps_per[label]:
            ev = ", ".join(f"{z.real:.3g}{z.imag:+.3g}j" for z in fp.evals)
            print(f"    r*={fp.r:.4f} m  psi*={fp.psi:.3f}  Omega={fp.Omega:.3f}  "
                  f"{fp.kind}  eig={ev}")

    r_mm_max = 1000.0 * float(np.nanpercentile(np.concatenate(r_all) if r_all else [0.01], 99.5))
    r_mm_max = max(r_mm_max, 1.0)
    n_reg = len(regimes)
    fig_w = max(3.2 * n_reg, 3.6)
    one = n_reg == 1                       # single regime: no per-panel title needed

    def panels(n, extra=0):
        fig, axes = plt.subplots(1, n, figsize=(fig_w + extra, 3.4), squeeze=False)
        return fig, axes[0]

    def panel_title(ax, text):
        if not one:
            ax.set_title(text)

    # ---- 04 reduced phase portrait ----
    fig, axes = panels(n_reg)
    for ax, (label, trials) in zip(axes, regimes):
        for tr in trials:
            rmm = 1000.0 * tr.r
            # transient
            trn = tr.valid & ~tr.asymptotic
            if trn.sum() >= 2:
                for p, rr in polyline_seam(tr.psi[trn], rmm[trn]):
                    ax.plot(p, rr, color=TRACK_COLOR, alpha=0.15, lw=0.6)
            # asymptotic
            m = tr.asymptotic
            if m.sum() >= 2:
                for p, rr in polyline_seam(tr.psi[m], rmm[m]):
                    ax.plot(p, rr, color=TRACK_COLOR, alpha=0.45, lw=1.3)
                add_time_arrows(ax, tr.t[m], tr.psi[m], rmm[m], args.arrow_dt, TRACK_COLOR)
            if tr.valid.any():
                i0 = np.where(tr.valid)[0][0]
                ax.plot(tr.psi[i0], rmm[i0], "s", mfc="none", mec=TRACK_COLOR, ms=4, zorder=5)
            rt, pt = terminal_state(tr)
            if np.isfinite(rt):
                ax.plot(pt, 1000.0 * rt, "o", color=TRACK_COLOR, ms=4, zorder=5)
        style_psi_axis(ax)
        ax.set_ylim(0, r_mm_max)
        ax.set_xlabel(r"$\psi$ (rad)")
        ax.set_ylabel(r"$r$ (mm)")
        panel_title(ax, label)
    fig.suptitle(st(r"reduced phase portrait: radius $r$ vs heading relative to radial $\psi$"))
    save(fig, "04_reduced_phase_portrait.png")

    # ---- 05 vector field + nullclines + FPs ----
    fig, axes = panels(n_reg, extra=1.0)
    for ax, (label, trials) in zip(axes, regimes):
        fld = fields[label]
        Psi, R = np.meshgrid(fld.psi_c, 1000.0 * fld.r_c)
        Omega_typ = np.nanmedian(np.abs(fld.phidot))
        if not np.isfinite(Omega_typ) or Omega_typ < 1e-6:
            Omega_typ = 1.0
        r_char = np.nanmedian(fld.r_c)
        U = fld.psidot / Omega_typ
        V = (fld.rdot / (r_char * Omega_typ)) * (1000.0 * r_char)   # mm / (char time)
        speed = np.hypot(U, V)
        if np.isfinite(speed).any():
            im = ax.pcolormesh(fld.psi_edges, 1000.0 * fld.r_edges, speed,
                               cmap="Greys", shading="flat", alpha=0.55,
                               vmin=0, vmax=np.nanpercentile(speed[np.isfinite(speed)], 95))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|v| (nondim)")
        skip = (slice(None, None, 2), slice(None, None, 2))
        Uq = np.where(fld.mask, U, np.nan)
        Vq = np.where(fld.mask, V, np.nan)
        ax.quiver(Psi[skip], R[skip], Uq[skip], Vq[skip],
                  color="k", angles="xy", scale_units="xy", scale=25, width=0.003,
                  minlength=0.0)
        psi_p, r_p, F_p, G_p, _ = wrap_grid_for_contour(fld)
        if np.isfinite(F_p).sum() > 4:
            ax.contour(psi_p, 1000.0 * r_p, F_p, levels=[0], colors="C1", linewidths=1.4)
        if np.isfinite(G_p).sum() > 4:
            ax.contour(psi_p, 1000.0 * r_p, G_p, levels=[0], colors="C2", linewidths=1.4)
        for fp in fps_per[label]:
            ax.plot(fp.psi, 1000.0 * fp.r, "D", color="C3", ms=5, zorder=6)
            ax.annotate(fp.kind, (fp.psi, 1000.0 * fp.r), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="C3")
        style_psi_axis(ax)
        ax.set_ylim(0, r_mm_max)
        ax.set_xlabel(r"$\psi$ (rad)")
        ax.set_ylabel(r"$r$ (mm)")
        panel_title(ax, label)
    fig.suptitle(st(r"reduced vector field (orange $\dot r=0$, green $\dot\psi=0$)"))
    save(fig, "05_vector_field.png")

    # ---- 06 drift map phidot ----
    fig, axes = panels(n_reg, extra=1.0)
    ph_vals = np.concatenate([fld.phidot[np.isfinite(fld.phidot)]
                              for fld in fields.values()]) if any(
        np.isfinite(f.phidot).any() for f in fields.values()) else np.array([1.0])
    vmax = float(np.nanpercentile(np.abs(ph_vals), 95)) if ph_vals.size else 1.0
    vmax = max(vmax, 1e-6)
    for ax, (label, _) in zip(axes, regimes):
        fld = fields[label]
        psi_p, r_p, _, _, H_p = wrap_grid_for_contour(fld)
        im = ax.pcolormesh(fld.psi_edges, 1000.0 * fld.r_edges, fld.phidot,
                           cmap="coolwarm", shading="flat", vmin=-vmax, vmax=vmax)
        if np.isfinite(H_p).sum() > 4:
            ax.contour(psi_p, 1000.0 * r_p, H_p, levels=[0], colors="k", linewidths=1.2)
        for fp in fps_per[label]:
            ax.plot(fp.psi, 1000.0 * fp.r, "k+", ms=7, zorder=5)
            ax.annotate(fr"$\Omega$={fp.Omega:.2f}", (fp.psi, 1000.0 * fp.r),
                        textcoords="offset points", xytext=(4, 4), fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\dot\phi$ (rad/s)")
        style_psi_axis(ax)
        ax.set_ylim(0, r_mm_max)
        ax.set_xlabel(r"$\psi$ (rad)")
        ax.set_ylabel(r"$r$ (mm)")
        panel_title(ax, label)
    fig.suptitle(st(r"orbital rate $\dot\phi(r,\psi)$"))
    save(fig, "06_drift_map.png")

    # ---- 07 chirality histograms ----
    fig, axes = plt.subplots(2, n_reg, figsize=(fig_w, 5.0), squeeze=False)
    Omegas, psis = {}, {}
    for i, (label, trials) in enumerate(regimes):
        Omegas[label] = np.array([mean_phidot_trial(tr) for tr in trials], float)
        psis[label] = np.array([terminal_state(tr)[1] for tr in trials], float)
    omax = np.nanmax(np.abs(np.concatenate(list(Omegas.values())))) if Omegas else 1.0
    omax = max(float(omax) * 1.2, 0.1)
    bins_o = np.linspace(-omax, omax, 25)
    for i, (label, _) in enumerate(regimes):
        axes[0, i].hist(Omegas[label][np.isfinite(Omegas[label])], bins=bins_o,
                        color=TRACK_COLOR, alpha=0.7, edgecolor="none")
        axes[0, i].axvline(0, color="0.6", lw=0.8)
        axes[0, i].set_xlim(-omax, omax)
        panel_title(axes[0, i], label)
        axes[0, i].set_xlabel(r"$\langle\dot\phi\rangle$ (rad/s)")
        axes[1, i].hist(psis[label][np.isfinite(psis[label])], bins=np.linspace(-np.pi, np.pi, 25),
                        color=TRACK_COLOR, alpha=0.7, edgecolor="none")
        style_psi_axis(axes[1, i])
        axes[1, i].set_xlabel(r"terminal $\psi$")
        mu = circ_mean(psis[label]) if np.isfinite(psis[label]).any() else np.nan
        sd = circ_std(psis[label])
        axes[1, i].axvline(mu, color="C3", lw=1.0)
        axes[1, i].set_title(fr"circ mean {mu:.2f}, sd {sd:.2f}")
    axes[0, 0].set_ylabel("trials")
    axes[1, 0].set_ylabel("trials")
    fig.suptitle(st(r"rotation direction: mean orbital rate and terminal $\psi$ per trial"))
    save(fig, "07_chirality.png")

    # ---- 08 reflection symmetry ----
    fig, axes = panels(n_reg)
    for ax, (label, trials) in zip(axes, regimes):
        for tr in trials:
            m = tr.asymptotic
            if m.sum() < 2:
                continue
            rmm = 1000.0 * tr.r[m]
            for p, rr in polyline_seam(tr.psi[m], rmm):
                ax.plot(p, rr, color=TRACK_COLOR, alpha=0.45, lw=1.1)
            for p, rr in polyline_seam(wrap(-tr.psi[m]), rmm):
                ax.plot(p, rr, color=MIRROR_COLOR, alpha=0.25, lw=1.0)
        S = symmetry_residual(fields[label])
        style_psi_axis(ax)
        ax.set_ylim(0, r_mm_max)
        ax.set_xlabel(r"$\psi$ (rad)")
        ax.set_ylabel(r"$r$ (mm)")
        ax.set_title(fr"{label}  $S_G$={S['S_G']:.2f}  $S_H$={S['S_H']:.2f}  $S_F$={S['S_F']:.2f}")
        print(f"  {label} symmetry residual: S_G={S['S_G']:.3f}  S_H={S['S_H']:.3f}  S_F={S['S_F']:.3f}")
    fig.suptitle(st(r"reflection test (blue: data, red: $\psi\to-\psi$)"))
    save(fig, "08_reflection_symmetry.png")

    # ---- 09 lab-frame trajectories ----
    fig, axes = plt.subplots(1, n_reg, figsize=(3.4 * n_reg, 3.4), squeeze=False)
    axes = axes[0]
    for ax, (label, trials) in zip(axes, regimes):
        # up to two trials per identified attractor (nearest FP), else first two
        fps = fps_per[label]
        chosen = []
        if fps:
            used = set()
            for fp in fps:
                best, best_d = None, np.inf
                for k, tr in enumerate(trials):
                    if k in used:
                        continue
                    rt, pt = terminal_state(tr)
                    if not np.isfinite(rt):
                        continue
                    d = np.hypot(rt - fp.r, wrap(pt - fp.psi))
                    if d < best_d:
                        best, best_d = k, d
                if best is not None:
                    used.add(best)
                    chosen.append(trials[best])
        if not chosen:
            chosen = trials[:2]
        for tr in chosen:
            tnorm = (tr.t - tr.t[0]) / max(tr.t[-1] - tr.t[0], 1e-9)
            ax.scatter(1e3 * tr.x, 1e3 * tr.y, c=tnorm, cmap="viridis", s=2, linewidths=0,
                       zorder=2)
            step = max(1, int(round(0.15 / max(tr.dt, 1e-3))))
            # heading γ as 5 mm arrows (trial.theta holds γ)
            ax.quiver(1e3 * tr.x[::step], 1e3 * tr.y[::step],
                      np.cos(tr.theta[::step]), np.sin(tr.theta[::step]),
                      color="0.3", alpha=0.5, angles="xy", scale_units="xy",
                      scale=0.2, width=0.004, zorder=3)
        ax.plot(1e3 * xc, 1e3 * yc, "k+", ms=8, zorder=5)
        for fp in fps:
            circ = plt.Circle((1e3 * xc, 1e3 * yc), 1e3 * fp.r, fill=False, color="C3",
                              lw=1.0, ls="--")
            ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        panel_title(ax, label)
    fig.suptitle(st(r"trajectories (color = time, arrows = heading $\gamma$)"))
    save(fig, "09_lab_trajectories.png")

    # ---- 10 isotropy audit ----
    fig, axes = plt.subplots(1, n_reg, figsize=(fig_w, 3.4), squeeze=False)
    axes = axes[0]
    for ax, (label, trials) in zip(axes, regimes):
        fld = fields[label]
        phi, eps = [], []
        obs, pred = [], []
        for tr in trials:
            m = tr.asymptotic & np.isfinite(tr.phidot)
            if m.sum() == 0:
                continue
            pr = lookup_field(fld, tr.psi[m], tr.r[m], "phidot")
            ok = np.isfinite(pr)
            phi.append(tr.phi[m][ok])
            eps.append(tr.phidot[m][ok] - pr[ok])
            obs.append(tr.phidot[m][ok]); pred.append(pr[ok])
        if not phi:
            ax.set_title(f"{label} (no data)")
            continue
        phi, eps = np.concatenate(phi), np.concatenate(eps)
        obs, pred = np.concatenate(obs), np.concatenate(pred)
        ax.scatter(phi, eps, s=6, c=TRACK_COLOR, alpha=0.15, linewidths=0)
        # Fourier of binned residual vs phi
        pe = np.linspace(-np.pi, np.pi, 65)
        pc = 0.5 * (pe[:-1] + pe[1:])
        idx = np.clip(np.digitize(wrap(phi), pe) - 1, 0, len(pc) - 1)
        with np.errstate(all="ignore"):
            mu = np.array([np.nanmean(eps[idx == j]) if np.any(idx == j) else 0.0
                           for j in range(len(pc))])
        mu = np.nan_to_num(mu)
        spec = np.abs(rfft(mu))
        freqs = rfftfreq(len(mu), d=1.0)
        # harmonic k is bin k of rfft (period 2pi -> k=1)
        amps = {h: float(spec[h]) / max(len(mu), 1) for h in range(1, min(9, len(spec)))}
        var_o = float(np.var(obs)) or 1.0
        R2 = 1.0 - float(np.var(obs - pred)) / var_o
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(PSI_TICKS); ax.set_xticklabels(PSI_TICKLABELS)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\dot\phi$ residual")
        amp_str = " ".join(fr"$a_{h}$={amps[h]:.3g}" for h in list(amps)[:4])
        ax.set_title(fr"{label}  $R^2$={R2:.3f}  {amp_str}")
        print(f"  {label} isotropy: R^2(phidot|{'(r,psi)'})={R2:.3f}  harmonics {amps}")
    fig.suptitle(st(r"isotropy check: $\dot\phi$ residual vs $\phi$"))
    save(fig, "10_isotropy_audit.png")

    # ---- 11 reconstruction check (one longest asymptotic trial per regime) ----
    fig, axes = plt.subplots(n_reg, 2, figsize=(7.0, 2.6 * n_reg), squeeze=False)
    for i, (label, trials) in enumerate(regimes):
        tr = max(trials, key=lambda t: int(t.asymptotic.sum()), default=None)
        ax0, ax1 = axes[i]
        if tr is None or tr.asymptotic.sum() < 5:
            ax0.set_title(f"{label}: no trial")
            continue
        m = tr.valid
        t, pu, hd = tr.t[m], tr.phi_u[m], tr.phidot[m]
        # integrate phidot from the first asymptotic sample
        rec = np.full_like(pu, np.nan)
        rec[0] = pu[0]
        for k in range(1, len(t)):
            rec[k] = rec[k - 1] + 0.5 * (hd[k] + hd[k - 1]) * (t[k] - t[k - 1])
        ax0.plot(t, pu, color="k", lw=1.2, label="measured")
        ax0.plot(t, rec, color=TRACK_COLOR, lw=1.0, alpha=0.8, label="reconstructed")
        ax0.set_ylabel(r"$\phi_u$ (rad)")
        panel_title(ax0, label)
        ax0.legend(frameon=False)
        ax1.plot(t, wrap(pu - rec), color=TRACK_COLOR, lw=1.0)
        ax1.axhline(0, color="0.6", lw=0.7)
        ax1.set_ylabel(r"residual (wrapped)")
        ax0.set_xlabel("t (s)"); ax1.set_xlabel("t (s)")
    fig.suptitle(st(r"reconstruction: $\int\dot\phi\,dt$ vs measured $\phi$"))
    save(fig, "11_reconstruction.png")

    # ---- 12 fixed-point summary ----
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    labels = [lab for lab, _ in regimes]
    cmap = plt.cm.tab10
    for i, (label, _) in enumerate(regimes):
        col = cmap(i % 10)
        fps = fps_per[label]
        if not fps:
            continue
        rs = [1000.0 * fp.r for fp in fps]
        ps = [fp.psi for fp in fps]
        Om = [abs(fp.Omega) for fp in fps]
        stable = ["stable" in fp.kind for fp in fps]
        for r, p, o, is_stable in zip(rs, ps, Om, stable):   # not `st`: shadows st()
            mfc = col if is_stable else "none"
            axes[0].plot(i, r, "o", mfc=mfc, mec=col, ms=5)
            axes[1].plot(i, p, "o", mfc=mfc, mec=col, ms=5)
            axes[2].plot(i, o, "o", mfc=mfc, mec=col, ms=5)
    for ax in axes:
        ax.set_xticks(range(n_reg))
        if one:                              # the figure title already names the regime
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel(r"$r^*$ (mm)"); axes[0].set_title("radius")
    axes[1].set_ylabel(r"$\psi^*$"); axes[1].set_title(r"heading rel. radial $\psi^*$")
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_yticks(PSI_TICKS); axes[1].set_yticklabels(PSI_TICKLABELS)
    axes[2].set_ylabel(r"$|\Omega|$ (rad/s)"); axes[2].set_title("orbital rate")
    fig.suptitle(st("fixed points per regime (filled = stable)"))
    save(fig, "12_fixed_points.png")

    # braid unit-test numbers
    for label, _ in regimes:
        fps = fps_per[label]
        if len(fps) >= 1:
            fp = fps[0]
            sep = 2.0 * fp.r * np.sin(fp.psi)
            print(f"  {label} braid check: 2 r* sin(psi*) = {sep:.4f} m  "
                  f"(cluster separation at theta=0); crossings of mirror pair at theta=±π/2")


def main():
    args = parse_args()
    plt.rcParams.update(STYLE)
    labels = args.labels or [param_label(file_prefix(p), args.l_step, args.d_step)
                             for p in args.robot_csv]
    if len(labels) != len(args.robot_csv):
        raise SystemExit("--labels must have one entry per CSV")
    outdir = args.outdir or (os.path.splitext(args.robot_csv[0])[0] + "_plots")
    os.makedirs(outdir, exist_ok=True)
    saved = []

    loaded = []
    for path, lab in zip(args.robot_csv, labels):
        df, meta, nid = load_csv(path)
        loaded.append((path, lab, df, meta, nid))

    meta0 = loaded[0][3]
    nid0 = loaded[0][4]
    src = ", ".join(file_prefix(p) for p in args.robot_csv)
    title = " | ".join(labels)                      # parameter title for every figure
    provenance = (f"heading source: {meta0.get('heading_source', 'tag')}   |   "
                  f"frame: {meta0.get('frame', '?')}   |   node {nid0}")
    print(f"Source: {src}")
    print(f"Provenance: {provenance}")
    stamp = provenance if args.provenance else ""

    def save(fig, name, tight=True):
        savefig(fig, outdir, name, args.dpi, saved, provenance=stamp, tight=tight, src=src,
                pdf=args.pdf)

    # ------------------------------------------------------------------ #
    # 01–03 existing lab-frame plots (all trials overlaid, not concatenated)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ends = []
    poincare_yn, poincare_yn1, poincare_xy = [], [], []
    xyzth = []
    xytht = []
    n_section = 0
    n_tracks_total = 0
    for path, lab, df, meta, nid in loaded:
        tracks = [int(t) for t in sorted(df["track"].dropna().unique())]
        n_tracks_total += len(tracks)
        for tr in tracks:
            sub = df[df["track"] == tr]
            x, y, th = x_series(sub, nid), y_series(sub, nid), heading_series(sub, nid)
            t = pd_to_float(sub["time"])
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(th) & np.isfinite(t)
            x, y, th, t = 1e3 * x[finite], 1e3 * y[finite], th[finite], t[finite]   # mm
            if y.size < 2:
                continue
            pts = np.stack([th, y], axis=-1).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            dth = np.abs(th[1:] - th[:-1])
            segs = segs[dth < np.pi]
            lc = LineCollection(segs, colors=[TRACK_COLOR], alpha=TRACK_ALPHA, linewidths=0.8)
            ax.add_collection(lc)
            ax.plot(th[0], y[0], "s", color=TRACK_COLOR, alpha=TRACK_ALPHA, ms=4, zorder=3)
            ends.append((th, y, TRACK_COLOR))
            x0, y0 = heading_zero_sample(th, x, y)
            n_section += len(y0)
            if y0.size:
                poincare_xy.append((x0, y0))
            if y0.size >= 2:
                poincare_yn.append(y0[:-1]); poincare_yn1.append(y0[1:])
            xyzth.append((x, y, wrap(th)))
            xytht.append((t, x, y, th))
    print(f"Tracks: {n_tracks_total}")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(PSI_TICKS); ax.set_xticklabels(PSI_TICKLABELS)
    ax.autoscale(axis="y")
    ax.set_xlabel(GAMMA_LABEL)
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\nphase portrait: heading vs y")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97) if stamp else (0, 0, 1, 1))
    fig.canvas.draw()
    for th, y, color in ends:
        end_arrow(ax, th, y, color, alpha=max(TRACK_ALPHA, 0.4))
    save(fig, "01_phase_portrait_y_heading.png", tight=False)

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    if poincare_yn:
        yn = np.concatenate(poincare_yn)
        yn1 = np.concatenate(poincare_yn1)
        ax.scatter(yn, yn1, s=8, c=TRACK_COLOR, alpha=0.35, zorder=3, edgecolors="none")
        lim = np.array([np.nanmin([yn.min(), yn1.min()]), np.nanmax([yn.max(), yn1.max()])])
        pad = 0.05 * (lim[1] - lim[0] or 1.0)
        lim = lim + np.array([-pad, pad])
        ax.plot(lim, lim, color="0.5", lw=0.8, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
    else:
        ax.text(0.5, 0.5, r"no $\gamma = 0$ crossings", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y_n$ at $\gamma = 0$ (mm)")
    ax.set_ylabel(r"$y_{n+1}$ (mm)")
    ax.set_title(f"{title}\nPoincaré map at " r"$\gamma = 0$" f" ({n_section} crossings)")
    save(fig, "02_poincare_heading0.png")

    fig = plt.figure(figsize=(5.0, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    any_pts = False
    for x0, y0 in poincare_xy:
        if x0.size == 0:
            continue
        any_pts = True
        n = np.arange(len(x0), dtype=float)
        ax.plot(x0, y0, n, color=TRACK_COLOR, alpha=TRACK_ALPHA, lw=0.8)
        ax.scatter(x0, y0, n, color=TRACK_COLOR, alpha=TRACK_ALPHA, s=6,
                   depthshade=False, linewidths=0)
    if not any_pts:
        ax.text2D(0.5, 0.5, r"no $\gamma = 0$ crossings", ha="center", va="center",
                  transform=ax.transAxes, color="0.5")
    ax.set_xlabel(r"x at $\gamma = 0$ (mm)")
    ax.set_ylabel(r"y at $\gamma = 0$ (mm)")
    ax.set_zlabel("return index $n$")
    ax.set_title(f"{title}\nPoincaré map at " r"$\gamma = 0$" f" ({n_section} crossings)")
    save(fig, "03_poincare3d_heading0.png", tight=False)

    fig = plt.figure(figsize=(5.0, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    for x, y, th in xyzth:
        cuts = np.where(np.abs(np.diff(th)) > np.pi)[0]
        bounds = np.concatenate([[0], cuts + 1, [len(th)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            ax.plot(x[a:b], y[a:b], th[a:b], color=TRACK_COLOR, alpha=TRACK_ALPHA, lw=0.8)
        ax.plot([x[0]], [y[0]], [th[0]], "s", color=TRACK_COLOR, alpha=TRACK_ALPHA, ms=3)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_zticks(PSI_TICKS)
    ax.set_zticklabels(PSI_TICKLABELS)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel(r"$\gamma$ (rad)")
    ax.set_title(f"{title}\nphase portrait: (x, y, " r"$\gamma$)")
    save(fig, "03b_phase_portrait_3d.png", tight=False)

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    segs_th, td_th = [], []
    segs_xy, td_xy = [], []
    xy_starts = []
    for t, x, y, th in xytht:
        if t.size < 5:
            continue
        dt = float(np.median(np.diff(t))) if t.size > 1 else np.nan
        thdot = savgol_deriv(np.unwrap(th), dt, args.savgol_window, args.savgol_poly)
        td = 0.5 * (thdot[:-1] + thdot[1:])
        keep_th = np.abs(th[1:] - th[:-1]) < np.pi
        pts_th = np.stack([th, y], axis=-1).reshape(-1, 1, 2)
        segs_th.append(np.concatenate([pts_th[:-1], pts_th[1:]], axis=1)[keep_th])
        td_th.append(td[keep_th])
        pts_xy = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
        segs_xy.append(np.concatenate([pts_xy[:-1], pts_xy[1:]], axis=1))
        td_xy.append(td)
        xy_starts.append((x[0], y[0]))
    vmax = 1.0
    if td_th:
        td_cat = np.concatenate(td_th)
        td_cat = td_cat[np.isfinite(td_cat)]
        if td_cat.size:
            vmax = max(float(np.nanpercentile(np.abs(td_cat), 95)), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    if segs_th:
        segs = np.concatenate(segs_th, axis=0)
        td = np.concatenate(td_th)
        ok = np.isfinite(td)
        lc = LineCollection(segs[ok], cmap="RdBu_r", norm=norm, array=td[ok],
                            alpha=0.75, linewidths=0.8)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label=GDOT_LABEL)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(PSI_TICKS)
    ax.set_xticklabels(PSI_TICKLABELS)
    ax.autoscale(axis="y")
    ax.set_xlabel(GAMMA_LABEL)
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\nphase portrait: heading vs y, colored by " r"$\dot\gamma$")
    save(fig, "03c_phase_portrait_gammadot.png")

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    if segs_xy:
        segs = np.concatenate(segs_xy, axis=0)
        td = np.concatenate(td_xy)
        ok = np.isfinite(td)
        lc = LineCollection(segs[ok], cmap="RdBu_r", norm=norm, array=td[ok],
                            alpha=0.75, linewidths=0.8)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label=GDOT_LABEL)
        for x0, y0 in xy_starts:
            ax.plot(x0, y0, "s", color="k", ms=3, zorder=3)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\ntrajectories, colored by " r"$\dot\gamma$")
    save(fig, "03d_phase_portrait_xy.png")

    # ------------------------------------------------------------------ #
    # 04–12 reduced-space figures
    # ------------------------------------------------------------------ #
    regimes = []
    for path, lab, df, meta, nid in loaded:
        regimes.append((lab, trial_list_from_df(df, nid, lab)))
    n_trials = sum(len(t) for _, t in regimes)
    print(f"\nReduction: {n_trials} trial(s) in {len(regimes)} regime(s)")
    xc, yc, r_min, _, _ = reduce_all(regimes, args)
    provenance_red = (provenance + f"   |   SG {args.savgol_window}s/p{args.savgol_poly}"
                      f"   |   c=({xc:.3f},{yc:.3f})")
    stamp_red = provenance_red if args.provenance else ""

    def save_red(fig, name, tight=True):
        savefig(fig, outdir, name, args.dpi, saved, provenance=stamp_red, tight=tight, src=src,
                pdf=args.pdf)

    plot_reduced(regimes, xc, yc, args, save_red, r_min, src=title)

    print(f"\nWrote {len(saved)} figures to {outdir}/")


if __name__ == "__main__":
    main()
