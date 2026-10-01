#!/usr/bin/env python3
"""
Steady orbits of a single node in an elastic well vs caster offset l.

Input: a directory of format_tracks_single.py outputs, one continuous run each. File names
are read token by token (split on ``_``): ``l<l>`` offset index, ``d<d>`` perpendicular
offset, a 1–2 digit token = replicate number (default 1), and the --spring-tag token
(default ``s``, also ``s<rep>``) = run with the alternative spring. Examples:
``l3_d0_3_robot.csv`` (l=3, rep 3), ``l4_d0_s_robot.csv`` (l=4, new spring).
The first --settle seconds are dropped. The orbit centre is the run's mean position.

Per run
  per-revolution  the unwrapped polar angle about the centre is cut at every full turn;
                  each revolution gives a period (→ frequency) and an rms radius
  R, f            median and IQR over revolutions of the rms radius and of 1/period
                  (a typical revolution; robust to stalls). Direction = sign of the net
                  polar rotation (+ CCW, − CW)
  f_net           revolutions / total time (includes stalls)
  f_heading       net heading turns / duration (equals f_net when the heading is locked
                  to the orbit)
  stall_frac      fraction of time spent in revolutions longer than 1.5× the median period

Run selection
  main figures    original-spring runs only; for the l listed in --reps only those
                  replicates (e.g. ``--reps 2:3 3:3``), otherwise every replicate
  spring figure   the --compare-l offsets (default 4 5): the main-figure runs vs every
                  new-spring run of the same l

Outputs (to --outdir, default <directory>/orbit_plots/):
  orbit_runs.png/.pdf       one row per l: x–y trajectory, heading γ(t) over the first
                            --tmax s, and cumulative heading turns over the whole run
  orbit_vs_offset.png/.pdf  (a) R vs l, (b) f vs l; one marker per l = mean over runs of
                            the per-run median, error bar = std across runs (single run:
                            half the within-run IQR)
  orbit_vs_offset_norm.png/.pdf
                            R and f on one 0–1 axis, each divided by its value at --norm-l
                            (default: largest l); error bars scaled by the same factor
  orbit_spring_compare.png/.pdf
                            one x–y panel per compared l (original vs new spring overlaid),
                            then R vs l and f vs l for both springs
  orbit_runs.csv (all runs, with spring / used_main flags), orbit_revolutions.csv

Example
-------
    python plot_orbit_vs_offset.py ../Data/300926/orbit_radius
    python plot_orbit_vs_offset.py ../Data/300926/orbit_radius --reps 2:3 3:3 --compare-l 4 5
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from format_tracks import wrap_angle
from plot_kinematics_tracks import load_robot, seam_segments


STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}


def parse_args():
    p = argparse.ArgumentParser(description="Orbit radius and frequency vs caster offset")
    p.add_argument("directory", nargs="?",
                   default="/Users/alexleffell/Documents/PhD/tplax/Data/300926/orbit_radius",
                   help="Directory of l<l>_d<d>[_<rep>][_s]_robot.csv files. "
                        "Default: Data/300926/orbit_radius")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/orbit_plots/")
    p.add_argument("--reps", type=str, nargs="*", default=[],
                   help="l:rep[,rep...] replicates used in the main figures for those l "
                        "(others: all original-spring replicates), e.g. 2:3 3:3 for the "
                        "orbit_radius folder. Default: all replicates")
    p.add_argument("--spring-tag", type=str, default="s",
                   help="File-name token marking the new-spring runs. Default: s")
    p.add_argument("--compare-l", type=int, nargs="+", default=[4, 5],
                   help="l indices in the spring comparison figure. Default: 4 5")
    p.add_argument("--spring-labels", type=str, nargs=2, default=["original spring", "new spring"],
                   help="Legend labels for the two springs. Default: 'original spring' "
                        "'new spring'")
    p.add_argument("--norm-l", type=int, default=None,
                   help="l index used as the reference (= 1) in orbit_vs_offset_norm. "
                        "Default: largest l")
    p.add_argument("--settle", type=float, default=2.0,
                   help="Seconds dropped from the start of each run. Default: 2")
    p.add_argument("--tmax", type=float, default=2.0,
                   help="Seconds of wrapped heading shown in orbit_runs. Default: 2")
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per l index (mm). Default: 5")
    p.add_argument("--l-color-max", type=float, default=5.0,
                   help="l index at the top of the viridis colormap (matches the other "
                        "figures). Default: 5")
    p.add_argument("--width", type=float, default=7.0, help="Figure width (in). Default: 7")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    return p.parse_args()


def parse_name(fname, spring_tag):
    """(l, d, rep, spring) from a *_robot.csv name, or None if no l token."""
    toks = fname[:-len("_robot.csv")].split("_")
    l, d, rep, spring = None, -1, 1, False
    spring_re = re.compile(rf"{re.escape(spring_tag)}(\d{{0,2}})")
    for tk in toks:
        if re.fullmatch(r"l\d+", tk):
            l = int(tk[1:])
        elif re.fullmatch(r"d\d+", tk):
            d = int(tk[1:])
        elif spring_re.fullmatch(tk):
            spring = True
            digits = spring_re.fullmatch(tk).group(1)
            if digits:
                rep = int(digits)
        elif re.fullmatch(r"\d{1,2}", tk) and l is not None:
            rep = int(tk)
    return None if l is None else (l, d, rep, spring)


def parse_reps(items):
    out = {}
    for it in items or []:
        l, reps = it.split(":")
        out[int(l)] = {int(r) for r in reps.split(",")}
    return out


def load_run(path, settle):
    df, nid, th_col, x_col, y_col = load_robot(path)
    t = df["time"].to_numpy(dtype=float)
    x, y, g = (df[c].to_numpy(dtype=float) for c in (x_col, y_col, th_col))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & (t >= t[0] + settle)
    if df["track"].nunique() > 1:
        print(f"  NOTE: {os.path.basename(path)} has {df['track'].nunique()} tracks; "
              "treating them as one run")
    return t[ok], x[ok], y[ok], g[ok]


def revolutions(t, r, phi_u):
    """Per full polar turn: (start index, end index, period s, rms radius m).

    Turn-completion times are linearly interpolated between frames, so periods are not
    quantized to the frame interval."""
    turns = np.abs(phi_u - phi_u[0]) / (2 * np.pi)
    k = np.floor(turns).astype(int)
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(k) > 0) + 1])
    tc = [float(t[0])]
    for c in cuts[1:]:
        u0, u1 = turns[c - 1], turns[c]
        f = (k[c] - u0) / (u1 - u0) if u1 != u0 else 0.0
        tc.append(float(t[c - 1] + f * (t[c] - t[c - 1])))
    return [(a, b, tc[i + 1] - tc[i], float(np.sqrt(np.mean(r[a:b] ** 2))))
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])) if b - a >= 3]


def analyze(path, args):
    t, x, y, g = load_run(path, args.settle)
    xc, yc = float(x.mean()), float(y.mean())
    x0, y0 = x - xc, y - yc
    r = np.hypot(x0, y0)
    phi_u = np.unwrap(np.arctan2(y0, x0))
    T = float(t[-1] - t[0])
    sign = 1.0 if phi_u[-1] >= phi_u[0] else -1.0
    revs = revolutions(t, r, phi_u)
    period = np.array([rv[2] for rv in revs])
    rad = np.array([rv[3] for rv in revs])
    g_u = np.unwrap(g)
    out = dict(t=t, x0=x0, y0=y0, g=g, g_turns=(g_u - g_u[0]) / (2 * np.pi), revs=revs,
               duration=T, n_rev=len(revs), sign=sign,
               f_net=sign * len(revs) / float(period.sum()) if period.size else 0.0,
               f_heading=float(g_u[-1] - g_u[0]) / (2 * np.pi) / T,
               r_min=float(r.min()), r_max=float(r.max()), stall_frac=0.0)
    if period.size:
        freq = 1.0 / period
        slow = period > 1.5 * np.median(period)
        out["stall_frac"] = float(period[slow].sum() / period.sum())
        for key, v in (("R", rad), ("f", freq)):
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            out[f"{key}_med"], out[f"{key}_lo"], out[f"{key}_hi"] = q2, q2 - q1, q3 - q2
    else:
        out.update(R_med=float(np.sqrt(np.mean(r ** 2))), R_lo=0.0, R_hi=0.0,
                   f_med=0.0, f_lo=0.0, f_hi=0.0)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save(fig, outdir, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=dpi)
    plt.close(fig)


def label_panels(fig, axes, dx=0.1):
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in axes) + 0.02
    for ax, letter in zip(axes, "abcdefgh"):
        fig.text(ax.get_position().x0 - dx, top, f"({letter})", fontweight="bold",
                 ha="left", va="bottom")


def plot_runs(runs, color, lab, args, outdir):
    ls = sorted({r["l"] for r in runs})
    fig, axes = plt.subplots(len(ls), 3, figsize=(args.width, 1.45 * len(ls) + 0.3),
                             gridspec_kw={"width_ratios": [0.62, 1.0, 1.0]}, squeeze=False)
    lim = 1e3 * max(np.max(np.hypot(r["x0"], r["y0"])) for r in runs) * 1.05
    for row, l in enumerate(ls):
        ax_xy, ax_g, ax_n = axes[row]
        grp = sorted([r for r in runs if r["l"] == l], key=lambda r: r["rep"])
        for k, r in enumerate(grp):
            ls_ = ("-", "--", ":")[k % 3]
            ax_xy.plot(1e3 * r["x0"], 1e3 * r["y0"], color=color[l], lw=0.3,
                       alpha=0.5 if k == 0 else 0.35, ls=ls_)
            keep = r["t"] <= r["t"][0] + args.tmax
            tt = r["t"][keep] - r["t"][0]
            first = True
            for u, v in seam_segments(wrap_angle(r["g"][keep]), tt):
                ax_g.plot(v, u, color=color[l], lw=0.8, ls=ls_,
                          label=f"run {r['rep']}" if first else None)
                first = False
            ax_n.plot(r["t"] - r["t"][0], r["g_turns"], color=color[l], lw=1.0, ls=ls_,
                      label=f"run {r['rep']}")
        ax_xy.plot(0, 0, "+", color="k", ms=4, mew=0.7)
        ax_xy.set_xlim(-lim, lim)
        ax_xy.set_ylim(-lim, lim)
        ax_xy.set_aspect("equal")
        ax_xy.set_ylabel(f"$l$ = {lab(l)}\n\ny (mm)")
        ax_g.set_ylim(-np.pi, np.pi)
        ax_g.set_yticks([-np.pi, 0, np.pi])
        ax_g.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax_g.set_ylabel(r"$\gamma$")
        ax_n.set_ylabel("heading turns")
        if len(grp) > 1:
            ax_n.legend(frameon=False, loc="best", handlelength=1.6)
        if row == len(ls) - 1:
            ax_xy.set_xlabel("x (mm)")
            ax_g.set_xlabel("time (s)")
            ax_n.set_xlabel("time (s)")
        else:
            for ax in (ax_xy, ax_g, ax_n):
                ax.tick_params(labelbottom=False)
    axes[0][0].set_title("trajectory")
    axes[0][1].set_title(f"heading, first {args.tmax:g} s")
    axes[0][2].set_title("cumulative heading rotation")
    fig.tight_layout(h_pad=0.4)
    save(fig, outdir, "orbit_runs", args.dpi)


def offset_axis(ax, ls, args):
    """x-axis spanning the measured offsets with half a step of margin, ticks at each l."""
    ax.set_xlim((min(ls) - 0.5) * args.l_step, (max(ls) + 0.5) * args.l_step)
    ax.set_xticks([l * args.l_step for l in ls])


def per_l_stats(runs, key, scale):
    """{l: (mean, err, n)}: mean of the per-run medians; err = std across runs (ddof=1),
    or the within-run IQR half-width when an l has a single run."""
    out = {}
    for l in sorted({r["l"] for r in runs}):
        grp = [r for r in runs if r["l"] == l]
        vals = np.array([abs(r[f"{key}_med"]) * scale for r in grp])
        if len(grp) > 1:
            err = float(vals.std(ddof=1))
        else:
            err = 0.5 * (grp[0][f"{key}_lo"] + grp[0][f"{key}_hi"]) * scale
        out[l] = (float(vals.mean()), err, len(grp))
    return out


def plot_per_l(ax, runs, key, scale, color, args, marker="o", dx=0.0, mec=None,
               line_ls="-"):
    """One marker per l (mean over runs, error bar = std across runs); returns max(top)."""
    st = per_l_stats(runs, key, scale)
    ls = sorted(st)
    for l in ls:
        m, e, _ = st[l]
        ax.errorbar(l * args.l_step + dx, m, yerr=e, fmt=marker, ms=4, capsize=2.5, lw=0.9,
                    color=color[l], mec=mec or color[l], mew=1.0, zorder=3)
    ax.plot([l * args.l_step + dx for l in ls], [st[l][0] for l in ls], color="0.5", lw=0.9,
            ls=line_ls, zorder=1)
    return max(m + e for m, e, _ in st.values())


def plot_vs_offset(runs, color, args, outdir):
    ls = sorted({r["l"] for r in runs})
    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(args.width * 0.62, 2.3))
    for key, ax, scale in (("R", ax_r, 1e3), ("f", ax_f, 1.0)):
        top = plot_per_l(ax, runs, key, scale, color, args)
        ax.set_xlabel(r"caster offset $l$ (mm)")
        offset_axis(ax, ls, args)
        ax.set_ylim(0, 1.1 * top)
    ax_r.set_ylabel(r"orbit radius $R$ (mm)")
    ax_f.set_ylabel("orbit frequency (Hz)")
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    label_panels(fig, (ax_r, ax_f))
    save(fig, outdir, "orbit_vs_offset", args.dpi)


def plot_vs_offset_norm(runs, color, args, outdir):
    """R and f on one 0–1 axis, each divided by its value at the reference offset."""
    ls = sorted({r["l"] for r in runs})
    l_ref = args.norm_l if args.norm_l in ls else max(ls)
    fig, ax = plt.subplots(figsize=(args.width * 0.4, 2.3))
    for key, scale, marker, line_ls, name in (("R", 1e3, "o", "-", r"radius $R$"),
                                              ("f", 1.0, "s", "--", r"frequency $f$")):
        st = per_l_stats(runs, key, scale)
        ref = st[l_ref][0]
        xs = [l * args.l_step for l in ls]
        ax.plot(xs, [st[l][0] / ref for l in ls], color="0.5", lw=0.9, ls=line_ls, zorder=1)
        for l in ls:
            m, e, _ = st[l]
            ax.errorbar(l * args.l_step, m / ref, yerr=e / ref, fmt=marker, ms=4, capsize=2.5,
                        lw=0.9, color=color[l], mec=color[l], mew=1.0, zorder=3)
        print(f"  normalized {key}: " + "  ".join(f"l={l}: {st[l][0] / ref:.3f}" for l in ls)
              + f"   (ref l={l_ref}: {ref:.4g})")
    ax.set_xlabel(r"caster offset $l$ (mm)")
    ax.set_ylabel(rf"value / value at $l$ = {l_ref * args.l_step:g} mm")
    offset_axis(ax, ls, args)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    h = [plt.Line2D([], [], marker="o", ls="-", color="0.5", mfc="0.3", mec="0.3", ms=4),
         plt.Line2D([], [], marker="s", ls="--", color="0.5", mfc="0.3", mec="0.3", ms=4)]
    ax.legend(h, [r"radius $R$", r"frequency $f$"], frameon=False, loc="lower right",
              handlelength=2.0, handletextpad=0.4)
    fig.tight_layout()
    save(fig, outdir, "orbit_vs_offset_norm", args.dpi)


def plot_spring_compare(orig, spring, color, lab, args, outdir):
    """x–y overlay per compared l, then R and f vs l for both springs."""
    ls = sorted({r["l"] for r in orig + spring})
    n = len(ls)
    fig, axes = plt.subplots(1, n + 2, figsize=(args.width, 2.3),
                             gridspec_kw={"width_ratios": [0.62] * n + [1.0, 1.0]})
    lim = 1e3 * max(np.max(np.hypot(r["x0"], r["y0"])) for r in orig + spring) * 1.05
    for ax, l in zip(axes[:n], ls):
        for r in [r for r in orig if r["l"] == l]:
            ax.plot(1e3 * r["x0"], 1e3 * r["y0"], color=color[l], lw=0.3, alpha=0.5)
        for r in [r for r in spring if r["l"] == l]:
            ax.plot(1e3 * r["x0"], 1e3 * r["y0"], color="k", lw=0.3, alpha=0.45, ls="--")
        ax.plot(0, 0, "+", color="k", ms=4, mew=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(rf"$l$ = {lab(l)}")
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    ax_r, ax_f = axes[n], axes[n + 1]
    for key, ax, scale in (("R", ax_r, 1e3), ("f", ax_f, 1.0)):
        tops = []
        # the two springs are different parameter sets at the same l: dodge them slightly
        if orig:
            tops.append(plot_per_l(ax, orig, key, scale, color, args, marker="o", dx=-0.7))
        if spring:
            tops.append(plot_per_l(ax, spring, key, scale, color, args, marker="s", dx=0.7,
                                   mec="k", line_ls="--"))
        ax.set_xlabel(r"caster offset $l$ (mm)")
        ax.set_xlim((min(ls) - 1) * args.l_step, (max(ls) + 1) * args.l_step)
        ax.set_xticks([l * args.l_step for l in ls])
        ax.set_ylim(0, 1.1 * max(tops))
    ax_r.set_ylabel(r"orbit radius $R$ (mm)")
    ax_f.set_ylabel("orbit frequency (Hz)")
    h = [plt.Line2D([], [], marker="o", ls="-", color="0.5", mfc="0.3", mec="0.3", ms=4),
         plt.Line2D([], [], marker="s", ls="--", color="0.5", mfc="0.3", mec="k", ms=4)]
    ax_f.legend(h, args.spring_labels, frameon=False, loc="lower right", handletextpad=0.4)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    label_panels(fig, list(axes), dx=0.06)
    save(fig, outdir, "orbit_spring_compare", args.dpi)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    plt.rcParams.update(STYLE)
    outdir = args.outdir or os.path.join(args.directory, "orbit_plots")
    os.makedirs(outdir, exist_ok=True)
    rep_sel = parse_reps(args.reps)

    runs = []
    for path in sorted(glob.glob(os.path.join(args.directory, "*_robot.csv"))):
        fname = os.path.basename(path)
        parsed = parse_name(fname, args.spring_tag)
        if parsed is None:
            print(f"  WARNING: no l token in {fname}; skipping")
            continue
        l, d, rep, spring = parsed
        res = analyze(path, args)
        res.update(name=fname[:-len("_robot.csv")], l=l, d=d, rep=rep, spring=spring)
        res["used_main"] = (not spring) and (l not in rep_sel or rep in rep_sel[l])
        runs.append(res)
        print(f"{res['name']}: l={l} rep {rep}{' [new spring]' if spring else ''}"
              f"{'' if res['used_main'] or spring else ' [excluded from main]'}  "
              f"{res['n_rev']} rev  R = {1e3 * res['R_med']:.1f} mm (IQR -"
              f"{1e3 * res['R_lo']:.1f}/+{1e3 * res['R_hi']:.1f})  f = {res['f_med']:.2f} Hz "
              f"(IQR -{res['f_lo']:.2f}/+{res['f_hi']:.2f})  "
              f"{'CCW' if res['sign'] > 0 else 'CW'}  net {res['f_net']:+.2f} Hz "
              f"(heading {res['f_heading']:+.2f})  stalled {100 * res['stall_frac']:.0f}% of time")
    if not runs:
        raise SystemExit(f"No l<l>_..._robot.csv files in {args.directory}")

    main_runs = [r for r in runs if r["used_main"]]
    for l in sorted({r["l"] for r in main_runs}):
        signs = {r["sign"] for r in main_runs if r["l"] == l}
        if len(signs) > 1:
            print(f"  NOTE: l={l} mixes CW and CCW runs; |R| and |f| are averaged together")
    print("\nper-l (mean ± std across runs):")
    for l, (mR, eR, n) in per_l_stats(main_runs, "R", 1e3).items():
        mf, ef, _ = per_l_stats(main_runs, "f", 1.0)[l]
        print(f"  l={l}: n={n}  R = {mR:.2f} ± {eR:.2f} mm   f = {mf:.3f} ± {ef:.3f} Hz")
    for l, reps in rep_sel.items():
        missing = reps - {r["rep"] for r in main_runs if r["l"] == l}
        if missing:
            print(f"  WARNING: --reps asks for l={l} rep {sorted(missing)} but no such "
                  "original-spring run was found")
    cmap = plt.get_cmap("viridis")
    color = {l: cmap(min(l / args.l_color_max, 1.0)) for l in sorted({r["l"] for r in runs})}

    def lab(l):
        return f"{l * args.l_step:g} mm"

    saved = []
    if main_runs:
        plot_runs(main_runs, color, lab, args, outdir)
        plot_vs_offset(main_runs, color, args, outdir)
        plot_vs_offset_norm(main_runs, color, args, outdir)
        saved += ["orbit_runs", "orbit_vs_offset", "orbit_vs_offset_norm"]
    orig_c = [r for r in main_runs if r["l"] in args.compare_l]
    spring_c = [r for r in runs if r["spring"] and r["l"] in args.compare_l]
    if spring_c:
        plot_spring_compare(orig_c, spring_c, color, lab, args, outdir)
        saved.append("orbit_spring_compare")
    else:
        print(f"  NOTE: no '{args.spring_tag}' runs for l in {args.compare_l}; "
              "skipping the spring comparison")

    pd.DataFrame([{k: r[k] for k in ("name", "l", "d", "rep", "spring", "used_main",
                                     "duration", "n_rev", "sign", "R_med", "R_lo", "R_hi",
                                     "f_med", "f_lo", "f_hi", "f_net", "f_heading",
                                     "stall_frac", "r_min", "r_max")}
                  | {"offset_mm": r["l"] * args.l_step} for r in runs]
                 ).to_csv(os.path.join(outdir, "orbit_runs.csv"), index=False)
    pd.DataFrame([dict(name=r["name"], l=r["l"], rep=r["rep"], spring=r["spring"], rev=i,
                       t_start=r["t"][a], period=per, freq=1.0 / per, radius=rad)
                  for r in runs for i, (a, b, per, rad) in enumerate(r["revs"])]
                 ).to_csv(os.path.join(outdir, "orbit_revolutions.csv"), index=False)
    print(f"\nWrote {', '.join(saved)} (.png/.pdf) and CSVs to {outdir}/")


if __name__ == "__main__":
    main()
