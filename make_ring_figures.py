#!/usr/bin/env python3
"""
Journal figures for the ring-robot l × d sweep (same style as the single-node figures:
8 pt, viridis colours for the caster offset l, CW blue / CCW red reserved for direction).

Inputs: a day folder with one sub-folder per parameter set (``l<i>_d<j>/``), each already run
through run_pipeline.sh and summarize_parameter_set.py (``parameter_set_runs.csv`` with a
``state`` column in every folder, and ``<run>_analysis.npz`` next to each video).

Figure 1 — fig_ring_states:  morphology selects the collective behaviour
  (a–d) one representative run per state (top view: centre-of-mass path and ring outlines),
  (e)   fraction of runs in each state over the l × d grid,
  (f–i) per-state CoM speed, rigid-body KE fraction, |body rotation| and |caster spin|.
Figure 2 — fig_strain_wave:  the strain-wave limit cycle
  (a) caster-heading kymograph and (b) bond-angle kymograph of a wave run,
  (c) phase portrait of the dominant shear pair, (d) windowed strain-wave W and heading-wave H
  over the whole run, (e) heading-twist rate vs strain-wave rate for every wave run (1:1 line).

Representative runs are chosen automatically (largest windowed |W| for the wave, highest
flocking share × speed for the flock, largest |ω| for the rotor, closest to the state medians
for mixed); override with --wave-run / --flock-run / --rotor-run / --mixed-run SET/RUN.
Directions are in the data's x-y axes (camera frame: y points down).

Example
-------
    python make_ring_figures.py ../Data/051126
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from summarize_parameter_set import STATES, STATE_COLORS

STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
WINDOW = {"wave": 0.6, "flock": 3.0, "rotor": 1.2, "mixed": 3.0}   # s shown per state


def parse_args():
    p = argparse.ArgumentParser(description="Journal figures for the ring l × d sweep")
    p.add_argument("day", help="Day folder with l*_d* parameter-set sub-folders")
    p.add_argument("--outdir", default=None, help="Default: <day>/figures/")
    p.add_argument("--l-step", type=float, default=5.0, help="mm per l index. Default: 5")
    p.add_argument("--d-step", type=float, default=6.43, help="mm per d index. Default: 6.43")
    p.add_argument("--settle", type=float, default=3.0, help="Release transient (s). Default: 3")
    p.add_argument("--t-start", type=float, default=8.0,
                   help="Start (s) of the windows shown in the trajectory / kymograph panels. "
                        "Default: 8")
    p.add_argument("--kymo-window", type=float, default=1.5,
                   help="Kymograph / phase-portrait window (s). Default: 1.5")
    for s in STATES:
        p.add_argument(f"--{s}-run", default=None, help=f"SET/RUN for the {s} example")
    p.add_argument("--width", type=float, default=7.0, help="Figure width (in). Default: 7")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def load_runs(day):
    rows = []
    for f in sorted(glob.glob(os.path.join(day, "l*_d*", "parameter_set_runs.csv"))):
        s = os.path.basename(os.path.dirname(f))
        r = pd.read_csv(f)
        r["set"] = s
        r["l"] = int(s.split("_")[0][1:])
        r["d"] = int(s.split("_")[1][1:])
        rows.append(r)
    if not rows:
        raise SystemExit(f"No parameter_set_runs.csv under {day}/l*_d*/ — run "
                         "summarize_parameter_set.py first.")
    R = pd.concat(rows, ignore_index=True)
    if "state" not in R:
        raise SystemExit("parameter_set_runs.csv has no 'state' column — re-run "
                         "summarize_parameter_set.py.")
    return R


def pick_examples(R, args):
    ex = {}
    for s in STATES:
        forced = getattr(args, f"{s}_run")
        if forced:
            st, run = forced.split("/", 1)
            ex[s] = R[(R.set == st) & (R.run == run)].iloc[0]
            continue
        sub = R[R.state == s]
        if sub.empty:
            continue
        if s == "wave":
            r = sub.loc[sub.wave_abs_win.idxmax()]
        elif s == "flock":
            r = sub.loc[(sub.heading_flock_share * sub.com_speed).idxmax()]
        elif s == "rotor":
            r = sub.loc[sub.omega_mean.abs().idxmax()]
        else:
            z = ((sub.rigid_KE_fraction - sub.rigid_KE_fraction.median()).abs()
                 + (sub.com_speed - sub.com_speed.median()).abs())
            r = sub.loc[z.idxmin()]
        ex[s] = r
    return ex


def npz_of(day, row):
    return np.load(os.path.join(day, row["set"], f"{row['run']}_analysis.npz"), allow_pickle=True)


def label(fig, ax, letter, dx=0.012, dy=0.008):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, f"({letter})", fontweight="bold", ha="right", va="bottom")


def set_label(row, args):
    return f"l = {row['l'] * args.l_step:g} mm, d = {row['d'] * args.d_step:.3g} mm"


# --------------------------------------------------------------------------- #
def draw_trajectory(ax, d, state, args):
    t = d["time"]
    order = [list(d["nodes"]).index(n) for n in d["ring_nodes"]]
    t0 = t[0] + max(args.settle, args.t_start)
    m = (t >= t0) & (t < t0 + WINDOW[state])
    if m.sum() < 3:
        m = t >= t[0] + args.settle
    P = 1e3 * d["pos"][m][:, order]                     # (T, N, 2) mm
    com = P.mean(axis=1)
    P, com = P - com[0], com - com[0]
    c = STATE_COLORS[state]
    if state in ("wave", "rotor"):
        for n in range(P.shape[1]):
            ax.plot(P[:, n, 0], P[:, n, 1], color=c, lw=0.4, alpha=0.5)
    idx = np.linspace(0, len(P) - 1, 6).astype(int)
    for k, i in enumerate(idx):
        poly = np.vstack([P[i], P[i][:1]])
        ax.plot(poly[:, 0], poly[:, 1], color=c, lw=0.9, alpha=0.25 + 0.75 * k / (len(idx) - 1))
    ax.plot(P[idx[-1], :, 0], P[idx[-1], :, 1], "o", color=c, ms=2.5, mec="k", mew=0.3)
    ax.plot(com[:, 0], com[:, 1], color="k", lw=0.8)
    ax.plot(com[-1, 0], com[-1, 1], ">", color="k", ms=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.add_artist(AnchoredSizeBar(ax.transData, 100, "100 mm", "lower left", frameon=False,
                                  borderpad=0.1, sep=2, size_vertical=0,
                                  fontproperties={"size": 6}))
    ax.set_title(f"{state}  ({WINDOW[state]:g} s)", fontsize=7.5, fontweight="bold",
                 color=c if state != "mixed" else "0.45")


def draw_state_map(ax, R, args):
    ls, ds = sorted(R.l.unique()), sorted(R.d.unique())
    for (l, d), g in R.groupby(["l", "d"]):
        j, i = ds.index(d), ls.index(l)
        x0 = j - 0.42
        for s in STATES:
            f = float((g.state == s).mean())
            if f > 0:
                ax.add_patch(plt.Rectangle((x0, i - 0.3), 0.84 * f, 0.6, color=STATE_COLORS[s], lw=0))
                x0 += 0.84 * f
        ax.add_patch(plt.Rectangle((j - 0.42, i - 0.3), 0.84, 0.6, fill=False, lw=0.4, ec="0.35"))
        ax.text(j + 0.42, i + 0.32, f"n={len(g)}", ha="right", va="bottom", fontsize=5.5,
                color="0.35")
    ax.set_xlim(-0.5, len(ds) - 0.5); ax.set_ylim(-0.5, len(ls) - 0.5)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([f"{d * args.d_step:.3g}" for d in ds])
    ax.set_yticks(range(len(ls))); ax.set_yticklabels([f"{l * args.l_step:g}" for l in ls])
    ax.set_xlabel("perpendicular offset d (mm)"); ax.set_ylabel("caster offset l (mm)")
    ax.spines[["top", "right"]].set_visible(True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in STATES], STATES,
              frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, columnspacing=0.8)


def draw_state_bars(ax, R, key, ylabel, absval=False):
    xs = np.arange(len(STATES))
    for x, s in zip(xs, STATES):
        v = R.loc[R.state == s, key].to_numpy(dtype=float)
        v = np.abs(v) if absval else v
        v = v[np.isfinite(v)]
        if v.size:
            ax.bar(x, v.mean(), 0.7, yerr=v.std(ddof=1) if v.size > 1 else 0, color=STATE_COLORS[s],
                   capsize=2, error_kw=dict(lw=0.7))
    ax.set_xticks(xs); ax.set_xticklabels([s[:5] for s in STATES], rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_ylim(0, None)


def fig_states(R, ex, args, out):
    fig = plt.figure(figsize=(args.width, 0.75 * args.width))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.35], hspace=0.35, wspace=0.35,
                  left=0.08, right=0.99, top=0.93, bottom=0.09)
    axes_traj = []
    for k, s in enumerate(STATES):
        ax = fig.add_subplot(gs[0, k])
        if s in ex:
            draw_trajectory(ax, npz_of(args.day, ex[s]), s, args)
            ax.text(0.5, -0.02, set_label(ex[s], args), transform=ax.transAxes, ha="center",
                    va="top", fontsize=6, color="0.35")
        else:
            ax.axis("off")
        axes_traj.append(ax)
    ax_map = fig.add_subplot(gs[1, 0:2])
    draw_state_map(ax_map, R, args)
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 2:4], hspace=0.75, wspace=0.6)
    bars = [("com_speed", "CoM speed (m/s)", False), ("rigid_KE_fraction", "rigid-body KE frac.", False),
            ("omega_mean", r"$|\langle\omega\rangle|$ (rad/s)", True),
            ("caster_spin_mean", "|caster spin| (rad/s)", True)]
    axes_bar = []
    for k, (key, yl, ab) in enumerate(bars):
        ax = fig.add_subplot(inner[k // 2, k % 2])
        draw_state_bars(ax, R, key, yl, ab)
        axes_bar.append(ax)
    fig.canvas.draw()
    for ax, ch in zip(axes_traj + [ax_map] + axes_bar, "abcdefghi"):
        label(fig, ax, ch)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_ring_states.{ext}"), dpi=args.dpi)
    plt.close(fig)


def fig_wave(R, ex, args, out):
    row = ex["wave"]
    d = npz_of(args.day, row)
    t = d["time"]
    t0 = t[0] + max(args.settle, args.t_start)
    m = (t >= t0) & (t < t0 + args.kymo_window)
    tt = t[m] - t0
    fig = plt.figure(figsize=(args.width, 0.66 * args.width))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.42,
                  left=0.08, right=0.94, top=0.93, bottom=0.1)
    top = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :],
                                  width_ratios=[1, 0.025, 0.24, 1, 0.025], wspace=0.05)
    N = d["ring_headings"].shape[1]
    ax_h = fig.add_subplot(top[0, 0])
    im = ax_h.imshow(np.degrees(d["ring_headings"][m]).T, aspect="auto", origin="lower",
                     cmap="twilight", vmin=-180, vmax=180, interpolation="nearest",
                     extent=[tt[0], tt[-1], -0.5, N - 0.5])
    ax_h.set_yticks(range(N)); ax_h.set_yticklabels(range(1, N + 1))
    ax_h.set_ylabel("node (ring order)"); ax_h.set_xlabel("time (s)")
    ax_h.set_title(f"caster heading γ  ({set_label(row, args)})")
    fig.colorbar(im, cax=fig.add_subplot(top[0, 1]), ticks=[-180, 0, 180]).set_label("γ (deg)",
                                                                                   labelpad=-2)
    ax_b = fig.add_subplot(top[0, 3])
    bd = np.degrees(d["bond_angle_dev"][m])
    lim = float(np.nanpercentile(np.abs(bd), 99)) or 1.0
    im2 = ax_b.imshow(bd.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                      interpolation="nearest", extent=[tt[0], tt[-1], -0.5, N - 0.5])
    ax_b.set_yticks(range(N)); ax_b.set_yticklabels(range(1, N + 1))
    ax_b.set_xlabel("time (s)"); ax_b.set_title("bond-angle deviation")
    ax_b.set_yticklabels([])
    fig.colorbar(im2, cax=fig.add_subplot(top[0, 4])).set_label("deg", labelpad=1)
    # phase portrait of the dominant shear pair
    ax_p = fig.add_subplot(gs[1, 0])
    j = int(d["wave_dominant_sector"])
    a, b = [int(i) for i in d["sector_modes"][j]]
    Q = (d["modal_disp"] - d["modal_disp_mean"][None, :])[m]
    z1, z2 = 1e3 * Q[:, a], 1e3 * Q[:, b]
    pts = np.c_[z1, z2].reshape(-1, 1, 2)
    cw = STATE_COLORS["wave"]
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), colors=cw, lw=0.8)
    ax_p.add_collection(lc)
    k = int(np.argmax(z2))
    ax_p.annotate("", xy=(z1[k + 1], z2[k + 1]), xytext=(z1[k - 1], z2[k - 1]),
                  arrowprops=dict(arrowstyle="-|>", color="k", lw=0, mutation_scale=10))
    r = 1.15 * max(np.abs(z1).max(), np.abs(z2).max())
    ax_p.set_xlim(-r, r); ax_p.set_ylim(-r, r); ax_p.set_aspect("equal")
    ax_p.axhline(0, color="0.85", lw=0.6); ax_p.axvline(0, color="0.85", lw=0.6)
    ax_p.set_xlabel(r"$Q_1$ (mm)"); ax_p.set_ylabel(r"$Q_2$ (mm)")
    ax_p.set_title(f"shear pair (m = 2), {args.kymo_window:g} s")
    # W and H over the run
    ax_w = fig.add_subplot(gs[1, 1])
    ax_w.axvspan(0, args.settle, color="0.92", lw=0)
    ax_w.plot(t - t[0], d["wave_order_win"], color=STATE_COLORS["wave"], lw=1.1, label="strain wave W")
    ax_w.plot(t - t[0], d["heading_wave_H_win"], color="0.25", lw=1.0, ls="--", label="heading wave H")
    ax_w.axhline(0, color="0.8", lw=0.6)
    ax_w.set_ylim(-1.05, 0.3); ax_w.set_xlabel("time (s)"); ax_w.set_ylabel("order parameter")
    ax_w.legend(frameon=False, loc="upper right")
    ax_w.set_title("whole run (grey: release)")
    # 1:1 locking over all wave runs
    ax_l = fig.add_subplot(gs[1, 2])
    W = R[(R.state == "wave") & np.isfinite(R.heading_spin) & np.isfinite(R.wave_phase_speed)]
    cmap = plt.get_cmap("viridis")
    for _, rr in W.iterrows():
        ax_l.plot(rr.wave_phase_speed, rr.heading_spin, "o", ms=4, mec="k", mew=0.3,
                  color=cmap(min(rr.l / 5.0, 1.0)))
    v = np.r_[W.wave_phase_speed, W.heading_spin]
    pad = 0.08 * (v.max() - v.min())
    lo, hi = v.min() - pad, v.max() + pad
    ax_l.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
    ax_l.set_xlim(lo, hi); ax_l.set_ylim(lo, hi); ax_l.set_aspect("equal")
    ax_l.set_xlabel(r"strain-wave rate $\Omega_W$ (rad/s)")
    ax_l.set_ylabel(r"heading-twist rate $\Omega_H$ (rad/s)")
    ax_l.set_title(f"all wave runs (n = {len(W)})")
    hl = [plt.Line2D([], [], marker="o", ls="", color=cmap(l / 5.0), mec="k", mew=0.3, ms=4)
          for l in sorted(W.l.unique())]
    ax_l.legend(hl + [plt.Line2D([], [], color="0.5", ls="--")],
                [f"l = {l * args.l_step:g} mm" for l in sorted(W.l.unique())] + ["1:1"],
                frameon=False, loc="upper left", fontsize=6)
    fig.canvas.draw()
    for ax, ch in zip([ax_h, ax_b, ax_p, ax_w, ax_l], "abcde"):
        label(fig, ax, ch)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_strain_wave.{ext}"), dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    plt.rcParams.update(STYLE)
    out = args.outdir or os.path.join(args.day, "figures")
    os.makedirs(out, exist_ok=True)
    R = load_runs(args.day)
    ex = pick_examples(R, args)
    for s, r in ex.items():
        print(f"{s:>5} example: {r['set']}/{r['run']}")
    fig_states(R, ex, args, out)
    if "wave" in ex:
        fig_wave(R, ex, args, out)
    print(f"Wrote fig_ring_states and fig_strain_wave (.pdf/.png) -> {out}/")


if __name__ == "__main__":
    main()
