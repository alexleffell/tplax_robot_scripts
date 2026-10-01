#!/usr/bin/env python3
"""
Fraction of trials orbiting CW vs CCW for single-node-in-a-well parameter sets.

Each parameter folder ``m<drive>_l<l>_d<d>/`` under DIRECTORY holds one merged
``<folder>_robot.csv`` (format_tracks_single.py output); every track in it is one
trial (release). Subfolders whose names do not parse (e.g. ``hopf``) are ignored.

Per trial (after dropping the first --settle seconds), with the well centre taken as
the median position over all trials of that parameter set:
  - orbiting  : polar angle about the centre winds ≥ --min-turns full turns and the
                median radius is ≥ --r-min,
  - direction : sign of the circulation L = ⟨x ẏ − y ẋ⟩ (L > 0 CCW, L < 0 CW),
  - check     : the sign of the net heading rotation (γ) is compared with the orbit
                direction; disagreements are reported.
Same classification as plot_hopf_bifurcation.py (chirality fractions).

Colours follow the phase-portrait convention (RdBu_r on γ̇): CW (γ̇ < 0) blue,
CCW (γ̇ > 0) red, no orbit grey.

Outputs (to --outdir): chirality_fraction.pdf/.png, chirality_trials.csv.

Example
-------
    python plot_chirality_fraction.py ../Data/200826
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_kinematics_tracks import load_robot
from reduce_single import savgol_deriv


NAME_RE = re.compile(r"^m(\d+)_l(\d+)_d(\d+)$")
C_CW, C_CCW, C_NONE = "#3a73b0", "#c0392b", "0.72"


def parse_args():
    p = argparse.ArgumentParser(description="CW / CCW / no-orbit trial fractions per (l, d)")
    p.add_argument("directory", nargs="?",
                   default="/Users/alexleffell/Documents/PhD/tplax/Data/200826",
                   help="Folder of m<drive>_l<l>_d<d>/ parameter folders. Default: Data/200826")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/chirality_plots/")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Seconds dropped from the start of each trial. Default: 1.0")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Minimum samples after settling to keep a trial. Default: 20")
    p.add_argument("--min-turns", type=float, default=1.0,
                   help="Polar turns about the centre required to count as orbiting. Default: 1")
    p.add_argument("--r-min", type=float, default=0.003,
                   help="Minimum median orbit radius (m) to count as orbiting. Default: 0.003")
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per l index (mm); 0 = label by index. Default: 5.0")
    p.add_argument("--d-step", type=float, default=0.0,
                   help="Perpendicular offset per d index (mm); 0 = label by index. Default: 0")
    p.add_argument("--savgol-window", type=float, default=0.35,
                   help="Savitzky–Golay window (s) for ẋ, ẏ. Default: 0.35")
    p.add_argument("--savgol-poly", type=int, default=3, help="Savitzky–Golay polyorder. Default: 3")
    p.add_argument("--width", type=float, default=3.4, help="Figure width (in). Default: 3.4")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    return p.parse_args()


def find_sets(directory):
    """[(m, l, d, name, csv_path)] for every parseable parameter folder."""
    out = []
    for sub in sorted(os.listdir(directory)):
        m = NAME_RE.match(sub)
        path = os.path.join(directory, sub, f"{sub}_robot.csv")
        if not m or not os.path.isdir(os.path.join(directory, sub)):
            continue
        if not os.path.isfile(path):
            hits = sorted(glob.glob(os.path.join(directory, sub, "*_robot.csv")))
            if not hits:
                print(f"  WARNING: no *_robot.csv in {sub}; skipping")
                continue
            path = hits[0]
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), sub, path))
    return out


def classify(path, args):
    """Per-trial dicts for one parameter set."""
    df, nid, th_col, x_col, y_col = load_robot(path)
    trials = []
    for tr_id, sub in df.groupby("track", sort=True):
        t = sub["time"].to_numpy(dtype=float)
        keep = t >= t[0] + args.settle
        x = sub[x_col].to_numpy(dtype=float)[keep]
        y = sub[y_col].to_numpy(dtype=float)[keep]
        th = sub[th_col].to_numpy(dtype=float)[keep]
        t = t[keep]
        ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(th)
        if ok.sum() < args.min_samples:
            continue
        trials.append(dict(track=int(tr_id), t=t[ok], x=x[ok], y=y[ok], th=th[ok]))
    if not trials:
        return []
    xc = float(np.median(np.concatenate([tr["x"] for tr in trials])))
    yc = float(np.median(np.concatenate([tr["y"] for tr in trials])))
    for tr in trials:
        x0, y0 = tr["x"] - xc, tr["y"] - yc
        dt = float(np.median(np.diff(tr["t"])))
        phi = np.unwrap(np.arctan2(y0, x0))
        tr["turns"] = float(phi[-1] - phi[0]) / (2 * np.pi)
        tr["radius"] = float(np.median(np.hypot(x0, y0)))
        xd = savgol_deriv(x0, dt, args.savgol_window, args.savgol_poly)
        yd = savgol_deriv(y0, dt, args.savgol_window, args.savgol_poly)
        tr["L"] = float(np.nanmean(x0 * yd - y0 * xd))
        tr["heading_turns"] = float(np.unwrap(tr["th"])[-1] - np.unwrap(tr["th"])[0]) / (2 * np.pi)
        tr["duration"] = float(tr["t"][-1] - tr["t"][0])
        orbit = abs(tr["turns"]) >= args.min_turns and tr["radius"] >= args.r_min
        tr["state"] = ("CCW" if tr["L"] > 0 else "CW") if orbit else "none"
    return trials


def label_for(l, d, args):
    ls = f"{l * args.l_step:g} mm" if args.l_step > 0 else f"{l}"
    ds = f"{d * args.d_step:g} mm" if args.d_step > 0 else f"{d}"
    return ls, ds


def plot_fractions(rows, path_base, args):
    rc = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
          "pdf.fonttype": 42}
    ls_all = sorted({r["l"] for r in rows})
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(args.width, 0.72 * args.width))
        xpos, ticks, gap, w = [], [], 0.6, 0.8
        x = 0.0
        for li, l in enumerate(ls_all):
            grp = sorted([r for r in rows if r["l"] == l], key=lambda r: r["d"])
            for r in grp:
                r["x"] = x
                xpos.append(x)
                x += 1.0
            ticks.append((np.mean([r["x"] for r in grp]), l))
            x += gap
        for r in rows:
            n = r["n"]
            fr = [r["CW"] / n, r["CCW"] / n, r["none"] / n] if n else [0, 0, 0]
            bottom = 0.0
            for f, c, cnt in zip(fr, (C_CW, C_CCW, C_NONE), (r["CW"], r["CCW"], r["none"])):
                ax.bar(r["x"], f, w, bottom=bottom, color=c, edgecolor="k", lw=0.5)
                if f >= 0.12:
                    ax.text(r["x"], bottom + f / 2, f"{cnt}", ha="center", va="center",
                            fontsize=6.5, color="white" if c != C_NONE else "0.2")
                bottom += f
            ax.text(r["x"], 1.02, f"n={n}", ha="center", va="bottom", fontsize=6.5, color="0.3")
        ax.set_xticks([r["x"] for r in rows])
        ax.set_xticklabels([f"$d$={label_for(r['l'], r['d'], args)[1]}" for r in rows])
        for xm, l in ticks:
            ax.text(xm, -0.17, rf"$l$ = {label_for(l, 0, args)[0]}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top")
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("fraction of trials")
        ax.set_xlim(min(xpos) - 0.7, max(xpos) + 0.7)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="k", lw=0.5)
                   for c in (C_CW, C_CCW, C_NONE)]
        ax.legend(handles, ["CW", "CCW", "no orbit"], frameon=False, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 1.07), handlelength=1.0,
                  columnspacing=1.0)
        fig.tight_layout()
        fig.savefig(path_base + ".pdf")
        fig.savefig(path_base + ".png", dpi=args.dpi)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(args.directory, "chirality_plots")
    os.makedirs(outdir, exist_ok=True)
    sets = find_sets(args.directory)
    if not sets:
        raise SystemExit(f"No m<drive>_l<l>_d<d>/ folders in {args.directory}")
    drives = sorted({s[0] for s in sets})
    if len(drives) > 1:
        print(f"  NOTE: multiple drives {drives}; bars are per (m, l, d) but labelled by (l, d)")

    rows, trial_rows = [], []
    for m, l, d, name, path in sets:
        trials = classify(path, args)
        counts = {k: sum(tr["state"] == k for tr in trials) for k in ("CW", "CCW", "none")}
        mismatch = sum(1 for tr in trials if tr["state"] != "none"
                       and np.sign(tr["heading_turns"]) != (1 if tr["state"] == "CCW" else -1))
        rows.append(dict(m=m, l=l, d=d, name=name, n=len(trials), **counts))
        print(f"{name}: {len(trials)} trials  CW {counts['CW']}  CCW {counts['CCW']}  "
              f"no orbit {counts['none']}"
              + (f"  ({mismatch} orbit/heading sign mismatches)" if mismatch else ""))
        for tr in trials:
            trial_rows.append((name, m, l, d, tr["track"], tr["duration"], tr["turns"],
                               tr["radius"], tr["L"], tr["heading_turns"], tr["state"]))

    plot_fractions(rows, os.path.join(outdir, "chirality_fraction"), args)
    csv_path = os.path.join(outdir, "chirality_trials.csv")
    with open(csv_path, "w") as fh:
        fh.write("set,m,l,d,track,duration_s,polar_turns,median_radius_m,L,heading_turns,state\n")
        for r in trial_rows:
            fh.write(",".join(v if isinstance(v, str) else f"{v:g}" for v in r) + "\n")
    print(f"\nWrote chirality_fraction.pdf/.png and {os.path.basename(csv_path)} to {outdir}/")


if __name__ == "__main__":
    main()
