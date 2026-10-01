#!/usr/bin/env python3
"""
Per-trial Hopf-sweep metrics from format_tracks_single.py output.

Each ``m*_l*_robot.csv`` is one (motor drive, length) condition; each ``track``
inside the file is a trial. Metrics are computed per trial, then averaged
(mean ± std) over trials of the same (m, l).

  1. mean |heading angular velocity|  (unwrap → Butterworth low-pass → dθ/dt, then abs)
  2. mean radius after centering at the trial's mean (x, y)
  3. mean |polar angular velocity|    (unwrap φ → same low-pass → dφ/dt, then abs)
  4. total |heading rotation|         (unwrap → same low-pass → |θ_end − θ_start|)

Figures 01–03 and 07 use the full trajectory; 04–06 and 08 use only the
steady state (first ``--settle`` seconds dropped, default 1 s).

Figures 09–13 do not replace those. A trial is *rotating* if, after settle,
the low-passed unwrapped heading accumulates ``--n-rev`` full turns (default
5). Rotating-trial ω and r are then computed on that first-N-revolution
window only (so stop time does not enter). Also: time to N revolutions
(among trials that get there) and the fraction of trials that get there.

Unless ``--skip-bifurcation`` is set, ``plot_hopf_bifurcation.py`` then adds
representative trajectories (14), R / R² diagrams with a fitted Dc (15–16),
and onset-frequency / chirality / spectra figures (17–22).

Example
-------
    python plot_hopf_metrics.py ../Data/200826/hopf
    python plot_hopf_bifurcation.py ../Data/250926
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
from scipy.signal import butter, filtfilt

from format_tracks import read_csv_comments


NAME_RE = re.compile(r"m(?P<m>\d+)_l(?P<l>\d+)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Trial-averaged heading/polar metrics vs motor drive")
    p.add_argument("directory", nargs="?",
                   default="/Users/alexleffell/Documents/PhD/tplax/Data/200826/hopf",
                   help="Directory of m*_l*_robot.csv files. "
                        "Default: Data/200826/hopf")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/hopf_metric_plots/")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="Low-pass cutoff (Hz) applied to unwrapped angles. Default: 5.")
    p.add_argument("--butter-order", type=int, default=4,
                   help="Butterworth filter order. Default: 4.")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Skip trials shorter than this. Default: 20.")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Seconds dropped from the start of each trial for the "
                        "steady-state metrics (plots 04–06, 08). Default: 1.")
    p.add_argument("--n-rev", type=int, default=5,
                   help="Heading revolutions that define a rotating trial and "
                        "the analysis window for plots 09–13. Default: 5.")
    p.add_argument("--dpi", type=int, default=130, help="Figure DPI. Default: 130")
    p.add_argument("--skip-bifurcation", action="store_true",
                   help="Do not write the Hopf representative / bifurcation figures.")
    p.add_argument("--n-show", type=int, default=6,
                   help="Trials overlaid in representative / spectra panels. Default: 6.")
    p.add_argument("--tmax", type=float, default=4.0,
                   help="Seconds of SS shown in representative time series. Default: 4.")
    p.add_argument("--savgol-window", type=float, default=0.35)
    p.add_argument("--savgol-poly", type=int, default=3)
    p.add_argument("--r-min", type=float, default=0.005,
                   help="Min radius (m) for polar-angle derivatives. Default: 0.005.")
    return p.parse_args()


def parse_ml(path):
    name = os.path.basename(path)
    if name.endswith("_robot.csv"):
        name = name[:-len("_robot.csv")]
    m = NAME_RE.search(name)
    if m is None:
        return None
    return int(m.group("m")), int(m.group("l"))


def lowpass(sig, dt, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass. Returns `sig` unchanged if it cannot run."""
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n < 8 or not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
        return sig
    fs = 1.0 / dt
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    if not np.isfinite(wn) or wn <= 0.0 or wn >= 1.0:
        return sig
    b, a = butter(order, wn, btype="low")
    # scipy.filtfilt default padlen is 3 * max(len(a), len(b)); length must be > padlen.
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return sig
    return filtfilt(b, a, sig, padlen=padlen)


def mean_angvel(angle, t, dt, cutoff_hz, order):
    """Mean angular velocity (rad/s) of a wrapped angle after low-pass on unwrap."""
    ok = np.isfinite(angle) & np.isfinite(t)
    if ok.sum() < 3:
        return np.nan
    u = np.unwrap(np.asarray(angle[ok], dtype=float))
    tt = np.asarray(t[ok], dtype=float)
    uf = lowpass(u, dt, cutoff_hz, order)
    udot = np.gradient(uf, tt)
    return float(np.nanmean(udot))


def total_rotation(angle, t, dt, cutoff_hz, order):
    """Net |Δθ| (rad) of a wrapped angle after low-pass on unwrap."""
    ok = np.isfinite(angle) & np.isfinite(t)
    if ok.sum() < 3:
        return np.nan
    u = np.unwrap(np.asarray(angle[ok], dtype=float))
    uf = lowpass(u, dt, cutoff_hz, order)
    return float(abs(uf[-1] - uf[0]))


def n_rev_window(t, th, dt, cutoff_hz, order, n_rev):
    """Time from window start until |Δθ| reaches n_rev turns, and crop end index.

    Heading is unwrapped and low-passed (same filter as the other heading
    metrics). Returns (time_to_n_rev, end) where ``end`` is a slice end into
    ``t`` (samples with t <= crossing). If the threshold is never reached,
    (nan, 0).
    """
    t = np.asarray(t, dtype=float)
    th = np.asarray(th, dtype=float)
    ok = np.isfinite(th) & np.isfinite(t)
    if ok.sum() < 3 or n_rev <= 0:
        return np.nan, 0
    tt = t[ok]
    uf = lowpass(np.unwrap(th[ok]), dt, cutoff_hz, order)
    mag = np.abs(uf - uf[0])
    target = n_rev * 2.0 * np.pi
    hit = np.where(mag >= target)[0]
    if hit.size == 0:
        return np.nan, 0
    i = int(hit[0])
    if i == 0:
        t_cross = float(tt[0])
    else:
        y0, y1 = mag[i - 1], mag[i]
        t0, t1 = float(tt[i - 1]), float(tt[i])
        frac = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
        t_cross = t0 + frac * (t1 - t0)
    time_to = t_cross - float(tt[0])
    end = int(np.searchsorted(t, t_cross, side="right"))
    return float(time_to), end


def crop_settle(t, x, y, th, settle):
    """Keep samples at least `settle` seconds after the trial start."""
    if t.size == 0:
        empty = t[:0]
        return empty, empty, empty, empty
    keep = t >= (t[0] + settle)
    return t[keep], x[keep], y[keep], th[keep]


def trial_metrics(t, x, y, th, cutoff_hz, order, min_samples):
    n = t.size
    if n < min_samples:
        return None
    dt = float(np.median(np.diff(t))) if n > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return None
    omega_th = abs(mean_angvel(th, t, dt, cutoff_hz, order))

    xc, yc = float(np.nanmean(x)), float(np.nanmean(y))
    dx, dy = x - xc, y - yc
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    mean_r = float(np.nanmean(r))
    omega_phi = abs(mean_angvel(phi, t, dt, cutoff_hz, order))
    dtheta = total_rotation(th, t, dt, cutoff_hz, order)
    return {
        "omega_heading": omega_th,
        "mean_r": mean_r,
        "omega_polar": omega_phi,
        "heading_rotation": dtheta,
        "n_samples": n,
        "duration": float(t[-1] - t[0]),
        "xc": xc,
        "yc": yc,
    }


def prefix_metrics(mets, prefix):
    if mets is None:
        return {
            f"{prefix}omega_heading": np.nan,
            f"{prefix}mean_r": np.nan,
            f"{prefix}omega_polar": np.nan,
            f"{prefix}heading_rotation": np.nan,
            f"{prefix}n_samples": 0,
            f"{prefix}duration": np.nan,
            f"{prefix}xc": np.nan,
            f"{prefix}yc": np.nan,
        }
    return {f"{prefix}{k}": v for k, v in mets.items()}


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


METRIC_KEYS = ("omega_heading", "mean_r", "omega_polar", "heading_rotation")

METRICS = [
    ("omega_heading", r"$\langle|\dot\theta|\rangle$ (rad/s)",
     "mean heading angular velocity", "01_omega_heading.png"),
    ("mean_r", r"$\langle r\rangle$ (m)",
     "mean radius (centered)", "02_mean_r.png"),
    ("omega_polar", r"$\langle|\dot\phi|\rangle$ (rad/s)",
     "mean polar angular velocity", "03_omega_polar.png"),
    ("heading_rotation", r"$|\Delta\theta|$ (rad)",
     "total heading rotation", "07_heading_rotation.png"),
]

METRICS_SS = [
    ("ss_omega_heading", r"$\langle|\dot\theta|\rangle$ (rad/s)",
     "mean heading angular velocity (steady state)", "04_omega_heading_ss.png"),
    ("ss_mean_r", r"$\langle r\rangle$ (m)",
     "mean radius (centered, steady state)", "05_mean_r_ss.png"),
    ("ss_omega_polar", r"$\langle|\dot\phi|\rangle$ (rad/s)",
     "mean polar angular velocity (steady state)", "06_omega_polar_ss.png"),
    ("ss_heading_rotation", r"$|\Delta\theta|$ (rad)",
     "total heading rotation (steady state)", "08_heading_rotation_ss.png"),
]

ROT_METRIC_KEYS = ("omega_heading", "mean_r", "omega_polar")


def rotating_plot_list(n_rev):
    n = int(n_rev)
    return [
        ("rot_omega_heading", r"$\langle|\dot\theta|\rangle$ (rad/s)",
         f"mean heading angular velocity (first {n} revolutions)",
         f"09_omega_heading_{n}rev.png"),
        ("rot_mean_r", r"$\langle r\rangle$ (m)",
         f"mean radius (centered, first {n} revolutions)",
         f"10_mean_r_{n}rev.png"),
        ("rot_omega_polar", r"$\langle|\dot\phi|\rangle$ (rad/s)",
         f"mean polar angular velocity (first {n} revolutions)",
         f"11_omega_polar_{n}rev.png"),
        ("time_to_nrev", "time (s)",
         f"time to {n} revolutions",
         f"12_time_to_{n}rev.png"),
        ("frac_nrev", "fraction",
         f"fraction of trials reaching {n} revolutions",
         f"13_frac_{n}rev.png"),
    ]


def plot_sweep(summary, lengths, key, ylabel, title, fname, outdir, dpi, ylim=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, length in enumerate(lengths):
        sub = summary[summary["l"] == length].sort_values("m")
        ax.errorbar(sub["m"], sub[f"{key}_mean"], yerr=sub[f"{key}_std"],
                    fmt="o-", lw=1.5, ms=6, capsize=4, color=f"C{i}",
                    label=fr"$l = {length}$")
    ax.set_xlabel("motor drive $m$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.directory, "*_robot.csv")))
    if not files:
        raise SystemExit(f"No *_robot.csv files in {args.directory}")
    outdir = args.outdir or os.path.join(args.directory, "hopf_metric_plots")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for path in files:
        parsed = parse_ml(path)
        if parsed is None:
            print(f"  WARNING: no m/l in {os.path.basename(path)}; skipping")
            continue
        m_drive, length = parsed
        df, nid, th_col, x_col, y_col = load_robot(path)
        src = os.path.basename(path)
        n_ok = n_ss = n_rot = 0
        for tr_id, sub in df.groupby("track", sort=True):
            t = sub["time"].to_numpy(dtype=float)
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            th = sub[th_col].to_numpy(dtype=float)
            mets = trial_metrics(t, x, y, th, args.cutoff, args.butter_order,
                                 args.min_samples)
            if mets is None:
                continue
            t_ss, x_ss, y_ss, th_ss = crop_settle(t, x, y, th, args.settle)
            mets_ss = trial_metrics(t_ss, x_ss, y_ss, th_ss, args.cutoff,
                                    args.butter_order, args.min_samples)
            dt_ss = (float(np.median(np.diff(t_ss))) if t_ss.size > 1 else np.nan)
            t_nrev, end = n_rev_window(t_ss, th_ss, dt_ss, args.cutoff,
                                       args.butter_order, args.n_rev)
            mets_rot = None
            if np.isfinite(t_nrev) and end >= args.min_samples:
                mets_rot = trial_metrics(
                    t_ss[:end], x_ss[:end], y_ss[:end], th_ss[:end],
                    args.cutoff, args.butter_order, args.min_samples)
            n_ok += 1
            if mets_ss is not None:
                n_ss += 1
            if mets_rot is not None:
                n_rot += 1
            rows.append({
                "file": src,
                "m": m_drive,
                "l": length,
                "track": int(tr_id),
                **mets,
                **prefix_metrics(mets_ss, "ss_"),
                **prefix_metrics(mets_rot, "rot_"),
                "reached_nrev": int(mets_rot is not None),
                "time_to_nrev": t_nrev if mets_rot is not None else np.nan,
            })
        print(f"{src}: m={m_drive} l={length}  {n_ok} trial(s)  "
              f"{n_ss} with t>{args.settle:.3g}s SS  "
              f"{n_rot} with {args.n_rev} heading rev")

    if not rows:
        raise SystemExit("No trials produced metrics.")

    trials = pd.DataFrame(rows)
    agg = {
        "n_trials": ("track", "count"),
        "n_trials_ss": ("ss_omega_heading", "count"),
        "n_trials_rot": ("rot_omega_heading", "count"),
        "frac_nrev_mean": ("reached_nrev", "mean"),
        "time_to_nrev_mean": ("time_to_nrev", "mean"),
        "time_to_nrev_std": ("time_to_nrev", "std"),
    }
    for k in METRIC_KEYS:
        agg[f"{k}_mean"] = (k, "mean")
        agg[f"{k}_std"] = (k, "std")
        agg[f"ss_{k}_mean"] = (f"ss_{k}", "mean")
        agg[f"ss_{k}_std"] = (f"ss_{k}", "std")
    for k in ROT_METRIC_KEYS:
        agg[f"rot_{k}_mean"] = (f"rot_{k}", "mean")
        agg[f"rot_{k}_std"] = (f"rot_{k}", "std")
    summary = trials.groupby(["m", "l"], as_index=False).agg(**agg)
    p = summary["frac_nrev_mean"].clip(0.0, 1.0)
    n = summary["n_trials"].clip(lower=1)
    summary["frac_nrev_std"] = np.sqrt(p * (1.0 - p) / n)
    # pandas std is sample std (ddof=1); a single trial → NaN. Use 0 for plotting.
    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)

    trials_csv = os.path.join(outdir, "metrics_per_trial.csv")
    summary_csv = os.path.join(outdir, "metrics_mean_std.csv")
    trials.to_csv(trials_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    lengths = sorted(summary["l"].unique())
    n_old = 0
    for key, ylabel, title, fname in METRICS + METRICS_SS:
        plot_sweep(summary, lengths, key, ylabel, title, fname, outdir, args.dpi)
        n_old += 1
    n_new = 0
    for key, ylabel, title, fname in rotating_plot_list(args.n_rev):
        ylim = (0.0, 1.05) if key == "frac_nrev" else None
        plot_sweep(summary, lengths, key, ylabel, title, fname, outdir, args.dpi,
                   ylim=ylim)
        n_new += 1

    print(f"\nWrote {n_old} original + {n_new} rotating-trial figures + "
          f"{os.path.basename(trials_csv)} + {os.path.basename(summary_csv)} "
          f"to {outdir}/")

    if not args.skip_bifurcation:
        from plot_hopf_bifurcation import run_bifurcation
        args.outdir = outdir
        run_bifurcation(args)


if __name__ == "__main__":
    main()
