#!/usr/bin/env python3
"""
Plot analysis scalars as a function of motor drive across a sweep of experiments.

Scans a directory for ``*_analysis.npz`` bundles (from analyze_modes.py), reads the
"drive" from each filename (the trailing integer, e.g. ``010326_med_225`` -> 225), and
plots every scalar quantity the analysis produces as a function of drive — one figure
per scalar. Also writes a `scalars_vs_drive.csv` table.

Scalars = the values in analyze_modes.py's "Scalar results" section: stored scalars
(D_r, ⟨ω⟩, persistence time, …) and time-averages of the key per-frame series
(order parameter, zero-mode fractions, participation ratio, energies, …).

Example
-------
    python plot_scalars_vs_drive.py ../Data/010326
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, npz_key, reduction): reduction is "scalar" (0-d value) or "mean" (nanmean of a series).
SCALARS = [
    ("mean order parameter",                 "order_param",              "mean"),
    ("caster diffusion D_r",                 "D_mean",                   "scalar"),
    ("mean zero-mode (rigid) KE fraction",   "zero_mode_KE_ratio",       "mean"),
    ("mean caster zero fraction (Laplacian)", "lap_zero_ratio",          "mean"),
    ("mean caster zero fraction (elastic)",  "elastic_zero_ratio",       "mean"),
    ("mean participation ratio",             "participation_ratio",      "mean"),
    ("mean spectral entropy",                "spectral_entropy",         "mean"),
    ("mean polarity-velocity coupling",      "coupling_pv",              "mean"),
    ("mean polarity-def.-velocity coupling", "coupling_pvdef",           "mean"),
    ("mean body angular velocity",           "mean_omega",               "scalar"),
    ("std body angular velocity",            "std_omega",                "scalar"),
    ("net polarization rotation rate",       "pol_rot_rate",             "scalar"),
    ("dominant-pair orbit chirality",        "orbit_chirality",          "scalar"),
    ("orientational integral corr. time",    "tau_persist",              "scalar"),
    ("mean internal (deformation) KE",       "KE_deform",                "mean"),
    ("mean rigid-body KE",                   "KE_zero",                  "mean"),
    ("mean bond alignment",                  "bond_align",               "mean"),
    ("mean ring winding number",             "winding",                  "mean"),
    ("mean active-force / CoM-vel alignment", "mean_force_vel_alignment", "scalar"),
    ("mean KE",                              "KE_total",                 "mean"),
    ("mean PE",                              "PE_total",                 "mean"),
    ("mean total energy",                    "E_total",                  "mean"),
]


def parse_drive(path, regex):
    name = os.path.basename(path)
    name = re.sub(r"_analysis\.npz$", "", name)
    if regex:
        m = re.search(regex, name)
        if m:
            return float(m.group(1) if m.groups() else m.group(0))
        return None
    nums = re.findall(r"\d+", name)
    return float(nums[-1]) if nums else None


def extract(npz, key, reduction):
    if key not in npz.files:
        return np.nan
    v = npz[key]
    if reduction == "scalar":
        return float(v)
    arr = np.asarray(v, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan


def parse_args():
    p = argparse.ArgumentParser(description="Plot analysis scalars vs motor drive")
    p.add_argument("directory", type=str, help="Directory containing *_analysis.npz files")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/drive_sweep_plots/")
    p.add_argument("--drive-regex", type=str, default=None,
                   help="Regex to extract drive from the filename (first group, else whole match). "
                        "Default: trailing integer.")
    p.add_argument("--dpi", type=int, default=130, help="Figure DPI. Default: 130")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.directory, "*_analysis.npz")))
    if not files:
        raise SystemExit(f"No *_analysis.npz files in {args.directory}")
    outdir = args.outdir or os.path.join(args.directory, "drive_sweep_plots")
    os.makedirs(outdir, exist_ok=True)

    # Collect drive + all scalars per file.
    drives, labels_data, prov = [], {lbl: [] for lbl, _, _ in SCALARS}, set()
    kept = []
    for f in files:
        drive = parse_drive(f, args.drive_regex)
        if drive is None:
            print(f"  WARNING: no drive parsed from {os.path.basename(f)}; skipping")
            continue
        npz = np.load(f, allow_pickle=True)
        drives.append(drive)
        kept.append(os.path.basename(f))
        for lbl, key, red in SCALARS:
            labels_data[lbl].append(extract(npz, key, red))
        if "heading_source" in npz.files and "angle_frame" in npz.files:
            prov.add(f"{str(npz['heading_source'])}/{str(npz['angle_frame'])}")

    drives = np.array(drives)
    order = np.argsort(drives)
    drives_sorted = drives[order]
    provenance = " | ".join(sorted(prov)) if prov else ""
    print(f"Found {len(kept)} experiments; drives = {drives_sorted.tolist()}")
    if len(prov) > 1:
        print(f"  NOTE: mixed provenance across files: {sorted(prov)}")

    # One figure per scalar.
    n_saved = 0
    for lbl, key, red in SCALARS:
        y = np.array(labels_data[lbl])[order]
        if not np.isfinite(y).any():
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(drives_sorted, y, "o-", lw=1.4, ms=6)
        ax.set_xlabel("motor drive"); ax.set_ylabel(lbl); ax.set_title(f"{lbl} vs drive")
        ax.grid(alpha=0.3)
        if provenance:
            fig.text(0.995, 0.005, provenance, ha="right", va="bottom",
                     fontsize=7, color="0.4", family="monospace")
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", lbl).strip("_")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        path = os.path.join(outdir, f"{safe}.png")
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        n_saved += 1

    # Table.
    csv_path = os.path.join(outdir, "scalars_vs_drive.csv")
    with open(csv_path, "w") as fh:
        header = ["file", "drive"] + [lbl for lbl, _, _ in SCALARS]
        fh.write(",".join(f'"{h}"' for h in header) + "\n")
        for i in order:
            row = [kept[i], f"{drives[i]:g}"] + [f"{labels_data[lbl][i]:g}" for lbl, _, _ in SCALARS]
            fh.write(",".join(row) + "\n")

    print(f"Wrote {n_saved} scalar-vs-drive figures + {os.path.basename(csv_path)} to {outdir}/")


if __name__ == "__main__":
    main()
