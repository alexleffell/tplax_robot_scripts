#!/usr/bin/env python3
"""
Compare analysis scalars across parameter sets, averaging over repeat runs.

Each input FOLDER is one parameter set containing multiple `*_analysis.npz` runs (repeats
of the same parameters). For every scalar the analysis produces, this averages over the
runs in a folder and plots the mean with std error bars, one figure per scalar, with the
folders as the x-axis categories (in the order given). Also writes `scalars_grouped.csv`
with mean/std/n per folder.

Reuses the scalar list and extractor from plot_scalars_vs_drive.py.

Example
-------
    python plot_scalars_grouped.py ../Data/130726_l1d1 ../Data/130726_l3d1
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_scalars_vs_drive import SCALARS, extract


def parse_args():
    p = argparse.ArgumentParser(description="Compare analysis scalars across parameter-set folders")
    p.add_argument("folders", nargs="+", help="One folder per parameter set (each with *_analysis.npz runs)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <parent of first folder>/scalar_comparison_plots/")
    p.add_argument("--sem", action="store_true",
                   help="Error bars = standard error of the mean (std/sqrt(n)) instead of std.")
    p.add_argument("--dpi", type=int, default=130, help="Figure DPI. Default: 130")
    return p.parse_args()


def main():
    args = parse_args()
    labels = [os.path.basename(os.path.normpath(f)) for f in args.folders]
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.folders[0])),
                                          "scalar_comparison_plots")
    os.makedirs(outdir, exist_ok=True)

    # Per folder: {scalar_label: array of per-run values}.
    per_folder, n_runs, prov = {}, {}, set()
    for folder, label in zip(args.folders, labels):
        files = sorted(glob.glob(os.path.join(folder, "*_analysis.npz")))
        if not files:
            print(f"  WARNING: no *_analysis.npz in {folder}; skipping")
        vals = {lbl: [] for lbl, _, _ in SCALARS}
        for f in files:
            npz = np.load(f, allow_pickle=True)
            for lbl, key, red in SCALARS:
                vals[lbl].append(extract(npz, key, red))
            if "heading_source" in npz.files and "angle_frame" in npz.files:
                prov.add(f"{str(npz['heading_source'])}/{str(npz['angle_frame'])}")
        per_folder[label] = {lbl: np.array(vals[lbl], dtype=float) for lbl, _, _ in SCALARS}
        n_runs[label] = len(files)
        print(f"{label}: {len(files)} runs")

    provenance = " | ".join(sorted(prov)) if prov else ""
    if len(prov) > 1:
        print(f"  NOTE: mixed provenance across runs: {sorted(prov)}")

    x = np.arange(len(labels))
    n_saved = 0
    for lbl, _, _ in SCALARS:
        means = np.array([np.nanmean(per_folder[l][lbl]) if per_folder[l][lbl].size else np.nan
                          for l in labels])
        stds = np.array([np.nanstd(per_folder[l][lbl]) if per_folder[l][lbl].size else np.nan
                         for l in labels])
        if args.sem:
            counts = np.array([max(np.isfinite(per_folder[l][lbl]).sum(), 1) for l in labels])
            errs = stds / np.sqrt(counts)
        else:
            errs = stds
        if not np.isfinite(means).any():
            continue
        fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4))
        ax.errorbar(x, means, yerr=errs, fmt="o-", lw=1.4, ms=7, capsize=5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("parameter set"); ax.set_ylabel(lbl)
        ax.set_title(f"{lbl}  (mean ± {'SEM' if args.sem else 'std'} over runs)")
        ax.grid(alpha=0.3)
        if provenance:
            fig.text(0.995, 0.005, provenance, ha="right", va="bottom",
                     fontsize=7, color="0.4", family="monospace")
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", lbl).strip("_")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(os.path.join(outdir, f"{safe}.png"), dpi=args.dpi)
        plt.close(fig)
        n_saved += 1

    # Table: one row per folder, mean & std for each scalar.
    csv_path = os.path.join(outdir, "scalars_grouped.csv")
    with open(csv_path, "w") as fh:
        header = ["folder", "n_runs"]
        for lbl, _, _ in SCALARS:
            header += [f'"{lbl} mean"', f'"{lbl} std"']
        fh.write(",".join(str(h) if not h.startswith('"') else h for h in header) + "\n")
        for l in labels:
            row = [l, str(n_runs[l])]
            for lbl, _, _ in SCALARS:
                v = per_folder[l][lbl]
                row += [f"{np.nanmean(v):g}" if v.size else "nan",
                        f"{np.nanstd(v):g}" if v.size else "nan"]
            fh.write(",".join(row) + "\n")

    print(f"\nWrote {n_saved} scalar comparison figures + {os.path.basename(csv_path)} to {outdir}/")


if __name__ == "__main__":
    main()
