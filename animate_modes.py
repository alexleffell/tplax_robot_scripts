#!/usr/bin/env python3
"""
Animate the elastic normal modes as a sanity check.

Reads the analysis bundle from analyze_modes.py and, for every NON-rigid-body mode
(deform_idx), animates the reference lattice oscillating along that mode shape:

    positions(t) = reference + A * sin(2*pi*f*t) * mode_shape

Springs (connections) are drawn as lines and nodes as markers; the undeformed lattice
is shown faintly for reference. By default all modes are shown in one grid animation
(one GIF); --per-mode writes a separate GIF per mode.

Each mode is scaled to a comparable visible amplitude (a fraction of the mean bond
length) so the shape is legible regardless of the eigenvector's raw magnitude.

Example
-------
    python animate_modes.py ../Data/090726/chiral_1_analysis.npz
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args():
    p = argparse.ArgumentParser(description="Animate elastic normal modes (sanity check)")
    p.add_argument("analysis_npz", type=str, help="*_analysis.npz from analyze_modes.py")
    p.add_argument("--outdir", type=str, default=None, help="Output dir. Default: <npz>_modes/")
    p.add_argument("--amp", type=float, default=0.4,
                   help="Max node displacement as a fraction of the mean bond length. Default: 0.4")
    p.add_argument("--cycles", type=int, default=2, help="Oscillation cycles per clip. Default: 2")
    p.add_argument("--fps", type=int, default=20, help="Animation frames per second. Default: 20")
    p.add_argument("--frames-per-cycle", type=int, default=24, help="Frames per cycle. Default: 24")
    p.add_argument("--per-mode", action="store_true",
                   help="Write one GIF per mode instead of a single grid GIF.")
    p.add_argument("--dpi", type=int, default=110, help="Figure DPI. Default: 110")
    return p.parse_args()


def draw_lattice(ax, pts, bonds, ref, faint=True):
    """Draw springs + nodes; return (list of bond Line2D, node scatter). Also draws the
    undeformed reference faintly if `faint`."""
    if faint:
        for (i, j) in bonds:
            ax.plot([ref[i, 0], ref[j, 0]], [ref[i, 1], ref[j, 1]],
                    color="0.85", lw=1, zorder=1)
    bond_lines = []
    for (i, j) in bonds:
        (ln,) = ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color="C0", lw=2, zorder=2)
        bond_lines.append(ln)
    scat = ax.scatter(pts[:, 0], pts[:, 1], c="k", s=30, zorder=3)
    return bond_lines, scat


def main():
    args = parse_args()
    d = np.load(args.analysis_npz, allow_pickle=True)
    outdir = args.outdir or (os.path.splitext(args.analysis_npz)[0] + "_modes")
    os.makedirs(outdir, exist_ok=True)

    ref = np.asarray(d["ref"], dtype=float)             # (N, 2)
    evecs = np.asarray(d["eigenvectors"], dtype=float)  # (2N, 2N)
    evals = np.asarray(d["eigenvalues"], dtype=float)
    deform_idx = [int(i) for i in d["deform_idx"]]
    nodes = [int(n) for n in d["nodes"]]
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    bonds = [(idx[int(a)], idx[int(b)]) for a, b in d["connections"]]

    # Visible amplitude: scale each mode so its largest node displacement = amp * mean bond length.
    bond_len = np.mean([np.linalg.norm(ref[i] - ref[j]) for i, j in bonds])
    total_frames = args.cycles * args.frames_per_cycle
    phase = np.sin(2 * np.pi * args.cycles * np.arange(total_frames) / total_frames)

    def mode_shape(mi):
        v = evecs[:, mi].reshape(N, 2)
        maxd = np.max(np.linalg.norm(v, axis=1))
        return v * (args.amp * bond_len / maxd) if maxd > 0 else v

    lim = np.max(np.abs(ref - ref.mean(0))) + args.amp * bond_len
    center = ref.mean(0)

    def title(mi):
        lam = max(float(evals[mi]), 0.0)
        return f"mode {mi}  (λ={float(evals[mi]):.3g}, ω={np.sqrt(lam):.3g})"

    def setup_ax(ax, mi):
        ax.set_aspect("equal")
        ax.set_xlim(center[0] - lim * 1.15, center[0] + lim * 1.15)
        ax.set_ylim(center[1] - lim * 1.15, center[1] + lim * 1.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title(mi), fontsize=9)

    saved = []

    if args.per_mode:
        for mi in deform_idx:
            shp = mode_shape(mi)
            fig, ax = plt.subplots(figsize=(4, 4))
            setup_ax(ax, mi)
            lines, scat = draw_lattice(ax, ref + phase[0] * shp, bonds, ref)

            def update(f, lines=lines, scat=scat, shp=shp):
                pts = ref + phase[f] * shp
                for (i, j), ln in zip(bonds, lines):
                    ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
                scat.set_offsets(pts)
                return (*lines, scat)

            anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps, blit=False)
            path = os.path.join(outdir, f"mode_{mi:02d}.gif")
            anim.save(path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
            plt.close(fig)
            saved.append(path)
            print(f"  wrote {path}")
    else:
        n = len(deform_idx)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
        shapes, panels = [], []
        for j in range(nrow * ncol):
            ax = axes[j // ncol][j % ncol]
            if j >= n:
                ax.axis("off"); continue
            mi = deform_idx[j]
            shp = mode_shape(mi)
            setup_ax(ax, mi)
            lines, scat = draw_lattice(ax, ref + phase[0] * shp, bonds, ref)
            shapes.append(shp); panels.append((lines, scat))
        fig.suptitle("Elastic normal modes (non-rigid) — oscillation", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        def update(f):
            artists = []
            for shp, (lines, scat) in zip(shapes, panels):
                pts = ref + phase[f] * shp
                for (i, j), ln in zip(bonds, lines):
                    ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
                scat.set_offsets(pts)
                artists += [*lines, scat]
            return artists

        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps, blit=False)
        path = os.path.join(outdir, "modes_grid.gif")
        anim.save(path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        plt.close(fig)
        saved.append(path)
        print(f"  wrote {path}")

    print(f"\nAnimated {len(deform_idx)} non-rigid modes -> {outdir}/")


if __name__ == "__main__":
    main()
