#!/usr/bin/env python3
"""
Plot the analysis bundle produced by analyze_modes.py.

Reads the ``*_analysis.npz`` and writes a set of diagnostic figures (PNG) to an
output directory. Figures:

  1. CoM trajectory (colored by time)
  2. CoM 2D probability density (arena histogram)
  3. Energies vs time (KE, PE, total)
  4. Zero-mode / deformation energy ratios vs time
  5. Caster zero-mode fractions vs time (graph-Laplacian & elastic bases)
  6. Orientation order parameter vs time
  7. Elastic eigenvalue spectrum + time-averaged modal kinetic energy
  8. First few deformation mode shapes (quiver on the reference lattice)
  9. Caster-angle MSD vs lag with the diffusion fit
 10. PSDs: order parameter, KE/PE, and a modal-energy PSD heatmap

Example
-------
    python plot_analysis.py ../Data/020525/020525_clipped_2_analysis.npz
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def parse_args():
    p = argparse.ArgumentParser(description="Plot analyze_modes.py output")
    p.add_argument("analysis_npz", type=str, help="*_analysis.npz from analyze_modes.py")
    p.add_argument("--outdir", type=str, default=None, help="Output dir. Default: <npz>_plots/")
    p.add_argument("--dpi", type=int, default=130, help="Figure DPI. Default: 130")
    p.add_argument("--n-modes", type=int, default=4, help="How many deformation mode shapes to draw. Default: 4")
    return p.parse_args()


def savefig(fig, outdir, name, dpi, saved):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  wrote {path}")


def main():
    args = parse_args()
    d = np.load(args.analysis_npz, allow_pickle=True)
    outdir = args.outdir or (os.path.splitext(args.analysis_npz)[0] + "_plots")
    os.makedirs(outdir, exist_ok=True)
    saved = []

    time = d["time"]
    nodes = d["nodes"]
    N = len(nodes)

    # 1. CoM trajectory, colored by time.
    com = d["com"]
    fig, ax = plt.subplots(figsize=(6, 6))
    pts = com.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=time[:-1], alpha=0.8)
    ax.add_collection(lc)
    ax.scatter(com[0, 0], com[0, 1], c="green", s=40, label="start", zorder=3)
    ax.scatter(com[-1, 0], com[-1, 1], c="red", s=40, label="end", zorder=3)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Center-of-mass trajectory")
    fig.colorbar(lc, ax=ax, label="time")
    ax.legend()
    savefig(fig, outdir, "01_com_trajectory.png", args.dpi, saved)

    # 2. CoM 2D PDF.
    hist, xe, ye = d["com_hist"], d["com_xedges"], d["com_yedges"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(hist.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="equal", cmap="magma")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("CoM probability density")
    fig.colorbar(im, ax=ax, label="density")
    savefig(fig, outdir, "02_com_pdf.png", args.dpi, saved)

    # 3. Energies vs time.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, d["KE_total"], label="KE", lw=1)
    ax.plot(time, d["PE_total"], label="PE", lw=1)
    ax.plot(time, d["E_total"], label="E total", lw=1.2, color="k")
    ax.set_xlabel("time"); ax.set_ylabel("energy"); ax.set_title("Energies")
    ax.legend()
    savefig(fig, outdir, "03_energies.png", args.dpi, saved)

    # 4. Zero-mode / deformation energy ratio vs time.
    fig, ax = plt.subplots(figsize=(9, 4))
    zr = d["zero_mode_KE_ratio"]
    ax.plot(time, zr, lw=1, label=f"zero-mode (rigid) KE fraction (mean {np.nanmean(zr):.3f})")
    ax.axhline(np.nanmean(zr), color="C0", ls="--", alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time"); ax.set_ylabel("fraction"); ax.set_title("Rigid-body KE fraction")
    ax.legend()
    savefig(fig, outdir, "04_zero_mode_KE_ratio.png", args.dpi, saved)

    # 5. Caster zero-mode fractions vs time.
    fig, ax = plt.subplots(figsize=(9, 4))
    lz, ez = d["lap_zero_ratio"], d["elastic_zero_ratio"]
    ax.plot(time, lz, lw=1, label=f"graph-Laplacian zero (mean {np.nanmean(lz):.3f})")
    ax.plot(time, ez, lw=1, label=f"elastic zero (mean {np.nanmean(ez):.3f})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time"); ax.set_ylabel("fraction")
    ax.set_title("Caster-angle energy in zero modes")
    ax.legend()
    savefig(fig, outdir, "05_caster_zero_fractions.png", args.dpi, saved)

    # 6. Orientation order parameter.
    fig, ax = plt.subplots(figsize=(9, 4))
    op = d["order_param"]
    kind = "nematic" if bool(d["is_nematic"]) else "polar"
    ax.plot(time, op, lw=1, color="C3")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time"); ax.set_ylabel(r"$\Psi$")
    ax.set_title(f"Orientation order parameter ({kind}), mean {np.nanmean(op):.3f}")
    savefig(fig, outdir, "06_order_parameter.png", args.dpi, saved)

    # 7. Eigenvalue spectrum + time-averaged modal KE.
    evals = d["eigenvalues"]
    modal_KE = d["modal_KE"]
    mean_modal_KE = modal_KE.mean(axis=0)
    rigid_idx = set(d["rigid_idx"].tolist())
    colors = ["C1" if i in rigid_idx else "C0" for i in range(len(evals))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(evals)), evals, color=colors)
    axes[0].set_xlabel("mode index"); axes[0].set_ylabel(r"$\lambda\ (=\omega^2)$")
    axes[0].set_title("Elastic eigenvalue spectrum (orange = rigid)")
    axes[1].bar(range(len(mean_modal_KE)), mean_modal_KE, color=colors)
    axes[1].set_xlabel("mode index"); axes[1].set_ylabel("time-avg modal KE")
    axes[1].set_title("Mean kinetic energy per mode")
    savefig(fig, outdir, "07_mode_spectrum.png", args.dpi, saved)

    # 8. Deformation mode shapes (quiver on reference lattice).
    ref = d["ref"]
    evecs = d["eigenvectors"]
    deform_idx = d["deform_idx"]
    nshow = min(args.n_modes, len(deform_idx))
    if nshow > 0:
        ncol = min(nshow, 4)
        nrow = int(np.ceil(nshow / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for j in range(nrow * ncol):
            ax = axes[j // ncol][j % ncol]
            if j >= nshow:
                ax.axis("off"); continue
            mi = int(deform_idx[j])
            disp = evecs[:, mi].reshape(N, 2)
            ax.scatter(ref[:, 0], ref[:, 1], c="k", s=15, zorder=3)
            ax.quiver(ref[:, 0], ref[:, 1], disp[:, 0], disp[:, 1],
                      angles="xy", scale_units="xy", color="C0")
            ax.set_aspect("equal")
            ax.set_title(f"mode {mi}, $\\lambda$={evals[mi]:.3g}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Deformation mode shapes")
        savefig(fig, outdir, "08_mode_shapes.png", args.dpi, saved)

    # 9. Caster-angle MSD with diffusion fit.
    lags = d["ang_msd_lags"]
    msd = d["ang_msd"]  # (N, nlag)
    D_mean = float(d["D_mean"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(N):
        ax.plot(lags, msd[i], color="C0", alpha=0.35, lw=0.8)
    ax.plot(lags, msd.mean(axis=0), color="k", lw=2, label="mean MSD")
    ax.plot(lags, 2 * D_mean * lags, color="C3", ls="--",
            label=fr"$2 D_r t$, $D_r$={D_mean:.3g}")
    ax.set_xlabel("lag time"); ax.set_ylabel(r"$\langle \Delta\theta^2\rangle$")
    ax.set_title("Caster-angle MSD (per node + mean)")
    ax.legend()
    savefig(fig, outdir, "09_caster_msd.png", args.dpi, saved)

    # 10. PSDs.
    f = d["psd_freq"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(f, d["psd_order"], color="C3")
    axes[0].set_xlabel("frequency"); axes[0].set_ylabel("PSD")
    axes[0].set_title("Order parameter PSD")
    axes[1].semilogy(f, d["psd_KE"], label="KE")
    axes[1].semilogy(f, d["psd_PE"], label="PE")
    axes[1].set_xlabel("frequency"); axes[1].set_ylabel("PSD")
    axes[1].set_title("KE / PE PSD"); axes[1].legend()
    mep = d["modal_energy_psd"]  # (2N, nf)
    with np.errstate(divide="ignore"):
        logmep = np.log10(mep + 1e-30)
    im = axes[2].imshow(logmep, origin="lower", aspect="auto",
                        extent=[f[0], f[-1], 0, mep.shape[0]], cmap="viridis")
    axes[2].set_xlabel("frequency"); axes[2].set_ylabel("mode index")
    axes[2].set_title("Modal-energy PSD (log10)")
    fig.colorbar(im, ax=axes[2], label=r"$\log_{10}$ PSD")
    savefig(fig, outdir, "10_psds.png", args.dpi, saved)

    # ------------------------------------------------------------------ #
    # Active-solid diagnostics
    # ------------------------------------------------------------------ #
    modal_disp = d["modal_disp"]
    evecs = d["eigenvectors"]

    # 11. Collective actuation: polarity-velocity coupling, actuation spectrum, participation.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(time, d["coupling_pv"], lw=1, label="full velocity")
    axes[0].plot(time, d["coupling_pvdef"], lw=1, label="deformation velocity")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_ylim(-1.02, 1.02); axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\cos(\angle(p, v))$")
    axes[0].set_title("Polarity–velocity coupling"); axes[0].legend()
    act = d["actuation_spectrum"]
    rigid_set = set(d["rigid_idx"].tolist())
    cols = ["C1" if i in rigid_set else "C0" for i in range(len(act))]
    axes[1].bar(range(len(act)), act, color=cols)
    axes[1].set_ylim(-1.02, 1.02); axes[1].set_xlabel("mode index")
    axes[1].set_ylabel(r"corr$(C_i, A_i)$")
    axes[1].set_title("Per-mode actuation (polarity–velocity co-projection)")
    axes[2].plot(time, d["participation_ratio"], lw=1, color="C2",
                 label=fr"PR (mean {np.nanmean(d['participation_ratio']):.2f})")
    axes[2].plot(time, d["spectral_entropy"], lw=1, color="C4",
                 label=fr"entropy (mean {np.nanmean(d['spectral_entropy']):.2f})")
    axes[2].set_xlabel("time"); axes[2].set_title("Mode condensation"); axes[2].legend()
    savefig(fig, outdir, "11_collective_actuation.png", args.dpi, saved)

    # 12. Phase portraits: dominant pair and the first two non-zero modes.
    dom = d["dominant_modes"]
    ftn = d["first_two_nonzero"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pair, lab in [(axes[0], dom, "dominant pair"),
                          (axes[1], ftn, "first two non-zero")]:
        ia, ib = int(pair[0]), int(pair[1])
        qa, qb = modal_disp[:, ia], modal_disp[:, ib]
        pts = np.stack([qa, qb], axis=-1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=time[:-1], alpha=0.8)
        ax.add_collection(lc); ax.autoscale()
        ax.scatter(qa[0], qb[0], c="green", s=40, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel(f"mode {ia} amplitude")
        ax.set_ylabel(f"mode {ib} amplitude")
        ax.set_title(f"Phase portrait ({lab})")
        fig.colorbar(lc, ax=ax, label="time")
    savefig(fig, outdir, "12_phase_portraits.png", args.dpi, saved)

    # 13. Equipartition / per-mode effective temperature.
    fig, ax = plt.subplots(figsize=(9, 4))
    modes_x = range(len(d["Teff_kin"]))
    ax.bar([x - 0.2 for x in modes_x], d["Teff_kin"], width=0.4, label=r"$T_{eff}$ (kinetic)")
    ax.bar([x + 0.2 for x in modes_x], d["Teff_pot"], width=0.4, label=r"$T_{eff}$ (potential)")
    ax.set_xlabel("mode index"); ax.set_ylabel("effective temperature")
    ax.set_title("Per-mode effective temperature (flat = equipartition)")
    ax.legend()
    savefig(fig, outdir, "13_equipartition.png", args.dpi, saved)

    # 14. Chirality: body angular velocity and net polarization angle.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(time, d["omega"], lw=1)
    axes[0].axhline(float(d["mean_omega"]), color="C3", ls="--",
                    label=fr"$\langle\omega\rangle$={float(d['mean_omega']):.3g}")
    axes[0].set_xlabel("time"); axes[0].set_ylabel(r"$\omega$")
    axes[0].set_title("Body angular velocity"); axes[0].legend()
    axes[1].plot(time, d["pol_angle"], lw=1, color="C5")
    axes[1].set_xlabel("time"); axes[1].set_ylabel("net polarization angle (unwrapped)")
    axes[1].set_title(fr"Polarization rotation (rate {float(d['pol_rot_rate']):.3g}, "
                      fr"orbit chirality {float(d['orbit_chirality']):.3g})")
    savefig(fig, outdir, "14_chirality.png", args.dpi, saved)

    # 15. Orientational autocorrelation and VACF.
    ct = d["corr_time"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ct, d["orient_acf"], lw=1.2)
    axes[0].axhline(1 / np.e, color="gray", ls=":", label="1/e")
    axes[0].axvline(float(d["tau_persist"]), color="C3", ls="--",
                    label=fr"$\tau$={float(d['tau_persist']):.3g}")
    axes[0].set_xlabel("lag time"); axes[0].set_ylabel(r"$\langle\cos\Delta\theta\rangle$")
    axes[0].set_title("Orientational autocorrelation"); axes[0].legend()
    axes[1].plot(ct, d["vacf"], lw=1.2, color="C1")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xlabel("lag time"); axes[1].set_ylabel("VACF")
    axes[1].set_title("Velocity autocorrelation")
    savefig(fig, outdir, "15_autocorrelations.png", args.dpi, saved)

    # 16. Net active force vs CoM velocity.
    af, vc = d["active_force"], d["v_com"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(time, d["force_vel_cos"], lw=1, color="C6")
    axes[0].axhline(float(d["mean_force_vel_alignment"]), color="C3", ls="--",
                    label=fr"mean {float(d['mean_force_vel_alignment']):.3f}")
    axes[0].set_ylim(-1.02, 1.02); axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\cos(\angle(F, v_{cm}))$")
    axes[0].set_title("Active force / CoM-velocity alignment"); axes[0].legend()
    sc = axes[1].scatter(np.linalg.norm(af, axis=1), np.linalg.norm(vc, axis=1),
                         c=time, cmap="viridis", s=8)
    axes[1].set_xlabel(r"$|F_{active}|$"); axes[1].set_ylabel(r"$|v_{cm}|$")
    axes[1].set_title("Active force vs CoM speed")
    fig.colorbar(sc, ax=axes[1], label="time")
    savefig(fig, outdir, "16_active_force.png", args.dpi, saved)

    # 17. Spatial polarity structure: bond alignment and ring winding.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(time, d["bond_align"], lw=1, color="C0")
    axes[0].set_ylim(-1.02, 1.02); axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\langle\cos(\theta_i-\theta_j)\rangle_{bonds}$")
    axes[0].set_title("Bond polarity alignment")
    if np.all(np.isfinite(d["winding"])):
        axes[1].plot(time, d["winding"], lw=1, color="C3")
        axes[1].set_title("Ring winding number")
    else:
        axes[1].text(0.5, 0.5, "winding N/A\n(non-ring topology)", ha="center", va="center")
        axes[1].set_title("Ring winding number")
    axes[1].set_xlabel("time"); axes[1].set_ylabel("winding")
    savefig(fig, outdir, "17_spatial_polarity.png", args.dpi, saved)

    print(f"\nWrote {len(saved)} figures to {outdir}/")


if __name__ == "__main__":
    main()
