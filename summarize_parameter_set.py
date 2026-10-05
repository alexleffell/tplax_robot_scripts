#!/usr/bin/env python3
"""
Phase-diagram metrics for one or more parameter-set folders (mean ± std over runs).

Each FOLDER is one parameter set (e.g. ``l2_d4/``) holding ~10 recordings that have been run
through the pipeline (run_pipeline.sh), i.e. one ``<video>_analysis.npz`` per video from the
current analyze_modes.py. For every run the first --settle seconds are dropped and the metrics
below are computed; each folder is then summarised by the mean, sample std and n over its runs.
Videos without an analysis bundle are listed and skipped.

Metrics (per run, after the settle cut)
  Motion type
    rigid_KE_fraction        ⟨KE_rigid / KE_total⟩           1 = moves rigidly, 0 = only deforms
    rot_fraction_of_rigid    ⟨KE_rot⟩ / ⟨KE_rigid⟩            spinning vs translating (rigid part)
    omega_mean, omega_abs, omega_std   body rotation rate ω (rad/s; + = CCW), |⟨ω⟩|, std(ω)
    com_speed                ⟨|v_cm|⟩ (position units / s)
    force_vel_alignment      ⟨cos(net caster push, CoM velocity)⟩
  Strain wave (dominant 2-D sector = largest mean deformation share)
    wave_W                   share · Λ  (signed; + = CCW travelling wave, body frame)
    wave_abs_win             ⟨|W_win|⟩ (--wave-window average; direction-blind)
    wave_time_frac           fraction of time |W_win| > --wave-threshold
    wave_m                   wavenumber of the dominant sector
    wave_share, wave_circulation, wave_amp_cv   share, Λ, CV of |z|
    wave_phase_speed         Ω (rad/s);  wave_speed_over_spin = Ω / mean caster spin (only when
                             wave_time_frac ≥ 0.5 and |caster spin| > --min-spin; else NaN)
  Caster order
    order_minus_null         ⟨polar order⟩ − random-heading baseline (≈0.37 for N=6)
    bond_alignment           ⟨cos(γ_i − γ_j)⟩ over springs
    winding_time_frac        fraction of time with |winding number| ≥ 1
    caster_spin_mean, caster_spin_abs   mean heading rotation rate over nodes (rad/s), its |·|
  Deformation
    rms_deformation          rms per-node body-frame deformation / ring radius
    sector_participation     ⟨1/Σ_j share_j²⟩  (number of active deformation patterns)
    spin_sync                mean off-diagonal Pearson correlation of caster spin rates γ̇
  Regularity
    rot_diffusion            heading MSD slope/2 after removing each node's mean spin (rad²/s)
    spectral_peak_frac       fraction of power in the dominant spectral peak of the dominant
                             sector's amplitude (→1 periodic limit cycle, small = irregular)

Outputs
  <folder>/parameter_set_runs.csv     one row per run
  --out (default <parent>/parameter_set_summary.csv)  one row per folder: l, d (parsed from the
                                      folder name, ``l<i>`` / ``d<i>``), n_runs, <metric>_mean, _std
  and a printed table per folder.

Example
-------
    python summarize_parameter_set.py ../Data/051026/l2_d4
    python summarize_parameter_set.py ../Data/051026/l*_d* --out ../Data/051026/phase_metrics.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter, welch


METRICS = [
    "rigid_KE_fraction", "rot_fraction_of_rigid", "omega_mean", "omega_abs", "omega_std",
    "com_speed", "force_vel_alignment",
    "wave_W", "wave_abs_win", "wave_time_frac", "wave_m", "wave_share", "wave_circulation",
    "wave_amp_cv", "wave_phase_speed", "wave_speed_over_spin",
    "order_minus_null", "bond_alignment", "winding_time_frac", "caster_spin_mean",
    "caster_spin_abs",
    "rms_deformation", "sector_participation", "spin_sync",
    "rot_diffusion", "spectral_peak_frac",
]


def parse_args():
    p = argparse.ArgumentParser(description="Phase-diagram metrics per parameter-set folder")
    p.add_argument("folders", nargs="+", help="Parameter-set folder(s) with *_analysis.npz runs")
    p.add_argument("--settle", type=float, default=3.0,
                   help="Seconds dropped from the start of every run. Default: 3")
    p.add_argument("--wave-window", type=float, default=1.0,
                   help="Averaging window (s) for W_win. Default: 1.0")
    p.add_argument("--wave-threshold", type=float, default=0.5,
                   help="|W_win| above this counts as 'in a strain wave'. Default: 0.5")
    p.add_argument("--min-spin", type=float, default=0.5,
                   help="|mean caster spin| (rad/s) below which wave_speed_over_spin is not "
                        "reported. Default: 0.5")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Savitzky–Golay window (frames) for caster spin rates. Default: 7")
    p.add_argument("--out", type=str, default=None,
                   help="Summary CSV. Default: <parent of first folder>/parameter_set_summary.csv")
    return p.parse_args()


def parse_ld(name):
    l = re.search(r"(?:^|_)l(\d+)(?=_|$)", name)
    d = re.search(r"(?:^|_)d(\d+)(?=_|$)", name)
    return (int(l.group(1)) if l else np.nan), (int(d.group(1)) if d else np.nan)


def sg_deriv(x, dt, window):
    w = int(window) + (1 - int(window) % 2)
    if w < 5 or x.shape[0] <= w:
        return np.gradient(x, dt, axis=0)
    return savgol_filter(x, w, 3, deriv=1, delta=dt, axis=0)


def run_metrics(path, args):
    d = np.load(path, allow_pickle=True)
    need = ["modal_disp", "modal_disp_dot", "sector_modes", "sector_dim", "ring_headings",
            "order_param_null"]
    missing = [k for k in need if k not in d.files]
    if missing:
        raise ValueError(f"made by an older analyze_modes.py (missing {missing}); re-run analyze")
    t = d["time"].astype(float)
    fps = float(d["fps"])
    dt = 1.0 / fps
    keep = t >= t[0] + args.settle
    if keep.sum() < 10:
        raise ValueError("too short after the settle cut")
    N = len(d["nodes"])
    ref = d["ref"]
    R = float(np.mean(np.linalg.norm(ref - ref.mean(axis=0), axis=1)))
    m = {}

    # --- motion type --------------------------------------------------------- #
    KE_tot, KE_rig = d["KE_total"][keep], d["KE_zero"][keep]
    v_cm = d["v_cm"][keep]
    KE_trans = 0.5 * N * (v_cm ** 2).sum(axis=1)
    m["rigid_KE_fraction"] = float(np.nanmean(d["zero_mode_KE_ratio"][keep]))
    m["rot_fraction_of_rigid"] = float(np.clip((KE_rig - KE_trans).mean() / KE_rig.mean(), 0, 1)) \
        if KE_rig.mean() > 0 else np.nan
    om = d["omega"][keep]
    m["omega_mean"] = float(np.nanmean(om))
    m["omega_abs"] = abs(m["omega_mean"])
    m["omega_std"] = float(np.nanstd(om))
    m["com_speed"] = float(np.nanmean(np.linalg.norm(d["v_com"][keep], axis=1)))
    m["force_vel_alignment"] = float(np.nanmean(d["force_vel_cos"][keep]))

    # --- strain wave (recomputed on the settled window) ------------------------ #
    Q, Qd = d["modal_disp"][keep], d["modal_disp_dot"][keep]
    deform = np.asarray(d["deform_idx"], dtype=int)
    Qd2 = (Q[:, deform] ** 2).sum(axis=1)
    smods, sdims, sms = d["sector_modes"], d["sector_dim"], d["sector_m"]
    win = max(3, int(round(args.wave_window * fps)))

    def sm(x):
        return uniform_filter1d(x, size=win, mode="nearest")

    shares_t = []
    best = None
    for j, (mods, dim) in enumerate(zip(smods, sdims)):
        mods = [int(i) for i in mods if i >= 0]
        q2 = (Q[:, mods] ** 2).sum(axis=1)
        shares_t.append(np.divide(q2, Qd2, out=np.zeros_like(q2), where=Qd2 > 0))
        if int(dim) != 2:
            continue
        z = Q[:, mods[0]] + 1j * Q[:, mods[1]]
        zd = Qd[:, mods[0]] + 1j * Qd[:, mods[1]]
        L = np.imag(np.conj(z) * zd)
        den = np.abs(z) * np.abs(zd)
        share = q2.mean() / Qd2.mean() if Qd2.mean() > 0 else 0.0
        if best is None or share > best["share"]:
            sQd2 = sm(Qd2)
            eta_w = np.divide(sm(q2), sQd2, out=np.zeros_like(q2), where=sQd2 > 0)
            sden = sm(den)
            circ_w = np.divide(sm(L), sden, out=np.zeros_like(L), where=sden > 0)
            amp = np.abs(z)
            best = dict(j=j, share=share, mods=mods,
                        lam=L.mean() / den.mean() if den.mean() > 0 else 0.0,
                        om=L.mean() / (amp ** 2).mean() if (amp ** 2).mean() > 0 else 0.0,
                        cv=amp.std() / amp.mean() if amp.mean() > 0 else np.nan,
                        W_win=eta_w * circ_w)
    if best is not None:
        m["wave_share"] = float(best["share"])
        m["wave_circulation"] = float(best["lam"])
        m["wave_W"] = m["wave_share"] * m["wave_circulation"]
        m["wave_abs_win"] = float(np.mean(np.abs(best["W_win"])))
        m["wave_time_frac"] = float(np.mean(np.abs(best["W_win"]) > args.wave_threshold))
        m["wave_m"] = float(np.round(sms[best["j"]], 3))
        m["wave_amp_cv"] = float(best["cv"])
        m["wave_phase_speed"] = float(best["om"])
    else:
        for k in ("wave_share", "wave_circulation", "wave_W", "wave_abs_win", "wave_time_frac",
                  "wave_m", "wave_amp_cv", "wave_phase_speed"):
            m[k] = np.nan
    P = np.array(shares_t).T
    ssum = (P ** 2).sum(axis=1)
    m["sector_participation"] = float(np.nanmean(np.divide(1.0, ssum, out=np.full_like(ssum, np.nan),
                                                           where=ssum > 0)))
    m["rms_deformation"] = float(np.sqrt(Qd2.mean() / N) / R)

    # spectral peak sharpness of the dominant sector's first amplitude
    if best is not None and keep.sum() >= 64:
        f, pxx = welch(Q[:, best["mods"][0]] - Q[:, best["mods"][0]].mean(), fs=fps,
                       nperseg=min(1024, int(keep.sum())))
        pxx, f = pxx[1:], f[1:]
        if pxx.sum() > 0:
            k = int(np.argmax(pxx))
            m["spectral_peak_frac"] = float(pxx[max(k - 1, 0):k + 2].sum() / pxx.sum())
        else:
            m["spectral_peak_frac"] = np.nan
    else:
        m["spectral_peak_frac"] = np.nan

    # --- caster order ---------------------------------------------------------- #
    m["order_minus_null"] = float(np.nanmean(d["order_param"][keep]) - float(d["order_param_null"]))
    m["bond_alignment"] = float(np.nanmean(d["bond_align"][keep]))
    wnd = d["winding"][keep]
    m["winding_time_frac"] = float(np.mean(np.abs(np.round(wnd)) >= 1)) \
        if np.isfinite(wnd).any() else np.nan
    H = np.asarray(d["ring_headings"])[keep]
    if H.shape[1] >= 2:
        tt = t[keep]
        Hu = np.unwrap(H, axis=0)
        spins = np.array([np.polyfit(tt, Hu[:, i], 1)[0] for i in range(H.shape[1])])
        m["caster_spin_mean"] = float(spins.mean())
        m["caster_spin_abs"] = float(np.abs(spins).mean())
        # rotational diffusion about each node's mean spin
        Ds = []
        for i in range(H.shape[1]):
            ud = Hu[:, i] - spins[i] * (tt - tt[0])
            lags = np.arange(1, max(3, int(len(ud) * 0.25)))
            msd = np.array([np.mean((ud[l:] - ud[:-l]) ** 2) for l in lags])
            nfit = max(2, len(lags) // 2)
            Ds.append(np.polyfit(lags[:nfit] * dt, msd[:nfit], 1)[0] / 2.0)
        m["rot_diffusion"] = float(np.mean(Ds))
        gd = sg_deriv(Hu, dt, args.smooth_window)
        C = np.corrcoef(gd.T)
        off = C[~np.eye(C.shape[0], dtype=bool)]
        m["spin_sync"] = float(np.nanmean(off))
    else:
        m["caster_spin_mean"] = m["caster_spin_abs"] = m["rot_diffusion"] = m["spin_sync"] = np.nan
    # Only meaningful when a wave is present most of the time and the casters actually spin.
    wave_on = np.isfinite(m["wave_time_frac"]) and m["wave_time_frac"] >= 0.5
    spinning = np.isfinite(m["caster_spin_mean"]) and abs(m["caster_spin_mean"]) > args.min_spin
    m["wave_speed_over_spin"] = (m["wave_phase_speed"] / m["caster_spin_mean"]
                                 if wave_on and spinning else np.nan)
    m["duration_used_s"] = float(t[keep][-1] - t[keep][0])
    return m


def summarize_folder(folder, args):
    files = sorted(glob.glob(os.path.join(folder, "*_analysis.npz")))
    videos = sorted(p for p in glob.glob(os.path.join(folder, "*.mp4"))
                    if not p[:-4].endswith("_tagged"))
    have = {os.path.basename(f)[:-len("_analysis.npz")] for f in files}
    missing = [os.path.basename(v)[:-4] for v in videos if os.path.basename(v)[:-4] not in have]
    rows = []
    for f in files:
        name = os.path.basename(f)[:-len("_analysis.npz")]
        try:
            rows.append(dict(run=name, **run_metrics(f, args)))
        except Exception as e:                     # report and keep going
            print(f"  SKIP {name}: {e}")
    return rows, missing


def main():
    args = parse_args()
    folders = [f for f in args.folders if os.path.isdir(f)]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(folders[0])),
                                   "parameter_set_summary.csv")
    summary = []
    for folder in folders:
        name = os.path.basename(os.path.normpath(folder))
        print(f"\n=== {name} ===")
        rows, missing = summarize_folder(folder, args)
        if missing:
            print(f"  {len(missing)} video(s) without *_analysis.npz (run the pipeline first): "
                  + ", ".join(missing))
        if not rows:
            print("  no usable runs")
            continue
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(folder, "parameter_set_runs.csv"), index=False)
        l, dd = parse_ld(name)
        rec = dict(parameter_set=name, l=l, d=dd, n_runs=len(df))
        print(f"  {len(df)} run(s), settle {args.settle:g} s\n")
        print(f"  {'metric':<24} {'mean':>11} {'std':>11}")
        for k in METRICS:
            v = df[k].to_numpy(dtype=float)
            mu = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            sd = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan
            rec[f"{k}_mean"], rec[f"{k}_std"] = mu, sd
            print(f"  {k:<24} {mu:>11.4g} {sd:>11.3g}")
        summary.append(rec)
    if summary:
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"\nWrote per-run tables to each folder (parameter_set_runs.csv) and the summary -> {out}")


if __name__ == "__main__":
    main()
