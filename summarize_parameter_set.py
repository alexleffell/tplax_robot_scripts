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
    rms_deformation          rms per-node body-frame deformation / ring radius (incl. static part)
    static_deformation_frac  share of ⟨|q|²⟩ that is the run's constant mean shape (excluded from
                             all sector shares and wave metrics, which use deviations from it)
    sector_participation     ⟨1/Σ_j share_j²⟩  (number of active deformation patterns)
    spin_sync                mean off-diagonal Pearson correlation of caster spin rates γ̇
  Heading wave (twisted states of the body-frame caster headings; see
  analyze_modes.heading_wave_stats)
    heading_H                share_q* · Λ_q*  (signed; ±1 = clean travelling twist)
    heading_abs_win          ⟨|H_win|⟩;  heading_time_frac: fraction of time |H_win| > --wave-threshold
    heading_q                dominant twist number q* (±1 = vortex, the winding number)
    heading_share, heading_circulation   share and circulation of q*
    heading_travel_speed     speed at which the heading pattern travels round the ring (rad/s,
                             CCW positive; NaN for q = 0 and the alias-ambiguous q = N/2)
    heading_flock_share      share of the uniform (q = 0, flocking) component
    heading_spin             rotation rate of ψ_q* (rad/s) = common caster spin in the twist
    heading_strain_freq_ratio    heading-wave rate / strain-wave rate (Ω_q* / Ω_strain); 1 = the
                             casters turn once per strain-wave cycle (1:1 locking). Pattern speeds
                             differ by geometry (a q-twist travels at Ω/q, an m-lobed strain
                             pattern at Ω/m), so rates, not pattern speeds, are compared. Only
                             when both waves are present ≥ 50 % of the time.
  Regularity
    rot_diffusion            heading MSD slope/2 after removing each node's mean spin (rad²/s)
    spectral_peak_frac       fraction of power in the dominant spectral peak of the dominant
                             sector's amplitude (→1 periodic limit cycle, small = irregular)

Outputs
  <folder>/parameter_set_runs.csv     one row per run
  --out (default <parent>/parameter_set_summary.csv)  one row per folder: l, d (parsed from the
                                      folder name, ``l<i>`` / ``d<i>``), n_runs, <metric>_mean, _std
  and a printed table per folder.
  <summary dir>/parameter_set_plots/<metric>.png   (with ≥ 2 parameter sets) mean ± std of each
                                      metric vs l/d, vs atan(l/d), and an l × d heatmap of the
                                      mean (discrete cells); l, d in mm from the folder indices
                                      (--l-step, --d-step); colour = l, marker = d. d = 0 sets
                                      appear only in the atan panel (90°) and the heatmap.
  State classification (per run; thresholds are options, see classify_run):
    rotor  |⟨ω⟩| ≥ --rotor-omega and CoM speed < --rotor-speed   (spins rigidly in place)
    wave   wave_time_frac ≥ --wave-frac and |heading_H| ≥ --wave-heading (locked strain wave)
    flock  heading_flock_share ≥ --flock-share and CoM speed ≥ --flock-speed (translates)
    mixed  otherwise
  -> 'state' column in each parameter_set_runs.csv; state_<s>_frac per set in the summary;
     state_means.csv (per-state mean ± std of key metrics over all runs);
     state_sensitivity.csv (each threshold ±20 %, one at a time: min/max of every fraction);
     parameter_set_plots/state_map.(png|pdf) (stacked bar of state fractions per l × d cell) and
     state_heatmaps.(png|pdf) (W, flocking share, |⟨ω⟩| over l × d).

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter, welch

from analyze_modes import heading_wave_stats


METRICS = [
    "rigid_KE_fraction", "rot_fraction_of_rigid", "omega_mean", "omega_abs", "omega_std",
    "com_speed", "force_vel_alignment",
    "wave_W", "wave_abs_win", "wave_time_frac", "wave_m", "wave_share", "wave_circulation",
    "wave_amp_cv", "wave_phase_speed", "wave_speed_over_spin",
    "order_minus_null", "bond_alignment", "winding_time_frac", "caster_spin_mean",
    "caster_spin_abs",
    "rms_deformation", "static_deformation_frac", "sector_participation", "spin_sync",
    "rot_diffusion", "spectral_peak_frac",
    "heading_H", "heading_abs_win", "heading_time_frac", "heading_q", "heading_share",
    "heading_circulation", "heading_travel_speed", "heading_flock_share",
    "heading_spin", "heading_strain_freq_ratio",
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
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per l index (mm). Default: 5")
    p.add_argument("--d-step", type=float, default=6.43,
                   help="Perpendicular offset per d index (mm). Default: 6.43")
    p.add_argument("--plot-dir", type=str, default=None,
                   help="Directory for metric-vs-geometry plots. Default: <summary dir>/"
                        "parameter_set_plots/")
    p.add_argument("--no-plots", action="store_true", help="Skip the metric-vs-geometry plots.")
    g = p.add_argument_group("state classification (per run; see classify_run)")
    g.add_argument("--rotor-omega", type=float, default=1.5,
                   help="rotor: |mean body rotation| ≥ this (rad/s). Default: 1.5")
    g.add_argument("--rotor-speed", type=float, default=0.15,
                   help="rotor: CoM speed < this (m/s). Default: 0.15")
    g.add_argument("--wave-frac", type=float, default=0.5,
                   help="wave: fraction of time in the strain wave ≥ this. Default: 0.5")
    g.add_argument("--wave-heading", type=float, default=0.3,
                   help="wave: |heading-wave H| ≥ this. Default: 0.3")
    g.add_argument("--flock-share", type=float, default=0.5,
                   help="flock: flocking (q=0) heading share ≥ this. Default: 0.5")
    g.add_argument("--flock-speed", type=float, default=0.25,
                   help="flock: CoM speed ≥ this (m/s). Default: 0.25")
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
    m["rms_deformation"] = float(np.sqrt((Q[:, deform] ** 2).sum(axis=1).mean() / N) / R)
    # shares and the wave use deviations from this run's mean shape (a static offset such as a
    # ring sitting slightly smaller than the rest spacing is not dynamics)
    Qm = Q.mean(axis=0)
    m["static_deformation_frac"] = float((Qm[deform] ** 2).sum() / (Q[:, deform] ** 2).sum(axis=1).mean())
    Q = Q - Qm[None, :]
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
    # --- heading wave (body frame, CCW ring order) -------------------------------- #
    if "ring_headings_body" in d.files and np.asarray(d["ring_headings_body"]).shape[1] >= 3:
        Hb = np.asarray(d["ring_headings_body"])[keep]
        hw = heading_wave_stats(Hb, dt, args.smooth_window, win)
        j = hw["j_star"]
        m.update(heading_H=hw["H"], heading_abs_win=hw["abs_win"],
                 heading_time_frac=float(np.mean(np.abs(hw["H_win"]) > args.wave_threshold)),
                 heading_q=float(hw["q_star"]), heading_share=float(hw["share"][j]),
                 heading_circulation=float(hw["circulation"][j]),
                 heading_travel_speed=float(hw["travel_speed"][j]),
                 heading_flock_share=hw["flock_share"])
        m["heading_spin"] = float(hw["spin"][j])
        both = (m["heading_time_frac"] >= 0.5 and np.isfinite(m["wave_time_frac"])
                and m["wave_time_frac"] >= 0.5)
        ws = m["wave_phase_speed"]
        m["heading_strain_freq_ratio"] = (m["heading_spin"] / ws
                                          if both and np.isfinite(ws) and abs(ws) > 1e-9 else np.nan)
    else:
        for k in ("heading_H", "heading_abs_win", "heading_time_frac", "heading_q",
                  "heading_share", "heading_circulation", "heading_travel_speed",
                  "heading_flock_share", "heading_spin", "heading_strain_freq_ratio"):
            m[k] = np.nan
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


STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
D_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


STATES = ["wave", "flock", "rotor", "mixed"]
STATE_COLORS = {"wave": "#e67e22", "flock": "#17a589", "rotor": "#7d3c98", "mixed": "0.78"}  # red/blue are reserved for CCW/CW
THRESH = ["rotor_omega", "rotor_speed", "wave_frac", "wave_heading", "flock_share", "flock_speed"]
STATE_MEAN_KEYS = ["wave_W", "heading_H", "heading_strain_freq_ratio", "caster_spin_mean",
                   "omega_mean", "com_speed", "rigid_KE_fraction", "heading_flock_share",
                   "bond_alignment", "order_minus_null"]


def thresholds(args):
    return {k: getattr(args, k) for k in THRESH}


def classify_run(r, th):
    """rotor: |ω| ≥ rotor_omega and CoM speed < rotor_speed (spins rigidly in place);
    wave: strain wave ≥ wave_frac of the time and |heading wave H| ≥ wave_heading (locked
    shear + heading twist); flock: q=0 heading share ≥ flock_share and CoM speed ≥ flock_speed;
    otherwise mixed. Checked in that order; NaN fails every test."""
    if abs(r["omega_mean"]) >= th["rotor_omega"] and r["com_speed"] < th["rotor_speed"]:
        return "rotor"
    if r["wave_time_frac"] >= th["wave_frac"] and abs(r["heading_H"]) >= th["wave_heading"]:
        return "wave"
    if r["heading_flock_share"] >= th["flock_share"] and r["com_speed"] >= th["flock_speed"]:
        return "flock"
    return "mixed"


def state_fractions(runs, th):
    st = runs.apply(lambda r: classify_run(r, th), axis=1)
    fr = pd.crosstab(runs["parameter_set"], st, normalize="index")
    return fr.reindex(columns=STATES, fill_value=0.0)


def state_sensitivity(runs, th, out_csv):
    """Re-classify with each threshold moved ±20 % (one at a time); report how far each
    set's state fractions move."""
    base = state_fractions(runs, th)
    lo, hi = base.copy(), base.copy()
    worst = {}
    for k in THRESH:
        for f in (0.8, 1.2):
            th2 = dict(th); th2[k] = th[k] * f
            fr = state_fractions(runs, th2).reindex(index=base.index, fill_value=0.0)
            lo, hi = np.minimum(lo, fr), np.maximum(hi, fr)
            worst[f"{k} x{f}"] = float((fr - base).abs().values.max())
    rows = []
    for sname in base.index:
        for stt in STATES:
            rows.append(dict(parameter_set=sname, state=stt, base=base.loc[sname, stt],
                             min=lo.loc[sname, stt], max=hi.loc[sname, stt]))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    span = (hi - lo).values
    print(f"\nState-classification sensitivity (each threshold ±20 %, one at a time): "
          f"largest change of any state fraction in any set {span.max():.2f}, "
          f"median {np.median(span):.2f}.")
    print("  largest change per perturbation: "
          + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6]))
    print(f"  -> {out_csv}")


def lmm_dmm(sm, args):
    sm = sm.copy()
    sm["l_mm"], sm["d_mm"] = sm["l"] * args.l_step, sm["d"] * args.d_step
    return sm


def heatmap(ax, sm, key, args, title=None, fmt="{:.2f}"):
    """Discrete l × d cells (no interpolation) of `key`, value printed in each cell."""
    ls, ds = sorted(sm["l"].unique()), sorted(sm["d"].unique())
    M = np.full((len(ls), len(ds)), np.nan)
    for _, r in sm.iterrows():
        M[ls.index(r["l"]), ds.index(r["d"])] = r[key]
    fin = M[np.isfinite(M)]
    if fin.size and fin.min() < 0 < fin.max():
        v = np.abs(fin).max(); cmap, vmin, vmax = plt.get_cmap("RdBu_r").copy(), -v, v
    else:
        cmap = plt.get_cmap("viridis").copy()
        vmin, vmax = (fin.min(), fin.max()) if fin.size else (0, 1)
    cmap.set_bad("0.9")
    im = ax.imshow(np.ma.masked_invalid(M), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    for i in range(len(ls)):
        for j in range(len(ds)):
            if np.isfinite(M[i, j]):
                x = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                dark = x < 0.35 or (cmap.name.startswith("RdBu") and x > 0.8)
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="w" if dark else "k")
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([f"{d * args.d_step:.3g}" for d in ds])
    ax.set_yticks(range(len(ls))); ax.set_yticklabels([f"{l * args.l_step:g}" for l in ls])
    ax.set_xlabel("d (mm)"); ax.set_ylabel("l (mm)")
    ax.spines[["top", "right"]].set_visible(True)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)


def plot_vs_geometry(sm, args, plot_dir):
    """One figure per metric: mean ± std over runs vs l/d, vs atan(l/d), and an l × d heatmap of
    the mean (one cell per parameter set). Colour = l, marker = d in the first two panels. d = 0
    sets appear only in the atan panel (90°) and the heatmap."""
    sm = sm[np.isfinite(sm["l"]) & np.isfinite(sm["d"])].copy()
    if len(sm) < 2:
        print("  (fewer than 2 parameter sets with l and d in their names; no geometry plots)")
        return
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update(STYLE)
    sm = lmm_dmm(sm, args)
    sm["ratio"] = np.where(sm["d_mm"] > 0, sm["l_mm"] / sm["d_mm"].where(sm["d_mm"] > 0), np.nan)
    sm["angle_deg"] = np.degrees(np.arctan2(sm["l_mm"], sm["d_mm"]))
    ls, ds = sorted(sm["l"].unique()), sorted(sm["d"].unique())
    cmap = plt.get_cmap("viridis")
    col = {l: cmap(i / max(len(ls) - 1, 1)) for i, l in enumerate(ls)}
    mk = {d: D_MARKERS[i % len(D_MARKERS)] for i, d in enumerate(ds)}
    n0 = int((sm["d_mm"] <= 0).sum())
    amin, amax = sm["angle_deg"].min(), sm["angle_deg"].max()
    pad = max(2.0, 0.05 * (amax - amin))
    metrics = METRICS + [f"state_{s}_frac" for s in STATES if f"state_{s}_frac_mean" in sm]
    for k in metrics:
        if f"{k}_mean" not in sm or not np.isfinite(sm[f"{k}_mean"]).any():
            continue
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 2.9),
                                         gridspec_kw={"width_ratios": [1, 1, 1.05]})
        a2.sharey(a1)
        for _, r in sm.iterrows():
            y, e = r[f"{k}_mean"], r[f"{k}_std"]
            e = e if np.isfinite(e) else 0.0
            kw = dict(yerr=e, fmt=mk[r["d"]], ms=5, capsize=2.5, lw=0.9, color=col[r["l"]],
                      mec="k", mew=0.4)
            if np.isfinite(r["ratio"]):
                a1.errorbar(r["ratio"], y, **kw)
            a2.errorbar(r["angle_deg"], y, **kw)
        a1.set_xlabel(r"$l/d$")
        a1.set_ylabel(k.replace("_", " "))
        if n0:
            a1.set_title(f"(d = 0 sets omitted: l/d undefined; n = {n0})", fontsize=7, color="0.4")
        a2.set_xlabel(r"$\arctan(l/d)$ (deg)")
        a2.set_xlim(amin - pad, amax + pad)
        hl = [plt.Line2D([], [], marker="o", ls="", color=col[l], mec="k", mew=0.4, ms=4)
              for l in ls]
        hd = [plt.Line2D([], [], marker=mk[d], ls="", color="0.6", mec="k", mew=0.4, ms=4)
              for d in ds]
        a1.legend(hl + hd, [f"l = {l * args.l_step:g} mm" for l in ls]
                  + [f"d = {d * args.d_step:.3g} mm" for d in ds], frameon=False, fontsize=6,
                  ncol=2, loc="best", handletextpad=0.2, columnspacing=0.6)
        heatmap(a3, sm, f"{k}_mean", args, title="mean over runs")
        fig.suptitle(f"{k}  (mean ± std over runs)", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{k}.png"), dpi=200)
        plt.close(fig)
    print(f"Wrote metric-vs-geometry plots -> {plot_dir}/  ({len(sm)} parameter sets)")


def plot_state_map(sm, args, plot_dir):
    """l × d grid; in each cell a stacked bar of the fraction of runs in each state."""
    sm = sm[np.isfinite(sm["l"]) & np.isfinite(sm["d"])]
    ls, ds = sorted(sm["l"].unique()), sorted(sm["d"].unique())
    fig, ax = plt.subplots(figsize=(1.25 * len(ds) + 1.6, 1.05 * len(ls) + 0.9))
    for _, r in sm.iterrows():
        j, i = ds.index(r["d"]), ls.index(r["l"])
        x0 = j - 0.42
        for stt in STATES:
            f = r.get(f"state_{stt}_frac_mean", 0.0)
            if f > 0:
                ax.add_patch(plt.Rectangle((x0, i - 0.32), 0.84 * f, 0.64, color=STATE_COLORS[stt],
                                           lw=0))
                if f >= 0.2:
                    ax.text(x0 + 0.42 * f, i, f"{f:.0%}", ha="center", va="center", fontsize=6,
                            color="w" if stt != "mixed" else "k")
                x0 += 0.84 * f
        ax.add_patch(plt.Rectangle((j - 0.42, i - 0.32), 0.84, 0.64, fill=False, lw=0.5,
                                   ec="0.4"))
        ax.text(j, i + 0.38, f"n={int(r['n_runs'])}", ha="center", va="bottom", fontsize=5.5,
                color="0.4")
    ax.set_xlim(-0.5, len(ds) - 0.5); ax.set_ylim(-0.5, len(ls) - 0.5)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([f"{d * args.d_step:.3g}" for d in ds])
    ax.set_yticks(range(len(ls))); ax.set_yticklabels([f"{l * args.l_step:g}" for l in ls])
    ax.set_xlabel("perpendicular offset d (mm)"); ax.set_ylabel("caster offset l (mm)")
    ax.spines[["top", "right"]].set_visible(True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in STATES], STATES,
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.set_title("fraction of runs in each state")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "state_map.png"), dpi=220)
    fig.savefig(os.path.join(plot_dir, "state_map.pdf"))
    plt.close(fig)


def plot_state_heatmaps(sm, args, plot_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, (k, t) in zip(axes, [("wave_W_mean", "strain wave W"),
                                 ("heading_flock_share_mean", "flocking share (heading q = 0)"),
                                 ("omega_abs_mean", r"body rotation $|\langle\omega\rangle|$ (rad/s)")]):
        heatmap(ax, sm, k, args, title=t)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "state_heatmaps.png"), dpi=220)
    fig.savefig(os.path.join(plot_dir, "state_heatmaps.pdf"))
    plt.close(fig)


def state_means(runs, out_csv):
    g = runs.groupby("state")
    rows = []
    for stt in STATES:
        if stt not in g.groups:
            continue
        sub = g.get_group(stt)
        rec = dict(state=stt, n_runs=len(sub))
        for k in STATE_MEAN_KEYS:
            rec[f"{k}_mean"] = float(np.nanmean(sub[k])) if np.isfinite(sub[k]).any() else np.nan
            rec[f"{k}_std"] = float(np.nanstd(sub[k], ddof=1)) if np.isfinite(sub[k]).sum() > 1 else np.nan
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(out_csv, index=False)
    print("\nPer-state means over all runs (mean ± std):")
    print(f"  {'state':<6} {'n':>4}  " + "  ".join(f"{k[:14]:>15}" for k in STATE_MEAN_KEYS))
    for _, r in t.iterrows():
        print(f"  {r['state']:<6} {int(r['n_runs']):>4}  " + "  ".join(
            f"{r[k + '_mean']:>7.3g} ± {r[k + '_std']:<5.2g}" for k in STATE_MEAN_KEYS))
    print(f"  -> {out_csv}")


def main():
    args = parse_args()
    th = thresholds(args)
    folders = [f for f in args.folders if os.path.isdir(f)]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(folders[0])),
                                   "parameter_set_summary.csv")
    outdir = os.path.dirname(os.path.abspath(out))
    summary, all_runs = [], []
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
        df["state"] = df.apply(lambda r: classify_run(r, th), axis=1)
        df.to_csv(os.path.join(folder, "parameter_set_runs.csv"), index=False)
        l, dd = parse_ld(name)
        all_runs.append(df.assign(parameter_set=name, l=l, d=dd))
        rec = dict(parameter_set=name, l=l, d=dd, n_runs=len(df))
        print(f"  {len(df)} run(s), settle {args.settle:g} s\n")
        print(f"  {'metric':<24} {'mean':>11} {'std':>11}")
        for k in METRICS:
            v = df[k].to_numpy(dtype=float)
            mu = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            sd = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan
            rec[f"{k}_mean"], rec[f"{k}_std"] = mu, sd
            print(f"  {k:<24} {mu:>11.4g} {sd:>11.3g}")
        counts = df["state"].value_counts()
        for stt in STATES:
            f = float(counts.get(stt, 0)) / len(df)
            rec[f"state_{stt}_frac_mean"], rec[f"state_{stt}_frac_std"] = f, np.nan
        print("  states: " + ", ".join(f"{stt} {int(counts.get(stt, 0))}" for stt in STATES))
        summary.append(rec)
    if not summary:
        return
    sm = pd.DataFrame(summary)
    sm.to_csv(out, index=False)
    print(f"\nWrote per-run tables to each folder (parameter_set_runs.csv, with a 'state' column) "
          f"and the summary -> {out}")
    runs = pd.concat(all_runs, ignore_index=True)
    state_means(runs, os.path.join(outdir, "state_means.csv"))
    if len(sm) > 1:
        state_sensitivity(runs, th, os.path.join(outdir, "state_sensitivity.csv"))
    if not args.no_plots:
        plot_dir = args.plot_dir or os.path.join(outdir, "parameter_set_plots")
        plot_vs_geometry(sm, args, plot_dir)
        smg = sm[np.isfinite(sm["l"]) & np.isfinite(sm["d"])]
        if len(smg) > 1:
            plot_state_map(smg, args, plot_dir)
            plot_state_heatmaps(smg, args, plot_dir)
            print(f"Wrote state_map and state_heatmaps (.png/.pdf) -> {plot_dir}/")


if __name__ == "__main__":
    main()
