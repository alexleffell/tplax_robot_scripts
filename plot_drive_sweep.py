#!/usr/bin/env python3
"""
Single node in an elastic well: behaviour vs motor drive (PWM) and caster offset.

Replaces plot_hopf_metrics.py + plot_hopf_bifurcation.py with one script, one
definition of the well centre, and the paper figure style (plot_kinematics_tracks.py).

Input: a directory of format_tracks_single.py outputs named ``...m<pwm>_l<l>[_d<d>]..._robot.csv``;
each file is one (drive, offset) condition and each ``track`` in it is one trial
(release). The first --settle seconds of every trial are dropped (release transient).

Well centre (one per offset l): mean position of the trials that do not rotate
(heading completes fewer than --n-rev turns); if every trial rotates, the trials at
the lowest drive. Every radius and polar angle below is measured from this centre.

Per trial
  R            rms distance from the centre, ⟨r²⟩^{1/2}
  |γ̇|          |mean heading rate| (unwrap → zero-phase Butterworth → d/dt)
  |Δγ|         net heading rotation (same filter)
  |φ̇|          |mean polar-angle rate| about the centre (samples with r ≥ --r-min)
  L, ω_L       circulation ⟨xẏ − yẋ⟩ and ω_L = L / ⟨r²⟩  (L > 0 CCW, L < 0 CW)
  orbiting     polar angle winds ≥ --min-turns turns and median r ≥ --orbit-r-min;
               direction from the sign of L (same rule as plot_chirality_fraction.py)
  n-rev window heading rate, R, polar rate and time over the first --n-rev heading turns
  peak freq    spectral peak of x(t) (Welch)

Onset drive per l
  m*_50        drive where the orbiting fraction first reaches 0.5 (linear interpolation);
               used for the vertical markers and for picking example drives
  square-root  R² = B + A·max(m − m*, 0) fitted to per-drive medians, plus linear-onset
               and step alternatives at the same m* (drawn on the R figure)

Outputs (to --outdir, default <directory>/drive_sweep_plots/):
  01_orbit_radius  02_orbit_radius_sq  03_orbit_fraction  04_heading_rate
  05_heading_rotation  06_polar_rate  07_orbit_rate_signed  08_orbit_rate_abs
  09_circulation  10_peak_freq  11–15 first-n-rev metrics
  16_trajectories_l<l>  17_spectra_l<l>
  drive_figure.pdf/.png  paper figure: (a) example trials below / above onset for one
                         offset, (b) fraction of trials orbiting CW vs PWM, (c) R vs PWM
  trials.csv  summary.csv  onset.csv

Example
-------
    python plot_drive_sweep.py ../Data/250926
    python plot_drive_sweep.py ../Data/250926 --pdf
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, welch

from plot_kinematics_tracks import load_robot
from reduce_single import geometric_circle, savgol_deriv


NAME_RE = re.compile(r"m(?P<m>\d+)_l(?P<l>\d+)(?:_d(?P<d>\d+))?", re.IGNORECASE)
C_CW, C_CCW, C_NONE = "#3a73b0", "#c0392b", "0.6"
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
PWM_LABEL = r"motor PWM $m$ (8-bit)"


def parse_args():
    p = argparse.ArgumentParser(description="Single-node behaviour vs motor drive and offset")
    p.add_argument("directory", nargs="?",
                   default="/Users/alexleffell/Documents/PhD/tplax/Data/250926",
                   help="Directory of *m<pwm>_l<l>*_robot.csv files. Default: Data/250926")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir. Default: <directory>/drive_sweep_plots/")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Seconds dropped from the start of each trial. Default: 1")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Minimum samples after settling to keep a trial. Default: 20")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="Butterworth low-pass cutoff (Hz) for unwrapped angles. Default: 5")
    p.add_argument("--butter-order", type=int, default=4, help="Butterworth order. Default: 4")
    p.add_argument("--n-rev", type=int, default=5,
                   help="Heading turns that define a rotating trial and the n-rev window. "
                        "Default: 5")
    p.add_argument("--min-turns", type=float, default=1.0,
                   help="Polar turns about the centre to count as orbiting. Default: 1")
    p.add_argument("--orbit-r-min", type=float, default=0.003,
                   help="Minimum median radius (m) to count as orbiting. Default: 0.003")
    p.add_argument("--r-min", type=float, default=0.005,
                   help="Minimum radius (m) for polar-angle rates and ω_L. Default: 0.005")
    p.add_argument("--savgol-window", type=float, default=0.35,
                   help="Savitzky–Golay window (s) for ẋ, ẏ, φ̇. Default: 0.35")
    p.add_argument("--savgol-poly", type=int, default=3, help="Savitzky–Golay order. Default: 3")
    p.add_argument("--l-step", type=float, default=5.0,
                   help="Caster offset per l index (mm); 0 = label by index. Default: 5")
    p.add_argument("--l-color-max", type=float, default=5.0,
                   help="l index mapped to the top of the viridis colormap (matches the rail "
                        "figure, l0–l5). Default: 5")
    p.add_argument("--n-show", type=int, default=6,
                   help="Trials overlaid in trajectory / spectra figures. Default: 6")
    p.add_argument("--tmax", type=float, default=4.0,
                   help="Seconds shown in trajectory time series. Default: 4")
    p.add_argument("--fig-lengths", type=int, nargs="+", default=None,
                   help="l indices shown in drive_figure (e.g. 2 3 4). Default: all")
    p.add_argument("--fig-example-l", type=int, default=3,
                   help="l index for the example trajectories in drive_figure. Default: 3")
    p.add_argument("--pdf", action="store_true", help="Also write a PDF of every figure.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def dt_of(t):
    d = np.diff(np.asarray(t, dtype=float))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.nan


def lowpass(sig, dt, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass; returns `sig` unchanged if it cannot run."""
    sig = np.asarray(sig, dtype=float)
    if sig.size < 8 or not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
        return sig
    wn = cutoff_hz / (0.5 / dt)
    if not 0.0 < wn < 1.0:
        return sig
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if sig.size <= padlen:
        return sig
    return filtfilt(b, a, sig, padlen=padlen)


def filtered_unwrap(angle, dt, cutoff, order):
    return lowpass(np.unwrap(np.asarray(angle, dtype=float)), dt, cutoff, order)


def n_rev_crossing(t, g_f, n_rev):
    """(time from start until |Δγ| reaches n_rev turns, slice end); (nan, 0) if never."""
    mag = np.abs(g_f - g_f[0])
    target = n_rev * 2.0 * np.pi
    hit = np.flatnonzero(mag >= target)
    if n_rev <= 0 or not hit.size:
        return np.nan, 0
    i = int(hit[0])
    if i == 0:
        tc = float(t[0])
    else:
        y0, y1 = mag[i - 1], mag[i]
        f = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
        tc = float(t[i - 1] + f * (t[i] - t[i - 1]))
    return tc - float(t[0]), int(np.searchsorted(t, tc, side="right"))


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    src: str
    m: float
    l: int
    d: int
    track: int
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    g: np.ndarray                       # heading γ (wrapped)
    g_f: np.ndarray = field(default_factory=lambda: np.array([]))
    rotating: bool = False
    t_nrev: float = np.nan
    end_nrev: int = 0
    # filled relative to the well centre
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_u: np.ndarray = field(default_factory=lambda: np.array([]))
    xdot: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_f: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_p: np.ndarray = field(default_factory=lambda: np.array([]))
    scal: dict = field(default_factory=dict)


def load_trials(directory, args):
    files = sorted(glob.glob(os.path.join(directory, "*_robot.csv")))
    if not files:
        raise SystemExit(f"No *_robot.csv files in {directory}")
    trials = []
    for path in files:
        mm = NAME_RE.search(os.path.basename(path))
        if mm is None:
            print(f"  WARNING: no m/l in {os.path.basename(path)}; skipping")
            continue
        m, l = float(mm.group("m")), int(mm.group("l"))
        d = int(mm.group("d")) if mm.group("d") is not None else -1
        df, nid, th_col, x_col, y_col = load_robot(path)
        n = 0
        for tr_id, sub in df.groupby("track", sort=True):
            t = sub["time"].to_numpy(dtype=float)
            keep = t >= t[0] + args.settle
            arr = [sub[c].to_numpy(dtype=float)[keep] for c in (x_col, y_col, th_col)]
            t = t[keep]
            ok = np.isfinite(t) & np.isfinite(arr[0]) & np.isfinite(arr[1]) & np.isfinite(arr[2])
            if ok.sum() < args.min_samples:
                continue
            tr = Trial(os.path.basename(path), m, l, d, int(tr_id), t[ok], arr[0][ok],
                       arr[1][ok], arr[2][ok])
            dt = dt_of(tr.t)
            tr.g_f = filtered_unwrap(tr.g, dt, args.cutoff, args.butter_order)
            tr.t_nrev, tr.end_nrev = n_rev_crossing(tr.t, tr.g_f, args.n_rev)
            tr.rotating = bool(np.isfinite(tr.t_nrev))
            trials.append(tr)
            n += 1
        print(f"{os.path.basename(path)}: m={m:g} l={l}  {n} trial(s)")
    if len({tr.d for tr in trials}) > 1:
        print(f"  NOTE: several d values {sorted({tr.d for tr in trials})}; grouping by l only")
    return trials


def well_centre(trials, l):
    group = [tr for tr in trials if tr.l == l]
    quiet = [tr for tr in group if not tr.rotating]
    if not quiet:
        m0 = min(tr.m for tr in group)
        quiet = [tr for tr in group if tr.m == m0]
    return (float(np.mean(np.concatenate([tr.x for tr in quiet]))),
            float(np.mean(np.concatenate([tr.y for tr in quiet]))))


def polar_rate(phi_u, r, dt, args):
    ok = np.isfinite(r) & (r >= args.r_min)
    if ok.sum() < 5 or not np.isfinite(dt):
        return np.nan
    return float(np.nanmean(savgol_deriv(phi_u, dt, args.savgol_window, args.savgol_poly)[ok]))


def fill_trial(tr, xc, yc, args, fmin=0.15):
    dt = dt_of(tr.t)
    x0, y0 = tr.x - xc, tr.y - yc
    tr.r = np.hypot(x0, y0)
    tr.phi_u = np.unwrap(np.arctan2(y0, x0))
    tr.xdot = savgol_deriv(x0, dt, args.savgol_window, args.savgol_poly)
    ydot = savgol_deriv(y0, dt, args.savgol_window, args.savgol_poly)
    r2 = float(np.mean(tr.r ** 2))
    L = float(np.nanmean(x0 * ydot - y0 * tr.xdot))
    turns = float(tr.phi_u[-1] - tr.phi_u[0]) / (2 * np.pi)
    orbit = abs(turns) >= args.min_turns and float(np.median(tr.r)) >= args.orbit_r_min
    s = dict(R=np.sqrt(r2), R2=r2, L=L, polar_turns=turns,
             omega_L=L / r2 if r2 > args.r_min ** 2 else np.nan,
             heading_rate=abs(float(np.mean(np.gradient(tr.g_f, tr.t)))),
             heading_rotation=abs(float(tr.g_f[-1] - tr.g_f[0])),
             polar_rate=abs(polar_rate(tr.phi_u, tr.r, dt, args)),
             orbiting=orbit, direction=("CCW" if L > 0 else "CW") if orbit else "none",
             R_fit=np.nan, peak_freq=np.nan, time_to_nrev=tr.t_nrev,
             heading_rate_nrev=np.nan, R_nrev=np.nan, polar_rate_nrev=np.nan)
    s["omega_abs"] = abs(s["omega_L"])
    s["R_mm"], s["R2_mm2"] = 1e3 * s["R"], 1e6 * s["R2"]
    if orbit and x0.size >= 8:
        try:
            _, _, Rfit, rms = geometric_circle(x0, y0)
            if np.isfinite(Rfit) and rms < max(3.0 * Rfit, 1e-4):
                s["R_fit"] = float(Rfit)
        except Exception:
            pass
    e = tr.end_nrev
    if tr.rotating and e >= args.min_samples:
        s["heading_rate_nrev"] = abs(float(np.mean(np.gradient(tr.g_f[:e], tr.t[:e]))))
        s["R_nrev"] = 1e3 * float(np.sqrt(np.mean(tr.r[:e] ** 2)))
        s["polar_rate_nrev"] = abs(polar_rate(tr.phi_u[:e], tr.r[:e], dt, args))
    if x0.size >= 64 and np.isfinite(dt):
        nper = max(32, int(min(512, max(64, 2 * (x0.size // 4)))) // 2 * 2)
        if nper < x0.size:
            f, pxx = welch(x0 - x0.mean(), fs=1.0 / dt, nperseg=nper, detrend="constant")
            tr.spec_f, tr.spec_p = f, pxx
            band = (f >= fmin) & (f <= 0.45 / dt)
            if band.any() and np.nanmax(pxx[band]) > 0:
                s["peak_freq"] = float(f[band][np.argmax(pxx[band])])
    tr.scal = s


# --------------------------------------------------------------------------- #
# Onset
# --------------------------------------------------------------------------- #
def rmse(y, yhat):
    dlt = np.asarray(yhat, float) - np.asarray(y, float)
    dlt = dlt[np.isfinite(dlt)]
    return float(np.sqrt(np.mean(dlt ** 2))) if dlt.size else np.nan


def fit_sqrt_onset(M, R2):
    """R² = B + A·max(M − m*, 0), A, B ≥ 0 (i.e. R ∝ √(m − m*) above onset)."""
    ok = np.isfinite(M) & np.isfinite(R2)
    M, R2 = M[ok], R2[ok]
    nan = dict(m_star=np.nan, A=np.nan, B=np.nan, rmse=np.nan)
    if M.size < 6:
        return nan
    lo, hi, ymin, ymax = M.min(), M.max(), np.nanmin(R2), np.nanmax(R2)
    best, cost = None, np.inf
    for m0 in np.linspace(lo, hi, 9):
        p0 = [m0, max(ymax - ymin, 1e-16) / max(hi - lo, 1.0), max(ymin, 0.0)]
        try:
            res = least_squares(lambda p: R2 - (p[2] + p[1] * np.maximum(M - p[0], 0.0)), p0,
                                bounds=([lo - 0.5 * (hi - lo), 0.0, 0.0],
                                        [hi + 0.5 * (hi - lo), np.inf, ymax + 1e-9]))
        except ValueError:
            continue
        if res.cost < cost:
            best, cost = res, res.cost
    if best is None:
        return nan
    ms, A, B = (float(v) for v in best.x)
    return dict(m_star=ms, A=A, B=B, rmse=rmse(R2, B + A * np.maximum(M - ms, 0.0)))


def fit_linear_onset(M, R, m_star):
    ok = np.isfinite(R)
    M, R = M[ok], R[ok]
    try:
        res = least_squares(lambda p: R - (p[1] + p[0] * np.maximum(M - m_star, 0.0)),
                            [max(np.ptp(R), 1e-6) / max(np.ptp(M), 1.0), max(R.min(), 0.0)],
                            bounds=([0.0, 0.0], [np.inf, R.max() + 1e-9]))
    except ValueError:
        return None
    A, B = (float(v) for v in res.x)
    return dict(A=A, B=B, rmse=rmse(R, B + A * np.maximum(M - m_star, 0.0)))


def fit_step(M, R, m_star):
    lo, hi = R[M < m_star], R[M >= m_star]
    B = float(np.nanmean(lo)) if lo.size else float(np.nanmean(R))
    C = float(np.nanmean(hi)) if hi.size else B
    return dict(B=B, C=C, rmse=rmse(R, np.where(M < m_star, B, C)))


def onset_from_fraction(ms, frac):
    """First drive where the orbiting fraction reaches 0.5 (linear interpolation)."""
    for i, f in enumerate(frac):
        if f >= 0.5:
            if i == 0:
                return np.nan
            f0, f1, m0, m1 = frac[i - 1], f, ms[i - 1], ms[i]
            return float(m0 + (0.5 - f0) / (f1 - f0) * (m1 - m0)) if f1 != f0 else float(m1)
    return np.nan


def pick_drives(ms, m_on, orbit_frac):
    """(below, near, above) example drives around the onset."""
    ms = np.array(sorted(set(ms)))
    if ms.size < 3:
        return tuple(float(v) for v in (ms[0], ms[0], ms[-1]))
    if not np.isfinite(m_on):
        return float(ms[0]), float(ms[ms.size // 2]), float(ms[-1])
    near = float(ms[np.argmin(np.abs(ms - m_on))])
    below_c = [m for m in ms if m < near and orbit_frac.get(m, 0) < 0.5]
    below = float(below_c[-1]) if below_c else float(ms[0])
    above_c = [m for m in ms if m > near and orbit_frac.get(m, 0) >= 0.5]
    above = float(above_c[len(above_c) // 2]) if above_c else float(ms[-1])
    return below, near, above


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
class Painter:
    def __init__(self, args, lengths, outdir):
        self.args, self.lengths, self.outdir = args, lengths, outdir
        cmap = plt.get_cmap("viridis")
        self.color = {l: cmap(min(l / args.l_color_max, 1.0)) for l in lengths}

    def lab(self, l):
        return f"{l * self.args.l_step:g} mm" if self.args.l_step > 0 else f"{l}"

    def save(self, fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, name + ".png"), dpi=self.args.dpi)
        if self.args.pdf:
            fig.savefig(os.path.join(self.outdir, name + ".pdf"))
        plt.close(fig)
        return name

    def onset_lines(self, ax, onset):
        for l in self.lengths:
            m_on = onset[l]["m50"]
            if np.isfinite(m_on):
                ax.axvline(m_on, color=self.color[l], lw=0.8, ls="-", alpha=0.35, zorder=0)

    def sweep(self, trials, key, ylabel, name, onset, ax=None, legend=True):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(3.4, 2.5))
        rng = np.random.default_rng(0)
        for l in self.lengths:
            grp = [tr for tr in trials if tr.l == l and np.isfinite(tr.scal[key])]
            if not grp:
                continue
            m = np.array([tr.m for tr in grp])
            v = np.array([tr.scal[key] for tr in grp])
            ax.scatter(m + rng.uniform(-1.5, 1.5, m.size), v, s=5, alpha=0.2,
                       color=self.color[l], lw=0, zorder=2)
            g = pd.DataFrame({"m": m, "v": v}).groupby("m")["v"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0.0), fmt="o-", ms=3,
                        lw=1.1, capsize=2, color=self.color[l], mec="k", mew=0.3,
                        ecolor=self.color[l], zorder=3, label=self.lab(l))
        self.onset_lines(ax, onset)
        ax.set_xlabel(PWM_LABEL)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(title=r"$l$", frameon=False, title_fontsize=7, handlelength=1.2)
        if own:
            return self.save(fig, name)

    def radius(self, trials, onset, squared=False):
        key = "R2_mm2" if squared else "R_mm"
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        self.sweep(trials, key, r"$R^2$ (mm$^2$)" if squared else r"$R=\langle r^2\rangle^{1/2}$ (mm)",
                   None, onset, ax=ax, legend=False)
        scale = 1e6 if squared else 1e3
        for l in self.lengths:
            f = onset[l]
            if not np.isfinite(f["sqrt"]["m_star"]):
                continue
            ms = np.array([tr.m for tr in trials if tr.l == l])
            M = np.linspace(ms.min(), ms.max(), 300)
            ms_, A, B = f["sqrt"]["m_star"], f["sqrt"]["A"], f["sqrt"]["B"]
            R2 = B + A * np.maximum(M - ms_, 0.0)
            ax.plot(M, (R2 if squared else np.sqrt(np.maximum(R2, 0))) * scale,
                    color=self.color[l], lw=1.4, zorder=4)
            if f["linear"]:
                Rl = f["linear"]["B"] + f["linear"]["A"] * np.maximum(M - ms_, 0.0)
                ax.plot(M, (Rl ** 2 if squared else Rl) * scale, color=self.color[l],
                        lw=0.9, ls="--", zorder=4)
            Rs = np.where(M < ms_, f["step"]["B"], f["step"]["C"])
            ax.plot(M, (Rs ** 2 if squared else Rs) * scale, color=self.color[l], lw=0.9,
                    ls=":", zorder=4)
        h_l = [plt.Line2D([], [], color=self.color[l], marker="o", ms=3, lw=1.1)
               for l in self.lengths]
        leg1 = ax.legend(h_l, [self.lab(l) for l in self.lengths], title=r"$l$",
                         frameon=False, loc="upper left", title_fontsize=7, handlelength=1.2)
        ax.add_artist(leg1)
        h_f = [plt.Line2D([], [], color="0.3", lw=1.4),
               plt.Line2D([], [], color="0.3", lw=0.9, ls="--"),
               plt.Line2D([], [], color="0.3", lw=0.9, ls=":")]
        ax.legend(h_f, [r"$\sqrt{m-m^*}$", "linear", "step"], frameon=False,
                  loc="lower right", handlelength=1.6)
        return self.save(fig, "02_orbit_radius_sq" if squared else "01_orbit_radius")

    def orbit_fraction(self, summary, onset):
        n = len(self.lengths)
        fig, axes = plt.subplots(1, n, figsize=(1.9 * n + 0.4, 2.3), sharey=True, squeeze=False)
        for ax, l in zip(axes[0], self.lengths):
            s = summary[summary["l"] == l].sort_values("m")
            ax.plot(s["m"], s["frac_cw"], "o-", color=C_CW, ms=3, lw=1.1, label="CW")
            ax.plot(s["m"], s["frac_ccw"], "s-", color=C_CCW, ms=3, lw=1.1, label="CCW")
            ax.plot(s["m"], 1 - s["frac_orbit"], "^-", color=C_NONE, ms=3, lw=1.1,
                    label="no orbit")
            if np.isfinite(onset[l]["m50"]):
                ax.axvline(onset[l]["m50"], color="k", lw=0.7, alpha=0.35, zorder=0)
            ax.set_title(rf"$l$ = {self.lab(l)}")
            ax.set_xlabel(PWM_LABEL)
            ax.set_ylim(-0.03, 1.03)
        axes[0][0].set_ylabel("fraction of trials")
        axes[0][-1].legend(frameon=False, loc="center right")
        return self.save(fig, "03_orbit_fraction")

    def trajectories(self, trials, l, drives, onset):
        tags = ("below onset", "near onset", "above onset")
        fig = plt.figure(figsize=(7.0, 8.6))
        gs = GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35,
                      left=0.09, right=0.99, top=0.93, bottom=0.05)
        above = self.pick(trials, l, drives[2])
        span = np.concatenate([tr.r for tr in above]) if above else np.array([])
        lim = max(float(np.nanpercentile(span, 97)) if span.size else 0.05, 0.01) * 1e3
        for col, (m, tag) in enumerate(zip(drives, tags)):
            axs = [fig.add_subplot(gs[row, col]) for row in range(5)]
            ax_xy, ax_t, ax_r, ax_phi, ax_ps = axs
            ax_xy.set_aspect("equal")
            for k, tr in enumerate(self.pick(trials, l, m)):
                c = f"C{k % 10}"
                keep = tr.t <= tr.t[0] + self.args.tmax
                t = tr.t[keep] - tr.t[0]
                x0 = tr.r[keep] * np.cos(tr.phi_u[keep]) * 1e3
                y0 = tr.r[keep] * np.sin(tr.phi_u[keep]) * 1e3
                ax_xy.plot(x0, y0, color=c, lw=0.7, alpha=0.8)
                ax_xy.plot(x0[0], y0[0], "s", color=c, ms=2)
                ax_t.plot(t, x0, color=c, lw=0.7, alpha=0.8)
                ax_t.plot(t, y0, color=c, lw=0.7, alpha=0.8, ls="--")
                ax_r.plot(t, tr.r[keep] * 1e3, color=c, lw=0.7, alpha=0.8)
                ax_phi.plot(t, tr.phi_u[keep] - tr.phi_u[keep][0], color=c, lw=0.7, alpha=0.8)
                ax_ps.plot(x0, tr.xdot[keep], color=c, lw=0.6, alpha=0.8)
            ax_xy.plot(0, 0, "+", color="k", ms=6)
            ax_xy.set_xlim(-lim, lim)
            ax_xy.set_ylim(-lim, lim)
            ax_xy.set_title(rf"$m$ = {m:g} ({tag})")
            ax_xy.set_xlabel("x (mm)")
            ax_xy.set_ylabel("y (mm)")
            for ax, yl in ((ax_t, "x, y (mm)"), (ax_r, r"$r$ (mm)"),
                           (ax_phi, r"polar angle $\phi$ (rad)")):
                ax.set_xlabel("time (s)")
                ax.set_ylabel(yl)
            ax_ps.set_xlabel("x (mm)")
            ax_ps.set_ylabel(r"$\dot x$ (m/s)")
            ax_ps.locator_params(axis="x", nbins=4)
            if col == 0:
                ax_t.plot([], [], color="0.3", lw=1, label="x")
                ax_t.plot([], [], color="0.3", lw=1, ls="--", label="y")
                ax_t.legend(frameon=False)
        fig.suptitle(rf"$l$ = {self.lab(l)}", y=0.985)
        fig.savefig(os.path.join(self.outdir, f"16_trajectories_l{l}.png"), dpi=self.args.dpi)
        if self.args.pdf:
            fig.savefig(os.path.join(self.outdir, f"16_trajectories_l{l}.pdf"))
        plt.close(fig)
        return f"16_trajectories_l{l}"

    def spectra(self, trials, l, drives):
        tags = ("below onset", "near onset", "above onset")
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
        for ax, m, tag in zip(axes, drives, tags):
            for k, tr in enumerate(self.pick(trials, l, m)):
                if tr.spec_f.size:
                    ax.loglog(tr.spec_f[1:], tr.spec_p[1:], color=f"C{k % 10}", lw=0.7, alpha=0.8)
            ax.set_title(rf"$m$ = {m:g} ({tag})")
            ax.set_xlabel("frequency (Hz)")
        axes[0].set_ylabel(r"PSD of $x(t)$")
        fig.suptitle(rf"$l$ = {self.lab(l)}")
        return self.save(fig, f"17_spectra_l{l}")

    def pick(self, trials, l, m):
        grp = sorted([tr for tr in trials if tr.l == l and tr.m == m],
                     key=lambda tr: (tr.src, tr.track))
        if len(grp) <= self.args.n_show:
            return grp
        idx = sorted(set(np.linspace(0, len(grp) - 1, self.args.n_show).round().astype(int)))
        return [grp[i] for i in idx]


def plot_drive_figure(P, trials, summary, onset, args):
    """Paper figure: (a) top-view trials below / above onset for one offset,
    (b) fraction of trials orbiting CW vs PWM, (c) orbit radius R vs PWM."""
    ls = [l for l in P.lengths if l in (args.fig_lengths or P.lengths)]
    ex = args.fig_example_l if args.fig_example_l in P.lengths else ls[len(ls) // 2]
    below, _, above = onset[ex]["drives"]
    fig = plt.figure(figsize=(7.0, 2.35))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.55, 1.0, 1.0], wspace=0.42, hspace=0.3,
                  left=0.02, right=0.99, top=0.9, bottom=0.2)
    lim = 25.0
    for row, (m, tag) in enumerate(((below, "below onset"), (above, "above onset"))):
        ax = fig.add_subplot(gs[row, 0])
        for tr in P.pick(trials, ex, m):
            keep = tr.t <= tr.t[0] + args.tmax
            x0 = tr.r[keep] * np.cos(tr.phi_u[keep]) * 1e3
            y0 = tr.r[keep] * np.sin(tr.phi_u[keep]) * 1e3
            ax.plot(x0, y0, color=P.color[ex], lw=0.6, alpha=0.8)
        ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)
        if row == 0:                        # scale bar where the trajectories leave room
            ax.plot([lim - 12, lim - 2], [-lim + 3, -lim + 3], color="k", lw=1.2)
            ax.text(lim - 7, -lim + 5, "10 mm", ha="center", va="bottom", fontsize=6)
        ax.set_title(f"$m$ = {m:g}", fontsize=7, pad=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_color("0.5")
        if row == 0:
            ax_a = ax
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])
    for l in ls:
        s = summary[summary["l"] == l].sort_values("m")
        p = s["frac_cw"].clip(0, 1)
        ax_b.errorbar(s["m"], p, yerr=np.sqrt(p * (1 - p) / s["n_trials"].clip(lower=1)),
                      fmt="o-", ms=3, lw=1.1, capsize=2, color=P.color[l], mec="k", mew=0.3,
                      label=P.lab(l))
        grp = [tr for tr in trials if tr.l == l]
        g = pd.DataFrame({"m": [tr.m for tr in grp], "R": [tr.scal["R_mm"] for tr in grp]}
                         ).groupby("m")["R"].agg(["mean", "std"])
        ax_c.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0.0), fmt="o-", ms=3, lw=1.1,
                      capsize=2, color=P.color[l], mec="k", mew=0.3, label=P.lab(l))
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_xlabel(PWM_LABEL)
    ax_b.set_ylabel("fraction of trials orbiting")
    ax_b.legend(title=r"$l$", frameon=False, loc="lower right", title_fontsize=7,
                handlelength=1.2)
    ax_c.set_ylim(0, None)
    ax_c.set_xlabel(PWM_LABEL)
    ax_c.set_ylabel(r"orbit radius $R$ (mm)")
    fig.canvas.draw()
    top = ax_b.get_position().y1 + 0.03
    for ax, lab, dx in ((ax_a, "a", 0.045), (ax_b, "b", 0.075), (ax_c, "c", 0.075)):
        fig.text(ax.get_position().x0 - dx, top, f"({lab})", fontweight="bold",
                 ha="left", va="bottom")
    base = os.path.join(P.outdir, "drive_figure")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=args.dpi)
    plt.close(fig)
    print(f"  drive_figure: offsets {ls}, example l={ex} at m={below:g} / {above:g}")
    return "drive_figure"


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    plt.rcParams.update(STYLE)
    outdir = args.outdir or os.path.join(args.directory, "drive_sweep_plots")
    os.makedirs(outdir, exist_ok=True)

    trials = load_trials(args.directory, args)
    if not trials:
        raise SystemExit("No trials.")
    lengths = sorted({tr.l for tr in trials})
    centres = {l: well_centre(trials, l) for l in lengths}
    for tr in trials:
        fill_trial(tr, *centres[tr.l], args)

    tdf = pd.DataFrame([dict(file=tr.src, m=tr.m, l=tr.l, d=tr.d, track=tr.track,
                             rotating=int(tr.rotating), duration=tr.t[-1] - tr.t[0], **tr.scal)
                        for tr in trials])
    keys = ["R", "R2", "heading_rate", "heading_rotation", "polar_rate", "omega_L",
            "omega_abs", "L", "peak_freq", "heading_rate_nrev", "R_nrev", "polar_rate_nrev",
            "time_to_nrev"]
    agg = {"n_trials": ("track", "count"), "frac_orbit": ("orbiting", "mean"),
           "frac_nrev": ("rotating", "mean")}
    for k in keys:
        agg[f"{k}_mean"] = (k, "mean")
        agg[f"{k}_std"] = (k, "std")
    summary = tdf.groupby(["m", "l"], as_index=False).agg(**agg)
    dirs = tdf.groupby(["m", "l"])["direction"]
    summary["frac_cw"] = dirs.apply(lambda s: (s == "CW").mean()).to_numpy()
    summary["frac_ccw"] = dirs.apply(lambda s: (s == "CCW").mean()).to_numpy()

    onset = {}
    print()
    for l in lengths:
        s = summary[summary["l"] == l].sort_values("m")
        ms, fo = s["m"].to_numpy(float), s["frac_orbit"].to_numpy(float)
        m50 = onset_from_fraction(ms, fo)
        med = tdf[tdf["l"] == l].groupby("m")["R"].median()
        M, R = med.index.to_numpy(float), med.to_numpy(float)
        sq = fit_sqrt_onset(M, R ** 2)
        lin = fit_linear_onset(M, R, sq["m_star"]) if np.isfinite(sq["m_star"]) else None
        stp = fit_step(M, R, sq["m_star"]) if np.isfinite(sq["m_star"]) else None
        onset[l] = dict(m50=m50, sqrt=sq, linear=lin, step=stp,
                        drives=pick_drives(ms, m50 if np.isfinite(m50) else sq["m_star"],
                                           dict(zip(ms, fo))))
        print(f"  l={l}: centre ({centres[l][0]:.4f}, {centres[l][1]:.4f})  "
              f"onset m*_50 = {m50:.1f}   sqrt-fit m* = {sq['m_star']:.1f}  "
              f"examples {onset[l]['drives']}")

    P = Painter(args, lengths, outdir)
    saved = [P.radius(trials, onset), P.radius(trials, onset, squared=True),
             P.orbit_fraction(summary, onset)]
    n = args.n_rev
    for key, ylabel, name in [
        ("heading_rate", r"$|\langle\dot\gamma\rangle|$ (rad/s)", "04_heading_rate"),
        ("heading_rotation", r"$|\Delta\gamma|$ (rad)", "05_heading_rotation"),
        ("polar_rate", r"$|\langle\dot\phi\rangle|$ (rad/s)", "06_polar_rate"),
        ("omega_L", r"$\omega_L = L/\langle r^2\rangle$ (rad/s)", "07_orbit_rate_signed"),
        ("omega_abs", r"$|\omega_L|$ (rad/s)", "08_orbit_rate_abs"),
        ("L", r"$L=\langle x\dot y - y\dot x\rangle$ (m$^2$/s)", "09_circulation"),
        ("peak_freq", "peak frequency of $x(t)$ (Hz)", "10_peak_freq"),
        ("heading_rate_nrev", rf"$|\langle\dot\gamma\rangle|$, first {n} turns (rad/s)",
         f"11_heading_rate_{n}rev"),
        ("R_nrev", rf"$R$, first {n} turns (mm)", f"12_orbit_radius_{n}rev"),
        ("polar_rate_nrev", rf"$|\langle\dot\phi\rangle|$, first {n} turns (rad/s)",
         f"13_polar_rate_{n}rev"),
        ("time_to_nrev", f"time to {n} heading turns (s)", f"14_time_to_{n}rev"),
    ]:
        saved.append(P.sweep(trials, key, ylabel, name, onset))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for l in lengths:
        s = summary[summary["l"] == l].sort_values("m")
        p = s["frac_nrev"].clip(0, 1)
        ax.errorbar(s["m"], p, yerr=np.sqrt(p * (1 - p) / s["n_trials"].clip(lower=1)),
                    fmt="o-", ms=3, lw=1.1, capsize=2, color=P.color[l], mec="k", mew=0.3,
                    label=P.lab(l))
    P.onset_lines(ax, onset)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(PWM_LABEL)
    ax.set_ylabel(f"fraction reaching {n} heading turns")
    ax.legend(title=r"$l$", frameon=False, title_fontsize=7, handlelength=1.2)
    saved.append(P.save(fig, f"15_frac_{n}rev"))
    for l in lengths:
        saved.append(P.trajectories(trials, l, onset[l]["drives"], onset))
        saved.append(P.spectra(trials, l, onset[l]["drives"]))
    saved.append(plot_drive_figure(P, trials, summary, onset, args))

    tdf.to_csv(os.path.join(outdir, "trials.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame([dict(l=l, offset_mm=l * args.l_step, m_star_50=o["m50"],
                       m_star_sqrt=o["sqrt"]["m_star"], A_R2=o["sqrt"]["A"],
                       B_R2=o["sqrt"]["B"], rmse_R2=o["sqrt"]["rmse"],
                       rmse_linear_R=(o["linear"] or {}).get("rmse"),
                       rmse_step_R=(o["step"] or {}).get("rmse"),
                       xc=centres[l][0], yc=centres[l][1])
                  for l, o in onset.items()]).to_csv(os.path.join(outdir, "onset.csv"),
                                                     index=False)
    print(f"\nWrote {len(saved)} figures + trials.csv, summary.csv, onset.csv to {outdir}/")


if __name__ == "__main__":
    main()
