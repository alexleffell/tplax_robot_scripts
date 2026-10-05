# Robot Video-Analysis Pipeline — Design Notes

This document records the design choices, justifications, and assumptions behind the
five-script analysis pipeline for the spring-connected motorized-caster-wheel robot.
Each node carries an AprilTag whose **position = node center** and whose **in-plane
angle = caster orientation** in the lab frame. Fixed corner tags (ids 26–29) mark the
arena.

## Pipeline overview (data flow)

```
checkerboard images ─► camera_calibration.py ─► calibration.npz (+ .json, overlays)
                                                      │
video + calibration.npz ─► apriltag_tracker.py ─► <video>_raw.csv   (per-tag, per-frame, camera frame)
                                                      │
<video>_raw.csv ─► format_tracks.py ─► <core>_robot.csv (+ .log)    (wide, interpolated, lab frame, body angle)
                                                      │
<core>_robot.csv ─► analyze_modes.py ─► <core>_analysis.npz (+ .txt) (modal / energetic / active-solid quantities)
                                                      │
<core>_analysis.npz ─► plot_analysis.py ─► <core>_analysis_plots/*.png
```

Separation of concerns is deliberate: **script 1** produces intrinsics; **script 2**
does detection + pose only (raw, camera frame); **script 3** does geometry/bookkeeping
(interpolation, lab frame, rigid-body fit); **script 4** does all physics calculations
and stores them; **script 5** only reads the bundle and draws. Calculations never live
in the plotter and plotting never lives in the analysis script.

A parallel **single-node** path uses the same tracker. One video can hold several
trajectories (the tag is covered by hand between resets); `format_tracks_single.py`
splits on long dropouts, then `plot_analysis_single.py` draws (no modal analysis yet).

```
<video>_raw.csv ─► format_tracks_single.py ─► <core>_robot.csv (+ .log)   (wide, per-track, + track column)
                                                      │
<core>_robot.csv ─► plot_analysis_single.py ─► <core>_robot_plots/*.png
```

## Global conventions & environment

- **Detector**: `pupil_apriltags`, family `tag16h5` (matches the physical printed tags).
  Chosen over `cv2.aruco` and the legacy `apriltag`/`dt-apriltags` packages because it
  wraps the reference AprilTag-3 detector, installs cleanly on macOS arm64, and avoids
  reprinting tags. (Trade-off acknowledged: `tag16h5` is a weak family — small Hamming
  distance, higher false-positive rate — so decode reliability depends on filtering.)
- **Pose**: `cv2.solvePnP` with `SOLVEPNP_IPPE_SQUARE` (purpose-built for a single planar
  square of known size).
- **Units**: SI where the tag size is in meters; positions inherit those units.
- **Environment**: `/opt/miniconda3/envs/tplax_env` (cv2 4.10, pupil_apriltags, scipy).
- **Metadata convention**: CSVs carry a commented `# key: value` header compatible with
  the notebooks' `read_csv_comments`, so downstream stages recover fps / topology / etc.

---

# 1. `camera_calibration.py`

**Purpose.** Standard checkerboard intrinsic calibration; emits camera matrix + distortion
coefficients for the tracker, plus QC artifacts.

**Design choices**
- **`findChessboardCornersSB` with a classic fallback.** SB (sector-based, radon-transform)
  is more robust to blur/uneven lighting and returns sub-pixel corners directly. Falls back
  to `findChessboardCorners` + `cornerSubPix` when SB fails, for maximum yield.
- **Rational distortion model** (`CALIB_RATIONAL_MODEL | CALIB_FIX_K6`), matching the
  original notebook calibration, appropriate for the moderately wide lens in use.
- **Outputs**: `.npz` (machine) + `.json` (human) + corner-overlay images (visual
  verification that corners were found correctly).
- **Reporting**: overall RMS and per-image reprojection error, with a threshold flag to
  identify bad frames to drop.
- **Fisheye is the default model** (this project uses a wide-angle lens). The pinhole/rational
  model cannot fit strong barrel distortion — symptoms are a stuck RMS (2–3 px) and an unstable
  focal length. `--pinhole` switches to the rectilinear model (with `--fix-aspect-ratio` /
  `--simple-distortion` sub-options). The saved calibration records `model` (fisheye|pinhole),
  which the tracker reads to choose the correct un-distortion.
- Guardrails: warns on resolution mismatch downstream, anisotropic `fx/fy` (pinhole only), and
  RMS > 1 px. Per-image error is reported as true per-point RMS (`norm/√N`).

**Assumptions**
- All images share one resolution (asserted); the board's inner-corner count (`--pattern`,
  default 7×10) and square size (`--square-size`) match the physical board.
- `--square-size` only scales extrinsics; the intrinsics used downstream are independent of
  it (documented default 1.0).

---

# 2. `apriltag_tracker.py`

**Purpose.** Detect tags and estimate per-tag pose on every frame; write raw camera-frame
data. No interpolation or reformatting (that is script 3's job).

**Design choices**
- **Loads intrinsics from `--calib`**; if omitted, falls back to hard-coded defaults with a
  warning (so the script never silently runs uncalibrated).
- **Two physical tag sizes** — node tags (`--tag-size`, 0.045 m) vs corner tags
  (`--corner-tag-size`, 0.037 m) — because the environment corner markers are a different
  size; the correct object-point square is chosen per tag id.
- **Detection filtering**: keep a detection only if `hamming <= --hamming-max` (default 0,
  i.e. perfect decode), `tag_id in --valid-tags`, and `decision_margin > --decision-margin-min`.
  This is necessary precisely because `tag16h5` is error-prone.
- **Optional `--subpix`**: refine tag corners with `cv2.cornerSubPix` before pose.
- **Raw output** columns `frame#, node_id, x, y, z, angle, hamming, decision_margin,
  reproj_err`, plus a metadata header (fps, total_frames, sizes, corner ids, calib source).
- **Optional annotated video** behind `--output-video` (uses `alpha=0` undistort crop — QC
  only; see note below).

## Per-calculation details

- **Pose (`x, y, z`)**: `solvePnP(SOLVEPNP_IPPE_SQUARE)` on the four tag corners with the
  correct object-point square. Object points are ordered top-left, top-right, bottom-right,
  bottom-left to match `pupil_apriltags`' corner ordering. The tracker reads the calibration
  `model`: for **fisheye** it first maps the tag corners through `cv2.fisheye.undistortPoints`
  (into pinhole-`K` pixels) and solves with `K` and no further distortion; for **pinhole** it
  passes `dist_coeffs` to `solvePnP` directly. Using the pinhole path on a fisheye lens gives
  garbage poses (this project's lens is fisheye).
- **Caster angle (`angle`)** — **corrected from the original code.** The in-plane rotation is
  `theta = atan2(R[1,0], R[0,0])` where `R, _ = cv2.Rodrigues(rvec)`. The original script used
  `atan2(rvec[1], rvec[0])`, which is the direction of the Rodrigues **axis**, not the in-plane
  rotation — a genuine bug that this rewrite fixes.
- **`reproj_err`**: per-detection RMS reprojection error via `cv2.projectPoints`, kept for QC.

**Assumptions**
- The camera is roughly overhead so that tag *z* is nearly constant and the in-plane angle is
  the meaningful caster orientation.
- The cropping seen in `--output-video` is cosmetic (undistort `alpha=0`); **detection runs on
  the full raw frame**, so no tracking data is lost for tags near the periphery. (Peripheral
  poses are corrected for distortion but carry more error, since calibration is least
  constrained at the image edges.)

---

# 3. `format_tracks.py`

**Purpose.** Turn the raw per-tag CSV into a wide, analysis-ready, one-row-per-frame table:
interpolate gaps, transform into the lab frame, compute centroid and body angle.

**Design choices**
- **Node set** = the tag ids that appear in `--connections` (0-indexed; default hub-and-spoke
  over nodes 0–6). Any other detected valid tag (including corner tags) becomes an
  `extra_tag_<id>`. `N_nodes` = number of nodes.
- **Interpolation before anything else**: per-node linear interpolation over all frames
  (`limit_direction="both"`), so missing detections are filled. Justified because downstream
  modal analysis needs a value at every node every frame; linear is the simplest defensible
  fill and gaps are reported in the log.
- **Lab frame via corner tags** (chosen over camera frame for the final output): a 2D
  **similarity transform** (rotation + translation + uniform scale, `estimateAffinePartial2D`)
  from the time-averaged corner-tag positions to an axis-aligned rectangle. Target rectangle
  dimensions come from `--arena-size` if given, else the observed mean corner spacing (which
  just axis-aligns while preserving scale). Similarity (not homography) is used because with a
  roughly overhead camera the corners already form a near-rectangle; a full homography would
  overfit. Caster angles are rotated by the transform's rotation so `theta` is lab-frame.
  - **Fallback**: if the corner tags aren't detected (or `--camera-frame` is passed), stay in
    the camera frame and log a warning.
- **`--camera-frame` flag** added on request to force camera-frame output entirely.
- **Output** column order (as specified): `time`, then per node
  `{id}_x,{id}_y,{id}_z,{id}_theta,{id}_angle`, then `centroid_x,centroid_y,body_angle,
  body_angle_incremental`, then `extra_tag_{id}_x/_y`. Metadata header carries nodes,
  connections, corner_locs, baseline, arena_size, frame label, fps.
- **Separate text log** with interpolation stats, per-node detection rate, corner-tag status,
  lab-transform residual, and body-angle skip counts.

## Per-calculation details

- **Centroid** = mean of node positions each frame (the natural, unambiguous translational
  coordinate; equals center of mass for equal masses).
- **`body_angle` (absolute)** via **Procrustes / Kabsch** fit of the node positions to an
  **idealized per-topology template** (from `robot_topology.py`). This is the standard
  least-squares (Eckart-frame) resolution of the rigid rotation of a *deformable* body — the
  rotation that minimizes residual deformation. Labeled tags remove the polygon's
  rotational-symmetry ambiguity. The template radius does **not** affect the fitted angle
  (Kabsch is scale-invariant).
  - **Topology dispatch** (`--topology`, default `auto` = by node count): **7 → `hub_spoke`**
    (hub at origin + 6 ring nodes on a circle), **6 → `ring`** (regular hexagon, no hub). New
    topologies are added in `robot_topology.py` as needed; the same module feeds
    `analyze_modes.py`'s Hessian reference so the geometry can't drift between the two. An
    **idealized** template (not the mean shape) is used deliberately: the robot is flexible, so
    a data-derived mean shape is not repeatable across experiments. The resolved topology is
    written to the CSV header (`topology`) and reused downstream. Unknown node counts warn and
    fall back to the (non-repeatable) mean shape.
- **`body_angle_incremental`** via reference-free frame-to-frame Kabsch, integrated and
  unwrapped. Requested because for a deformable body the absolute angle depends on the (choice
  of) reference, whereas the integrated frame-to-frame rotation is reference-free and more
  robust under large deformation. Both are output so they can be compared.
- **Per-node `_theta`** = lab-frame caster angle; **per-node `_angle`** = body-relative
  (`wrap(theta − body_angle)`).

### Optional micro-controller sensor merge (`--sensor-csv`)

Some experiments also stream a sensor CSV (`timestamp_s, timestamp_us, node_id,
encoder_value, angle_value, motor_command`). When provided, it supplies the heading; when
absent, the pipeline behaves exactly as above (both datasets supported; the output schema is
a superset, so scripts 4–5 need no changes).

- **Different mounting when sensor is present.** In these experiments the AprilTag is fixed to
  the node **body**, not the caster. So the tag angle measures the node-base/body orientation
  (and is *not* expected to match the caster). The magnetic-encoder `angle_value` **is** the
  caster heading in the body frame directly. Confirmed empirically: sensor vs tag caster-angle
  velocity correlation ≈ 0, and the tag in-plane angle jitters ~9°/frame (solvePnP's
  worst-constrained DOF on a small `tag16h5` marker) — which is exactly why the sensor is used.
- **Heading mapping**: `{n}_angle` (body) = raw sensor `angle_value`; `{n}_theta` (lab) =
  `body_angle + sensor_angle`. No tag alignment (the tag isn't the caster). The hardware zero
  was set at a roughly aligned orientation (noisy); cross-node angle statistics inherit that
  per-node zero noise. `--sensor-angle-units {rad,deg}` (default rad); the raw range is logged
  so the units can be verified.
- **Clock & sync**: ESP-NOW broadcasts a shared epoch at start, so all nodes share one clock;
  each node is interpolated onto the video frame grid. A single additive offset pins the
  globally earliest `motor_command != 0` to the first video-motion frame (any-node speed over
  an auto noise-floor threshold; `--motion-onset-frame` / `--motion-threshold` override).
  Validated on real data: motor-on at frame 401 vs motion detected at 402, cross-correlation
  peak at ~0 lag.
- **Pre-sync/glitch rows** (a node's local uptime before the epoch broadcast → tiny
  `timestamp_s`) are dropped (`timestamp_s < 1e9` when an epoch clock is present).
- **Coverage**: heading is sensor-only — NaN outside the synced sensor window or for a node
  absent from the sensor file (logged). `encoder_value` and `motor_command` are carried through
  as extra `{n}_encoder` / `{n}_motor` columns.
- **Metadata**: `heading_source` (sensor|tag), and when sensor is used `tag_mount=body`,
  `sensor_sync_offset_s`, `motion_onset_frame`, `motor_onset_time_s`, `sensor_nodes`,
  `sensor_angle_units`. `analyze_modes.py` zeros any all-NaN heading column (with a warning)
  so the linear algebra doesn't propagate NaN.

**Assumptions**
- Exactly 4 corner tags define the arena; ring node ids are arranged in ascending order around
  the polygon (else the template is wrong — a non-hub-and-spoke topology falls back to the
  time-averaged mean shape with a warning).
- Linear interpolation is acceptable for the observed gap sizes; large gaps are surfaced in the
  log rather than silently trusted.
- Sensor experiments: tag body-fixed (heading from encoder); node bases share the body rotation
  (`body_angle`) for the lab-frame reconstruction. The robot is at rest before actuation so the
  motor-onset ↔ first-motion sync anchor is valid.

---

# 3b. `format_tracks_single.py`

**Purpose.** Same role as `format_tracks.py` for a **one-node** robot whose video contains
several trajectories. The operator covers the AprilTag with a hand while resetting the node
and uncovers it before the next run. Input is the raw CSV from `apriltag_tracker.py`.

**Design choices**
- **Node id** = `--node-id` if given, else the unique detected non-corner tag. Multiple
  non-corner tags is an error (pass `--node-id`). Corner / other tags remain `extra_tag_<id>`.
- **Split on long dropouts**, not on pose jumps: a new track starts when the node is missing
  for `>= --min-gap-seconds` (default 0.5 s; `--min-gap-frames` overrides). Brief tracking
  glitches (1–3 frames) stay inside a track. Hand-cover gaps are **omitted** from the output
  (not interpolated across). Tracks shorter than `--min-track-seconds` (default 0.25 s) are
  dropped. The log prints a gap histogram so the threshold can be retuned.
- **Interpolation only inside a track**, on the frame grid from first to last detection of
  that track. Positions: linear. Angle: unwrap → interpolate → re-wrap, so fills do not jump
  across ±π.
- **Lab frame** reuses `format_tracks.py`'s corner-tag similarity transform (and the
  `--camera-frame` / missing-corner fallback). Extra tags are interpolated over the whole
  video so corners that stay visible during a hand-cover still define the transform.
- **Output** matches `format_tracks.py` column order, with `track` (1-indexed) after `time`.
  For one node the heading *is* the body orientation: `body_angle` = `{id}_theta` = lab (or
  camera) tag angle; `{id}_angle` = 0; centroid = node position; `body_angle_incremental`
  resets to 0 at each track start. Header adds `topology: single`, `n_tracks`,
  `min_gap_frames`, `node_id`. No sensor merge (tag heading only).

## Per-calculation details

- **Track boundaries**: sorted unique detection frames `f_i`; split where
  `f_{i+1} − f_i − 1 >= min_gap_frames`.
- **`{id}_theta` / `body_angle`**: tag in-plane angle, plus the lab-transform rotation when
  the lab frame is used.
- **`body_angle_incremental`**: unwrapped heading minus the track's first-frame heading.

**Assumptions**
- The tag is actually lost during a hand-cover. If the detector still reports the tag
  (through the hand), gap-splitting will not cut trajectories — lower the gap threshold or
  split on pose quality instead.
- Linear in-track interpolation is acceptable for the short dropouts that remain.

---

# 4. `analyze_modes.py`

**Purpose.** Modal and kinematic analysis of a **ring** robot (relaxed springs, no
pre-stress), stored in one `.npz` bundle plus a `.txt` summary with checks.

## Foundational choices

- **Model**: N nodes on a regular ring (template from `robot_topology.py`, counter-clockwise in
  connection-cycle order), relaxed central-force springs `k n̂n̂ᵀ` per bond (no tension term — the
  reference is an exact equilibrium, so the Hessian is the exact small-deformation operator),
  optional harmonic bond bending `--kappa` about the ideal interior angle (Hessian
  `κ Σ ∇θ∇θᵀ`). Unit mass. `k = 1` and unit mass are arbitrary: KE and PE are in different
  units and are **not summed** (no total energy, no effective temperature / equipartition —
  the system is driven and dissipative).
- **Body frame for every modal projection.** Mode shapes are defined on the template (node k at
  angle 2πk/N). Each frame the best-fit rotation β(t) (closed-form 2-D Kabsch, template →
  observed) is removed before projecting displacements (`Q`), deformation velocities (`A`) and
  polarities (`C`). (Before this fix, lab-frame vectors were projected onto body-frame modes,
  which mixes radial and tangential components by sin β and invalidated band/condensation
  results; the harmonic-PE check had been failing at 0.5–388 on every run.)
- **Rigid / mechanism split.** The λ≈0 block is re-based into 3 analytic rigid modes + the
  mechanisms (3 for a 6-ring with κ = 0; lifted to finite λ when κ > 0).
- **Symmetry-adapted sectors.** The ring is invariant under the one-node rotation S
  (`(S q)_{k+1} = R(2π/N) q_k`), which commutes with the Hessian (checked, logged). Inside each
  degenerate band the eigenvectors are re-based by the real Schur form of S into sectors:
  2-D sectors (S acts as a rotation by φ = 2πm/N, 0 < φ < π; pair oriented so that a pattern
  travelling counter-clockwise in the body frame turns z = Q₁ + iQ₂ counter-clockwise) and 1-D
  sectors (m = 0 or N/2). Sector identities (m) do not depend on k or κ; κ only sets their λ and
  the radial/tangential mix within a sector. For a 6-ring with κ = 0: soft band = m=2 shear
  pair + m=3 mechanism; stiff sectors m=0 (breathing), m=1, m=2, m=3.
- **Mode cache** in `--modes-dir` keyed by nodes, connections, k, κ (+ radius when κ > 0), and
  checked against the stored reference shape; the cache holds raw `eigh` output and the rigid
  split / symmetry adaptation is applied after loading.
- **Heading frame** `--angle-frame {lab,body}` affects only the order parameter, kymographs,
  autocorrelations, diffusion and bond alignment; modal projections always use body-frame
  polarity `(cos(γ−β), sin(γ−β))`.

## Per-calculation details

- **Strain-wave order parameter** (the condensation measure for a driven ring). For each 2-D
  sector j with complex amplitude z_j = Q_{j,1} + iQ_{j,2} (body-frame displacements):
  `share_j(t) = |z_j|²/Σ_deform Q²` (fraction of the deformation in the sector),
  `circ_j(t) = Im(z̄ż)/(|z||ż|)` (+1 CCW travelling wave, −1 CW, 0 standing/noise),
  `Λ_j = ⟨Im z̄ż⟩/⟨|z||ż|⟩`, phase speed `Ω_j = ⟨Im z̄ż⟩/⟨|z|²⟩` (the pattern turns at Ω_j/m in
  the body frame), amplitude CV, and `W_j(t) = share_j·circ_j`; `W_win` uses `--wave-window`
  averages of numerator and denominator. The dominant sector is the 2-D sector with the largest
  mean share; `wave_order = share·Λ` (signed), `wave_order_abs = ⟨|W_win|⟩` (direction-blind,
  for runs that switch direction). A strain-wave limit cycle gives |W| → 1 (e.g. chiral_1_trim:
  m = 2, share 0.93, Λ = −0.975, W = −0.91, robust to κ = 0 vs 0.005).
- **Sector participation ratio** `1/Σ_j share_j²` — λ- and basis-independent condensation
  measure (per-mode PR and the banded PR are kept for comparison; per-mode PR is arbitrary
  inside degenerate bands).
- **Rigid-body KE fraction** — CoM translation + least-squares rotation about the instantaneous
  centroid, removed orthogonally; rotation-invariant.
- **Potential energy** — springs `½k(L − L_ref)²` + bending `½κ(θ − θ_ref)²` (ring interior
  angles), `PE_total = PE_spring + PE_bend`.
- **Order parameter** — polar `|⟨e^{iγ}⟩|` (or `--nematic`), reported with the finite-N
  random-heading baseline `order_param_null` (≈0.37 for N = 6).
- **Heading field on the Laplacian** — complex field e^{iγ} projected on the graph-Laplacian
  modes; the uniform-mode fraction equals (polar order)². (Raw wrapped angles were previously
  projected, which is gauge-dependent.)
- **Polarity overlap with the soft band** (`elastic_zero_ratio`) — body-frame polarity power in
  the lowest non-rigid band (the mechanisms when κ = 0). Previously identically 0 for κ > 0.
- **Rotational diffusion** — per-node heading MSD after removing that node's mean spin
  (least-squares slope), `D_r = slope/2`; `spin_per_node` stores the spin rates. (A raw MSD of a
  spinning caster is ballistic and gives no diffusion coefficient.)
- **Orientational autocorrelation / integral time** — for spinning headings ≈ 1/ω (a
  decorrelation-by-rotation time). **VACF**, **velocity / γ̇ correlation matrices**, **bond
  alignment**, **ring winding** (ring order from the connection cycle), **kymographs** (heading;
  interior-angle deviation from 180(N−2)/N), **CoM PDF** — lab frame, unchanged.
- **Polarity–velocity coupling / actuation spectrum** — kept, but with no-slip wheels the node
  velocity is slaved to the heading, so values near 1 are kinematic; departures measure caster
  swing (l·γ̇) and slip, not elastic mode selection.
- **PSDs** — order parameter, KE, PE, per-node γ̇, and modal **amplitudes** Q (an energy PSD
  shows a mode oscillating at f at 2f).
- **Gap interpolation** — positions linearly; angles on the unwrapped series.

### Checks (printed to the summary)

- **Harmonic PE** `½Σλ_iQ_i²` vs spring+bending PE, `median|Δ| / mean PE` — ≪ 1 when the
  deformation sits in stiff modes; with κ = 0 a large-amplitude mechanism stretches springs at
  second order (≈0.5 on chiral_1_trim), which the harmonic model correctly misses.
- **Rigid leakage** of deformation KE into rigid modes — small; grows with deformation amplitude
  because rigid motion is removed in the current (deformed) shape.
- **Ring symmetry** `‖SK − KS‖/‖K‖` (≈1e-16 for κ = 0, ≈1e-11 with the finite-difference
  bending Hessian).

**Assumptions** — a single ring of identical nodes; the template is a regular N-gon; small
enough deformation for the linear modes to be a useful basis (sectors stay well defined at
large amplitude because they are fixed by symmetry, not by λ).

---

# 5. `plot_analysis.py`

**Purpose.** Read the `.npz` bundle and render diagnostic figures. Contains **no
calculations** — it only visualizes stored quantities.

**Design choices**
- Matplotlib with the `Agg` backend (headless save to `<npz>_plots/`), configurable DPI.
- Trajectories/orbits colored by time via `LineCollection` for readability.
- Rigid modes highlighted in a distinct color in the spectral/actuation bar charts.
- Degenerate/near-degenerate deformation-mode structure is visible in the mode-shape quiver
  panel — a correctness signal (e.g. the hexagon's degenerate pair).
- **Provenance stamping.** The plotter reads `heading_source` (sensor|tag) and `angle_frame`
  (lab|body) from the `.npz` (`analyze_modes.py` stores both). Every figure gets a monospace
  footer `heading source: … | angle frame: …`, and every filename is suffixed
  `__<heading_source>_<angle_frame>` (e.g. `06_order_parameter__sensor_body.png`). This means
  lab- and body-frame runs land in distinct files even in the same output directory and can
  never be visually confused. Older bundles lacking the fields render as `?` — re-run
  `analyze_modes.py` to repopulate them.

**Figures**
1. CoM trajectory (time-colored)  2. CoM 2D PDF  3. Energies (KE and PE = springs + bending,
separate panels)  4. Rigid-body KE fraction  5. Polar order² (Laplacian uniform mode) and
polarity overlap with the soft band  6. Orientation order parameter (+ random-heading
baseline)  7. Eigenvalue spectrum + per-mode KE  8. Deformation mode shapes (quiver)
9. Heading MSD about the mean spin + diffusion fit  10. PSDs (order parameter, KE/PE,
modal-amplitude heatmap)  11. Collective actuation (coupling, actuation spectrum,
condensation)  12. Phase portraits of the dominant and the softest 2-D wave sectors (oriented
pairs: a circle = travelling wave)  13. Strain-wave order parameter W(t) + per-sector share and
circulation  14. Chirality (ω + polarization angle)  15. Orientational ACF + VACF  16. Active force vs CoM velocity  17. Spatial polarity
(bond alignment + winding)  18. Per-node angular-velocity PSD (7 curves)  19. Pairwise node
velocity correlation heatmap  20. Pairwise heading angular-velocity correlation heatmap
21. Heading kymograph over the ring (time × node, hue = caster angle)  22. Interior bond-angle
(∠ABC) deviation kymograph (time × node), centered on the regular n-gon interior angle.

---

# 5b. `plot_analysis_single.py`

**Purpose.** Read the formatted CSV from `format_tracks_single.py` and render diagnostic
figures. Contains **no calculations** beyond wrapping/unwrapping already-stored heading for
display. More figures will be added later.

**Design choices**
- Matplotlib `Agg`, default outdir `<csv>_plots/`, configurable DPI — same pattern as
  `plot_analysis.py`.
- Footer stamps heading source, frame (lab|camera), and node id.
- Tracks are **not** connected to each other.

**Figures**
1. Phase portrait of heading vs y, all tracks overlaid. Heading is wrapped to
   \([-\pi, \pi]\) (branch-cut segments dropped). Square = track start; arrow = track end
   (oriented along the path in display space). One color per track.

---

# Running the pipeline

Environment: `/opt/miniconda3/envs/tplax_env/bin/python` (cv2 4.10, pupil_apriltags, scipy).
Worked example: `240226_med_low_1` (has a sensor CSV).

```bash
PY=/opt/miniconda3/envs/tplax_env/bin/python
cd /Users/alexleffell/Documents/PhD/tplax/tplax_robot_scripts

# 1. Calibration (once per camera setup)
$PY camera_calibration.py ../Data/temp_room_calibration/ --pattern 7 10 --glob '*.bmp' --square-size 1.0
#   -> calibration.npz (+ .json, overlays/)

# 2. Tracker (per video)  [--output-video for QC overlay, --subpix for corner refinement]
$PY apriltag_tracker.py ../Data/240226/240226_med_low_1.mp4 \
    --calib ../Data/temp_room_calibration/calibration.npz
#   -> ..._raw.csv

# 3. Format  (add --sensor-csv when the micro-controller file exists)
$PY format_tracks.py ../Data/240226/240226_med_low_1_raw.csv --baseline 0.1689 \
    --sensor-csv ../Data/240226/240226_med_low_1.csv
#   -> ..._robot.csv (+ ..._robot.log)

# 4. Analyze  (--angle-frame body when the sensor provides the heading; lab otherwise)
$PY analyze_modes.py ../Data/240226/240226_med_low_1_robot.csv --kappa 0.005 --vel-smooth-window 7 --angle-frame body
#   -> ..._analysis.npz (+ ..._analysis.txt); modes cached in /Users/.../tplax_paper

# 5. Plot
$PY plot_analysis.py ../Data/240226/240226_med_low_1_analysis.npz
#   -> ..._analysis_plots/*.png  (footer-stamped + filename-tagged with heading/frame)
```

Per-experiment knobs to remember:
- **`--baseline`** (steps 3–4): node-to-node rest distance (m); sets the reference lattice.
- **`--kappa`** (step 4): bond-bending stiffness; sector identities do not depend on it.
- **`--angle-frame body`** (step 4): use whenever the sensor is the heading source (tag body-fixed).
- **`--sensor-angle-units`** (step 3): default `rad` (correct for the observed [0, 2π] range).
- **`--camera-frame`** (step 3): skip the lab transform if corner tags are unusable.
- Tag family/sizes (step 2) default to `tag16h5` / 0.045 m / 0.037 m — override per dataset.

Steps 3→4→5 are re-run while iterating on analysis; steps 1–2 are done once per camera/video.

Single-node (after the same tracker step; no `analyze_modes.py`):

```bash
$PY format_tracks_single.py ../Data/180826/_2026-08-18_15_27_42_802_raw.csv
#   -> ..._robot.csv (+ ..._robot.log); retune --min-gap-seconds from the gap histogram in the log

$PY plot_analysis_single.py ../Data/180826/_2026-08-18_15_27_42_802_robot.csv
#   -> ..._robot_plots/01_phase_portrait_y_heading.png
```

---

# Command-line argument reference

Every argument of every script (also available via `python <script>.py --help`). Positional
arguments are required; all `--flags` are optional with the defaults shown.

## `camera_calibration.py`

| Argument | Default | Description |
|---|---|---|
| `image_dir` (positional) | — | Directory containing the calibration images. |
| `--pattern COLS ROWS` | `7 10` | Number of **inner** checkerboard corners. |
| `--square-size` | `1.0` | Physical square edge length; scales extrinsics only (intrinsics used downstream are unaffected). |
| `--glob` | `*.bmp` | Filename glob within `image_dir`. |
| `--output` | `<image_dir>/calibration.npz` | Output `.npz` path (a sibling `.json` is also written). |
| `--overlay-dir` | `<image_dir>/overlays/` | Where corner-overlay QC images are written. |
| `--pinhole` | off (default = fisheye) | Use the rectilinear pinhole model instead of the default fisheye model. |
| `--fix-aspect-ratio` | off | (pinhole only) Force `fx == fy`. |
| `--simple-distortion` | off | (pinhole only) Standard 5-coeff distortion instead of the rational 8-coeff model. |
| `--error-threshold` | `1.0` | Per-image reprojection error (px) above which a warning is logged. |

## `apriltag_tracker.py`

| Argument | Default | Description |
|---|---|---|
| `video_path` (positional) | — | Input video file. |
| `--calib` | built-in defaults (+warning) | Calibration `.npz` from `camera_calibration.py`. |
| `--families` | `tag16h5` | AprilTag family (e.g. `tag36h11`); must match the printed tags. |
| `--tag-size` | `0.045` | Node tag edge length (m). |
| `--corner-tag-size` | `0.037` | Corner/environment tag edge length (m). |
| `--corner-ids` | `26 27 28 29` | Tag ids treated as environment corner tags. |
| `--valid-tags` | `0..13 + 26..29` | Tag ids to keep (all others discarded). |
| `--nthreads` | `4` | Detector threads. |
| `--quad-decimate` | `1.0` | Detector downsampling; raise to speed up, keep at 1 for `tag36h11`. |
| `--quad-sigma` | `0.0` | Gaussian blur applied before detection. |
| `--subpix` | off | Refine tag corners with `cv2.cornerSubPix` before `solvePnP`. |
| `--hamming-max` | `0` | Max allowed decode Hamming distance (kept if `hamming <=` this). |
| `--decision-margin-min` | `1.0` | Minimum decision margin to keep a detection. |
| `--output` | `<video>_raw.csv` | Output raw CSV path. |
| `--output-video` | off | Also write an undistorted, tag-annotated QC video. |
| `--undistort-alpha` | `1.0` | Output-video undistort alpha: `1` keeps the full frame (black borders), `0` crops/zooms the distorted periphery. |

## `format_tracks.py`

| Argument | Default | Description |
|---|---|---|
| `raw_csv` (positional) | — | Raw CSV from `apriltag_tracker.py`. |
| `--connections` | hub-and-spoke over `0..6` | Python-literal list of `(i, j)` node connections; defines the node set and template. |
| `--corner-ids` | raw-CSV header, else `26 27 28 29` | Corner tag ids used to build the lab frame. |
| `--arena-size W H` | observed corner spacing | Real lab-rectangle dimensions for the lab transform. |
| `--camera-frame` | off | Skip the corner-tag lab transform; keep camera-frame positions. |
| `--sensor-csv` | none | Micro-controller CSV; if given, its (body-frame) `angle_value` supplies the heading and encoder/motor are carried through. |
| `--sensor-angle-units` | `rad` | Units of the sensor `angle_value` column (`rad` or `deg`). |
| `--motion-onset-frame` | auto | Manual video motion-onset frame for sensor sync (overrides auto-detection). |
| `--motion-threshold` | auto | Manual speed threshold for motion-onset detection (overrides the auto noise floor). |
| `--baseline` | derived from data | Node-to-node template radius (m); does **not** affect the fitted body angle. |
| `--topology` | `auto` (7=hub_spoke, 6=ring) | Reference topology for the body-angle template (from `robot_topology.py`). |
| `--fps` | raw-CSV header, else 30 | Override the frame rate. |
| `--output` | `<raw>_robot.csv` | Output wide CSV path. |
| `--log` | `<raw>_robot.log` | Text log path (stats + warnings). |

## `format_tracks_single.py`

| Argument | Default | Description |
|---|---|---|
| `raw_csv` (positional) | — | Raw CSV from `apriltag_tracker.py`. |
| `--node-id` | unique non-corner detected tag | Robot tag id. |
| `--corner-ids` | raw-CSV header, else `26 27 28 29` | Corner tag ids used to build the lab frame. |
| `--arena-size W H` | observed corner spacing | Real lab-rectangle dimensions for the lab transform. |
| `--camera-frame` | off | Skip the corner-tag lab transform; keep camera-frame positions. |
| `--min-gap-seconds` | `0.5` | Dropout this long (or longer) starts a new trajectory. Ignored if `--min-gap-frames` is given. |
| `--min-gap-frames` | from `--min-gap-seconds` | Dropout length in frames that starts a new trajectory. |
| `--min-track-seconds` | `0.25` | Drop trajectories shorter than this. |
| `--fps` | raw-CSV header, else 30 | Override the frame rate. |
| `--output` | `<raw>_robot.csv` | Output wide CSV path. |
| `--log` | `<raw>_robot.log` | Text log path (stats + warnings). |

## `analyze_modes.py`

| Argument | Default | Description |
|---|---|---|
| `robot_csv` (positional) | — | Formatted CSV from `format_tracks.py` (ring robot). |
| `--k` | `1.0` | Uniform spring constant (sets the λ scale; arbitrary units). |
| `--kappa` | `0` (off) | Bond-bending stiffness about the ideal interior angle. Lifts the mechanisms; rigid modes stay at zero; sector identities unchanged. |
| `--baseline` | CSV header, else derived | Template circumradius (m); else from the mean observed spring length. |
| `--topology` | `auto` (CSV header / node count) | Reference topology; the wave-sector analysis requires `ring`. |
| `--angle-frame` | `lab` | Heading frame for order parameter, kymographs, autocorrelations, diffusion. Modal projections always use the body frame. |
| `--nematic` | off (polar) | Nematic \|⟨e^{2iγ}⟩\| instead of polar. |
| `--vel-smooth-window` | `0` (off) | Savitzky–Golay window (odd frames) for velocities and modal-amplitude derivatives; `0` = central differences. |
| `--wave-window` | `1.0` | Averaging window (s) for the windowed strain-wave order parameter `W_win(t)`. |
| `--modes-dir` | `/Users/.../tplax_paper` | Shared normal-mode cache directory. |
| `--recompute-modes` | off | Recompute and overwrite the cached modes for this lattice. |
| `--bins` | `50` | Bins per axis for the CoM 2D histogram. |
| `--zero-mode-tol` | `1e-6` | \|λ\| below which a mode counts as a zero mode. |
| `--output` | `<robot>_analysis.npz` | Output analysis bundle. |
| `--summary` | `<robot>_analysis.txt` | Output summary (params, checks, scalars). |

## `plot_analysis.py`

| Argument | Default | Description |
|---|---|---|
| `analysis_npz` (positional) | — | `*_analysis.npz` from `analyze_modes.py`. |
| `--outdir` | `<npz>_plots/` | Output directory for the figures. |
| `--dpi` | `130` | Figure DPI. |
| `--n-modes` | `4` | Number of deformation mode shapes to draw. |

## `plot_analysis_single.py`

| Argument | Default | Description |
|---|---|---|
| `robot_csv` (positional) | — | Formatted CSV from `format_tracks_single.py`. |
| `--outdir` | `<csv>_plots/` | Output directory for the figures. |
| `--dpi` | `130` | Figure DPI. |

---

# Known limitations / notes carried forward

- **`tag16h5`** is a weak family; certain ids (observed: tag 1) can be intermittently detected.
  Loosening `--hamming-max` to 1 recovers marginal decodes at the cost of more false positives;
  reprinting in `tag36h11` is the robust fix (same pose accuracy — corner localization is
  border-based — with far better decode robustness, at the cost of needing enough pixels per
  tag).
- **Peripheral poses** are the least accurate (largest distortion, weakest calibration
  constraint), independent of the cosmetic undistort crop.
- **Intermittently-detected tags** get heavy interpolation, which suppresses their apparent
  caster diffusion — interpret per-node stats for such tags with care.
- **No total energy** is computed: the system is active and dissipative, and KE (unit mass)
  and PE (units of k) are not commensurate.
- **Lab-frame vs body-frame heading** changes only diffusion, autocorrelations and the
  kymographs; modal projections always use the body frame and the order parameter is
  invariant. Default is lab. Provenance (heading source +
  angle frame) is stamped on every figure and into every plot filename so the two are never
  confused.
- **Sensor-present experiments use a body-fixed tag**, so the tag caster-angle is not the caster
  heading and (correctly) does not correlate with the encoder; the encoder is the heading source
  and per-node hardware zeros are "roughly aligned but noisy", so cross-node angle statistics
  inherit that per-node zero noise.
- **Single-node track splitting** only sees lost detections. If the tag is still decoded
  during a hand-cover, the gap never exceeds `--min-gap-seconds` and the video stays one
  track; check the gap histogram in the `format_tracks_single.py` log.
