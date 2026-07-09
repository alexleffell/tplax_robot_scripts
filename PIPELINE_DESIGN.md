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
- **`body_angle` (absolute)** via **Procrustes / Kabsch** fit of the node positions to a
  **regular-polygon template** (hub at origin, ring nodes on a circle of radius `--baseline`
  in ascending id order). This is the standard least-squares (Eckart-frame) resolution of the
  rigid rotation of a *deformable* body — the rotation that minimizes residual deformation.
  Labeled tags remove the polygon's rotational-symmetry ambiguity. The template radius does
  **not** affect the fitted angle (Kabsch rotation is scale-invariant); `--baseline` only sets
  the template's radius and is otherwise cosmetic here.
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

# 4. `analyze_modes.py`

**Purpose.** Compute all physical quantities for a polar active solid and store them in one
`.npz` bundle (plus a `.txt` summary with sanity checks). All heavy physics lives here.

## Foundational choices

- **Normal modes = passive elastic modes** (Hermitian spring Hessian), eigen-decomposed.
  This is the basis Baconnier–Dauchot project onto for "collective actuation"; the
  non-normality that *selects* a mode lives in the coupled position+polarity stability
  operator, which is not needed to *measure* the dynamics. Documented caveat: "normal mode"
  here means the passive one.
- **Reference (equilibrium) configuration** = the regular-polygon template built from
  `--baseline` (or data-derived if absent). Chosen over the mean-observed shape by user
  preference; radius is derived from the mean hub→ring distance when `--baseline` is omitted.
- **Spring parameters**: global `--k` and `--l0`. `--l0 = None` means each bond is relaxed at
  its own equilibrium length (zero pre-tension).
- **Unit mass** throughout (`KE = ½Σ|v|²`); velocities from finite differences of node
  positions at `dt = 1/fps`.
- **"Zero modes" = the 3 rigid-body modes** (2 translations + 1 rotation). Their KE is kept for
  the ratio; the deformation spectrum is computed after removing rigid-body motion.
- **Shared, cached normal modes.** Modes are loaded from
  `/Users/alexleffell/Documents/PhD/tplax/tplax_paper` (`--modes-dir`) keyed by a lattice
  signature (nodes + canonicalized connections + `k` + `l0`, plus radius only when `l0`
  introduces tension — relaxed-lattice eigenvectors are radius-independent). On a hit, the
  stored eigenvalues/eigenvectors are reused verbatim so **ordering and sign are identical
  across every experiment on the same lattice**; on a miss they are computed with a canonical
  sign convention (largest-magnitude component positive) and written there. This is the fix for
  the fact that `eigh` is not guaranteed repeatable — especially across degenerate subspaces
  (e.g. the hexagon's degenerate λ pair), which are only defined up to rotation within the
  subspace until pinned by the cache.
- **Caster-angle frame** selectable via `--angle-frame {lab,body}` (default `lab`, using
  `{n}_theta`; `body` uses `{n}_angle` with rigid rotation removed). Affects the order
  parameter, both angle projections, and the diffusion coefficient. Default is lab because the
  raw tracker measures lab-frame caster orientation; body frame is offered because it removes
  the robot's rigid rotation from the intrinsic caster dynamics.

## Per-calculation details

- **Center-of-mass trajectory** — directly from the `centroid_x/y` columns.

- **Elastic Hessian & normal modes** — central-force spring stiffness at the reference config:
  each bond contributes `k·n̂n̂ᵀ` longitudinally plus `(t/L)(I − n̂n̂ᵀ)` transversely, where
  `t = k(L − l0)` is the equilibrium tension. Eigen-decomposition gives eigenvalues (= ω²,
  unit mass) and orthonormal eigenvectors (modes), ordered ascending.

- **Node-velocity modal projection** — done with **rigid-body motion removed** (the
  deformation spectrum), per the requirement. Rigid removal uses the mechanics decomposition
  (subtract CoM velocity + best-fit angular velocity ω from `Σ r×v / Σ|r|²`), which is
  orthogonal (`v = v_rigid ⊕ v_def`). Modal amplitudes `A = v_def·U`; modal energies `½A²`.

- **Zero-mode KE ratio** — `KE_zero / KE_total` with `KE_zero = KE_total − KE_deform` using the
  orthogonal split. Numerator is the rigid-body KE (zero modes kept, per the requirement);
  denominator is total KE. This measures how rigid-body-like the motion is.

- **Caster-angle projection, two bases** (per request):
  - **Graph-Laplacian**: project the angle vector `θ` (N scalars) onto the network Laplacian
    eigenvectors; the zero mode is the uniform field, tying the zero-mode fraction to global
    alignment. Zero-mode ratio = energy in Laplacian null space / `|θ|²`.
  - **Elastic modes**: embed the caster field as a polarity vector `p = (cosθ, sinθ)` (a 2N
    vector in the same space as velocity) and project onto the elastic eigenvectors. This
    measures how the active driving overlaps each mechanical mode; the zero-mode (rigid)
    fraction = how rigid-body-like the driving is.

- **Spring potential energy** — per spring `½k(L − l0)²` (rest length = `l0`, or the per-bond
  reference length when `l0=None`); summed to `PE_total`.

- **Total energy** — `E = KE_total + PE_total`. (Noted: not conserved — motors inject energy,
  friction dissipates — so `E` is an observable, not an invariant.)

- **Orientation order parameter** — **polar** `Ψ = |⟨e^{iθ}⟩|` by default (matches the
  notebook's magnetization); `--nematic` switches to `|⟨e^{2iθ}⟩|`. Rotation-invariant, so the
  angle-frame choice does not change it.

- **Caster-angle diffusion `D_r`** — from the linear regime of the per-caster unwrapped-angle
  MSD, `D_r = slope/2`, reported per node and mean. (In the lab frame this is contaminated by
  the body's rigid rotation; use `--angle-frame body` for the intrinsic caster diffusion.)

- **CoM 2D histogram / PDF** — normalized 2D histogram of the centroid in the (lab) arena
  frame, `--bins` per axis.

- **PSDs** — Welch (`scipy.signal.welch`, fps sampling) of the order parameter, KE, PE, and
  each modal energy time series.

### Active-solid diagnostics (added for polar active solids)

- **Polarity–velocity coupling** — cosine similarity of the polarity field with the full and
  the deformation velocity (`coupling_pv`, `coupling_pvdef`). This is the collective-actuation
  order parameter: whether the active driving aligns with the mechanical response.
- **Per-mode actuation spectrum** — temporal correlation `corr(C_i, A_i)` between the polarity
  projection and the velocity projection on each mode; tests whether polarity and motion
  condense on the *same* mode.
- **Participation ratio & spectral entropy** — of the deformation modal energy; a low
  participation ratio signals energy condensing into few modes (selective actuation).
- **Modal displacement amplitudes `Q`** — body-frame deformation (`pos − aligned reference`,
  aligned via centroid + `body_angle`) projected onto the modes; used for phase portraits and
  the harmonic-PE check.
- **Phase portraits** — dominant deformation-mode pair (by displacement variance) *and* the
  first two non-zero modes (modes 3 vs 4) explicitly. A closed orbit is the collective-actuation
  signature; a degenerate pair traversed in quadrature is the canonical actuated state.
- **Equipartition / per-mode effective temperature** — `T_kin = 2⟨KE_i⟩` and
  `T_pot = λ_i⟨Q_i²⟩`; deviation from flat is the non-equilibrium signature.
- **Chirality** — mean/std body angular velocity `⟨ω⟩`, net-polarization rotation rate, and the
  signed orbit chirality `⟨q_a q̇_b − q_b q̇_a⟩` of the dominant pair.
- **Orientational autocorrelation + persistence time** and **VACF** — complementary to `D_r`;
  persistence time from the 1/e crossing of `⟨cos Δθ⟩`.
- **Net active force vs CoM motion** — `F = Σ(cosθ, sinθ)` vs `v_cm`, alignment time series and
  scatter; how efficiently alignment converts to locomotion.
- **Spatial polarity structure** — bond alignment `⟨cos(θ_i − θ_j)⟩` over connections, and the
  ring **winding number** of the caster field.

### Sanity checks (printed to the summary)

Equivalent quantities that must agree, verified to machine precision on validated data:
- `Σ modal KE == KE_deform` (Parseval of the deformation velocity);
- `v_rigid · v_def == 0` (orthogonality of the rigid-body removal);
- `KE_zero + KE_deform == KE_total`;
- `Σ caster proj² == |θ|²` (graph-Laplacian) and `== |p|²` (elastic) (Parseval of both bases);
- `KE + PE == E_total`;
- harmonic PE vs spring PE (approximate small-deformation cross-check, using body-frame `Q`);
- participation ratio within `[1, 2N]`, `|coupling| ≤ 1` (bounds).

**Assumptions**
- Deformations are small enough for the linearized (harmonic) mode picture to be informative;
  the harmonic-PE cross-check quantifies how far this holds.
- The passive Hessian is a valid basis for describing (not predicting) the dynamics.
- Ring node ids are ordered consistently with the template; non-star topologies fall back to a
  mean-shape reference with a warning.
- A near-zero third eigenvalue can appear if `--l0` differs slightly from the reference bond
  length (tiny pre-tension); the 3 lowest modes are always treated as rigid regardless.

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
1. CoM trajectory (time-colored)  2. CoM 2D PDF  3. Energies (KE/PE/E)  4. Rigid-body KE
fraction  5. Caster zero-mode fractions (both bases)  6. Orientation order parameter
7. Eigenvalue spectrum + per-mode KE  8. Deformation mode shapes (quiver)  9. Caster MSD +
diffusion fit  10. PSDs (order parameter, KE/PE, modal-energy heatmap)  11. Collective
actuation (coupling, actuation spectrum, condensation)  12. Phase portraits (dominant pair +
first two non-zero modes)  13. Per-mode effective temperature  14. Chirality (ω + polarization
angle)  15. Orientational ACF + VACF  16. Active force vs CoM velocity  17. Spatial polarity
(bond alignment + winding)  18. Per-node angular-velocity PSD (7 curves)  19. Pairwise node
velocity correlation heatmap  20. Pairwise heading angular-velocity correlation heatmap.

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
$PY analyze_modes.py ../Data/240226/240226_med_low_1_robot.csv --k 1.0 --l0 0.1689 --angle-frame body
#   -> ..._analysis.npz (+ ..._analysis.txt); modes cached in /Users/.../tplax_paper

# 5. Plot
$PY plot_analysis.py ../Data/240226/240226_med_low_1_analysis.npz
#   -> ..._analysis_plots/*.png  (footer-stamped + filename-tagged with heading/frame)
```

Per-experiment knobs to remember:
- **`--baseline`** (steps 3–4): node-to-node rest distance (m); sets the reference lattice.
- **`--k` / `--l0`** (step 4): spring constant / rest length. Omit `--l0` for a relaxed lattice
  (no pre-tension; exact zero rotation mode).
- **`--angle-frame body`** (step 4): use whenever the sensor is the heading source (tag body-fixed).
- **`--sensor-angle-units`** (step 3): default `rad` (correct for the observed [0, 2π] range).
- **`--camera-frame`** (step 3): skip the lab transform if corner tags are unusable.
- Tag family/sizes (step 2) default to `tag16h5` / 0.045 m / 0.037 m — override per dataset.

Steps 3→4→5 are re-run while iterating on analysis; steps 1–2 are done once per camera/video.

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
| `--fps` | raw-CSV header, else 30 | Override the frame rate. |
| `--output` | `<raw>_robot.csv` | Output wide CSV path. |
| `--log` | `<raw>_robot.log` | Text log path (stats + warnings). |

## `analyze_modes.py`

| Argument | Default | Description |
|---|---|---|
| `robot_csv` (positional) | — | Formatted CSV from `format_tracks.py`. |
| `--k` | `1.0` | Uniform spring constant. |
| `--l0` | each bond's equilibrium length | Uniform spring rest length; omit for a relaxed network (no pre-tension). |
| `--baseline` | CSV header, else derived | Template radius (m). |
| `--angle-frame` | `lab` | Caster-angle frame for order parameter/projections/diffusion: `lab` uses `{n}_theta`, `body` uses `{n}_angle`. |
| `--nematic` | off (polar) | Use nematic order parameter \|⟨e^{2iθ}⟩\| instead of polar. |
| `--vel-smooth-window` | `0` (off) | Savitzky-Golay window (odd frames) for the velocity estimate; velocity = SG analytic derivative (deriv=1). `0` = plain central difference. Suppresses the finite-difference noise floor in KE. |
| `--modes-dir` | `/Users/.../tplax_paper` | Shared normal-mode cache directory (keyed by lattice). |
| `--recompute-modes` | off | Recompute and overwrite the cached modes for this lattice. |
| `--bins` | `50` | Bins per axis for the CoM 2D histogram. |
| `--zero-mode-tol` | `1e-6` | Eigenvalue tolerance for reporting how many modes are numerically zero. |
| `--output` | `<robot>_analysis.npz` | Output analysis bundle. |
| `--summary` | `<robot>_analysis.txt` | Output summary (params, sanity checks, scalars). |

## `plot_analysis.py`

| Argument | Default | Description |
|---|---|---|
| `analysis_npz` (positional) | — | `*_analysis.npz` from `analyze_modes.py`. |
| `--outdir` | `<npz>_plots/` | Output directory for the figures. |
| `--dpi` | `130` | Figure DPI. |
| `--n-modes` | `4` | Number of deformation mode shapes to draw. |

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
- **Total energy is not conserved** (active, dissipative system); treat `E_total` as an
  observable.
- **Lab-frame vs body-frame caster analysis** materially changes diffusion and the modal angle
  projections; the order parameter is invariant. Default is lab. Provenance (heading source +
  angle frame) is stamped on every figure and into every plot filename so the two are never
  confused.
- **Sensor-present experiments use a body-fixed tag**, so the tag caster-angle is not the caster
  heading and (correctly) does not correlate with the encoder; the encoder is the heading source
  and per-node hardware zeros are "roughly aligned but noisy", so cross-node angle statistics
  inherit that per-node zero noise.
