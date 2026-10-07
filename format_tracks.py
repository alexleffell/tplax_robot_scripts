#!/usr/bin/env python3
"""
Format raw AprilTag tracks into an analysis-ready per-frame table.

Takes the raw CSV from apriltag_tracker.py and:
  1. interpolates missing per-node detections over all frames (linear, both directions);
  2. transforms positions into the LAB frame using the fixed corner tags (falls back
     to the camera frame if the corner tags are not detected);
  3. computes the centroid and TWO body-angle conventions per frame:
       - body_angle             : absolute Procrustes (Kabsch) fit to a regular-polygon
                                   template built from --connections and --baseline;
       - body_angle_incremental : reference-free frame-to-frame Kabsch, integrated.
     Because the robot deforms, the rigid rotation is only least-squares-defined; the
     Kabsch fit is the standard (Eckart-frame) resolution, and labeled tags remove the
     polygon's symmetry ambiguity.

Outputs a wide CSV (one row per frame) with a commented metadata header compatible with
the notebook's ``read_csv_comments``, plus a text log of statistics and warnings.

Column order:
    time,
    {id}_x, {id}_y, {id}_z, {id}_theta, {id}_angle    (per node, in connection order)
    centroid_x, centroid_y, body_angle, body_angle_incremental,
    extra_tag_{id}_x, extra_tag_{id}_y                (per extra tag)

Example
-------
    python format_tracks.py ../Data/020525/020525_clipped_2_raw.csv \
        --connections "[(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]"
"""

import argparse
import ast
import os
from io import StringIO

import cv2
import numpy as np
import pandas as pd

from robot_topology import (circumradius_from_spacing, default_rest_spacing,
                            infer_ring_order, reference_template, report_ring_direction,
                            resolve_topology, ring_cycle, ring_direction)

DEFAULT_CONNECTIONS = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                       (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    """Writes messages to both stdout and a log file."""

    def __init__(self, path):
        self._fh = open(path, "w")

    def __call__(self, msg=""):
        print(msg)
        self._fh.write(str(msg) + "\n")

    def close(self):
        self._fh.close()


# --------------------------------------------------------------------------- #
# IO helpers (mirror the notebook's read_csv_comments convention)
# --------------------------------------------------------------------------- #
def read_csv_comments(path):
    metadata = {}
    data_lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                key, value = line[2:].strip().split(":", 1)
                key, value = key.strip(), value.strip()
                try:
                    metadata[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    metadata[key] = value
            else:
                data_lines.append(line)
    df = pd.read_csv(StringIO("".join(data_lines)))
    df.attrs = metadata
    return df


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #
def interpolate_tracks(df, total_frames, log):
    """Fill missing per-node frames with linear interpolation (both directions)."""
    interpolated = []
    node_ids = df["node_id"].unique()
    gap_sizes = []
    pct_interp = []

    for nid in node_ids:
        node = df[df["node_id"] == nid].copy()
        grid = pd.DataFrame({"frame#": range(total_frames), "node_id": nid})
        merged = pd.merge(grid, node, on=["node_id", "frame#"], how="left")
        for col in ("x", "y", "z", "angle"):
            merged[col] = merged[col].interpolate(method="linear", limit_direction="both")
        interpolated.append(merged)

        orig = sorted(set(node["frame#"]))
        interp_times = set(range(total_frames))
        pct_interp.append(100.0 * (len(interp_times) - len(orig)) / max(len(interp_times), 1))
        for i in range(len(orig) - 1):
            gap = orig[i + 1] - orig[i] - 1
            if gap > 0:
                gap_sizes.append(gap)

    out = pd.concat(interpolated, ignore_index=True).sort_values(["node_id", "frame#"])

    log("\n=== Interpolation statistics ===")
    log(f"Original detections     : {len(df)}")
    log(f"After interpolation      : {len(out)}")
    log(f"Tags (unique ids)        : {len(node_ids)}")
    log(f"Added (interpolated) rows: {len(out) - len(df)}")
    log(f"Avg % interpolated / tag : {np.mean(pct_interp):.2f}%")
    log(f"Max % interpolated / tag : {np.max(pct_interp):.2f}%")
    log(f"Avg gap size (frames)    : {np.mean(gap_sizes) if gap_sizes else 0:.2f}")
    log(f"Max gap size (frames)    : {np.max(gap_sizes) if gap_sizes else 0}")
    return out


def pivot_series(df, total_frames, value):
    """Return a (total_frames x n_ids) DataFrame of `value`, indexed by frame, columns=node_id."""
    wide = df.pivot_table(index="frame#", columns="node_id", values=value)
    return wide.reindex(range(total_frames))


# --------------------------------------------------------------------------- #
# Lab-frame transform
# --------------------------------------------------------------------------- #
def order_quad(points):
    """Order 4 points counter-clockwise, starting at the one nearest the origin corner."""
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    order = np.argsort(ang)
    ordered = points[order]
    start = int(np.argmin(ordered.sum(axis=1)))  # smallest x+y
    ordered = np.roll(ordered, -start, axis=0)
    order = np.roll(order, -start)
    return ordered, order


def compute_lab_transform(mean_corners, arena_size, log):
    """2D similarity transform (2x3 M) mapping mean corner points to an axis-aligned rectangle.

    Returns (M, residual_rms) or (None, None) if it cannot be built.
    """
    if len(mean_corners) != 4:
        log(f"WARNING: expected 4 corner tags for the lab frame, found "
            f"{len(mean_corners)}; staying in the camera frame.")
        return None, None

    src, _ = order_quad(mean_corners)
    if arena_size is not None:
        w, h = float(arena_size[0]), float(arena_size[1])
    else:
        # Side lengths from the observed quad (edges 0-1,2-3 -> width; 1-2,3-0 -> height).
        w = 0.5 * (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3]))
        h = 0.5 * (np.linalg.norm(src[2] - src[1]) + np.linalg.norm(src[3] - src[0]))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst)
    if M is None:
        log("WARNING: similarity transform estimation failed; staying in the camera frame.")
        return None, None

    proj = (M[:, :2] @ src.T).T + M[:, 2]
    residual = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    return M, residual


def apply_transform(M, x, y):
    """Apply a 2x3 affine transform to arrays x, y (same shape)."""
    if M is None:
        return x, y
    xa = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ya = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return xa, ya


# --------------------------------------------------------------------------- #
# Procrustes / Kabsch
# --------------------------------------------------------------------------- #
def kabsch_angle(A, B):
    """Rotation angle (rad) of the least-squares rotation mapping A -> B (both Nx2)."""
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0]))


def mean_shape_template(nodes, xw, yw):
    # Only include detected nodes; undetected ones have no column in the pivot.
    return {n: (float(np.nanmean(xw[n].values)), float(np.nanmean(yw[n].values)))
            for n in nodes if n in xw.columns}


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def project_to_table_plane(raw, node_ids, log, min_points=50):
    """Replace node tag positions by the intersection of each tag's viewing ray with the table
    plane.

    solvePnP depth for a small tag is noisy (~1–2 % of the distance frame to frame), and the
    tracker's x, y are depth × ray direction, so depth noise leaks into the in-plane position.
    All node tags ride on the same (flat) table, so: fit one plane to every node detection of
    the video (SVD, two rounds of 3·MAD outlier rejection), then place each detection where its
    ray (tvec direction = ray through the tag centre in undistorted coordinates) meets that
    plane. The plane is the average over many thousand detections, so the metric scale is
    kept while the per-frame depth noise is removed. Corner/extra tags are left unchanged.
    """
    m = (raw["node_id"].isin(node_ids) & np.isfinite(raw["x"]) & np.isfinite(raw["y"])
         & np.isfinite(raw["z"]) & (raw["z"] > 0))
    P = raw.loc[m, ["x", "y", "z"]].to_numpy(dtype=float)
    if len(P) < min_points:
        log(f"\nPlanar projection skipped: only {len(P)} node detections.")
        return raw
    keep = np.ones(len(P), dtype=bool)
    for _ in range(3):
        c = P[keep].mean(axis=0)
        _, _, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = Vt[-1] * (1.0 if Vt[-1][2] > 0 else -1.0)
        res = (P - c) @ n
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res - np.median(res[keep])) <= max(3 * mad, 1e-9)
    dist = float(n @ c)
    rays = P / P[:, 2:3]                                    # (x/z, y/z, 1)
    t = dist / (rays @ n)
    Pp = rays * t[:, None]

    def jitter(df):
        js = []
        for _, g in df.groupby("node_id"):
            g = g.sort_values("frame#")
            consecutive = np.diff(g["frame#"].to_numpy()) == 1
            if consecutive.sum() > 10:
                dx = np.diff(g[["x", "y"]].to_numpy(), 2, axis=0)
                ok = consecutive[1:] & consecutive[:-1]
                js.append(np.median(np.linalg.norm(dx[ok], axis=1)))
        return float(np.median(js)) if js else np.nan

    before = jitter(raw.loc[m])
    out = raw.copy()
    out.loc[m, ["x", "y", "z"]] = Pp
    after = jitter(out.loc[m])
    tilt = np.degrees(np.arccos(abs(n[2])))
    log(f"\nPlanar projection onto the table plane (from {keep.sum()} of {len(P)} node detections): "
        f"distance {dist:.4f} m, tilt {tilt:.2f}° from the image plane, depth scatter about the "
        f"plane {1e3 * np.std(res[keep]):.1f} mm (rms).")
    log(f"  median frame-to-frame position jitter (2nd difference): {1e3 * before:.2f} mm -> "
        f"{1e3 * after:.2f} mm")
    return out


def circ_mean(angles):
    """Circular mean of a 1D array of angles (rad); NaN if empty."""
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


# --------------------------------------------------------------------------- #
# Micro-controller sensor merge (optional)
# --------------------------------------------------------------------------- #
def detect_motion_onset(xw, yw, nodes, fps, total_frames, threshold=None, min_run=3):
    """First frame where ANY node's speed exceeds a threshold (auto from noise floor).

    Returns (onset_frame, threshold, speed_series)."""
    speeds = []
    for n in nodes:
        if n in xw.columns:
            x, y = xw[n].values, yw[n].values
            speeds.append(np.hypot(np.gradient(x), np.gradient(y)) * fps)
    if not speeds:
        return 0, 0.0, np.zeros(total_frames)
    s = np.nan_to_num(np.nanmax(np.vstack(speeds), axis=0))
    if threshold is None:
        srt = np.sort(s)
        base = np.median(srt[:max(1, int(0.2 * len(srt)))])       # quiet-period noise floor
        mad = np.median(np.abs(s - base))
        threshold = max(base + 6 * 1.4826 * mad, base * 3, 1e-9)
    above = s > threshold
    onset = 0
    for i in range(len(above) - min_run + 1):
        if np.all(above[i:i + min_run]):
            onset = i
            break
    return onset, float(threshold), s


def load_and_sync_sensor(path, nodes, total_frames, fps, xw, yw, body_angle, args, log):
    """Load the micro-controller CSV, sync it to the video, and interpolate each node's
    caster heading (body frame) onto the video frame grid. Returns a dict of per-node arrays
    or None if the file has no actuation (cannot sync).

    Geometry (sensor-present experiments): the AprilTag is fixed to the node BODY, not the
    caster, so the tag angle is the node-base orientation and does NOT measure the caster.
    The magnetic-encoder ``angle_value`` IS the caster heading in the body frame directly.
    Hence the sensor angle is used raw (no tag alignment): body-frame heading = sensor angle
    (up to the hardware zero, which was set at a roughly aligned orientation), and lab-frame
    heading = body_angle + sensor angle.

    Sync: video and sensor share a real-time clock (ESP-NOW broadcast). The globally
    earliest motor_command != 0 is pinned to the first video-motion frame -> single offset.
    Heading is sensor-only: NaN outside sensor coverage.
    """
    sdf = pd.read_csv(path)
    required = {"timestamp_s", "timestamp_us", "node_id", "encoder_value",
                "angle_value", "motor_command"}
    missing = required - set(sdf.columns)
    if missing:
        raise SystemExit(f"Sensor CSV missing columns: {sorted(missing)}")
    # Drop pre-sync / glitch rows: before the base-station epoch broadcast a node reports its
    # local uptime (small timestamp_s), which would corrupt the shared-clock interpolation.
    n_dropped = 0
    if sdf["timestamp_s"].max() > 1e9:
        bad = sdf["timestamp_s"] < 1e9
        n_dropped = int(bad.sum())
        if n_dropped:
            sdf = sdf[~bad].copy()
    sdf = sdf.assign(t=sdf["timestamp_s"].astype(float) + sdf["timestamp_us"].astype(float) * 1e-6)
    ang_scale = np.pi / 180.0 if args.sensor_angle_units == "deg" else 1.0

    log("\n=== Sensor merge ===")
    log(f"Sensor CSV              : {path}  ({len(sdf)} rows, {n_dropped} pre-sync/glitch dropped)")
    log(f"angle_value raw range   : [{sdf['angle_value'].min():.4g}, {sdf['angle_value'].max():.4g}] "
        f"(units={args.sensor_angle_units}; check this looks right)")

    nz = sdf[sdf["motor_command"] != 0]
    if nz.empty:
        log("WARNING: motor_command is never non-zero; cannot sync. Falling back to tag heading.")
        return None
    motor_onset = float(nz["t"].min())

    if args.motion_onset_frame is not None:
        onset_frame, thr = int(args.motion_onset_frame), None
    else:
        onset_frame, thr, _ = detect_motion_onset(xw, yw, nodes, fps, total_frames,
                                                   args.motion_threshold)
    offset = onset_frame / fps - motor_onset
    log(f"Motor onset (sensor t)  : {motor_onset:.4f} s")
    log(f"Video motion onset frame: {onset_frame}"
        + (f" (auto, speed thr={thr:.4g})" if thr is not None else " (manual)"))
    log(f"Sync offset (video-sensor): {offset:+.4f} s")

    tf = np.arange(total_frames) / fps
    out = {"theta_lab": {}, "angle_body": {}, "encoder": {}, "motor": {},
           "offset": offset, "onset_frame": onset_frame, "motor_onset": motor_onset,
           "threshold": thr, "sensor_nodes": [], "coverage": {}}

    def interp_mask(tv, vals):
        v = np.interp(tf, tv, vals)
        v[(tf < tv.min()) | (tf > tv.max())] = np.nan
        return v

    for n in nodes:
        sub = sdf[sdf["node_id"] == n].sort_values("t")
        if len(sub) < 2:
            continue
        tv = sub["t"].values + offset
        # Raw sensor angle is the body-frame caster heading (tag is body-fixed here, so
        # there is no tag caster reference to align to).
        ab = wrap_angle(interp_mask(tv, sub["angle_value"].values.astype(float) * ang_scale))
        out["angle_body"][n] = ab
        out["theta_lab"][n] = wrap_angle(body_angle + ab)   # lab = body orientation + caster
        out["encoder"][n] = interp_mask(tv, sub["encoder_value"].values.astype(float))
        out["motor"][n] = interp_mask(tv, sub["motor_command"].values.astype(float))
        out["coverage"][n] = int(np.isfinite(ab).sum())
        out["sensor_nodes"].append(n)

    if not out["sensor_nodes"]:
        log("WARNING: no usable per-node sensor data; falling back to tag heading.")
        return None
    log("Heading = raw sensor caster angle (tag is body-fixed; not used for the heading).")
    log(f"Sensor nodes            : {out['sensor_nodes']}")
    log(f"Per-node frame coverage : "
        + ", ".join(f"{n}:{out['coverage'][n]}/{total_frames}" for n in out["sensor_nodes"]))
    absent = [n for n in nodes if n not in out["sensor_nodes"]]
    if absent:
        log(f"WARNING: nodes with no sensor data (heading -> NaN, sensor-only): {absent}")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Format raw AprilTag tracks into a wide per-frame table")
    p.add_argument("raw_csv", type=str, help="Raw CSV from apriltag_tracker.py")
    p.add_argument("--connections", type=str, default=repr(DEFAULT_CONNECTIONS),
                   help="List of (i, j) node connection tuples (Python literal). "
                        "Default: hub-and-spoke over nodes 0..6")
    p.add_argument("--corner-ids", type=int, nargs="+", default=None,
                   help="Corner tag ids for the lab frame. Default: from raw-CSV header, else 26 27 28 29")
    p.add_argument("--arena-size", type=float, nargs=2, default=None, metavar=("W", "H"),
                   help="Real lab-rectangle dimensions. If omitted, observed corner spacing is used.")
    p.add_argument("--camera-frame", action="store_true",
                   help="Skip the corner-tag lab-frame transform and keep all positions in the "
                        "camera frame. (Corner tags still appear as extra tags.)")
    p.add_argument("--strict-connections", action="store_true",
                   help="Always use --connections as given. Default: for a ring, if the nodes' "
                        "physical cyclic order (inferred from positions) differs from "
                        "--connections, use the inferred order and warn (catches swapped tags).")
    p.add_argument("--pnp-depth", action="store_true",
                   help="Use the raw solvePnP translations for node positions. Default: project "
                        "each node tag's viewing ray onto the table plane fitted to all node "
                        "detections, which removes per-frame PnP depth noise (several mm of "
                        "spurious in-plane jitter otherwise).")
    p.add_argument("--sensor-csv", type=str, default=None,
                   help="Optional micro-controller CSV (timestamp_s, timestamp_us, node_id, "
                        "encoder_value, angle_value, motor_command). If given, its (body-frame) "
                        "angle_value is used for the heading and encoder/motor are carried through.")
    p.add_argument("--sensor-angle-units", choices=["rad", "deg"], default="rad",
                   help="Units of the sensor angle_value column. Default: rad.")
    p.add_argument("--motion-onset-frame", type=int, default=None,
                   help="Manual video motion-onset frame for sensor sync (overrides auto-detection).")
    p.add_argument("--motion-threshold", type=float, default=None,
                   help="Manual speed threshold for motion-onset detection (overrides auto noise floor).")
    p.add_argument("--baseline", type=float, default=None,
                   help="Rest node-to-node (spring) distance (m) for the template. Default: the "
                        "measured value for the topology (robot_topology.DEFAULT_REST_SPACING, "
                        "0.153 m for the 6-node ring). Recorded in the CSV header and used by "
                        "analyze_modes.py; it does not affect the fitted body_angle.")
    p.add_argument("--topology", type=str, default="auto",
                   help="Reference topology for the body-angle template: 'auto' (by node count: "
                        "7=hub_spoke, 6=ring), or an explicit name from robot_topology.py.")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps (else read from the raw-CSV header, else 30).")
    p.add_argument("--output", type=str, default=None, help="Output CSV. Default: <raw>_robot.csv")
    p.add_argument("--log", type=str, default=None, help="Log file. Default: <raw>_robot.log")
    return p.parse_args()


def main():
    args = parse_args()
    core = args.raw_csv[:-8] if args.raw_csv.endswith("_raw.csv") else os.path.splitext(args.raw_csv)[0]
    out_csv = args.output or (core + "_robot.csv")
    log = Logger(args.log or (core + "_robot.log"))

    connections = ast.literal_eval(args.connections)
    connections = [tuple(c) for c in connections]
    using_default_connections = (args.connections == repr(DEFAULT_CONNECTIONS))

    raw = read_csv_comments(args.raw_csv)
    meta = raw.attrs
    fps = args.fps or meta.get("fps", 30)
    total_frames = int(meta.get("total_frames", raw["frame#"].max() + 1))
    corner_ids = args.corner_ids or meta.get("corner_ids", [26, 27, 28, 29])
    corner_ids = list(corner_ids)

    log(f"Raw CSV     : {args.raw_csv}")
    log(f"Detections  : {len(raw)}")
    log(f"fps         : {fps}")
    log(f"total_frames: {total_frames}")
    log(f"connections : {connections}")
    log(f"corner_ids  : {corner_ids}")

    nodes = sorted({n for c in connections for n in c})
    detected = sorted(raw["node_id"].unique())
    node_set = set(nodes)
    corner_set = set(corner_ids)
    extra_tags = sorted(d for d in detected if d not in node_set)
    log(f"\nNodes (from connections): {nodes}  (N_nodes={len(nodes)})")
    log(f"Detected tag ids        : {detected}")
    log(f"Extra tags              : {extra_tags}")

    missing_nodes = [n for n in nodes if n not in detected]
    if missing_nodes:
        log(f"WARNING: node(s) never detected in any frame: {missing_nodes} "
            f"(their columns will be NaN).")
        if using_default_connections:
            log("!" * 70)
            log(f"NOTE: --connections was not given, so the DEFAULT 7-node hub-and-spoke is in "
                f"use. The node set (hence N_nodes and topology) comes from --connections, NOT "
                f"from the detected tags {detected}. If this is a different lattice, pass "
                f"--connections for it (e.g. a 6-ring: "
                f"\"[(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]\").")
            log("!" * 70)

    # Planar projection of node positions (removes PnP depth noise; see project_to_table_plane).
    if not args.pnp_depth:
        raw = project_to_table_plane(raw, nodes, log)
    else:
        log("\nPlanar projection disabled (--pnp-depth): using raw solvePnP translations.")

    # Interpolate, then pivot to per-frame arrays.
    interp = interpolate_tracks(raw, total_frames, log)
    xw = pivot_series(interp, total_frames, "x")
    yw = pivot_series(interp, total_frames, "y")
    zw = pivot_series(interp, total_frames, "z")
    aw = pivot_series(interp, total_frames, "angle")

    # Per-node detection rate (fraction of frames with a real, pre-interpolation detection).
    log("\n=== Per-node detection rate (pre-interpolation) ===")
    counts = raw.groupby("node_id")["frame#"].nunique()
    for n in nodes:
        c = int(counts.get(n, 0))
        log(f"  node {n:>3}: {c}/{total_frames} frames ({100.0 * c / total_frames:.1f}%)")

    # --- Lab frame -------------------------------------------------------- #
    if args.camera_frame:
        M, residual = None, None
        log("\nLab frame: DISABLED via --camera-frame; keeping the camera frame.")
    else:
        present_corners = [cid for cid in corner_ids if cid in xw.columns]
        mean_corners = np.array([[np.nanmean(xw[cid].values), np.nanmean(yw[cid].values)]
                                 for cid in present_corners])
        M, residual = compute_lab_transform(mean_corners, args.arena_size, log)
    frame_label = "lab" if M is not None else "camera"
    if M is not None:
        rot = float(np.arctan2(M[1, 0], M[0, 0]))
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        log(f"\nLab frame: similarity transform fitted "
            f"(rotation={np.degrees(rot):.2f} deg, scale={scale:.4f}, "
            f"corner-fit RMS residual={residual:.4g}).")
    else:
        rot = 0.0
        log("\nLab frame: using CAMERA frame (no valid corner-tag transform).")

    # Apply transform to every tag's position; rotate caster angles into the lab frame.
    for col in xw.columns:
        xw[col], yw[col] = apply_transform(M, xw[col].values, yw[col].values)
    aw = aw + rot  # in-plane caster angle expressed in the lab frame

    corner_locs = [(float(np.nanmean(xw[cid].values)), float(np.nanmean(yw[cid].values)))
                   if cid in xw.columns else (np.nan, np.nan) for cid in corner_ids]

    # --- Template for absolute body angle --------------------------------- #
    # Idealized per-topology reference (repeatable across experiments; a flexible robot's
    # mean shape is not). Radius is irrelevant to the body angle (Kabsch is scale-invariant).
    if args.baseline is None:
        args.baseline = default_rest_spacing(args.topology, len(nodes))
    radius = (circumradius_from_spacing(args.baseline, args.topology, len(nodes))
              if args.baseline else 1.0)
    # Ring order: the springs join physical neighbours, so the true connections are given by the
    # nodes' cyclic order around the ring. If --connections disagrees (e.g. two tags swapped
    # between builds) use the order inferred from the positions, unless --strict-connections.
    connections_given = list(connections)
    ring_support = np.nan
    if (resolve_topology(args.topology, len(nodes)) == "ring"
            and all(n in xw.columns for n in nodes)):
        xy_all = np.stack([np.stack([xw[n].values, yw[n].values], axis=-1) for n in nodes], axis=1)
        inferred, ring_support = infer_ring_order(xy_all, nodes)
        if inferred is None:
            log("WARNING: could not infer the ring order (no frame with every node).")
        else:
            inf_edges = {frozenset((inferred[i], inferred[(i + 1) % len(inferred)]))
                         for i in range(len(inferred))}
            given_edges = {frozenset(c) for c in connections}
            if inf_edges == given_edges:
                log(f"Ring order matches --connections ({100 * ring_support:.1f}% of frames).")
            elif args.strict_connections:
                log(f"WARNING: --connections {connections} do not match the physical ring order "
                    f"{inferred} ({100 * ring_support:.1f}% of frames); keeping them because "
                    "--strict-connections is set. Modal/wave results will be wrong.")
            elif ring_support < 0.5:
                log(f"WARNING: --connections {connections} do not match the most common ring order "
                    f"{inferred}, but that order holds in only {100 * ring_support:.1f}% of frames; "
                    "keeping --connections. Check the tag labelling.")
            else:
                connections = [(inferred[i], inferred[(i + 1) % len(inferred)])
                               for i in range(len(inferred))]
                log("!" * 70)
                log(f"WARNING: --connections {connections_given} do not follow the physical ring "
                    f"order. Using the order inferred from the positions: {inferred} "
                    f"({100 * ring_support:.1f}% of frames), i.e. connections {connections}. "
                    "(Likely swapped/mislabelled tags; use --strict-connections to override.)")
                log("!" * 70)
    # Ring direction: build the template with the same rotational sense as the data (a
    # mirror-image template cannot be fitted by a rotation), and warn on non-monotonic order.
    ring_dir = 1
    if resolve_topology(args.topology, len(nodes)) == "ring":
        cyc = ring_cycle(nodes, connections)
        if cyc is None:
            log("WARNING: ring topology but the connections do not form one cycle through "
                "every node; template uses ascending id order.")
        elif all(n in xw.columns for n in cyc):
            xy = np.stack([np.stack([xw[n].values, yw[n].values], axis=-1) for n in cyc], axis=1)
            info = ring_direction(xy, range(len(cyc)))
            report_ring_direction(info, cyc, log)
            ring_dir = info["direction"]
    template, topology = reference_template(nodes, connections, args.topology, radius=radius,
                                            direction=ring_dir)
    if template is None:
        log(f"WARNING: no built-in template for topology '{topology}' ({len(nodes)} nodes); "
            f"falling back to the (non-repeatable) mean shape. Add it to robot_topology.py.")
        template = mean_shape_template(nodes, xw, yw)
    else:
        log(f"Reference template: topology={topology}, rest node spacing "
            f"{args.baseline if args.baseline else 'unset'} m, circumradius {radius:.4f} m "
            f"(the radius does not affect the body angle).")
    tmpl_nodes = [n for n in nodes if n in template]
    tmpl_pts = np.array([template[n] for n in tmpl_nodes])

    # --- Rigid-body fit (arrays: centroid, body angle) -------------------- #
    centroid_x = np.full(total_frames, np.nan)
    centroid_y = np.full(total_frames, np.nan)
    body_angle = np.full(total_frames, np.nan)
    body_angle_incr = np.full(total_frames, np.nan)
    prev_pts, prev_ids, incr_accum = None, None, 0.0
    skipped_abs = skipped_incr = 0

    for t in range(total_frames):
        node_pos = {}
        for n in nodes:
            if n in xw.columns:
                x, y = xw.at[t, n], yw.at[t, n]
                if np.isfinite(x) and np.isfinite(y):
                    node_pos[n] = (x, y)
        if node_pos:
            centroid_x[t] = np.mean([p[0] for p in node_pos.values()])
            centroid_y[t] = np.mean([p[1] for p in node_pos.values()])

        shared = [n for n in tmpl_nodes if n in node_pos]
        if len(shared) >= 2:
            A = np.array([template[n] for n in shared])
            B = np.array([node_pos[n] for n in shared])
            body_angle[t] = kabsch_angle(A, B)
        else:
            skipped_abs += 1

        cur_ids = sorted(node_pos.keys())
        if prev_pts is not None:
            common = [n for n in cur_ids if n in prev_ids]
            if len(common) >= 2:
                A = np.array([prev_pts[n] for n in common])
                B = np.array([node_pos[n] for n in common])
                incr_accum += kabsch_angle(A, B)
            else:
                skipped_incr += 1
        body_angle_incr[t] = incr_accum
        prev_pts, prev_ids = node_pos, set(cur_ids)

    log("\n=== Body-angle fit ===")
    log(f"Frames skipped (absolute, <2 nodes)   : {skipped_abs}")
    log(f"Frames skipped (incremental, <2 shared): {skipped_incr}")

    # --- Optional sensor heading merge ------------------------------------ #
    heading_source = "tag"
    sensor = None
    if args.sensor_csv:
        sensor = load_and_sync_sensor(args.sensor_csv, nodes, total_frames, fps,
                                      xw, yw, body_angle, args, log)
        if sensor is not None:
            heading_source = "sensor"
    log(f"\nHeading source: {heading_source}")

    # --- Per-frame assembly ----------------------------------------------- #
    rows = []
    for t in range(total_frames):
        row = {"time": t / fps}
        for n in nodes:
            has_pos = n in xw.columns
            row[f"{n}_x"] = xw.at[t, n] if has_pos else np.nan
            row[f"{n}_y"] = yw.at[t, n] if has_pos else np.nan
            row[f"{n}_z"] = zw.at[t, n] if (has_pos and n in zw.columns) else np.nan
            if heading_source == "sensor" and n in sensor["theta_lab"]:
                row[f"{n}_theta"] = sensor["theta_lab"][n][t]
                row[f"{n}_angle"] = sensor["angle_body"][n][t]
            else:
                theta = aw.at[t, n] if has_pos else np.nan
                row[f"{n}_theta"] = theta
                row[f"{n}_angle"] = (wrap_angle(theta - body_angle[t])
                                     if (np.isfinite(theta) and np.isfinite(body_angle[t]))
                                     else np.nan)
            if heading_source == "sensor":
                row[f"{n}_encoder"] = sensor["encoder"][n][t] if n in sensor["encoder"] else np.nan
                row[f"{n}_motor"] = sensor["motor"][n][t] if n in sensor["motor"] else np.nan

        row["centroid_x"] = centroid_x[t]
        row["centroid_y"] = centroid_y[t]
        row["body_angle"] = body_angle[t]
        row["body_angle_incremental"] = body_angle_incr[t]

        for e in extra_tags:
            row[f"extra_tag_{e}_x"] = xw.at[t, e] if e in xw.columns else np.nan
            row[f"extra_tag_{e}_y"] = yw.at[t, e] if e in yw.columns else np.nan

        rows.append(row)

    robot_df = pd.DataFrame(rows)

    # --- Write CSV with metadata header ----------------------------------- #
    attrs = {
        "n_nodes": len(nodes),
        "nodes": nodes,
        "connections": connections,
        "connections_given": connections_given,
        "ring_order_support": (None if not np.isfinite(ring_support) else round(float(ring_support), 4)),
        "corner_ids": corner_ids,
        "corner_locs": corner_locs,
        "extra_tags": extra_tags,
        "baseline": args.baseline,
        "ring_direction": ring_dir,
        "arena_size": list(args.arena_size) if args.arena_size else None,
        "frame": frame_label,
        "fps": fps,
        "heading_source": heading_source,
        "topology": topology,
    }
    if sensor is not None:
        attrs["sensor_angle_units"] = args.sensor_angle_units
        attrs["sensor_sync_offset_s"] = round(sensor["offset"], 6)
        attrs["motion_onset_frame"] = sensor["onset_frame"]
        attrs["motor_onset_time_s"] = round(sensor["motor_onset"], 6)
        attrs["sensor_nodes"] = sensor["sensor_nodes"]
        attrs["tag_mount"] = "body"   # tag is body-fixed when the sensor provides the heading
    with open(out_csv, "w") as f:
        for key, value in attrs.items():
            f.write(f"# {key}: {value}\n")
        robot_df.to_csv(f, index=False)

    log(f"\nWrote {len(robot_df)} frames x {len(robot_df.columns)} columns to {out_csv}")
    log("Done.")
    log.close()


if __name__ == "__main__":
    main()
