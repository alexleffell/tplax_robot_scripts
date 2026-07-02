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


def build_template(nodes, connections, baseline, xw, yw, log):
    """Regular-polygon template: hub at origin, ring nodes on a circle in ascending id order.

    Returns a dict {node_id: (x, y)} or None if a regular template can't be inferred
    (caller then falls back to the mean shape).
    """
    degree = {n: 0 for n in nodes}
    for a, b in connections:
        degree[a] += 1
        degree[b] += 1
    hub = min((n for n in nodes), key=lambda n: (-degree[n], n))
    ring = [n for n in nodes if n != hub]

    # Star topology check: hub connects to every ring node.
    conn_set = {frozenset(c) for c in connections}
    is_star = all(frozenset((hub, n)) in conn_set for n in ring)
    if not is_star:
        log(f"WARNING: connections are not hub-and-spoke (hub={hub}); "
            f"falling back to a time-averaged mean-shape template for body_angle.")
        return None

    r = baseline
    if r is None:
        # Mean hub->ring distance over all frames.
        dists = []
        for n in ring:
            dx = xw[n].values - xw[hub].values
            dy = yw[n].values - yw[hub].values
            dists.append(np.nanmean(np.sqrt(dx ** 2 + dy ** 2)))
        r = float(np.nanmean(dists)) if dists else 1.0
        log(f"Derived baseline (mean hub->ring distance): {r:.4f}")

    template = {hub: (0.0, 0.0)}
    m = len(ring)
    for k, n in enumerate(ring):
        theta = 2.0 * np.pi * k / m
        template[n] = (r * np.cos(theta), r * np.sin(theta))
    log(f"Regular-polygon template: hub={hub}, ring={ring}, radius={r:.4f} "
        f"(radius does not affect the fitted angle).")
    return template


def mean_shape_template(nodes, xw, yw):
    return {n: (float(np.nanmean(xw[n].values)), float(np.nanmean(yw[n].values))) for n in nodes}


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


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
    p.add_argument("--baseline", type=float, default=None,
                   help="Node-to-node distance for the template. If omitted, derived from data. "
                        "(Only sets the template radius; does not affect the fitted body_angle.)")
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
    template = build_template(nodes, connections, args.baseline, xw, yw, log)
    if template is None:
        template = mean_shape_template(nodes, xw, yw)
    tmpl_nodes = [n for n in nodes if n in template]
    tmpl_pts = np.array([template[n] for n in tmpl_nodes])

    # --- Per-frame assembly ----------------------------------------------- #
    rows = []
    prev_pts = None
    prev_ids = None
    incr_accum = 0.0
    skipped_abs = 0
    skipped_incr = 0

    for t in range(total_frames):
        row = {"time": t / fps}

        # Node positions available this frame (post-interpolation, non-NaN).
        node_pos = {}
        for n in nodes:
            if n in xw.columns:
                x, y = xw.at[t, n], yw.at[t, n]
                if np.isfinite(x) and np.isfinite(y):
                    node_pos[n] = (x, y)

        centroid_x = np.mean([p[0] for p in node_pos.values()]) if node_pos else np.nan
        centroid_y = np.mean([p[1] for p in node_pos.values()]) if node_pos else np.nan

        # Absolute body angle: Kabsch(template -> observed) over shared nodes.
        shared = [n for n in tmpl_nodes if n in node_pos]
        if len(shared) >= 2:
            A = np.array([template[n] for n in shared])
            B = np.array([node_pos[n] for n in shared])
            body_angle = kabsch_angle(A, B)
        else:
            body_angle = np.nan
            skipped_abs += 1

        # Incremental body angle: Kabsch(prev -> curr) over shared nodes, integrated.
        cur_ids = sorted(node_pos.keys())
        if prev_pts is not None:
            common = [n for n in cur_ids if n in prev_ids]
            if len(common) >= 2:
                A = np.array([prev_pts[n] for n in common])
                B = np.array([node_pos[n] for n in common])
                incr_accum += kabsch_angle(A, B)
            else:
                skipped_incr += 1
        body_angle_incremental = incr_accum
        prev_pts, prev_ids = node_pos, set(cur_ids)

        # Per-node columns.
        for n in nodes:
            if n in xw.columns:
                row[f"{n}_x"] = xw.at[t, n]
                row[f"{n}_y"] = yw.at[t, n]
                row[f"{n}_z"] = zw.at[t, n] if n in zw.columns else np.nan
                theta = aw.at[t, n] if n in aw.columns else np.nan
                row[f"{n}_theta"] = theta
                row[f"{n}_angle"] = wrap_angle(theta - body_angle) if np.isfinite(body_angle) else np.nan
            else:
                row[f"{n}_x"] = row[f"{n}_y"] = row[f"{n}_z"] = np.nan
                row[f"{n}_theta"] = row[f"{n}_angle"] = np.nan

        row["centroid_x"] = centroid_x
        row["centroid_y"] = centroid_y
        row["body_angle"] = body_angle
        row["body_angle_incremental"] = body_angle_incremental

        for e in extra_tags:
            row[f"extra_tag_{e}_x"] = xw.at[t, e] if e in xw.columns else np.nan
            row[f"extra_tag_{e}_y"] = yw.at[t, e] if e in yw.columns else np.nan

        rows.append(row)

    robot_df = pd.DataFrame(rows)

    log("\n=== Body-angle fit ===")
    log(f"Frames skipped (absolute, <2 nodes)   : {skipped_abs}")
    log(f"Frames skipped (incremental, <2 shared): {skipped_incr}")

    # --- Write CSV with metadata header ----------------------------------- #
    attrs = {
        "n_nodes": len(nodes),
        "nodes": nodes,
        "connections": connections,
        "corner_ids": corner_ids,
        "corner_locs": corner_locs,
        "extra_tags": extra_tags,
        "baseline": args.baseline,
        "arena_size": list(args.arena_size) if args.arena_size else None,
        "frame": frame_label,
        "fps": fps,
    }
    with open(out_csv, "w") as f:
        for key, value in attrs.items():
            f.write(f"# {key}: {value}\n")
        robot_df.to_csv(f, index=False)

    log(f"\nWrote {len(robot_df)} frames x {len(robot_df.columns)} columns to {out_csv}")
    log("Done.")
    log.close()


if __name__ == "__main__":
    main()
