#!/usr/bin/env python3
"""
Format raw AprilTag tracks from a single-node robot into an analysis-ready table.

Takes the raw CSV from apriltag_tracker.py. Unlike format_tracks.py (multi-node, one
continuous run), this script:

  1. identifies the single robot node (the non-corner tag, or --node-id);
  2. splits the node's detections into separate trajectories wherever tracking is
     lost for long enough to be a hand-cover reset (not a brief dropout);
  3. interpolates only *inside* each trajectory (small dropouts);
  4. transforms positions into the LAB frame using the fixed corner tags (falls
     back to the camera frame if they are not detected);
  5. writes a wide CSV in the same column convention as format_tracks.py, plus a
     ``track`` column (1-indexed trajectory number). Hand-cover gaps are omitted.

Heading is the tag in-plane angle. For one node that is also the body orientation:
body_angle = lab heading, {id}_theta = lab heading, {id}_angle = 0.

Example
-------
    python format_tracks_single.py ../Data/180826/_2026-08-18_15_27_42_802_raw.csv
    python format_tracks_single.py ../Data/180826/foo_raw.csv --min-gap-seconds 0.5
"""

import argparse
import os

import numpy as np
import pandas as pd

from format_tracks import (
    Logger,
    apply_transform,
    compute_lab_transform,
    interpolate_tracks,
    pivot_series,
    read_csv_comments,
    wrap_angle,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Format single-node AprilTag tracks, splitting trajectories on long dropouts")
    p.add_argument("raw_csv", type=str, help="Raw CSV from apriltag_tracker.py")
    p.add_argument("--node-id", type=int, default=None,
                   help="Robot tag id. Default: the unique non-corner detected tag.")
    p.add_argument("--corner-ids", type=int, nargs="+", default=None,
                   help="Corner tag ids for the lab frame. Default: from raw-CSV header, else 26 27 28 29")
    p.add_argument("--arena-size", type=float, nargs=2, default=None, metavar=("W", "H"),
                   help="Real lab-rectangle dimensions. If omitted, observed corner spacing is used.")
    p.add_argument("--camera-frame", action="store_true",
                   help="Skip the corner-tag lab-frame transform and keep all positions in the "
                        "camera frame. (Corner tags still appear as extra tags.)")
    p.add_argument("--min-gap-seconds", type=float, default=0.5,
                   help="A dropout this long (or longer) starts a new trajectory. Default: 0.5 s. "
                        "Ignored if --min-gap-frames is given.")
    p.add_argument("--min-gap-frames", type=int, default=None,
                   help="Dropout length in frames that starts a new trajectory. Overrides "
                        "--min-gap-seconds.")
    p.add_argument("--min-track-seconds", type=float, default=0.25,
                   help="Drop trajectories shorter than this. Default: 0.25 s.")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps (else read from the raw-CSV header, else 30).")
    p.add_argument("--output", type=str, default=None, help="Output CSV. Default: <raw>_robot.csv")
    p.add_argument("--log", type=str, default=None, help="Log file. Default: <raw>_robot.log")
    return p.parse_args()


def split_tracks(frames, min_gap_frames):
    """Split a sorted 1-D array of detected frames into trajectories.

    A new track starts when the number of missing frames between consecutive
    detections is >= min_gap_frames.
    """
    frames = np.asarray(sorted(set(int(f) for f in frames)), dtype=int)
    if frames.size == 0:
        return []
    if frames.size == 1:
        return [frames]
    missing = np.diff(frames) - 1
    breaks = np.where(missing >= min_gap_frames)[0]
    segments = []
    start = 0
    for b in breaks:
        segments.append(frames[start:b + 1])
        start = b + 1
    segments.append(frames[start:])
    return segments


def interpolate_track(node_df, frames, nid):
    """Linear-fill x, y, z and (unwrapped) angle over a contiguous frame grid."""
    grid = pd.DataFrame({"frame#": frames, "node_id": nid})
    merged = pd.merge(grid, node_df, on=["node_id", "frame#"], how="left")
    for col in ("x", "y", "z"):
        merged[col] = merged[col].interpolate(method="linear", limit_direction="both")

    ang = merged["angle"].to_numpy(dtype=float)
    finite = np.isfinite(ang)
    unwrapped = np.full(len(ang), np.nan)
    if finite.any():
        idx = np.arange(len(ang))
        unwrapped[finite] = np.unwrap(ang[finite])
        if finite.all():
            filled = unwrapped
        else:
            filled = np.interp(idx, idx[finite], unwrapped[finite])
        merged["angle_unwrapped"] = filled
        merged["angle"] = wrap_angle(filled)
    else:
        merged["angle_unwrapped"] = unwrapped
    return merged


def main():
    args = parse_args()
    core = args.raw_csv[:-8] if args.raw_csv.endswith("_raw.csv") else os.path.splitext(args.raw_csv)[0]
    out_csv = args.output or (core + "_robot.csv")
    log = Logger(args.log or (core + "_robot.log"))

    raw = read_csv_comments(args.raw_csv)
    meta = raw.attrs
    fps = float(args.fps or meta.get("fps", 30))
    total_frames = int(meta.get("total_frames", raw["frame#"].max() + 1))
    corner_ids = list(args.corner_ids or meta.get("corner_ids", [26, 27, 28, 29]))
    corner_set = set(corner_ids)

    detected = sorted(int(n) for n in raw["node_id"].unique())
    non_corner = [n for n in detected if n not in corner_set]
    if args.node_id is not None:
        nid = int(args.node_id)
    elif len(non_corner) == 1:
        nid = non_corner[0]
    elif len(non_corner) == 0:
        raise SystemExit("No non-corner tag detections; pass --node-id explicitly.")
    else:
        raise SystemExit(
            f"Multiple non-corner tags detected {non_corner}; pass --node-id to choose one.")

    extra_tags = sorted(d for d in detected if d != nid)
    nodes = [nid]
    min_gap = (args.min_gap_frames if args.min_gap_frames is not None
               else max(1, int(round(args.min_gap_seconds * fps))))
    min_track = max(1, int(round(args.min_track_seconds * fps)))

    log(f"Raw CSV     : {args.raw_csv}")
    log(f"Detections  : {len(raw)}")
    log(f"fps         : {fps}")
    log(f"total_frames: {total_frames}")
    log(f"node_id     : {nid}")
    log(f"corner_ids  : {corner_ids}")
    log(f"extra_tags  : {extra_tags}")
    log(f"min gap     : {min_gap} frames ({min_gap / fps:.3f} s)")
    log(f"min track   : {min_track} frames ({min_track / fps:.3f} s)")

    node_raw = raw[raw["node_id"] == nid].drop_duplicates(subset=["frame#"], keep="first")
    det_frames = node_raw["frame#"].to_numpy()
    n_det = int(pd.Series(det_frames).nunique())
    log("\n=== Per-node detection rate (pre-interpolation) ===")
    log(f"  node {nid:>3}: {n_det}/{total_frames} frames ({100.0 * n_det / total_frames:.1f}%)")

    # Gap report (before splitting), so the threshold can be tuned from the log.
    ordered = np.sort(np.unique(det_frames.astype(int)))
    if ordered.size >= 2:
        missing = np.diff(ordered) - 1
        nz = missing[missing > 0]
        log("\n=== Dropout gaps (node) ===")
        log(f"Number of dropouts      : {len(nz)}")
        log(f"Max gap (frames / s)    : {int(nz.max()) if len(nz) else 0} / "
            f"{(nz.max() / fps if len(nz) else 0):.3f}")
        log(f"Gaps >= split threshold : {int((nz >= min_gap).sum()) if len(nz) else 0}")
        if len(nz):
            uniq, cnt = np.unique(nz, return_counts=True)
            log("Gap histogram (missing frames: count) — only gaps that split tracks:")
            for g, c in zip(uniq, cnt):
                if g >= min_gap:
                    log(f"  {int(g):5d} ({g / fps:.3f} s): {int(c)}")

    segments = split_tracks(ordered, min_gap)
    kept, dropped = [], []
    for seg in segments:
        (kept if len(seg) >= min_track else dropped).append(seg)

    log("\n=== Trajectory split ===")
    log(f"Candidate tracks        : {len(segments)}")
    log(f"Kept (>= min track)     : {len(kept)}")
    log(f"Dropped (too short)     : {len(dropped)}")
    for i, seg in enumerate(kept, start=1):
        n_miss = int((seg[-1] - seg[0] + 1) - len(seg))
        log(f"  track {i:>3}: frames {int(seg[0])}–{int(seg[-1])}  "
            f"({len(seg)} detections, {n_miss} interpolated, "
            f"{(seg[-1] - seg[0] + 1) / fps:.2f} s)")
    for seg in dropped:
        log(f"  dropped : frames {int(seg[0])}–{int(seg[-1])} ({len(seg)} detections)")

    if not kept:
        raise SystemExit("No trajectories survived the gap/length filters.")

    # Corners / extra tags: interpolate over the whole video (they stay visible
    # during a hand-cover) so the lab-frame transform and extra_tag columns are defined.
    extra_raw = raw[raw["node_id"].isin(extra_tags)]
    extra_xw = extra_yw = None
    if len(extra_raw):
        extra_interp = interpolate_tracks(extra_raw, total_frames, log)
        extra_xw = pivot_series(extra_interp, total_frames, "x")
        extra_yw = pivot_series(extra_interp, total_frames, "y")

    # --- Lab frame -------------------------------------------------------- #
    if args.camera_frame:
        M, residual = None, None
        log("\nLab frame: DISABLED via --camera-frame; keeping the camera frame.")
    elif extra_xw is None:
        M, residual = None, None
        log("\nLab frame: no extra/corner tags detected; staying in the camera frame.")
    else:
        present_corners = [cid for cid in corner_ids if cid in extra_xw.columns]
        mean_corners = np.array([[np.nanmean(extra_xw[cid].values), np.nanmean(extra_yw[cid].values)]
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

    if extra_xw is not None:
        for col in extra_xw.columns:
            extra_xw[col], extra_yw[col] = apply_transform(M, extra_xw[col].values, extra_yw[col].values)

    corner_locs = [(float(np.nanmean(extra_xw[cid].values)), float(np.nanmean(extra_yw[cid].values)))
                   if extra_xw is not None and cid in extra_xw.columns else (np.nan, np.nan)
                   for cid in corner_ids]

    # --- Per-track interpolation + assembly -------------------------------- #
    rows = []
    n_interp = 0
    for track_i, seg in enumerate(kept, start=1):
        grid_frames = np.arange(int(seg[0]), int(seg[-1]) + 1, dtype=int)
        filled = interpolate_track(node_raw, grid_frames, nid)
        n_interp += int(len(grid_frames) - len(seg))

        x, y = apply_transform(M, filled["x"].to_numpy(dtype=float), filled["y"].to_numpy(dtype=float))
        z = filled["z"].to_numpy(dtype=float)
        heading_unw = filled["angle_unwrapped"].to_numpy(dtype=float) + rot
        heading = wrap_angle(heading_unw)
        incr = heading_unw - heading_unw[0]

        for k, t in enumerate(grid_frames):
            row = {
                "time": t / fps,
                "track": track_i,
                f"{nid}_x": x[k],
                f"{nid}_y": y[k],
                f"{nid}_z": z[k],
                f"{nid}_theta": heading[k],
                f"{nid}_angle": 0.0,
                "centroid_x": x[k],
                "centroid_y": y[k],
                "body_angle": heading[k],
                "body_angle_incremental": incr[k],
            }
            for e in extra_tags:
                row[f"extra_tag_{e}_x"] = (extra_xw.at[t, e]
                                           if extra_xw is not None and e in extra_xw.columns
                                           else np.nan)
                row[f"extra_tag_{e}_y"] = (extra_yw.at[t, e]
                                           if extra_yw is not None and e in extra_yw.columns
                                           else np.nan)
            rows.append(row)

    robot_df = pd.DataFrame(rows)
    # Stable column order matching format_tracks.py, with `track` after `time`.
    front = ["time", "track",
             f"{nid}_x", f"{nid}_y", f"{nid}_z", f"{nid}_theta", f"{nid}_angle",
             "centroid_x", "centroid_y", "body_angle", "body_angle_incremental"]
    extra_cols = [c for e in extra_tags for c in (f"extra_tag_{e}_x", f"extra_tag_{e}_y")]
    robot_df = robot_df[front + extra_cols]

    log("\n=== Interpolation statistics (within tracks only) ===")
    log(f"Original node detections: {n_det}")
    log(f"Output rows             : {len(robot_df)}")
    log(f"Interpolated (in-track) : {n_interp}")
    log(f"Tracks written          : {len(kept)}")

    attrs = {
        "n_nodes": 1,
        "nodes": nodes,
        "connections": [],
        "corner_ids": corner_ids,
        "corner_locs": corner_locs,
        "extra_tags": extra_tags,
        "baseline": None,
        "arena_size": list(args.arena_size) if args.arena_size else None,
        "frame": frame_label,
        "fps": fps,
        "heading_source": "tag",
        "topology": "single",
        "n_tracks": len(kept),
        "min_gap_frames": min_gap,
        "node_id": nid,
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
