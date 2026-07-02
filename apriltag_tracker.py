#!/usr/bin/env python3
"""
AprilTag tracker.

Loads a video, detects AprilTags on every frame with pupil_apriltags, estimates
each tag's pose via solvePnP (using camera intrinsics from camera_calibration.py),
and writes a raw per-tag/per-frame CSV in the CAMERA frame. Reformatting into the
analysis-ready wide table (interpolation, lab frame, body angle) is done separately
by format_tracks.py.

The per-tag ``angle`` is the caster's in-plane rotation, computed from the pose
rotation matrix as ``atan2(R[1,0], R[0,0])`` -- NOT ``atan2(rvec[1], rvec[0])``,
which is the direction of the Rodrigues axis and is wrong.

Output CSV has a commented metadata header (``# key: value``) readable by the
notebook's ``read_csv_comments``; format_tracks.py uses it to recover fps / total_frames.

Example
-------
    python apriltag_tracker.py ../Data/020525/020525_clipped_2.mp4 \
        --calib ../Data/temp_room_calibration/calibration.npz --output-video
"""

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from pupil_apriltags import Detector

# Fallback intrinsics (air-table calibration) used only when --calib is omitted.
DEFAULT_CAMERA_MATRIX = np.array([
    [3.21321589e+03, 0.00000000e+00, 1.49873167e+03],
    [0.00000000e+00, 3.01845806e+03, 1.01978088e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
])
DEFAULT_DIST_COEFFS = np.array(
    [[-2.96506920e-01, 3.89225183e-01, 1.04457586e-03, -5.88244099e-02, -1.14800693e+00]])


def parse_args():
    p = argparse.ArgumentParser(description="AprilTag tracker (raw pose output)")
    p.add_argument("video_path", type=str, help="Path to the input video file")
    p.add_argument("--calib", type=str, default=None,
                   help="Path to calibration .npz from camera_calibration.py. "
                        "If omitted, built-in defaults are used (with a warning).")
    p.add_argument("--families", type=str, default="tag16h5", help="AprilTag family. Default: tag16h5")
    p.add_argument("--tag-size", type=float, default=0.045, help="Node tag edge length (m). Default: 0.045")
    p.add_argument("--corner-tag-size", type=float, default=0.037,
                   help="Corner (environment) tag edge length (m). Default: 0.037")
    p.add_argument("--corner-ids", type=int, nargs="+", default=[26, 27, 28, 29],
                   help="Tag ids that are environment corner tags. Default: 26 27 28 29")
    p.add_argument("--valid-tags", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 27, 28, 29],
                   help="Tag ids to keep. Default: 0..13 + 26..29")
    p.add_argument("--nthreads", type=int, default=4, help="Detector threads. Default: 4")
    p.add_argument("--quad-decimate", type=float, default=1.0, help="quad_decimate. Default: 1.0")
    p.add_argument("--quad-sigma", type=float, default=0.0, help="quad_sigma. Default: 0.0")
    p.add_argument("--subpix", action="store_true",
                   help="Refine tag corners with cv2.cornerSubPix before solvePnP.")
    p.add_argument("--hamming-max", type=int, default=0,
                   help="Max allowed hamming distance (kept if hamming <= this). Default: 0")
    p.add_argument("--decision-margin-min", type=float, default=1.0,
                   help="Minimum decision_margin. Default: 1.0")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: <video>_raw.csv")
    p.add_argument("--output-video", action="store_true",
                   help="Also write an undistorted, tag-annotated video.")
    return p.parse_args()


def load_calibration(path):
    if path is None:
        print("WARNING: no --calib given; using built-in default intrinsics.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS, "builtin_default"
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"], os.path.abspath(path)


def object_points(size):
    """Square tag corners (m) in the tag frame, ordered to match pupil_apriltags corners
    (top-left, top-right, bottom-right, bottom-left)."""
    h = size / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float32)


def caster_angle(rvec):
    """In-plane caster rotation from the pose rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def main():
    args = parse_args()
    start_time = time.time()

    if not os.path.exists(args.video_path):
        raise SystemExit(f"Error: input video not found at {args.video_path}")

    camera_matrix, dist_coeffs, calib_source = load_calibration(args.calib)
    core_path = os.path.splitext(args.video_path)[0]
    output_csv = args.output or (core_path + "_raw.csv")
    output_video = (core_path + "_tagged.mp4") if args.output_video else None

    corner_ids = set(args.corner_ids)
    valid_tags = set(args.valid_tags)
    obj_node = object_points(args.tag_size)
    obj_corner = object_points(args.corner_tag_size)

    detector = Detector(
        families=args.families,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=1,
        decode_sharpening=0.0,
        debug=0,
    )

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {total_frames} frames, {frame_width}x{frame_height}, {fps} FPS")

    out = None
    if output_video:
        out_dir = os.path.dirname(output_video)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
        print(f"Writing annotated video to: {os.path.abspath(output_video)}")

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    results = []
    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(image)

        for r in tags:
            if not (r.hamming <= args.hamming_max
                    and r.tag_id in valid_tags
                    and r.decision_margin > args.decision_margin_min):
                continue

            img_points = np.array(r.corners, dtype=np.float32)
            if args.subpix:
                refined = img_points.reshape(-1, 1, 2).copy()
                cv2.cornerSubPix(image, refined, (5, 5), (-1, -1), subpix_criteria)
                img_points = refined.reshape(-1, 2)

            obj_points = obj_corner if r.tag_id in corner_ids else obj_node
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not success:
                continue

            proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
            reproj_err = float(cv2.norm(img_points, proj.reshape(-1, 2), cv2.NORM_L2) / len(obj_points))

            results.append({
                "frame#": frame_count,
                "node_id": int(r.tag_id),
                "x": float(tvec[0][0]),
                "y": float(tvec[1][0]),
                "z": float(tvec[2][0]),
                "angle": caster_angle(rvec),
                "hamming": int(r.hamming),
                "decision_margin": float(r.decision_margin),
                "reproj_err": reproj_err,
            })

            if out is not None:
                pts = [tuple(map(int, c)) for c in r.corners]
                for i in range(4):
                    cv2.line(image, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
                center = tuple(map(int, r.center))
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID: {r.tag_id}", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if out is not None:
            h, w = image.shape[:2]
            new_cm, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
            undist = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_cm)
            flip = cv2.flip(undist, 0)
            out.write(cv2.cvtColor(flip, cv2.COLOR_GRAY2BGR))

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    if out is not None:
        out.release()
        print(f"Annotated video saved to: {os.path.abspath(output_video)}")

    df = pd.DataFrame(results, columns=[
        "frame#", "node_id", "x", "y", "z", "angle",
        "hamming", "decision_margin", "reproj_err"])

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "families": args.families,
        "tag_size": args.tag_size,
        "corner_tag_size": args.corner_tag_size,
        "corner_ids": list(args.corner_ids),
        "calib_source": calib_source,
    }
    with open(output_csv, "w") as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        df.to_csv(f, index=False)

    print(f"\nWrote {len(df)} detections to {output_csv}")
    print("Processing complete. --- %.2f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
