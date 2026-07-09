#!/usr/bin/env python3
"""
Camera calibration from checkerboard images.

Loads every image in a directory, detects the checkerboard corners, runs OpenCV
camera calibration, and saves the camera matrix + distortion coefficients (and a
few QC artifacts) so the AprilTag tracker can consume them.

Outputs
-------
- <output>.npz : camera_matrix, dist_coeffs, image_size, rms, pattern, square_size
- <output>.json: same values, human-readable
- <overlay-dir>/*: input images with the detected corners drawn, for verification

Example
-------
    python camera_calibration.py ../Data/temp_room_calibration/ \
        --pattern 7 10 --glob '*.bmp' --square-size 1.0
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Checkerboard camera calibration")
    p.add_argument("image_dir", type=str, help="Directory containing calibration images")
    p.add_argument("--pattern", type=int, nargs=2, default=[7, 10],
                   metavar=("COLS", "ROWS"),
                   help="Number of INNER corners (cols rows). Default: 7 10")
    p.add_argument("--square-size", type=float, default=1.0,
                   help="Physical checkerboard square edge length. Only scales the "
                        "extrinsics; the intrinsics used downstream are unaffected. Default: 1.0")
    p.add_argument("--glob", type=str, default="*.bmp",
                   help="Filename glob within image_dir. Default: '*.bmp'")
    p.add_argument("--output", type=str, default=None,
                   help="Output .npz path. Default: <image_dir>/calibration.npz")
    p.add_argument("--overlay-dir", type=str, default=None,
                   help="Directory for corner-overlay images. Default: <image_dir>/overlays/")
    p.add_argument("--pinhole", action="store_true",
                   help="Calibrate with the standard PINHOLE model. Default is the FISHEYE model "
                        "(this project's wide-angle lens); use --pinhole only for a rectilinear lens.")
    p.add_argument("--fix-aspect-ratio", action="store_true",
                   help="(pinhole only) Force fx == fy (square pixels). Better-conditioned when the "
                        "board views don't cleanly constrain the aspect ratio.")
    p.add_argument("--simple-distortion", action="store_true",
                   help="(pinhole only) Use the standard 5-coefficient distortion model instead of "
                        "the rational 8-coefficient model, which can overfit modest image sets.")
    p.add_argument("--error-threshold", type=float, default=1.0,
                   help="Per-image reprojection error (px) above which a warning is logged. Default: 1.0")
    return p.parse_args()


def find_corners(gray, pattern):
    """Return (found, corners) using findChessboardCornersSB with a classic fallback."""
    # SB (sector-based) returns sub-pixel corners directly and is more robust.
    flags_sb = (cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_EXHAUSTIVE
                + cv2.CALIB_CB_ACCURACY)
    ret, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags_sb)
    if ret:
        return True, corners

    # Fallback: classic detector + sub-pixel refinement.
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_FAST_CHECK
             + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners
    return False, None


def main():
    args = parse_args()
    pattern = tuple(args.pattern)  # (cols, rows) inner corners

    output = args.output or os.path.join(args.image_dir, "calibration.npz")
    overlay_dir = args.overlay_dir or os.path.join(args.image_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    glob_pattern = os.path.join(args.image_dir, args.glob)
    images = sorted(glob.glob(glob_pattern))
    if not images:
        raise SystemExit(f"No images matched {glob_pattern!r}")
    print(f"Found {len(images)} images matching {glob_pattern!r}")

    # World coordinates of the checkerboard corners (z=0), scaled by square size.
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []   # 3D points in world space
    imgpoints = []   # 2D points in image plane
    used_images = []
    failed_images = []
    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  WARNING: could not read {fname}")
            failed_images.append(fname)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (w, h)
        elif gray.shape[::-1] != image_size:
            print(f"  WARNING: {os.path.basename(fname)} size {gray.shape[::-1]} "
                  f"!= {image_size}; skipping")
            failed_images.append(fname)
            continue

        found, corners = find_corners(gray, pattern)
        if not found:
            print(f"  WARNING: checkerboard NOT found in {os.path.basename(fname)}")
            failed_images.append(fname)
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_images.append(fname)

        overlay = img.copy()
        cv2.drawChessboardCorners(overlay, pattern, corners, found)
        out_name = os.path.join(overlay_dir, os.path.basename(fname))
        cv2.imwrite(out_name, overlay)

    if len(objpoints) < 3:
        raise SystemExit(f"Only {len(objpoints)} usable images; need at least 3.")

    print(f"\nCalibrating from {len(objpoints)} images "
          f"({len(failed_images)} skipped)...")

    model = "pinhole" if args.pinhole else "fisheye"
    print("Calibration model:", model)
    per_image_errors = []
    if model == "fisheye":
        # Wide-angle/fisheye lenses need the fisheye projection model; a pinhole/rational
        # fit cannot capture the barrel distortion (inflated RMS, unstable focal length).
        n_ok = len(objpoints)
        camera_matrix = np.zeros((3, 3))
        dist_coeffs = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3)) for _ in range(n_ok)]
        tvecs = [np.zeros((1, 1, 3)) for _ in range(n_ok)]
        fe_obj = [op.reshape(1, -1, 3) for op in objpoints]
        fe_img = [ip.reshape(1, -1, 2) for ip in imgpoints]
        fe_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        try:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                fe_obj, fe_img, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, fe_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6))
        except cv2.error as e:
            raise SystemExit(f"Fisheye calibration failed ({str(e)[:120]}). Try pruning blurry / "
                             "extreme-angle frames, then re-run.")
    else:
        flags = 0 if args.simple_distortion else (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K6)
        print("Distortion model:", "standard 5-coeff" if args.simple_distortion else "rational 8-coeff")
        if args.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO   # force fx == fy (square pixels)
            print("Fixing aspect ratio (fx == fy).")
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None, flags=flags)
        # Per-image RMS reprojection error (per point, in px): norm / sqrt(n_points).
        for i in range(len(used_images)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            per_image_errors.append(cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj)))

    print(f"\nOverall RMS reprojection error: {rms:.4f} px")
    if per_image_errors:
        print("Per-image RMS reprojection error:")
        for fname, err in zip(used_images, per_image_errors):
            flag = "  <-- high" if err > args.error_threshold else ""
            print(f"  {os.path.basename(fname):40s} {err:.4f} px{flag}")
    print("\nCamera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", np.ravel(dist_coeffs))

    # Guardrails. Aspect check only for pinhole (fisheye fx~fy by construction).
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    if model == "pinhole" and abs(fx / fy - 1.0) > 0.10:
        print("\n" + "!" * 70)
        print(f"WARNING: fx/fy = {fx / fy:.2f} (fx={fx:.0f}, fy={fy:.0f}) is far from 1.0.")
        print("Real cameras have ~square pixels. This is a DEGENERATE calibration (coplanar board")
        print("views), OR the lens is wide-angle/fisheye -- if so, re-run with --fisheye.")
        print("!" * 70)
    if float(rms) > 1.0:
        hint = ("Prune high-error frames; ensure the board is sharp and fills more of the frame."
                if model == "fisheye" else
                "Prune high-error frames (or, for a wide/fisheye lens, drop --pinhole).")
        print(f"\nWARNING: overall RMS reprojection error {float(rms):.2f} px is high (want < ~0.5 px). "
              + hint)

    # Save results.
    camera_matrix = np.asarray(camera_matrix)
    dist_coeffs = np.asarray(dist_coeffs)
    np.savez(
        output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms=np.array(float(rms)),
        pattern=np.array(pattern),
        square_size=np.array(args.square_size),
        model=model,
    )

    json_path = os.path.splitext(output)[0] + ".json"
    json_blob = {
        "model": model,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(image_size),
        "rms": float(rms),
        "pattern": list(pattern),
        "square_size": args.square_size,
        "n_images_used": len(objpoints),
        "n_images_failed": len(failed_images),
        "per_image_errors": [float(e) for e in per_image_errors],
        "failed_images": [os.path.basename(f) for f in failed_images],
    }
    with open(json_path, "w") as f:
        json.dump(json_blob, f, indent=2)

    print(f"\nSaved calibration to {output}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved corner overlays to {overlay_dir}/")
    if failed_images:
        print(f"\nWARNING: {len(failed_images)} image(s) skipped (no board / unreadable):")
        for f in failed_images:
            print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
