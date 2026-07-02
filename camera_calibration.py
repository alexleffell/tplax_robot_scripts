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
    p.add_argument("--fisheye", action="store_true",
                   help="Also run the fisheye calibration model and report its result.")
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

    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K6
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)

    # Per-image reprojection error.
    print(f"\nOverall RMS reprojection error: {rms:.4f} px")
    print("Per-image reprojection error:")
    per_image_errors = []
    for i, fname in enumerate(used_images):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_image_errors.append(err)
        flag = "  <-- high" if err > args.error_threshold else ""
        print(f"  {os.path.basename(fname):40s} {err:.4f} px{flag}")

    print("\nCamera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs.ravel())

    # Optional fisheye model.
    fisheye_result = None
    if args.fisheye:
        try:
            n_ok = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            fe_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
            fe_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
            # fisheye.calibrate wants object points shaped (1, N, 3).
            fe_obj = [op.reshape(1, -1, 3) for op in objpoints]
            fe_img = [ip.reshape(1, -1, 2) for ip in imgpoints]
            fe_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                        + cv2.fisheye.CALIB_CHECK_COND
                        + cv2.fisheye.CALIB_FIX_SKEW)
            fe_rms, _, _, _, _ = cv2.fisheye.calibrate(
                fe_obj, fe_img, image_size, K, D, fe_rvecs, fe_tvecs, fe_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            fisheye_result = {"rms": float(fe_rms), "K": K.tolist(), "D": D.tolist()}
            print(f"\nFisheye RMS: {fe_rms:.4f} px")
            print("Fisheye K:\n", K)
            print("Fisheye D:\n", D.ravel())
        except cv2.error as e:
            print(f"\nWARNING: fisheye calibration failed: {e}")

    # Save results.
    save_kwargs = dict(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms=np.array(rms),
        pattern=np.array(pattern),
        square_size=np.array(args.square_size),
    )
    if fisheye_result is not None:
        save_kwargs["fisheye_K"] = np.array(fisheye_result["K"])
        save_kwargs["fisheye_D"] = np.array(fisheye_result["D"])
    np.savez(output, **save_kwargs)

    json_path = os.path.splitext(output)[0] + ".json"
    json_blob = {
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
    if fisheye_result is not None:
        json_blob["fisheye"] = fisheye_result
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
