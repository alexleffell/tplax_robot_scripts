#!/usr/bin/env bash
# Run the full analysis pipeline over every .mp4 in a data directory.
#
# Per-video stages: apriltag_tracker -> format_tracks -> analyze_modes -> plot_analysis.
# Defaults match the 010326 batch: no output video, tag16h5, camera frame, and a
# per-video sensor CSV (same base name, ".csv") used for the heading.
#
# Usage:
#   bash run_pipeline.sh [DATA_DIR]
#   FORCE=1 bash run_pipeline.sh            # re-run the (slow) tracker even if *_raw.csv exists
#   CALIB=/path/to/calibration.npz FAMILIES=tag36h11 bash run_pipeline.sh /path/to/data
set -u

PY=/opt/miniconda3/envs/tplax_env/bin/python
SCRIPTS=/Users/alexleffell/Documents/PhD/tplax/tplax_robot_scripts
DATA_DIR="${1:-/Users/alexleffell/Documents/PhD/tplax/Data/010326}"
CALIB="${CALIB:-/Users/alexleffell/Documents/PhD/tplax/Data/temp_room_calibration/calibration.npz}"
FAMILIES="${FAMILIES:-tag16h5}"
FORCE="${FORCE:-0}"        # 1 = re-run tracker even if *_raw.csv already exists

cd "$SCRIPTS"
shopt -s nullglob

echo "Data dir : $DATA_DIR"
echo "Calib    : $CALIB"
echo "Families : $FAMILIES"
[ -f "$CALIB" ] || { echo "ERROR: calibration file not found: $CALIB"; exit 1; }

for mp4 in "$DATA_DIR"/*.mp4; do
    base="${mp4%.mp4}"
    name="$(basename "$base")"
    case "$name" in *_tagged) continue ;; esac   # skip annotated QC videos

    raw="${base}_raw.csv"
    robot="${base}_robot.csv"
    npz="${base}_analysis.npz"
    sensor="${base}.csv"

    echo
    echo "================ $name ================"

    # 1. Tracker (expensive): skip if *_raw.csv exists unless FORCE=1.  Settings: no output video.
    if [ "$FORCE" = "1" ] || [ ! -f "$raw" ]; then
        echo "[track] $(basename "$mp4")"
        $PY apriltag_tracker.py "$mp4" --calib "$CALIB" --families "$FAMILIES" \
            || { echo "  tracker FAILED for $name"; continue; }
    else
        echo "[track] skip (found $(basename "$raw"); set FORCE=1 to redo)"
    fi

    # 2. Format: camera frame; use the matching sensor CSV for the heading if present.
    fmt=(--camera-frame)
    if [ -f "$sensor" ]; then
        fmt+=(--sensor-csv "$sensor")
    else
        echo "  WARNING: no sensor CSV ($(basename "$sensor")); using tag heading"
    fi
    echo "[format] ${fmt[*]}"
    $PY format_tracks.py "$raw" "${fmt[@]}" || { echo "  format FAILED"; continue; }

    # 3. Analyze: body frame (heading comes from the sensor when present).
    #    --vel-smooth-window suppresses finite-difference velocity noise (tune per fps;
    #    7 frames ~ 0.24 s at 29 fps). Add --k / --l0 / --baseline for a specific spring model.
    echo "[analyze] --angle-frame body --vel-smooth-window 7"
    $PY analyze_modes.py "$robot" --angle-frame body --vel-smooth-window 7 \
        || { echo "  analyze FAILED"; continue; }

    # 4. Plot.
    echo "[plot]"
    $PY plot_analysis.py "$npz" || { echo "  plot FAILED"; continue; }
done

echo
echo "All done."
