#!/usr/bin/env bash
# Run the single-node pipeline over every .mp4 in a data directory.
#
# Per-video stages: apriltag_tracker -> format_tracks_single.
# Defaults match the single-node / fisheye setup:
#   tag36h11, hamming-max 2, valid-tags 0, corner-tag-size 0.045,
#   calibration_fisheye.npz. No output video. Override any of
#   CALIB/FAMILIES/HAMMING/VALID_TAGS/CORNER_TAG_SIZE via the environment.
#
# Usage:
#   bash run_pipeline_single.sh [DATA_DIR]
#   FORCE=1 bash run_pipeline_single.sh            # re-run the tracker even if *_raw.csv exists
#   CALIB=/path/to/calibration.npz bash run_pipeline_single.sh /path/to/data
#   STAGES=track bash run_pipeline_single.sh /path/to/data    # tracker only
#   STAGES=format bash run_pipeline_single.sh /path/to/data   # re-run formatting only
set -u

PY=/opt/miniconda3/envs/tplax_env/bin/python
SCRIPTS=/Users/alexleffell/Documents/PhD/tplax/tplax_robot_scripts
DATA_DIR="${1:-/Users/alexleffell/Documents/PhD/tplax/Data/200826/m150_l1_d3}"
CALIB="${CALIB:-/Users/alexleffell/Documents/PhD/tplax/Data/single_node_calib/calibration_fisheye.npz}"
FAMILIES="${FAMILIES:-tag36h11}"
HAMMING="${HAMMING:-2}"
VALID_TAGS="${VALID_TAGS:-0}"
CORNER_TAG_SIZE="${CORNER_TAG_SIZE:-0.045}"
FORCE="${FORCE:-0}"        # 1 = re-run tracker even if *_raw.csv already exists
STAGES="${STAGES:-track format}"   # space-separated subset of the two stages

for s in $STAGES; do
    case "$s" in track|format) ;;
        *) echo "ERROR: unknown stage '$s' (valid: track format)"; exit 1 ;; esac
done
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

cd "$SCRIPTS"
shopt -s nullglob

echo "Data dir         : $DATA_DIR"
echo "Calib            : $CALIB"
echo "Families         : $FAMILIES"
echo "Hamming max      : $HAMMING"
echo "Valid tags       : $VALID_TAGS"
echo "Corner tag size  : $CORNER_TAG_SIZE"
echo "Stages           : $STAGES"
[ -d "$DATA_DIR" ] || { echo "ERROR: data directory not found: $DATA_DIR"; exit 1; }
if has_stage track; then
    [ -f "$CALIB" ] || { echo "ERROR: calibration file not found: $CALIB"; exit 1; }
fi

n_vid=0
for mp4 in "$DATA_DIR"/*.mp4; do
    base="${mp4%.mp4}"
    name="$(basename "$base")"
    case "$name" in *_tagged) continue ;; esac   # skip annotated QC videos

    raw="${base}_raw.csv"
    n_vid=$((n_vid + 1))

    echo
    echo "================ $name ================"

    # 1. Tracker (expensive): skip if *_raw.csv exists unless FORCE=1.
    if has_stage track; then
        if [ "$FORCE" = "1" ] || [ ! -f "$raw" ]; then
            echo "[track] $(basename "$mp4")"
            $PY apriltag_tracker.py "$mp4" --calib "$CALIB" --families "$FAMILIES" \
                --hamming-max "$HAMMING" --valid-tags $VALID_TAGS \
                --corner-tag-size "$CORNER_TAG_SIZE" \
                || { echo "  tracker FAILED for $name"; continue; }
        else
            echo "[track] skip (found $(basename "$raw"); set FORCE=1 to redo)"
        fi
    fi

    # 2. Format: split on hand-cover gaps; lab frame if corners are in the raw CSV,
    #    otherwise camera-frame fallback.
    if has_stage format; then
        if [ ! -f "$raw" ]; then
            echo "  format SKIPPED (no $(basename "$raw"); run the track stage first)"
            continue
        fi
        echo "[format] format_tracks_single.py $(basename "$raw")"
        $PY format_tracks_single.py "$raw" || { echo "  format FAILED"; continue; }
    fi
done

if [ "$n_vid" -eq 0 ]; then
    echo "ERROR: no .mp4 files in $DATA_DIR"
    exit 1
fi

echo
echo "All done."
