#!/usr/bin/env bash
# Run the full analysis pipeline over every .mp4 in a data directory.
#
# Per-video stages: apriltag_tracker -> format_tracks -> analyze_modes -> plot_analysis.
# Defaults: no output video, tag36h11, the air-table fisheye calibration, 6-node ring
# connections, corner tags 583-586 kept for the lab-frame transform, lab-frame positions +
# caster angle, bending stiffness kappa=0.005, and a per-video sensor CSV (same base name,
# ".csv") used for the heading when present.  Override any of CALIB/FAMILIES/CONNECTIONS/
# CORNER_IDS/VALID_TAGS via the environment.
#
# Usage:
#   bash run_pipeline.sh [DATA_DIR]
#   FORCE=1 bash run_pipeline.sh            # re-run the (slow) tracker even if *_raw.csv exists
#   CALIB=/path/to/calibration.npz FAMILIES=tag36h11 bash run_pipeline.sh /path/to/data
#   STAGES=track bash run_pipeline.sh /path/to/data        # tracker only
#   STAGES="analyze plot" bash run_pipeline.sh /path/to/data   # re-run later stages only
set -u

PY=/opt/miniconda3/envs/tplax_env/bin/python
SCRIPTS=/Users/alexleffell/Documents/PhD/tplax/tplax_robot_scripts
DATA_DIR="${1:-/Users/alexleffell/Documents/PhD/tplax/Data/010326}"
CALIB="${CALIB:-/Users/alexleffell/Documents/PhD/tplax/Data/070726/070726_airtable_calibration/calibration.npz}"
FAMILIES="${FAMILIES:-tag36h11}"
CONNECTIONS="${CONNECTIONS:-[(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]}"   # default: 6-node ring
CORNER_IDS="${CORNER_IDS:-583 584 585 586}"                           # environment/arena corner tags
# Tags the tracker keeps (node ids + corner ids). Corners MUST be listed or they get filtered
# out and the lab-frame transform can't run.
VALID_TAGS="${VALID_TAGS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 ${CORNER_IDS}}"
FORCE="${FORCE:-0}"        # 1 = re-run tracker even if *_raw.csv already exists
STAGES="${STAGES:-track format analyze plot}"   # space-separated subset of the four stages

for s in $STAGES; do
    case "$s" in track|format|analyze|plot) ;;
        *) echo "ERROR: unknown stage '$s' (valid: track format analyze plot)"; exit 1 ;; esac
done
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

cd "$SCRIPTS"
shopt -s nullglob

echo "Data dir    : $DATA_DIR"
echo "Calib       : $CALIB"
echo "Families    : $FAMILIES"
echo "Connections : $CONNECTIONS"
echo "Corner ids  : $CORNER_IDS"
echo "Stages      : $STAGES"
if has_stage track; then
    [ -f "$CALIB" ] || { echo "ERROR: calibration file not found: $CALIB"; exit 1; }
fi

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
    if has_stage track; then
        if [ "$FORCE" = "1" ] || [ ! -f "$raw" ]; then
            echo "[track] $(basename "$mp4")"
            $PY apriltag_tracker.py "$mp4" --calib "$CALIB" --families "$FAMILIES" \
                --corner-ids $CORNER_IDS --valid-tags $VALID_TAGS \
                || { echo "  tracker FAILED for $name"; continue; }
        else
            echo "[track] skip (found $(basename "$raw"); set FORCE=1 to redo)"
        fi
    fi

    # 2. Format: lab frame (corner-tag transform, camera fallback); given connections;
    #    use the matching sensor CSV for the heading if present.
    if has_stage format; then
        [ -f "$raw" ] || { echo "  format: missing $(basename "$raw"); run the track stage first"; continue; }
        fmt=(--connections "$CONNECTIONS")
        if [ -f "$sensor" ]; then
            fmt+=(--sensor-csv "$sensor")
        else
            echo "  WARNING: no sensor CSV ($(basename "$sensor")); using tag heading"
        fi
        echo "[format] ${fmt[*]}"
        $PY format_tracks.py "$raw" "${fmt[@]}" || { echo "  format FAILED"; continue; }
    fi

    # 3. Analyze: lab frame, with bending stiffness kappa=0.005.
    #    --vel-smooth-window suppresses finite-difference velocity noise (tune per fps).
    #    Add --k / --l0 / --baseline / --connections for a specific spring model / topology.
    if has_stage analyze; then
        [ -f "$robot" ] || { echo "  analyze: missing $(basename "$robot"); run the format stage first"; continue; }
        echo "[analyze] --angle-frame lab --kappa 0.005 --vel-smooth-window 7"
        $PY analyze_modes.py "$robot" --angle-frame lab --kappa 0.005 --vel-smooth-window 7 \
            || { echo "  analyze FAILED"; continue; }
    fi

    # 4. Plot.
    if has_stage plot; then
        [ -f "$npz" ] || { echo "  plot: missing $(basename "$npz"); run the analyze stage first"; continue; }
        echo "[plot]"
        $PY plot_analysis.py "$npz" || { echo "  plot FAILED"; continue; }
    fi
done

echo
echo "All done."
