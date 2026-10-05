#!/usr/bin/env bash
# Run the full analysis pipeline over every .mp4 in a data directory.
#
# Stages: apriltag_tracker -> format_tracks -> analyze_modes -> plot_analysis.
# Tracking (the slow stage) runs PARALLEL videos at a time, each with its own log
# (<video>_track.log); format/analyze/plot then run per video in order.
# Defaults: no output video, tag36h11, sub-pixel corner refinement, decodes with up to 2 bit
# errors accepted, the air-table fisheye calibration, 6-node ring
# connections, corner tags 583-586 kept for the lab-frame transform, lab-frame positions +
# caster angle, bending stiffness kappa=0.005, and a per-video sensor CSV (same base name,
# ".csv") used for the heading when present.  Override any of CALIB/FAMILIES/CONNECTIONS/
# CORNER_IDS/VALID_TAGS/PARALLEL/TRACK_THREADS/QUAD_DECIMATE/SUBPIX/HAMMING via the environment.
#
# Usage:
#   bash run_pipeline.sh [DATA_DIR]
#   FORCE=1 bash run_pipeline.sh            # re-run the (slow) tracker even if *_raw.csv exists
#   CALIB=/path/to/calibration.npz FAMILIES=tag36h11 bash run_pipeline.sh /path/to/data
#   STAGES=track bash run_pipeline.sh /path/to/data        # tracker only
#   STAGES="analyze plot" bash run_pipeline.sh /path/to/data   # re-run later stages only
#   PARALLEL=4 QUAD_DECIMATE=1.5 bash run_pipeline.sh /path/to/data   # faster tracking
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
# Tracking parallelism: one tracker process saturates at ~4 detector threads, so run several
# videos at once. PARALLEL x TRACK_THREADS ~ number of CPU cores is a good target.
PARALLEL="${PARALLEL:-3}"              # videos tracked simultaneously (1 = sequential)
TRACK_THREADS="${TRACK_THREADS:-4}"    # detector threads per tracker
QUAD_DECIMATE="${QUAD_DECIMATE:-1.0}"  # 1.5 ~25% faster with no detection loss (tested); 2.0 drops tags
SUBPIX="${SUBPIX:-1}"                  # 1 = refine tag corners with cv2.cornerSubPix before solvePnP
HAMMING="${HAMMING:-2}"                # max decode bit errors accepted (tag36h11 codes differ by >= 11 bits)

for s in $STAGES; do
    case "$s" in track|format|analyze|plot) ;;
        *) echo "ERROR: unknown stage '$s' (valid: track format analyze plot)"; exit 1 ;; esac
done
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
case "$HAMMING" in ''|*[!0-9]*) echo "ERROR: HAMMING must be a non-negative integer"; exit 1 ;; esac
case "$PARALLEL" in ''|*[!0-9]*|0) echo "ERROR: PARALLEL must be a positive integer"; exit 1 ;; esac

cd "$SCRIPTS"
shopt -s nullglob

echo "Data dir    : $DATA_DIR"
echo "Calib       : $CALIB"
echo "Families    : $FAMILIES"
echo "Connections : $CONNECTIONS"
echo "Corner ids  : $CORNER_IDS"
echo "Stages      : $STAGES"
echo "Tracking    : $PARALLEL at a time, $TRACK_THREADS threads each, quad_decimate $QUAD_DECIMATE,"\
     "subpix $SUBPIX, hamming <= $HAMMING"
track_extra=(--hamming-max "$HAMMING")
[ "$SUBPIX" = "1" ] && track_extra+=(--subpix)
if has_stage track; then
    [ -f "$CALIB" ] || { echo "ERROR: calibration file not found: $CALIB"; exit 1; }
fi

videos=()
for mp4 in "$DATA_DIR"/*.mp4; do
    case "$(basename "${mp4%.mp4}")" in *_tagged) continue ;; esac   # skip annotated QC videos
    videos+=("$mp4")
done
[ ${#videos[@]} -gt 0 ] || { echo "No .mp4 files in $DATA_DIR"; exit 0; }

# ---------------------------------------------------------------------------- #
# 1. Tracking, PARALLEL videos at a time (skip if *_raw.csv exists unless FORCE=1).
# ---------------------------------------------------------------------------- #
if has_stage track; then
    fail_list="$(mktemp -t track_failures)"
    track_one() {
        local mp4="$1" base="${1%.mp4}" name
        name="$(basename "$base")"
        if $PY apriltag_tracker.py "$mp4" --calib "$CALIB" --families "$FAMILIES" \
                --nthreads "$TRACK_THREADS" --quad-decimate "$QUAD_DECIMATE" "${track_extra[@]}" \
                --corner-ids $CORNER_IDS --valid-tags $VALID_TAGS > "${base}_track.log" 2>&1; then
            echo "[track] done    $name"
        else
            echo "[track] FAILED  $name  (see $(basename "${base}_track.log"))"
            echo "$name" >> "$fail_list"
        fi
    }
    echo
    echo "================ tracking ${#videos[@]} video(s) ================"
    for mp4 in "${videos[@]}"; do
        name="$(basename "${mp4%.mp4}")"
        if [ "$FORCE" != "1" ] && [ -f "${mp4%.mp4}_raw.csv" ]; then
            echo "[track] skip    $name (found _raw.csv; set FORCE=1 to redo)"
            continue
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 1; done
        echo "[track] start   $name"
        track_one "$mp4" &
    done
    wait
    if [ -s "$fail_list" ]; then
        echo "Tracking failed for: $(tr '\n' ' ' < "$fail_list")"
    fi
    rm -f "$fail_list"
fi

# ---------------------------------------------------------------------------- #
# 2-4. Format, analyze, plot (per video, in order).
# ---------------------------------------------------------------------------- #
for mp4 in "${videos[@]}"; do
    base="${mp4%.mp4}"
    name="$(basename "$base")"
    raw="${base}_raw.csv"
    robot="${base}_robot.csv"
    npz="${base}_analysis.npz"
    sensor="${base}.csv"

    has_stage format || has_stage analyze || has_stage plot || break
    echo
    echo "================ $name ================"

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
    #    Add --k / --baseline for a specific spring model (connections come from the CSV).
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
