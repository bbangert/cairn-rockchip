#!/bin/sh
# Float baseline of the three MIT checkpoints on COCO val2017 (see baseline_mit.py). Long: run it
# yourself. Resumable -- a variant whose .json exists is skipped -- and self-logging.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/work/mit/venv/bin/python"
OUT="$HERE/work/mit/baseline"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

step() { # name mode resample
    stem=$("$PY" -B -I "$HERE/baseline_mit.py" "$1" --mode "$2" --resample "$3" --print-stem) || {
        log "FAILED $1 $2 $3 (stem)"
        exit 1
    }
    if [ -f "$OUT/$stem.json" ]; then
        log "skip $stem (done)"
        return 0
    fi
    log "start $stem"
    # -B: the MIT clone must stay clean of bytecode
    if "$PY" -B -I "$HERE/baseline_mit.py" "$1" --mode "$2" --resample "$3" --out-dir "$OUT" \
        --check-mit-decode 8 >>"$LOG" 2>&1; then
        log "end $stem"
    else
        log "FAILED $stem"
        exit 1
    fi
}

log "begin"
for m in v9-t v9-s; do
    step $m centre lanczos-pil
    step $m centre bilinear-cv2
    step $m origin bilinear-cv2
    step $m stretch bilinear-cv2
done
step v9-m centre lanczos-pil
step v9-m origin bilinear-cv2

if ! "$PY" -B -I - "$OUT" >>"$LOG" 2>&1 <<'PY'
import glob, json, os, sys
print(f"{'variant':<44} {'AP':>6} {'AP50':>6} {'AP75':>6}")
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    r = json.load(open(p))
    print(f"{os.path.basename(p)[:-5]:<44} {r['AP']:>6} {r['AP50']:>6} {r['AP75']:>6}")
PY
then
    log "FAILED summary"
    exit 1
fi
log "DONE"
