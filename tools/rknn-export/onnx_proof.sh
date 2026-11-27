#!/bin/sh
# Step-1 proof of the MIT YOLOv9 ONNX exports (export_mit_yolov9.py, onnx_proof.py): the RepConv fusion check,
# export, float baselines (8-bit-snapped, and unsnapped as the AP reference), then onnxruntime vs PyTorch on
# full COCO val2017, each result gated to within 0.05 AP of its PyTorch baseline. Every exported model has a
# proof. Long: run it yourself. Resumable -- an export whose .onnx.json exists is skipped, a baseline whose .json
# exists is skipped, a proof is skipped while its result carries the model's current onnx_sha256 (and still passes
# the AP gate) -- and self-logging; the last line of run.log is DONE or FAILED <step> (FAILED signal after
# INT/TERM/HUP; a SIGKILL leaves no final line).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/work/mit/venv/bin/python"
OUT="$HERE/work/mit/onnx"
BASE="$HERE/work/mit/baseline"
LOG="$OUT/run.log"
mkdir -p "$OUT" "$BASE"
export PYTHONDONTWRITEBYTECODE=1  # the MIT clone must stay clean

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }
trap 'log "FAILED signal"; exit 1' INT TERM HUP

stem_of() { # script args... -> basename of the result the script would write
    "$PY" -B -I "$@" --print-stem 2>>"$LOG"
}

run() { # step result-file command...
    step="$1"; done_file="$2"; shift 2
    if [ -f "$done_file" ]; then
        log "skip $step (done)"
        return 0
    fi
    log "start $step"
    if "$@" >>"$LOG" 2>&1; then
        log "end $step"
    else
        log "FAILED $step"
        exit 1
    fi
}

fuse_check() { # name; always run: it is cheap and tests code that can change under an existing export
    log "start fuse check $1"
    if "$PY" -B -I "$HERE/fuse_repconv.py" "$1" >>"$LOG" 2>&1; then
        log "end fuse check $1"
    else
        log "FAILED fuse check $1"
        exit 1
    fi
}

export_model() { # name tag flags...
    name="$1"; tag="$2"; shift 2
    run "export $name$tag" "$OUT/$name$tag.onnx.json" \
        "$PY" -B -I "$HERE/export_mit_yolov9.py" "$name" --out "$OUT/$name$tag.onnx" "$@"
}

baseline() { # name flags...  (the stem is asked of baseline_mit.py, so the skip test cannot drift from its output)
    name="$1"; shift
    stem=$(stem_of "$HERE/baseline_mit.py" "$name" --mode origin --resample bilinear-cv2 "$@") || {
        log "FAILED baseline $name $* (stem)"
        exit 1
    }
    run "baseline $stem" "$BASE/$stem.json" \
        "$PY" -B -I "$HERE/baseline_mit.py" "$name" --mode origin --resample bilinear-cv2 "$@" --out-dir "$BASE"
}

proof() { # name tag [size]
    name="$1"; tag="$2"; size="${3:-640}"
    model="$OUT/$name$tag.onnx"
    common="--mode origin --resample bilinear-cv2"
    bstem=$(stem_of "$HERE/baseline_mit.py" "$name" $common --size "$size") || {
        log "FAILED onnx $name$tag (baseline stem)"
        exit 1
    }
    # one invocation shape for the currency check and the run; --is-current: 0 skip, 1 run, else refused or gate failed
    set -- "$name" "$model" $common --compare 20 --out-dir "$OUT" --baseline "$BASE/$bstem.json"
    "$PY" -B -I "$HERE/onnx_proof.py" "$@" --is-current >>"$LOG" 2>&1
    case $? in
        0) log "skip onnx $name$tag (result current)"; return 0 ;;
        1) ;;
        *) log "FAILED onnx $name$tag (existing result)"; exit 1 ;;
    esac
    log "start onnx $name$tag"
    if "$PY" -B -I "$HERE/onnx_proof.py" "$@" >>"$LOG" 2>&1; then
        log "end onnx $name$tag"
    else
        log "FAILED onnx $name$tag"
        exit 1
    fi
}

log "begin"
for m in v9-t v9-s v9-m; do fuse_check $m; done

export_model v9-t ""
export_model v9-t -fused --fuse
export_model v9-t -fused-logits --fuse --logits
export_model v9-s -fused --fuse
export_model v9-m -fused --fuse
export_model v9-t -fused-640x384 --fuse --size 640x384

for m in v9-t v9-s v9-m; do
    baseline $m
    baseline $m --snap-scores
done
baseline v9-t --size 640x384

proof v9-t ""
proof v9-t -fused
proof v9-t -fused-logits
proof v9-s -fused
proof v9-m -fused
proof v9-t -fused-640x384 640x384

if ! "$PY" -B -I - "$OUT" "$BASE" >>"$LOG" 2>&1 <<'PY'
import glob, json, os, sys
rows = [p for d in sys.argv[1:] for p in sorted(glob.glob(os.path.join(d, "*origin-bilinear-cv2*.json")))
        if not p.endswith(".onnx.json") and "-limit" not in p]
print(f"{'variant':<58} {'AP':>6} {'AP50':>6} {'AP75':>6}")
for p in rows:
    r = json.load(open(p))
    print(f"{os.path.basename(p)[:-5]:<58} {r['AP']:>6} {r['AP50']:>6} {r['AP75']:>6}")
PY
then
    log "FAILED table"
    exit 1
fi
log "DONE"
