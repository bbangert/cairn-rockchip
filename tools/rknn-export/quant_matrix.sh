#!/bin/sh
# quant_matrix.sh: the v9-t INT8 matrix (P3-T2..T4) and its class-branch round (P3-T5: tensors pinned to
# float16, and a clamped export), or with RUNG one other model's short round; each build analysed
# (work/mit/quant/builds) and scored on COCO val2017 in the toolkit's simulator under both rule sets, beside the
# float ONNX scored on the same images. Long: run it
# detached. One toolkit or onnxruntime job at a time. Resumable: a step whose output exists is skipped, and a
# scoring run resumes chunk by chunk. A failed step logs `FAILED <step>` and the run goes on to the next; the
# last line of work/mit/quant/matrix.log is DONE (or FAILED signal after INT/TERM/HUP). A signal to the script
# alone leaves its python child running: launch it with setsid and stop the whole group,
# `kill -TERM -- -PGID` (PGID = the script's pid under setsid). A second launch on the same QUANT_DIR is
# refused while $QUANT_DIR/.lock exists; a SIGKILLed run leaves it behind, to be removed by hand.
#   LIMIT=N         score the first N of the sorted val2017 ids (default: all 5000)
#   SWEEP_ALGO=a    the algorithm for the calibration-count sweep and the single-factor variants; by default
#                   the best reference-rules AP of the three coco baseline builds, decided once and kept in
#                   $QUANT_DIR/sweep_algo so a resumed run does not change its mind
#   MMSE_N=N        frames the v9-t matrix's baseline mmse builds calibrate on (default 50: the vendor guide's
#                   20-50, and mmse's build time grows with the count — v9-t, 20 frames: 400 s, 2026-11-27, this host)
#   QUANT_DIR=DIR   outputs somewhere other than work/mit/quant
#   HOME_FRAMES=DIR the 285 home calibration frames, if work/mit/quant/calib/home640 is not built yet
#                   (core's tools/qdq-export/out-20260820/calib)
#   RUNG=v9-s|v9-m  instead of the v9-t matrix, that model's float reference, FP16, the unclamped normal/200
#                   build and the clamped round (export, float reference, normal/200, mmse/100). Results are
#                   named by model, so they share QUANT_DIR, its lock and the summary; sweep_algo is not read.
#   CLAMP_K=K       the rung's clamp (v9-s default 32; v9-m has none: on 40 coco frames 32 moved its float cls
#                   by 0.014 and 64 by 2e-5, clipping only stride 8)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TPY="$HERE/venv/bin/python"
MPY="$HERE/work/mit/venv/bin/python"
Q="${QUANT_DIR:-$HERE/work/mit/quant}"
ONNX="$HERE/work/mit/onnx"
ANN="$HERE/work/coco/annotations/instances_val2017.json"
B="$Q/builds" S="$Q/sim"
LOG="$Q/matrix.log"
LIMIT=${LIMIT:-}
MMSE_N=${MMSE_N:-50}
RUNG=${RUNG:-v9-t}
case "$RUNG" in
    v9-t) ;;
    v9-s) K=${CLAMP_K:-32} ;;
    v9-m) K=${CLAMP_K:?RUNG=v9-m needs CLAMP_K (see the header)} ;;
    *) echo "RUNG=$RUNG: v9-s or v9-m (unset: the v9-t matrix)" >&2; exit 1 ;;
esac
LIM=${LIMIT:+--limit $LIMIT}
mkdir -p "$Q/calib" "$Q/parity" "$B" "$S"
export PYTHONDONTWRITEBYTECODE=1

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }
if ! mkdir "$Q/.lock" 2>/dev/null; then
    echo "$Q/.lock exists: another quant_matrix.sh is running on $Q (remove it if none is)" >&2
    exit 1
fi
trap 'rmdir "$Q/.lock" 2>/dev/null' EXIT
trap 'log "FAILED signal"; exit 1' INT TERM HUP

step() { # name done-file command...: skip when done-file exists, log FAILED and carry on when the command fails
    # sh has no locals: these names must not be ones a caller of step uses
    st_name="$1"; st_done="$2"; shift 2
    if [ -e "$st_done" ]; then log "skip $st_name (done)"; return 0; fi
    log "start $st_name"
    if "$@" >>"$LOG" 2>&1; then log "end $st_name"; return 0; fi
    log "FAILED $st_name"; return 1
}

prep() { # set-dir source [--list FILE]
    out="$1"; src="$2"; shift 2
    step "prep $(basename "$out")" "$out/set.json" "$TPY" -I -B "$HERE/prep_calib.py" "$src" "$out" 640 png rgb "$@"
}

score() { # stem label
    for r in "" -core; do
        step "score $2$r" "$1$r.json" "$MPY" -I -B "$HERE/score_coco.py" "$ANN" "$1$r.npz" \
            --meta "$1$r.meta.json" --out "$1$r.json" --label "$2$r"
    done
}

float_ref() { # onnx-basename: the float baseline on exactly the images the builds are scored on
    stem=$("$TPY" -I -B "$HERE/sim_coco.py" $LIM --out-dir "$S" --print-stem onnx "$ONNX/$1.onnx") || { log "FAILED stem $1"; return; }
    step "ort $1" "$S/$stem-core.meta.json" "$TPY" -I -B "$HERE/sim_coco.py" $LIM --out-dir "$S" onnx "$ONNX/$1.onnx" &&
        score "$S/$stem" "$1"
}

variant() { # name onnx-basename mode [convert.py options]: one build, analysed and scored in the same process
    name="$1"; model="$ONNX/$2.onnx"; mode="$3"; shift 3
    extra=""; [ "$mode" = i8 ] && extra="--ranges --analysis 3"
    stem=$("$TPY" -I -B "$HERE/sim_coco.py" $LIM --out-dir "$S" --print-stem rknn "$name" "$model" "$mode") || { log "FAILED stem $name"; return; }
    step "build+sim $name" "$S/$stem-core.meta.json" "$TPY" -I -B "$HERE/sim_coco.py" $LIM --out-dir "$S" \
        rknn "$name" "$model" "$mode" --std 255,255,255 --parity "$Q/parity/heldout640" --kind mit --refuse-cpu $extra \
        --out "$B" "$@" && score "$S/$stem" "$name"
}

clamp_round() { # model K: the class-clamped export (export_mit_yolov9.py --class-clamp), pure INT8
    cr_m="$1"; cr_k="$2"; cr_f="$1-fused-clamp$2"
    step "export $cr_f" "$ONNX/$cr_f.onnx.json" "$MPY" -B -I "$HERE/export_mit_yolov9.py" "$cr_m" \
        --out "$ONNX/$cr_f.onnx" --fuse --class-clamp "$cr_k" &&
        float_ref "$cr_f"
    variant "$cr_m-clamp${cr_k}_i8_normal_coco200" "$cr_f" i8 --algo normal --calib "$Q/calib/coco640" --calib-count 200
    variant "$cr_m-clamp${cr_k}_i8_mmse_coco100" "$cr_f" i8 --algo mmse --calib "$Q/calib/coco640" --calib-count 100
}

count() { [ "$1" = mmse ] && echo "$MMSE_N" || echo 200; } # algo: the frames its baseline build calibrates on

best_algo() { # the coco baseline with the highest reference-rules AP, decided once and persisted in $Q/sweep_algo
    if [ -s "$Q/sweep_algo" ]; then cat "$Q/sweep_algo"; return; fi
    "$MPY" -I -B - "$S" "${LIMIT:-}" "$MMSE_N" "$Q/sweep_algo" <<'PY'
import json, os, sys
d, lim, mmse_n, keep = sys.argv[1:]
tail = f"-origin-bilinear-cv2{'-limit' + lim if lim else ''}.json"
have, missing = {}, []
for algo in ("normal", "mmse", "kl_divergence"):
    p = os.path.join(d, f"v9-t_i8_{algo}_coco{mmse_n if algo == 'mmse' else 200}-sim{tail}")
    if os.path.exists(p):
        have[algo] = json.load(open(p))["AP"]
    else:
        missing.append(algo)
best = max(have, key=have.get) if have else "normal"
why = f"best of {have}" + (f"; no result for {missing}, so chosen without them" if missing else "")
print(why, file=sys.stderr)
with open(keep + ".tmp", "w") as f:
    f.write(best + "\n")
os.replace(keep + ".tmp", keep)
print(best)
PY
}

log "begin LIMIT=${LIMIT:-all} RUNG=$RUNG"
prep "$Q/calib/coco640" "$HERE/work/coco/train2017-calib" --list "$HERE/coco_calib_ids.txt"
if [ "$RUNG" = v9-t ]; then
    if [ ! -f "$Q/calib/home640/set.json" ] && [ -z "${HOME_FRAMES:-}" ]; then
        log "FAILED prep home640 (set HOME_FRAMES)"
    else
        prep "$Q/calib/home640" "${HOME_FRAMES:-unused}"
    fi
fi
prep "$Q/parity/heldout640" "$HERE/work/final/heldout"

if [ "$RUNG" != v9-t ]; then
    float_ref "$RUNG-fused"
    variant "${RUNG}_fp16" "$RUNG-fused" fp16 --calib "$Q/calib/coco640"
    variant "${RUNG}_i8_normal_coco200" "$RUNG-fused" i8 --algo normal --calib "$Q/calib/coco640" --calib-count 200
    clamp_round "$RUNG" "$K"
else
    float_ref v9-t-fused
    # convert.py refuses --parity without --calib (its frames' sources are checked against calibration's); fp16 calibrates on nothing
    variant v9-t_fp16 v9-t-fused fp16 --calib "$Q/calib/coco640"
    for algo in normal mmse kl_divergence; do
        variant "v9-t_i8_${algo}_coco$(count $algo)" v9-t-fused i8 --algo $algo --calib "$Q/calib/coco640" --calib-count "$(count $algo)"
    done
    if [ -n "${SWEEP_ALGO:-}" ]; then A=$SWEEP_ALGO; log "sweep algorithm $A (SWEEP_ALGO)"
    else A=$(best_algo 2>>"$LOG"); log "sweep algorithm $A ($Q/sweep_algo)"; fi
    N=$(count "$A")
    SWEEP="50 100 500"; [ "$A" = mmse ] && SWEEP="20 100"
    for n in $SWEEP; do
        variant "v9-t_i8_${A}_coco$n" v9-t-fused i8 --algo "$A" --calib "$Q/calib/coco640" --calib-count $n
    done
    for algo in normal mmse kl_divergence; do
        variant "v9-t_i8_${algo}_home$(count $algo)" v9-t-fused i8 --algo $algo --calib "$Q/calib/home640" --calib-count "$(count $algo)"
    done
    float_ref v9-t-fused-logits
    variant "v9-t-logits_i8_${A}_coco$N" v9-t-fused-logits i8 --algo "$A" --calib "$Q/calib/coco640" --calib-count "$N"
    float_ref v9-t
    variant "v9-t-unfused_i8_${A}_coco$N" v9-t i8 --algo "$A" --calib "$Q/calib/coco640" --calib-count "$N"
    variant "v9-t_i8_${A}_coco${N}_hybrid" v9-t-fused i8 --algo "$A" --calib "$Q/calib/coco640" --calib-count "$N" --hybrid
    variant "v9-t_i8_${A}_coco${N}_opt0" v9-t-fused i8 --algo "$A" --calib "$Q/calib/coco640" --calib-count "$N" --opt-level 0
    # P3-T5. A pinned op's output is cast back to INT8 for an INT8 consumer, so class_conv.1's float16 needs
    # class_conv.2 pinned too (measured: class_conv.1 alone did nothing). fnmatch's * crosses /: cls12hi pins
    # both at strides 16 and 32 (6 tensors), cls12 at all three (9). The clamp32 export clips class_conv.1's conv
    # outputs instead and stays pure INT8.
    F16_HI='/net/class_conv/class_conv.1/*_1/*,/net/class_conv/class_conv.1/*_2/*,/net/class_conv/class_conv.2_1/*,/net/class_conv/class_conv.2_2/*'
    F16_ALL='/net/class_conv/class_conv.1/*,/net/class_conv/class_conv.2*'
    variant v9-t_i8_normal_coco200_f16-cls12hi v9-t-fused i8 --algo normal --calib "$Q/calib/coco640" --calib-count 200 --float16 "$F16_HI"
    variant v9-t_i8_normal_coco200_f16-cls12 v9-t-fused i8 --algo normal --calib "$Q/calib/coco640" --calib-count 200 --float16 "$F16_ALL"
    variant v9-t_i8_mmse_coco100_f16-cls12 v9-t-fused i8 --algo mmse --calib "$Q/calib/coco640" --calib-count 100 --float16 "$F16_ALL"
    clamp_round v9-t 32
fi

if ! "$MPY" -I -B - "$S" "$B" >>"$LOG" 2>&1 <<'PY'
import glob, json, os, sys
s, b = sys.argv[1:]
rows = []
for p in sorted(glob.glob(os.path.join(s, "*.json"))):
    if p.endswith(".meta.json"):
        continue
    r = json.load(open(p))
    name = os.path.basename(p)[:-5]
    bd = r.get("build") or {}
    if "outputs" not in bd:  # a result scored before the runner kept the build's record with it
        bj = os.path.join(b, (bd.get("name") or "-") + ".json")
        bd = json.load(open(bj)) if os.path.exists(bj) else {}
    f16 = bd.get("float16_layers", "")
    # max_score: the probability a cls output can reach, through the sigmoid for a logits export
    ceil = min((o.get("max_score", o["max_representable"]) for o in (bd.get("outputs") or []) if o["name"].startswith("cls") and o.get("quantized")), default=None)
    rows.append((name, r["AP"], r["AP50"], r["AP75"], r["n_images"], f16, "" if ceil is None else f"{ceil:.3f}"))
print(f"{'result':<72} {'AP':>6} {'AP50':>6} {'AP75':>6} {'imgs':>5} {'f16':>4} {'clsmax':>6}")
with open(os.path.join(s, "summary.tsv"), "w") as f:
    f.write("result\tAP\tAP50\tAP75\timages\tfloat16_layers\tcls_max_representable\n")
    for row in rows:
        print(f"{row[0]:<72} {row[1]:>6} {row[2]:>6} {row[3]:>6} {row[4]:>5} {row[5]!s:>4} {row[6]:>6}")
        f.write("\t".join(map(str, row)) + "\n")
PY
then
    log "FAILED summary"
fi
log "DONE"
