# rknn-export — RK3566 (RKNN) rung pipeline

Builds the INT8 `.rknn` rungs for the HA Green's NPU (RK3566,
rknn-toolkit2 2.3.2), each beside the FP32 ONNX it was converted from, and
scores the toolkit's simulator against that ONNX on frames calibration never
saw. Host only. Nothing built here enters git: `work/` and `venv/` are
ignored, and so is any `.onnx`, `.rknn`, `.pt`, `.npy` or `.npz` wherever it
lands.

Simulator parity measures INT8 quantisation error, not the NPU's numerics.
The board is a separate measurement (`board/`).

A path called core's is in a checkout of
[cairn-nvr](https://github.com/bbangert/cairn-nvr); `build_rungs.sh` takes
it from `CAIRN_CORE_PATH`, which the commands below need absolute.

## Why the rungs look the way they do

This toolkit gives an INT8 output tensor one scale. Everything else follows.

- **Stock Ultralytics heads detect nothing.** `[1, 4+nc, A]` (and yolo26's
  end-to-end `[1, 300, 6]`) carries pixel-range boxes and 0..1 scores in one
  tensor, so the scores quantise to zero: all six models convert, run, keep a
  box cosine of 0.998, and emit zero detections ≥ 0.3. FP16 of the same
  graphs is exact in the simulator. On the board, FP16 builds of the v8
  and v11 stock heads and of yolo26's end-to-end head ran, and only
  yolo26n's was scored: its scores and classes matched and its boxes came
  back near the origin, the runtime logging
  `GatherElements op: Out of range value in index tensor` on every pass
  (`board/results/m3.log` has its timing; the comparison was not kept).
- **The split head is the fix.** `export_ultralytics_rknn.py … split` cuts
  the head before DFL/decode and emits nine tensors, `box{i}`, `cls{i}`,
  `sum{i}` for strides 8/16/32. `box` is `[1, 4*reg_max, H, W]` (64 channels
  of DFL logits for v8/v11; 4 ltrb distances in stride units for yolo26,
  taken from its one2one head), `cls` is the sigmoided scores, `sum` is
  airockchip's early-reject channel. The export sets those names and asserts
  names and shapes before the file reaches its destination; all nine survive
  into the `.rknn`. `cairn-detect` reads them as `--model-profile
  yolov8split` (v8/v11) or `yolo26split`, which bind `box*` and `cls*` by
  those names.
- **YOLOX survives as one tensor** because its box columns are raw
  regressions, not pixels. The price is a coarse score: boxes and scores
  still share a scale, and the step is 0.025–0.028 (listed under the parity
  table).
- **yolox_nano does not survive INT8** (recall 0.00; the `mmse` quantiser
  does not rescue it). rfdetr_nano INT8 is unusable (score error 0.09–0.25)
  and RT-DETR-l fails at build. rfdetr_nano FP16 passes the simulator and
  crashes librknnrt 2.3.2 on the board: `Unsupport CPU op: GridSample`,
  then a segfault (the last lines of `board/results/m3.log`, which do not
  name the model).
- **FP16 is not a rung.** It is exact in parity and three times slower on
  the board end to end: yolox_s 640 takes 101 ms a frame INT8 and 310 ms
  FP16, 65 of them converting the input (`board/results/m1.log`).

## Environments

- `venv` (here, ignored): `python3.11 -m venv venv && venv/bin/pip install
  -r requirements.txt`. Runs every script in this directory except the
  Ultralytics export and the YOLOv9 stages.
  `onnx` is pinned because pip's latest breaks `load_onnx`.
- core's `tools/qdq-export/export-venv`: runs `export_ultralytics_rknn.py`
  only. The AGPL boundary is the one core's `tools/qdq-export/README.md`
  describes: you
  install and run Ultralytics, and neither its weights nor anything exported
  from them enters git or an image.
- core's `plugins/cairn-detect/model/quant-venv`: runs the held-out census
  (core's `tools/qdq-export/calib_census.py`).
- `work/mit/venv` (`sources.py mit`): torch, onnxruntime and the MIT
  repo's dependencies; runs the YOLOv9 stages.
- `tests/requirements.txt`: pytest's, in a venv of its own (see Tests).

Scripts run as `venv/bin/python -I`. `-I` keeps the script directory off
`sys.path`, which is why the ones that import a sibling insert it by hand.

## Input conventions

Mean and std are folded into the `.rknn`, so the model takes raw uint8.

| family | resize | channels | mean / std |
|---|---|---|---|
| yolox | letterbox, pad 114, content at the origin | BGR | 0 / 1 |
| yolov8, yolo11, yolo26 | stretch | RGB | 0 / 255 |
| rfdetr (trial only) | stretch | RGB | ImageNet |
| yolov9 (MIT) | letterbox, pad 114, content at the origin | RGB | 0 / 255 |

`prep_calib.py` writes each set as NCHW uint8 `.npy` — the toolkit rejects
NHWC `.npy`, and image files would hide the channel order — with a
`set.json` recording every frame's source geometry and an id hashed over the
frame bytes. That id is the calibration set id in the manifest.
The YOLOv9 sets (policy `png`) are the scoring pipeline's own canvases as
PNG at the input size; the toolkit reads a PNG as RGB (checked: a file
written R=200, B=20 calibrated model channel 0 to 200).

## Calibration and held-out frames

Both sets come from core's `tools/qdq-export/capture_frames.sh` (full resolution,
sampled across the whole clip) and both are operator-local.

**Calibration** is the 285-frame set the QNN pipeline already vetted,
core's `tools/qdq-export/out-20260820/calib`: 36 clips, 2560x1920 ×120, 1920x1080
×105, 640x480 ×60, a person in 24 % of frames. The conversion trial's own
set (135 frames) had no substream and one 2.5-hour window — the composition
core's `tools/qdq-export/README.md` warns against — so the rungs are not calibrated
on it.

**Held-out** is 90 frames from 20 clips, none of them a clip calibration
drew from (`convert.py` refuses to build a rung whose two sets share one):
2560x1920 ×40, 1920x1080 ×30, 1024x576 ×20. Parity is reported per geometry
because a single pooled number hides a geometry that fails. Three limits on
what it can say:

- Held out by clip, not by scene. The 15 event clips come from one of
  calibration's two event cameras, inside the same 2.5 hours; one ends 4 s
  before a calibration clip from that camera begins, and another overlaps
  in time a calibration clip from the other. This is parity on unseen
  frames of familiar scenes.
- A person scores ≥ 0.5 in 11 of the 90 frames, none of them in the
  substream group; the one held-out substream clip never shows one. The
  person columns count cells whose FP32 person score is ≥ 0.3, nearly all
  of them in those frames. Read them to ±0.1, and substream person parity
  as unmeasured.
- The any-class counts are mostly the same parked cars, frame after frame.

## Build

```sh
./build_rungs.sh apache   # the three YOLOX rungs; no Ultralytics needed
./build_rungs.sh pack     # the six split rungs; needs export-venv and work/src/weights/*.pt
./build_rungs.sh all
```

It needs ffmpeg, `CAIRN_CORE_PATH`, the event clips in core's
`data/events/reolink_main/`, core's `tools/qdq-export/out-20260820/` (the
calibration frames and its clips) and the YOLOX release files in `work/src/`
(hashes in core's `plugins/cairn-detect/README.md`).

Steps skip when their output exists, so a failed run resumes. The log is
`work/final/build.log`. Per rung, `work/final/build/` holds `NAME.out` (the
toolkit's stdout, operator table included), `NAME.build.log` (its debug
log), `NAME.json` and `NAME.parity.json`. A rung takes 30–115 s to build
and 18–42 s to score. `work/rungs/` ends up holding the `.rknn` files,
their twins, `MANIFEST.sha256` (`sha256sum -c` format) and `MANIFEST.json`
(bytes, license, twin, toolkit, target, calibration set id). `manifest.py`
lists a rung only if the hash its build recorded is the hash of the file.

The YOLOX twins are the 0.1.1rc0 release files, bytes unchanged
(`work/src/yolox_{tiny,s}.onnx`). `yolox_tiny_640.onnx` is the 416 release
file with its input size rewritten (`onnx_edit.py resize`); the 640 rung is
converted from that file, so it has a reference to be scored against.

**A manifest hash names one build, not a recipe.** Two builds of
`yolox_tiny_416_i8` from identical inputs differ in 105 bytes within the
first 2.6 kB and are identical after it; their parity results are
byte-identical. Publish the files the manifest hashed, not a rebuild.

## Parity

Simulator vs twin, pooled over the 90 held-out frames. A positive is a cell
whose FP32 score is ≥ 0.3; recall is the share of those the simulator also
scores ≥ 0.3. Person recall is given overall, then for 1920x1080 and
2560x1920. CPU ops excludes the input and output operators. Box error is
mean absolute error at positive cells, in input pixels. Max is the highest
score seen, FP32 → simulator: a ceiling baked in by calibration would show
as a gap.

| rung | MB | CPU ops | any: n, recall | person: n, recall | person 16:9 / 4:3 | person mean abs Δscore | box px | max |
|---|---|---|---|---|---|---|---|---|
| yolox_tiny_416_i8 | 6.3 | 3 | 1392, 0.93 | 93, 0.87 | 0.88 / 0.84 | 0.071 | 0.95 | .92 → .92 |
| yolox_tiny_640_i8 | 6.4 | 3 | 2020, 0.95 | 100, 0.94 | 0.99 / 0.82 | 0.066 | 0.84 | .93 → .92 |
| yolox_s_640_i8 | 10.4 | 3 | 1772, 0.94 | 96, 1.00 | 1.00 / 1.00 | 0.020 | 0.72 | .93 → .95 |
| yolov8n_split_640_i8 | 4.8 | 0 | 1550, 0.91 | 87, 0.83 | 0.80 / 0.89 | 0.110 | 0.51 | .88 → .87 |
| yolov8s_split_640_i8 | 13.2 | 0 | 2406, 0.93 | 122, 0.92 | 0.91 / 1.00 | 0.037 | 0.33 | .90 → .89 |
| yolo11n_split_640_i8 | 4.6 | 4 | 1328, 0.84 | 78, 0.92 | 0.89 / 1.00 | 0.072 | 0.78 | .90 → .92 |
| yolo11s_split_640_i8 | 11.9 | 4 | 2390, 0.86 | 100, 0.75 | 0.76 / 0.73 | 0.155 | 0.71 | .95 → .94 |
| yolo26n_split_640_i8 | 4.6 | 8 | 153, 0.63 | 8, 0.88 | 0.83 / 1.00 | 0.109 | 0.82 | .91 → .93 |
| yolo26s_split_640_i8 | 12.2 | 8 | 245, 0.73 | 13, 0.77 | 0.75 / 0.80 | 0.162 | 0.82 | .95 → .94 |

- No rung shows a score ceiling.
- YOLOX score steps: 0.0278 (tiny 416), 0.0256 (tiny 640), 0.0253 (s 640).
- **yolox_tiny at 640 passed its bar**, set before the run: person recall
  ≥ 0.85 overall and no more than 0.05 below the 416 rung. It scored 0.94
  against 0.87, with the same output cosine (0.969). It is built and
  published but is not an `rk3566-tier1` rung: on the board, with one
  context, it held no more cameras than yolox_s
  (`docs/evidence/rk3566-capacity-20261118.md`).
- yolo11n loses stride 32 for non-person classes (any-class recall 0.42
  there); yolo11s loses a quarter of person cells at every stride it has
  them.
- yolo26's one2one head fires once per object, hence the small counts. Its
  any-class recall (0.63, 0.73) is the worst of the set and its person
  sample is too thin to rescue it.

## Apache rungs

The rows a fetch manifest would pin (toolkit 2.3.2, target `rk3566`,
`quantized_algorithm=normal`):

| file | bytes | sha256 | calibration set |
|---|---|---|---|
| yolox_tiny_416_i8.rknn | 6273103 | `832df464bc84259d8d612bb7e9d4f6dd8573acb94f9847e051cce4c438c78304` | `5047ae0e6484f100` |
| yolox_tiny.onnx | 20219662 | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` | |
| yolox_tiny_640_i8.rknn | 6402703 | `cd146e38777ea0c38e20c114c4166fb9b7553c379e20651a999167acb0bb044a` | `920782f4e1e36e81` |
| yolox_tiny_640.onnx | 20219702 | `6066fa79efe783a554a6f7e13792e0ee9246bc0c7b27d6eddad3f43bdd000a61` | |
| yolox_s_640_i8.rknn | 10414863 | `c61524bca75d6f5df3fd255c8f27b5eaca544c94cbd05d4ce835a82a48733282` | `920782f4e1e36e81` |
| yolox_s.onnx | 35858002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` | |

## Toolkit traps

- The simulator only runs a graph built in the same process; a `.rknn`
  loaded with `load_rknn` cannot be simulated. So `convert.py` scores parity
  itself right after the build, and the check on the exported file is that
  `load_rknn` accepts it.
- Every build drops `check*_*.onnx` files (20–40 MB each) into the current
  directory. `convert.py` changes into its `--out` directory first.
- `accuracy_analysis` reads one frame per model input, so `--analysis N`
  calls it N times; each call dumps ~0.75 GB of text tensors, which
  `convert.py` parses and deletes. The yolox and split rung builds do not
  use it.
- Per-tensor ranges, scales and zero points are not reported by `build`.
  `--ranges` gets them by building through
  `hybrid_quantization_step1(proposal=False)` + `step2`, one calibration
  whose cfg is the model's own. `--hybrid` exists only in `build`, so with
  it step 1 runs as a second calibration pass beside the build (v9-t
  normal, 2026-11-27: its output scales were the simulator's output steps
  from `build`, and step 1 + 2 gave `build`'s parity counts exactly). That
  pass knows nothing of the layers `--hybrid` moved to float16:
  `NAME.ranges.json` then carries `"plan": "before auto_hybrid"` and
  `NAME.json` takes no output scales or flags from it.
- A `.rknn` hash does not identify a build: two builds from the same
  inputs differed in 200-240 bytes that read as heap addresses (v9-t
  normal and kl_divergence, 2026-11-27) with equal cfgs and parity.
  `convert.graph_id`, the outputs on a fixed input, is the identity.
- CPU fallback is never a warning. It shows only as `CPU` rows in the
  operator table the toolkit prints to stdout (`NAME.out`, not
  `NAME.build.log`), which is what `summarize.py` counts.

## The conversion trial

`run.sh`, `batch_ultra.sh`, `parity_e2e.py`, `parity_detr.py` and the `v8`,
`v8t`, `e2e` kinds of `parity.py` reproduce the matrix behind the first
section, on five calibration frames per model. Those kinds print per-frame
means, not the pooled counts above.

```sh
venv/bin/python -I prep_calib.py work/frames work/calib/lb640_bgr 640 letterbox bgr   # also lb416_bgr, st640_rgb, st384_rgb
(cd work/src/weights && "$CAIRN_CORE_PATH"/tools/qdq-export/export-venv/bin/python -I \
  ../../../export_ultralytics_rknn.py yolov8n stock 640 ../yolov8n_stock.onnx)        # stock | split | e2e
./run.sh yolox_s_640_i8 work/src/yolox_s.onnx i8 --calib work/calib/lb640_bgr
venv/bin/python -I parity.py yolox_s_640_i8 yolox
./batch_ultra.sh                                  # six models × stock, split × INT8, FP16
./run.sh rfdetr_nano_384_fp16 work/src/rfdetr_nano.onnx fp16 --mean 123.675,116.28,103.53 \
  --std 58.395,57.12,57.375 --calib work/calib/st384_rgb --size 384x384 --input-name pixel_values
```

## YOLOv9 (MIT)

YOLOv9-t, -s and -m come from `MultimediaTechLab/YOLO`, the MIT-licensed
rewrite, never from the GPL-3.0 original or an AGPL package, so what is
built from them is MIT. Nothing fetched for this path enters git.

**Sources.** `sources.lock` pins the repo at one commit, the three release
checkpoints by bytes and sha256, and COCO val2017 (both archives, the
annotation file, a digest over all 5000 images). `python3 -I sources.py mit`
clones that commit, fetches and checks the checkpoints and builds
`work/mit/venv` with `--require-hashes` (locked for x86_64 Linux, CPython
3.11, CPU torch). `sources.py coco` fills `work/coco/` and refuses to start
without room for the download (1.1 GB), the extraction (0.85 GB) and
20 GiB of headroom. `sources.py verify` re-hashes everything and compares
the venv's `pip freeze` with the requirements. Only v9-t's hash is one
GitHub publishes; the rest were recorded on first fetch.

**The v9-t checkpoint does not fit the pinned code.** Its class-branch neck
is 80 wide where the code builds 128. MIT's loader only warns and leaves
the class head untrained. `mit_model.load` builds that one width as the
checkpoint has it and loads strictly, refusing a missing, reshaped or stray
key. Every stage must load through it.

**The cut.** Per stride (8/16/32) the model ends at the head's last convs:
`box{0,1,2}` `[1, 64, h, w]`, raw DFL logits, and `cls{0,1,2}`
`[1, 80, h, w]`, sigmoided; the export adds `sum{0,1,2}` as the split rungs
carry them unless `--no-sum` (or `--logits`, raw class logits). The DFL
and anchor tail quantises badly in INT8, so it is cut, as for the split
rungs. The host decodes, with `decode_split.py` as the one reference every
stage scores through; on the board, core's `SplitDfl` binds the outputs by
name.

**Stages.** Every stage feeds RGB scaled to 0..1 and runs as
`work/mit/venv/bin/python -B -I`; `-B` keeps bytecode out of the MIT clone,
which the exporter refuses unless it is clean at the pinned commit. The two
scripts are long, resumable and log to their directory's `run.log`, ending
`DONE` or `FAILED <step>`. Run one torch or onnxruntime job at a time.

- `./baseline_mit.sh` scores each checkpoint in PyTorch on val2017 into
  `work/mit/baseline/`, per resize (centred or top-left letterbox, stretch)
  and resampler (PIL LANCZOS, cv2 bilinear), first checking the cut and
  decode against MIT's own on 8 images. One variant: `baseline_mit.py v9-t --mode
  origin --resample bilinear-cv2 [--size 640x384] [--snap-scores]`;
  `--snap-scores` rounds class scores to k/256, an 8-bit output's cost alone.
- `export_mit_yolov9.py v9-s --out work/mit/onnx/v9-s-fused.onnx --fuse
  [--size WxH]` writes opset 12. `--fuse` folds each RepConv into one conv;
  sizes are multiples of 32. A graph with the wrong outputs, a Softmax, an
  unfolded BatchNorm, a tensor above 4-D or a dead branch is deleted. The
  sidecar `FILE.onnx.json` records checkpoint, commit and file hashes.
- `fuse_repconv.py v9-s` exits non-zero if a RepConv is left or any output
  moves more than 1e-3 × max(1, max |plain|) on six inputs.
- `./onnx_proof.sh` runs the fusion check, six exports, the top-left
  baselines it gates against, then `onnx_proof.py` per export on
  onnxruntime CPU. It exits non-zero when an output on 20 images differs
  from PyTorch by more than 1e-3 × max(1, max |reference|), when AP is more
  than 0.05 from the baseline, or when the baseline, sidecar or checkpoint
  does not match the run.

AP on full val2017 (5000 images), top-left letterbox, cv2 bilinear:

| rung | PyTorch | ONNX, fused |
|---|---|---|
| v9-t 640 | 38.21 | 38.21 |
| v9-s 640 | 46.56 | 46.56 |
| v9-m 640 | 51.14 | 51.14 |
| v9-t 640x384 | 35.88 | 35.88 |

- Top-left is within 0.2 AP of centred (centred LANCZOS 38.26 / 46.69 /
  51.14); stretch costs 1.7–1.8 (v9-t 36.50, v9-s 44.80). The resampler
  moves AP by 0.06 at most.
- 8-bit class scores cost 0.05 AP at most: 38.17 / 46.51 / 51.09.

`manifest.py` labels a YOLOv9 file MIT only through the exporter's sidecar
recording its hash; without one the manifest is refused.

**Quantization.** Run with `venv/bin/python -I -B` unless marked.

- `coco_calib.py fetch coco_calib_ids.txt work/coco/train2017-calib`
  downloads the 500 train2017 images the tracked id list names (seeded
  draw, `coco_calib.py select` redraws it from `instances_train2017.json`);
  `prep_calib.py SRC OUT 640 png rgb [--list coco_calib_ids.txt]` writes a
  set. `convert.py --calib-count N` calibrates on its first N frames.
- `convert.py NAME ONNX i8 --std 255,255,255 --calib CALIB_SET --calib-count 200
  --algo normal|mmse|kl_divergence [--hybrid] [--opt-level 0] --kind mit
  --parity HELDOUT_SET --refuse-cpu --ranges --analysis 3 --out DIR` leaves
  beside
  `NAME.rknn`: `NAME.json` (settings, hashes, op targets, warnings,
  float16 layers, each output's scale and zero point), `NAME.out` and
  `NAME.build.log`, `NAME.ranges.json` (clipped box tensors, score ceilings
  under 1.0, Concat inputs under a quarter of their output's range),
  `NAME.analysis.json` (per-layer `entire` and `single` cosine, worst
  ranked) and `NAME.parity.json`, whose verdict (`parity.MIT_BAR`) is the
  fast check for a rebuild. With `--ranges` (not `--hybrid`) the build is
  `hybrid_quantization_step1` + `step2`, so the cfg is the model's own and
  calibration runs once (v9-t normal: same parity counts as `build`).
- `sim_coco.py [--limit N] rknn NAME ONNX i8 [convert.py options]` builds
  (with convert.py's `--parity`, `--ranges`, `--analysis` on the same
  graph) and scores on val2017 in the simulator (v9-t: 6.0 images/s, one
  core of this host, 2026-11-27); `sim_coco.py onnx ONNX` is the
  float ONNX on the same images. Each writes detections for both rule sets:
  the reference protocol, and `-core`, core's shipping decode (one class per
  box, IoU 0.45, 300 candidates, 32 per image, wire rounding; v9-t float
  ONNX on val2017, 2026-11-27: 36.35 AP and 141,566 detections, as the Rust
  decode). At a 0.001 floor that is the decode, not what a camera emits at
  `min_score`; for the logits export it is no shipping path (core's
  SplitDfl applies no sigmoid). Score each with
  `work/mit/venv/bin/python -I -B score_coco.py ANN STEM.npz --meta
  STEM.meta.json --out STEM.json`.
- `./quant_matrix.sh` runs the v9-t matrix, each build analysed and scored,
  and writes `work/mit/quant/sim/summary.tsv`; detached, resumable, log
  `work/mit/quant/matrix.log`. `LIMIT=N` scores the first N images; an mmse
  build calibrates on `MMSE_N` frames (50), the others on 200.
- `./build_rungs.sh mit` builds the t/s/m rungs into `work/rungs` with
  `MIT_ALGO`, `MIT_CALIB`, `MIT_CALIB_COUNT`.

## board/

Probes that ran on the Green during bring-up, with the board's `python3` and
numpy and nothing from the toolkit. Each prints `JSON:`-prefixed records, and
`results/` holds what they printed; the other lines there come from the
commands around them (`DONE`, `[exit 0]`, `:ok`, a checksum) or, in `m3.log`,
from librknnrt's stderr and the shell reporting the segfault. The models were trial
builds, not the pinned rungs.

- `rk.py` is a ctypes harness over librknnrt: `info`, `bench` (200 timed
  calls, then 60 s sustained with NPU load, temperature and free memory
  sampled), `input` (the ways to hand over a frame, timed and checked against
  each other), `acc` (float outputs for the host) and `worker` (the child
  `conc.py` starts). `matrix.sh` runs `bench` over `models/NAME.rknn` into
  `m1.log`, `m2.log` and `m3.log`; `tab.py` prints those a line per record.
- `zcin.py` and `zc.py` try zero-copy I/O through `rknn_create_mem` and
  `rknn_set_io_mem`. A zero-copy input gives the outputs `rknn_inputs_set`
  gives, at cosine 1.000 (`zcin.log`). A zero-copy output does not: `zc.py`
  read the native buffer back at cosine 0.05 under every layout it tried
  (`zc.log`), so nothing here shows how to read one.
- `conc.py` and `conc.sh` run one context per process, started on one epoch
  (`conc.log`).
- `heaps.py` and `heapcap.py` allocate from each dma-heap, report what that
  takes from CMA, and count the 1080p NV12 buffers the 32 MiB
  `rk-dma-heap-cma` pool yields (`heaps.log`, `heapcap.log`).
- `dec.py` runs K concurrent `mpi_dec_test` decodes with free CMA sampled
  (`dec.log`). That log was taken while `dec.py` still read the decoders'
  pipes one after another. Its 1080p rows at K = 2 and 4 show the decoders
  finishing in order, 2.2 s apart, so each waited on a full pipe: read those
  aggregate rates as a floor. The 2560x1920 pair finished together.
- `syms.py` checks the exported symbols of librknnrt, librga and
  librockchip_mpp against a list of calls a board library might make, drawn up
  before one was written. No log is kept.
- `cpu_ort.py` times ONNX Runtime on the CPU, for the NPU's margin over it.
  It has not been run on the board.

`compare.py` runs on the host and answers whether the NPU agrees with the
simulator: it scores an `rk.py acc` dump against the simulator and FP32
tensors of the same build, `$RKNN_OUT/NAME.parity.npz` (default `work/rknn`).
A `run.sh` build writes that file for five calibration frames, and `acc` must
be given the same five in the same order. `build_rungs.sh` keeps no tensors,
so a rung from `work/rungs` cannot be scored this way.

## Tests

`python -m pytest tools/rknn-export` covers the parity arithmetic, the ONNX
edits and the letterbox size rule, the YOLOv9 decode, scoring, graph checks
and source fetch, the pieces whose errors would be silent, and the
manifest's refusals. `tests/requirements.txt` installs none of the toolkit,
onnxruntime, torch or PIL, so under it the checks that need them (a resized
model run, the RepConv fold and loader in torch, the PIL resampler) are
skipped; it does install headless OpenCV, at the toolkit venv's version, so
the check that a calibration PNG is the scoring canvas runs.
