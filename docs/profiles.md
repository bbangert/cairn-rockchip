# Profiles on the Rockchip board

What this board adds to a hardware profile. The schema, the ladder rules and
every message that is not about `rknn` or `rkmpp` are core's
[profile-authoring guide](https://github.com/bbangert/cairn-nvr/blob/main/docs/profile-authoring.md);
the measurements behind the rules here are [npu-rknn.md](npu-rknn.md).

## The menus

`Cairn.Rockchip.Board` offers the backends `ort` and `rknn` and the decoders
`auto`, `sw` and `rkmpp` (MPP decode + RGA scale). Its `auto` never picks
`rkmpp`: a profile names it or does not get it. There is no `qnn`: this
board has no HTP, so a profile naming it is refused with this board's name.

| backend | artifact key | runs a fused-NMS export | takes dynamic shapes | needs a CPU twin | experimental |
|---|---|---|---|---|---|
| `ort` | `model.onnx` | yes | yes | no | no |
| `rknn` | `model.rknn` | no | no | yes | yes |

`rknn`'s fused-NMS and dynamic-shape columns mirror the `RKNN` descriptor in
`native/cairn_rockchip/src/rknn.rs`, which carries the evidence. `rknn` stays
experimental until it has soaked, so a profile naming it says
`experimental: true` and a group running it says `allow_experimental: true`.

## An `rknn` profile

```yaml
# <profile_dirs entry>/my-board.yml
experimental: true
backend: rknn
model:
  onnx: models/yolox_tiny.onnx           # the CPU twin
  rknn: models/yolox_tiny_rk3566.rknn
model_profile: yolox
input_size: 416
decoder: rkmpp
labels: models/coco.names
```

An `.rknn` is compiled from an `.onnx` by a host-side toolchain
([`tools/rknn-export`](../tools/rknn-export/README.md)). Only librknnrt runs
it, so `rknn` also needs `onnx:`: the FP32 export the `.rknn` was converted
from, its **CPU twin**. The node's health check judges the accelerator
against one CPU pass of the same model, and that pass is measured on the
twin. No `onnx:` on such a profile — or on any rung of such a ladder — fails
the load; one that names a file not on disk loads with a warning and no
health ratio.

An `.rknn` has its mean and std folded in at conversion and is fed the frame's
bytes, so the `model_profile:` beside it must name the family it was converted
as: the channel order and scale are the ones in "Input conventions" in
[`tools/rknn-export/README.md`](../tools/rknn-export/README.md), and nothing
at load can check them. A family with per-channel statistics folded in
(`rfdetr`) is refused at open: the packer has no path for it.

**The artifact pins `input_size:` and config load cannot check it.**
rknn-toolkit2 compiles the model at one input geometry, and config load
cannot open an `.rknn` to look. The engine's probe-load refuses a model whose
own shape contradicts the size it was given, after the config has loaded,
and no camera detects. Set it to the geometry you converted at.

## `decoder: rkmpp`

Naming a decoder is a requirement (core's guide says why): without MPP, RGA
or the dma-heap the camera's detect branch goes dark with the reason on
`cameras:status`, while recording carries on. On RK3566 the software
fallback would not even carry the fleet: software decode of one 15 fps
substream ran at the edge of keeping up on the HA Green. The board also says
so at config load: it warns when a profile names `rkmpp` and
`/dev/mpp_service`, `/dev/rga` or `/dev/dma_heap/system` is missing, or the
board NIF did not load.

## Which families convert

`yolox`, `yolov8split` and `yolo26split` converted to INT8 with rknn-toolkit2
2.3.2, ran on the RK3566 NPU and detected. A profile pairing `rknn` with any
other family is refused unless it declares `experimental: true`, and then
loads with a warning that gives the same reason:

- **The stock whole head** (`yolov8` and its row: `yolov9`, `yolo11`,
  `yolo26`) carries pixel-range boxes and 0..1 scores in one tensor. INT8
  gives a tensor one scale, which zeroes the scores: yolov8, yolo11 and
  yolo26 converted and detected nothing in the toolkit's simulator (yolov9
  was not tried). For a v8, v11 or 26 model, export the split head and
  name `yolov8split` (or `yolo26split`) instead.
- **The end-to-end head** (`yolov10`), measured on yolo26's export of it:
  no detections in INT8 in the simulator, and in FP16 on the board the
  right scores and classes with wrong boxes.
- **`rfdetr`**: its INT8 scores are unusable in the simulator, and no
  INT8 build was run on the board. Its FP16 build crashed librknnrt 2.3.2
  there: the runtime has no GridSample op.

The verdict is per decode contract, not per model size: a yolox_nano
build passes this rule and detected nothing in INT8 in the simulator.
FP16 is not a way around it. The backend opens INT8 artifacts only and
refuses an FP16 build at load; the FP16 results above come from the
bench harness, where FP16 also ran about three times slower than INT8
end to end (yolox_s at 640: 310 ms a frame against 101).
[`tools/rknn-export/README.md`](../tools/rknn-export/README.md) has the
measurements.

## Sizing

**RKNN is sequential per context, and the engine holds three.** librknnrt
documents no async submission and no batching: one inference at a time per
context. Contexts of one model overlap on the RK3566, so the engine holds
three and passes from different cameras run on whichever is free; the
count is the board's (`CONTEXTS` in `native/cairn_rockchip/src/rknn.rs`),
and nothing in a profile can change it — there is no batch size and no
concurrency field. `rk3566-tier1`'s yolox_s budget is measured at three
contexts and its yolox_tiny budget at two; at three no decoder dropped a
sample at that count
([`docs/evidence/rk3566-capacity-20261126.md`](evidence/rk3566-capacity-20261126.md)).
Measure a budget for your own rknn ladder at the engine's count.

A measurement records the CPU governor it ran under; RK3566's budgets were
taken at the CPU governor as found, `schedutil`.

## `rk3566-tier1`

The profile this board ships, in `priv/profiles/`, with its reasoning in the
file:

| backend | rate | stages | notes |
|---|---|---|---|
| `rknn` | derived per rung | none (tier 1) | two fetched Apache YOLOX rungs with baked CPU twins, `decoder: rkmpp`, claim 13 cameras; boundary-measured, yolox_s at three contexts and yolox_tiny at two ([`docs/evidence/rk3566-capacity-20261126.md`](evidence/rk3566-capacity-20261126.md)) |

It is `experimental: true` because `rknn` has not proven out in soak: a group
naming it needs your `allow_experimental: true`. For a ladder file of your
own it is the worked example of budget provenance: both budgets measured,
each note naming its runs.

## Messages about `rknn` and `rkmpp`

Errors:

```
profile my-board: model_profile yolov10 is not a proven rknn conversion: its
  end-to-end output detects nothing in INT8 in the toolkit's simulator and
  has wrong boxes in FP16 on the board, both measured on yolo26's export of
  that head — experimental: true loads it regardless (measurements:
  https://github.com/bbangert/cairn-rockchip/blob/main/tools/rknn-export/README.md)
profile my-board: model artifact models/gone.rknn does not exist or is not a
  regular file (relative paths resolve against this node's working directory)
plugin detect: unknown profile "rk3566" — no such file in priv/profiles, the
  board's profiles or any profile_dirs entry
plugin detect: profile my-board uses backend rknn, which is experimental on
  the rockchip board — a profile naming it must declare experimental: true
plugin detect: profile my-board uses backend rknn, which is experimental
  — set allow_experimental: true on this plugin group to run it anyway
plugin detect: profile my-board names no model.rknn artifact for its rknn
  backend — the engine takes its model from the profile alone
plugin detect: profile my-board runs backend rknn, whose artifact the CPU
  baseline cannot run, but it names no model.onnx — name the model's FP32
  ONNX twin there
plugin detect: profile my-board runs backend "rknn", which needs librknnrt.so,
  not downloaded yet: the last fetch failed: <why> — the download is retried
  until it succeeds; no camera can run on this group before then
plugin detect: profile rk3566-tier1 needs yolox_s_640_i8.rknn (ladder rung 1),
  not downloaded yet: the fetch has not been tried — the download is retried
  until it succeeds; no camera can run on this group before then
```

The last two are errors only while a camera runs on the group, else
warnings.

Warnings:

```
plugin detect: profile my-board names decoder "rkmpp", which the rockchip
  board provides but cannot use on this host: /dev/rga is missing
profile my-board: CPU twin models/gone.onnx does not exist or is not a regular
  file — rknn will load and detect, but its health cannot be judged against
  the CPU
profile my-board: model_ladder rung 2: experimental: true loads model_profile
  yolov10, which is not a proven rknn conversion: <the same reason>
```
