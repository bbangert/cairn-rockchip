# RKNN, MPP and RGA on rk3566 / rk3576

The research record behind this board's backend-capability table
(`backends/0` in `Cairn.Rockchip.Board`) and the `RKNN` descriptor it
mirrors (`native/cairn_rockchip/src/rknn.rs`). The QNN half of the same
research, and the backend seam both sit behind, are core's
[docs/npu-backends.md](https://github.com/bbangert/cairn-nvr/blob/main/docs/npu-backends.md).

Measured on a Home Assistant Green (RK3566), 2026-11-15 to 2026-11-18; the
raw logs behind it are kept outside this repository. Figures still marked as
claims are third parties' and are in no profile.

## Decode: the kernel flavor decides

Which video decoder Cairn can open depends on the kernel flavor the board's
image runs and on which ffmpeg it ships, and the two Rockchip SoCs answer
differently enough that findings do not transfer between them (D-M6,
confirmed by spike 0.4):

| board | kernel flavor | decode driver | reachable from Cairn's ffmpeg as | `decoder:` value |
|---|---|---|---|---|
| RK3576 | mainline 7.0+ | `rkvdec` VDPU383, V4L2 **stateless** (request API) | not by `h264_v4l2m2m`; unsettled | none: no profile, not measured |
| RK3566 | vendor BSP 6.1 | MPP, `/dev/mpp_service` | only in the `ffmpeg-rockchip` fork (`h264_rkmpp`) | `rkmpp`, this board's own MPP binding, not ffmpeg |

The NPU half forks less than the decode half does: RK3566 and RK3576 share one
runtime (`librknnrt`), one converter (rknn-toolkit2, only `--target` differs)
and one artifact format — so a second Rockchip SoC should fork on decode
driver, kernel flavor and model budget, and reuse the RKNN backend. That is
an expectation: the backend has run on RK3566 only. (RK3566's 1 TOPS NPU is
RKNPU2-family, *not* the older rknpu1/`librknn_api` line — an earlier plan
label said otherwise.)

`rk3566-tier1` names `rkmpp`, which this board's `auto` never picks, and a
profile that names a decoder gets it or refuses ([profiles.md](profiles.md)).
On RK3566 the surrounding work was plumbing rather than silicon. The add-on's
six device nodes all open (`/dev/mpp_service`, `/dev/rga`, `/dev/dri/card1`
and `renderD129` — the NPU is a DRM device on this kernel, not `/dev/rknpu` —
and two dma-heaps), but MPP also needs the device tree: with `/sys/firmware`
masked, as a container has it by default, MPP matches no SoC and every
decoder open fails, while RKNN opens in the same container and hides the
cause. The add-on sets `devicetree: true` for it.

## RKNN

- Compilation is host-side Python only: rknn-toolkit2 (x86) converts
  ONNX→.rknn, INT8 (calibration dataset required) or FP16; INT4 on rk3576
  only. No dynamic shapes. YOLOv5-v11 officially covered by the model zoo (v8/v11
  as exports with the head cut before its box decode — the nine-tensor
  layout `tools/rknn-export`'s split export emits). **Trialed with
  rknn-toolkit2 and librknnrt 2.3.2:** in INT8 on the RK3566, yolox_s,
  yolox_tiny and both split heads (`yolov8split`, `yolo26split`) run and
  detect. In INT8 in the toolkit's simulator, the stock Ultralytics whole
  heads and the end-to-end head detect nothing (one output scale zeroes the
  scores) and rfdetr's scores are unusable. In FP16 on the board, the
  end-to-end head's boxes are wrong and rfdetr crashes the runtime (no
  GridSample op); Cairn's backend opens INT8 artifacts only —
  [`tools/rknn-export/README.md`](../tools/rknn-export/README.md).
- Runtime from Rust: the crate opens librknnrt itself with `dlopen`
  (`rknn/sys.rs`, written against the 2.3.2 API); rknpu2-rs is not used. No
  async and no batching — one inference at a time per context. Contexts of
  one model do overlap (measured, below); Cairn's engine holds one. No
  perf-mode API equivalent to QNN's, and none was needed: every budget was
  taken under the governors as found (`schedutil`, `rknpu_ondemand`).
  `rknn_set_core_mask` is not bound: nothing would set it.
- Frames in and out are copies. Input goes through `rknn_inputs_set`.
  Importing the scaler's dma-buf instead (`rknn_set_io_mem`) gave
  bit-identical outputs over 200 frames and needed no sync, but saved about
  1 ms a sample, some 2 %, so the shipped chain does not use it. Output comes
  back as floats from `rknn_outputs_get`: a native output buffer read back at
  cosine 0.05 under every layout tried
  ([`tools/rknn-export/README.md`](../tools/rknn-export/README.md)).
- Per-board reality:
  - rk3566 (~1 TOPS): on the HA Green, in the pipeline with one context,
    INT8 yolox_tiny@416 holds nine cameras at 2 fps (16.57 passes/s carried,
    46.8 ms a pass) and yolox_s@640 four (7.36, 120.0 ms) —
    `rk3566-tier1`'s budgets are those camera counts at the loader's
    1.875/s floor rate, 16.875 and 7.5. A third to just over half
    of each pass is CPU time in the calling thread; one context leaves the
    NPU 40-51 % busy at saturation, and a second context of the same model
    adds 57 % (tiny@416) or 34 % (s@640):
    [`docs/evidence/rk3566-capacity-20261118.md`](evidence/rk3566-capacity-20261118.md).
  - rk3576 (6 TOPS, INT4-capable): not measured. No profile ships for it.
- **Preprocessing: no Rust crate for RGA** (Rockchip's 2D accelerator for
  resize/color-convert, librga C API, dma_fd zero-copy), so the crate
  binds librga by hand (`rga.rs`); `rkmpp` scales there. On the Green
  (librga 1.10.5, RGA2) a 1080p picture scales to 416² or 640² in 4.4-4.9 ms. librga
  converts with BT.601 and takes full or limited range from the destination
  buffer's format, not from the stream, so the binding picks it per picture
  from `mpp_frame_get_color_range`; the first build did not, and expanded
  full-range video as if it were limited.

## Known-broken combinations → profile validation rules

1. RKNN backend + a family whose conversion was not measured to detect
   (stock whole heads, the end-to-end head, rfdetr) → reject unless the
   profile declares `experimental: true`, and warn with the reason when it
   does.
2. RKNN backend + an artifact whose input is not NHWC INT8 (an FP16 build)
   → refused by the backend when it opens the model. Config load cannot see
   inside the file.

That RKNN runs one inference at a time per context is not a rule: no profile
field can ask for async or batching, so it is a sizing fact
([profiles.md](profiles.md)).

## Behind the backend seam

RKNN is a second backend behind the stage library's `Detector::open`/`detect`
seam, registered from this repository's crate. It takes no per-backend
session options through the seam: its one setting, the runtime's path, is a
NIF argument or `CAIRN_RKNN_LIBRARY`. A profile's `model:` names its artifact
under `rknn:`.

## Sources

- https://github.com/airockchip/rknn_model_zoo
- https://github.com/boundarybitlabs/rknpu2-rs
- https://docs.ultralytics.com/integrations/rockchip-rknn
- https://github.com/airockchip/librga
- https://bliiot.com/info-detail/rk3576-deep-dive-6-tops-npu-meets-unbeatable-cost-performance-in-aiot
- Frigate discussions (rk3566 real-world inference times)
