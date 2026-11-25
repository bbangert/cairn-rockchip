# cairn-rockchip

The Rockchip board for [Cairn](https://github.com/bbangert/cairn-nvr): MPP
hardware decode and RGA scaling for `decoder: rkmpp`, and the RKNN engine for
`backend: rknn`, as the OTP application `cairn_rockchip` and its NIF. Measured
on the RK3566 of the HA Green; the profile it ships, `rk3566-tier1`, carries
that board's budgets (`docs/evidence/rk3566-capacity-20261118.md`).

## How it attaches

Core's release ships no board. This project is built into a *board
directory*, which the release loads at boot from `CAIRN_BOARD_DIR`; it is not
rebuilt for it. The rules a board follows, and why each exists, are core's
[docs/boards.md](https://github.com/bbangert/cairn-nvr/blob/main/docs/boards.md).

`CORE_REF` names the core version tag this board is built against (a full
commit sha while developing, which has no base image), and everything taken
from core must be that commit: `cairn_stage` (the board contract) as a
sparse Mix git dependency, `cairn-detect` and `cairn-ort` as Cargo git
dependencies (`tag =` for a tag, `rev =` for a sha), and the release the
directory attaches to. `mix.exs` reads `CORE_REF`; Cargo cannot, so
`tools/check-core-ref.sh` checks that `Cargo.toml`, `Cargo.lock` and
`mix.lock` name it and lock one commit, and that `Cargo.lock` holds one copy
of each core crate. When `CORE_REF` moves, seed `mix.lock` from core's
`mix.lock` at the new ref before `mix deps.get`: the directory task refuses a
dependency the release also ships at another version.

## Building and testing

Erlang and Elixir are the versions `mise.toml` pins, which must be the
release's: the directory task refuses another ERTS.

```sh
mix deps.get
mix compile --warnings-as-errors
mix format --check-formatted
mix test
mix dialyzer
```

`mix test` loads `priv/native/libcairn_rockchip.so` if it is there and skips
the checks that need it if not. The crate, from `native/cairn_rockchip/`:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
mkdir -p ../../priv/native && cp target/release/libcairn_rockchip.so ../../priv/native/
```

Run cargo from that directory: its `.cargo/config.toml` points the build at
an FFmpeg 7.1 under `/opt/ffmpeg7`, which `cairn-detect` binds through
`rsmpeg`. For a distro's FFmpeg 7.1 set `FFMPEG_PKG_CONFIG_PATH` to its
pkg-config directory and `RUSTFLAGS=""` (no rpath). An aarch64 build for
the image takes `--features ort-load-dynamic` (`Dockerfile`).

`tools/board-test.sh` builds the NIF and the board directory against a
core release and boots the release with it attached, on x86: the NIF opens
librknnrt, MPP and librga only when an engine or decoder is opened. Give it
`REL=<built release>` or `CAIRN_CORE_PATH=<core checkout>`, where it builds
the release. `tools/attached-test.sh` (needs `CAIRN_CORE_PATH`) runs
`test/attached`, core's application with this board attached, in the core
checkout's test environment.

CI (`.github/workflows/ci.yml`) runs all of the above and
`tools/check-core-ref.sh`, in the image core's Dockerfile at `CORE_REF`
builds the release in, and fails when `mise.toml` names another toolchain.
`.github/workflows/nightly.yml` repeats the build and boot against core's
`main` instead of `CORE_REF`; it is advisory and gates nothing.

`tools/rknn-export/` builds and scores the `.rknn` rungs on a host
(its README); `tools/board-pipeline/` and `docs/bench.md` are the capacity
cells run on the board.

## The image

`Dockerfile` builds the linux/arm64 add-on image `FROM`
`ghcr.io/bbangert/cairn-nvr-base-aarch64:<version>`, the version being
`CORE_REF` without its `v` (`tools/core-ref.sh`). It adds the MPP and librga
userspace (cross-built from pinned source, MPP without its ION allocator),
the board directory with `CAIRN_BOARD_DIR` pointing at it, the probe on the
`PATH`, and the CPU twins core's Rockchip image bakes: the profile's two
Megvii exports and the tiny file at 640. It takes its versions as build
arguments and has no defaults for them:

```sh
tools/check-toolchain.sh   # prints version, elixir, otp, debian, contract, …
docker buildx build --platform linux/arm64 \
  --build-arg CORE_VERSION=<version> --build-arg ELIXIR_VERSION=<elixir> \
  --build-arg OTP_VERSION=<otp> --build-arg DEBIAN_TAG=<debian> \
  --build-arg CAIRN_CONTRACT=<contract> -t cairn-rockchip-aarch64:dev .
```

`.github/workflows/image.yml` builds it on pull requests and on dispatch,
never pushes it, and uploads it as the `cairn-rockchip-image` artifact, a
`docker load` tarball for a bench board. Before the build,
`tools/check-toolchain.sh` holds the base image's labels to this repository:
`cairn.elixir.version` and `cairn.otp.version` to `mise.toml`,
`cairn.contract` to the contract `cairn_stage` has at `CORE_REF`, and
`org.opencontainers.image.revision` to the commit the locks hold. Then three
guards stand between the board and a release it does not fit:

1. **The loader** (core, at every boot) refuses a board directory whose
   contract is not the release's, naming both numbers.
2. **The image boots** (`tools/image-smoke.sh`, the `boot` job, under qemu):
   the loader attaches the board, its NIF loads, a config naming
   `rk3566-tier1` loads, and the node is still up a minute later. No NPU,
   MPP or RGA is touched.
3. **The release's `.rel`** (in the build): `mix cairn_stage.board` writes
   the board directory against the `.rel` copied from the base image, and
   refuses another ERTS or an application the release ships at another
   version; the manifest's contract must also equal the label's.

## Developing against a core checkout

With core checked out beside this repository as `../cairn-nvr`:

- **Mix**: `CAIRN_CORE_PATH=../cairn-nvr` switches `cairn_stage` to the
  checkout's `libs/cairn_stage`.
- **Cargo**: put the patch in `.cargo/config.toml` at this repository's
  root, which git ignores. Cargo merges it with the crate's own committed
  config, and its paths are relative to this repository's root:

  ```toml
  [patch."https://github.com/bbangert/cairn-nvr.git"]
  cairn-detect = { path = "../cairn-nvr/plugins/cairn-detect" }
  cairn-ort = { path = "../cairn-nvr/plugins/cairn-ort" }
  ```

  The patch rewrites `Cargo.lock` to drop the git sources, and `--locked`
  then fails. Restore the lock before committing (`git checkout
  native/cairn_rockchip/Cargo.lock`); `tools/check-core-ref.sh` refuses a
  lock with either core crate off the git source.

Neither switch may reach a committed lock.

## Not in this repository

Rockchip's runtime and the models are fetched, not carried:

- **librknnrt** (2.3.2) is not redistributable. Core's artifact store
  downloads it at first start from Rockchip's `rknn-toolkit2` repository, at
  the commit and sha256 `Cairn.Rockchip.Board.artifacts/1` pins, and points
  `CAIRN_RKNN_LIBRARY` at it.
- **The `.rknn` rungs** of `rk3566-tier1` are fetched the same way from the
  `models-rk3566-1` release of cairn-nvr, each pinned by its `sha256:` in
  `priv/profiles/rk3566-tier1.yml`. Their CPU twins and labels are named
  relative to the node's working directory (`models/`). Core's base image
  provides the labels and not the twins: an image for this board adds
  Megvii's two YOLOX ONNX exports (`tools/rknn-export/README.md`, Apache
  rungs).
- No `rknn_api.h` or other Rockchip header, and no MPP or librga source: the
  NIF transcribes the calls it makes and opens librknnrt, MPP and librga with
  `dlopen` at run time; MPP and librga are the board's userspace.
- No weights, ONNX exports or `.rknn` files; `test/support/fixtures/models/`
  holds text stand-ins named like models.

## Licence

Apache-2.0; see `LICENSE`.
