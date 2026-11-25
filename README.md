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

CI (`.github/workflows/ci.yml`) runs all of the above,
`tools/check-core-ref.sh` and `tools/check-release.sh`, in the image core's Dockerfile at `CORE_REF`
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

`.github/workflows/image.yml` builds it on pull requests, on dispatch and
on a `v*` tag, and uploads it as the `cairn-rockchip-image` artifact, a
`docker load` tarball for a bench board. Only a tag pushes it (see
Releasing). Before the build,
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

## Releasing and installing

`cairn_rockchip/config.yaml` is the add-on vagus's store reads from this
repository (`repository.yaml` names the repository). The store offers the
`version:` that file has on `main` and pulls
`ghcr.io/bbangert/cairn-rockchip-aarch64:<version>` anonymously, so the
package must be public (set once, after the first push).

To release, change the `version:` in `mix.exs` and in
`cairn_rockchip/config.yaml` to the same value in one commit on `main`, and
push the tag `v<version>` with it. The tag runs `image.yml`, whose publish
job pushes the image that `boot` booted under qemu, not a rebuild, as
`<version>` (no `latest`). Nothing orders the two refs for the store: it
offers the new version from its first reload after the commit is on `main`,
and the image exists some ten minutes after the tag is pushed; an install
in between fails to pull. The same holds for 0.1.0 until its tag is pushed:
no image of this repository has been published yet. Never bump the add-on's `version:` in a
commit that is not about to be tagged.

Two checks refuse a release, both `tools/check-release.sh`. On pull requests
and pushes to `main` (`ci.yml`) it fails when the add-on's `version:` and
`mix.exs`'s differ, or when the repository tracks any other
`config.yaml`/`config.yml`/`config.json` (the store would read it as an
add-on). Given the tag, as the first step of `image.yml`, it also fails when
the tag is not `v` plus that version or `CORE_REF` is a commit sha, before
anything is built. A tag whose boot job fails publishes nothing.

To install, add `https://github.com/bbangert/cairn-rockchip` as a repository
in the vagus store, without a `#branch` (the store derives the add-on's
slug from the URL, so another URL installs a second add-on with its own
empty data), then install "Cairn NVR (Rockchip)"; see
`cairn_rockchip/README.md`. The store downloads the whole default-branch
archive and caps it at 32 MiB, so no models or large fixtures belong in git.

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
