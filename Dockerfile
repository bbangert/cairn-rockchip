# syntax=docker/dockerfile:1
# The Rockchip add-on image (linux/arm64 only): core's base image, the
# release with no board, plus the MPP and librga userspace, this board built
# as a board directory, its probe on the PATH and the profile's CPU twins.
# Rust and the vendor libraries cross-compile on the build host; the board
# directory is written under emulation, on the toolchain the release was
# built on, and the runtime's checks run on the target platform.
#
# The ARGs but CAIRN_REVISION are required and have no default:
# tools/check-toolchain.sh derives them from CORE_REF and checks them against
# the base image's labels, and image.yml passes its output. An unset
# version leaves a FROM naming no image, which fails the build; the contract
# is tested where it is read.
ARG CORE_VERSION
ARG ELIXIR_VERSION
ARG OTP_VERSION
ARG DEBIAN_TAG
# cairn.contract on the base image; the board directory's manifest must match.
ARG CAIRN_CONTRACT
# This repository's commit, for the image's revision label: the base's label
# names core's.
ARG CAIRN_REVISION=

FROM ghcr.io/bbangert/cairn-nvr-base-aarch64:${CORE_VERSION} AS base

# ---------------------------------------------------------------------------
# The board NIF and its probe, cross-built against core's crates at CORE_REF
# (Cargo.lock, --locked). Runs on the build platform, links for the target,
# as core's own crates are built for its image. The Rust is the one the
# board's CI gates run on.
FROM --platform=$BUILDPLATFORM rust:1.100.0-trixie AS board-native
ARG TARGETARCH

# Trixie's FFmpeg is 7.1, the rsmpeg feature cairn-detect binds and the
# sonames the base image's runtime carries; rusty_ffmpeg's build script
# pkg-configs every libav*/libsw* library, so all their dev packages.
RUN <<'EOS'
set -eux
[ "$TARGETARCH" = "arm64" ] \
  || { echo "this image is linux/arm64 only (TARGETARCH=$TARGETARCH)" >&2; exit 1; }
dpkg --add-architecture arm64
apt-get update
apt-get install -y --no-install-recommends \
  gcc-aarch64-linux-gnu libc6-dev-arm64-cross pkg-config clang file \
  libavcodec-dev:arm64 libavformat-dev:arm64 libavutil-dev:arm64 \
  libswscale-dev:arm64 libswresample-dev:arm64 libavfilter-dev:arm64 \
  libavdevice-dev:arm64
rustup target add aarch64-unknown-linux-gnu
rm -rf /var/lib/apt/lists/*
EOS

WORKDIR /src/native/cairn_rockchip
COPY native/cairn_rockchip ./
# The environment overrides the crate's .cargo/config.toml, which points at
# an x86 /opt/ffmpeg7 and bakes its rpath: the target-specific rustflags
# replace `[build] rustflags`, and a set FFMPEG_PKG_CONFIG_PATH wins over
# `[env]`. The arm64 FFmpeg is on the default search path, so no rpath.
RUN <<'EOS'
set -eux
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS=""
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export FFMPEG_PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig
export PKG_CONFIG_ALLOW_CROSS=1
cargo build --release --locked --target aarch64-unknown-linux-gnu --features ort-load-dynamic
out=target/aarch64-unknown-linux-gnu/release
# Apart: rockchip/ goes onto the PATH whole, rockchip-nif/ into the board's
# priv/native.
mkdir -p /dist/rockchip /dist/rockchip-nif
cp "$out/cairn-rockchip-probe" /dist/rockchip/
cp "$out/libcairn_rockchip.so" /dist/rockchip-nif/
for f in /dist/rockchip/cairn-rockchip-probe /dist/rockchip-nif/libcairn_rockchip.so; do
  file "$f" | grep -q aarch64 || { echo "$f is not aarch64" >&2; exit 1; }
done
EOS

# ---------------------------------------------------------------------------
# Rockchip vendor userspace, cross-built from pinned source: MPP (hardware
# video codec API; Apache-2.0 + MIT) and librga (RGA 2D engine API;
# Apache-2.0). librknnrt is not built, fetched or copied here: its licence
# does not permit redistribution, so `Cairn.Artifacts` downloads it on the
# board at run time.
#
# The crate's src/mpp/sys.rs and src/rga/sys.rs are ABI-pinned to exactly
# these sources: MPP at rockchip-linux/mpp c08762ebfadeb4e986d2fed993bc7a54862d3ebe
# and librga 1.10.5 (JeffyCN/mirrors 57a1067a246c71fa6c9a355d1668884fda155dd5).
# A pin bump re-derives those files' layout assertions first.
FROM --platform=$BUILDPLATFORM debian:${DEBIAN_TAG}-slim AS rockchip-libs
ARG TARGETARCH

# g++ for MPP too, although the library is C: its CMakeLists adds -fPIC and
# -march only once it has identified the C++ compiler as GNU, and the link
# fails on non-PIC objects without it. librga carries its own copy of the
# DRM uapi headers and issues the ioctls itself, so no libdrm is needed.
RUN <<'EOS'
set -eux
[ "$TARGETARCH" = "arm64" ] || exit 0
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl cmake meson ninja-build \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross
rm -rf /var/lib/apt/lists/*
EOS

RUN <<'EOS'
set -eux
# Off arm64 the stage is these empty directories: the runtime stage refuses
# that platform before it copies from here.
mkdir -p /opt/rockchip/lib /opt/rockchip/licenses /dist/models
[ "$TARGETARCH" = "arm64" ] || exit 0
mpp_rev=c08762ebfadeb4e986d2fed993bc7a54862d3ebe
curl -fsSL -o /tmp/mpp.tar.gz \
  https://github.com/rockchip-linux/mpp/archive/${mpp_rev}.tar.gz
echo "82da3d7ea5926021d48fcaf56cde2e6cb39c3eaad11e06f3cf2a9bfbb3133852 /tmp/mpp.tar.gz" | sha256sum -c -
curl -fsSL -o /tmp/librga.tar.gz \
  https://github.com/JeffyCN/mirrors/archive/57a1067a246c71fa6c9a355d1668884fda155dd5.tar.gz
echo "c539db8a4059d13975e12f596fbe6fd0ba818288cc067300c5e00f3a9f0442e0 /tmp/librga.tar.gz" | sha256sum -c -
mkdir /tmp/mpp /tmp/librga
tar -xzf /tmp/mpp.tar.gz -C /tmp/mpp --strip-components=1
tar -xzf /tmp/librga.tar.gz -C /tmp/librga --strip-components=1

# MPP's ION allocator compiles in Android's ion.h, which carries plain GPLv2
# text. The add-on grants no /dev/ion, so that allocator could never be
# chosen here. The files go too, so a pin bump that brings a reference back
# fails the build.
sed -i '/allocator\/allocator_ion\.c/d' /tmp/mpp/osal/CMakeLists.txt
sed -i '/MPP_BUFFER_TYPE_ION) != 0) ? allocator_ion :/d' /tmp/mpp/osal/mpp_allocator.c
rm /tmp/mpp/osal/allocator/allocator_ion.c /tmp/mpp/osal/allocator/ion.h

# The root path keeps MPP's find_library(mvec) probe in the arm64 sysroot
# instead of the build host's.
cat > /tmp/aarch64.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
cmake -S /tmp/mpp -B /tmp/mpp-out -G Ninja -DCMAKE_TOOLCHAIN_FILE=/tmp/aarch64.cmake \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF
# librockchip_mpp alone: not the legacy librockchip_vpu, the static archive
# or the test utils.
cmake --build /tmp/mpp-out --target rockchip_mpp -j"$(nproc)"
cp -a /tmp/mpp-out/mpp/librockchip_mpp.so* /opt/rockchip/lib/
# A shared object links with undefined symbols, so a leftover reference to the
# removed allocator would only fail at load time on the board.
aarch64-linux-gnu-nm -D /opt/rockchip/lib/librockchip_mpp.so.1 > /tmp/mpp.syms
if grep -q allocator_ion /tmp/mpp.syms; then
  echo "librockchip_mpp still references allocator_ion" >&2
  exit 1
fi

cat > /tmp/aarch64.meson <<'EOF'
[binaries]
c = 'aarch64-linux-gnu-gcc'
cpp = 'aarch64-linux-gnu-g++'
ar = 'aarch64-linux-gnu-ar'
strip = 'aarch64-linux-gnu-strip'

[host_machine]
system = 'linux'
cpu_family = 'aarch64'
cpu = 'aarch64'
endian = 'little'
EOF
meson setup /tmp/librga-out /tmp/librga --cross-file /tmp/aarch64.meson \
  --buildtype=release --prefix=/opt/rockchip --libdir=lib
meson compile -C /tmp/librga-out
DESTDIR=/tmp/librga-stage meson install -C /tmp/librga-out
cp -a /tmp/librga-stage/opt/rockchip/lib/librga.so* /opt/rockchip/lib/

cp /tmp/mpp/LICENSES/Apache-2.0 /opt/rockchip/licenses/mpp-Apache-2.0
cp /tmp/mpp/LICENSES/MIT /opt/rockchip/licenses/mpp-MIT
cp /tmp/librga/COPYING /opt/rockchip/licenses/librga-COPYING
printf '%s\n' \
  'librockchip_mpp is built from rockchip-linux/mpp at commit' \
  "$mpp_rev with the ION allocator removed:" \
  'osal/allocator/allocator_ion.c and osal/allocator/ion.h are deleted, and' \
  'their references dropped from osal/CMakeLists.txt and osal/mpp_allocator.c.' \
  > /opt/rockchip/licenses/mpp-CHANGES
# Both libraries compile in their own copies of MIT DRM uapi headers, each
# with its own copyright holders, and MIT's notice has to travel with the
# object code. drm.h opens with an author comment, so the cut is the end of
# the comment holding the permission text, not of the first comment.
for f in mpp/osal/linux/drm.h mpp/osal/linux/drm_mode.h \
  librga/core/3rdparty/libdrm/include/drm/drm.h \
  librga/core/3rdparty/libdrm/include/drm/drm_mode.h \
  librga/core/3rdparty/libdrm/include/drm/drm_fourcc.h; do
  echo "$f:"
  awk '{print} /Permission is hereby granted/{f=1} f&&/\*\//{exit}' "/tmp/$f"
  echo
done > /opt/rockchip/licenses/drm-uapi-headers-NOTICE
# A pin bump that reshapes a header must not quietly drop its notice.
[ "$(grep -c 'Permission is hereby granted' /opt/rockchip/licenses/drm-uapi-headers-NOTICE)" = 5 ]

# The board crate binds these exact names — every symbol that
# native/cairn_rockchip/src/mpp/sys.rs and src/rga/sys.rs load. Names C
# callers write, such as wrapbuffer_handle or mpp_buffer_get_fd, are macros,
# and importbuffer_fd and improcess have C++-mangled overloads beside the
# extern "C" one, so only an exact unmangled match counts.
need() {
  lib=$1
  shift
  for sym in "$@"; do
    aarch64-linux-gnu-nm -D --defined-only "$lib" | awk '{print $NF}' | grep -qxF "$sym" \
      || { echo "$lib does not export $sym" >&2; exit 1; }
  done
}
need /opt/rockchip/lib/librga.so.2 \
  importbuffer_fd releasebuffer_handle wrapbuffer_handle_t \
  improcess imfill_t imStrError_t querystring
need /opt/rockchip/lib/librockchip_mpp.so.1 \
  mpp_create mpp_init mpp_destroy \
  mpp_packet_init mpp_packet_set_pts mpp_packet_set_eos mpp_packet_deinit \
  mpp_frame_get_width mpp_frame_get_height mpp_frame_get_hor_stride \
  mpp_frame_get_ver_stride mpp_frame_get_errinfo mpp_frame_get_discard \
  mpp_frame_get_eos mpp_frame_get_info_change mpp_frame_get_fmt \
  mpp_frame_get_color_range \
  mpp_frame_get_pts mpp_frame_get_buf_size mpp_frame_get_buffer mpp_frame_deinit \
  mpp_buffer_get_fd_with_caller mpp_buffer_group_get mpp_buffer_group_put \
  mpp_buffer_group_clear mpp_buffer_group_limit_config
EOS

# FP32 YOLOX exports (Apache-2.0). The two Megvii published are the CPU
# twins of rk3566-tier1's rungs, whose `.rknn` files the release fetches at
# first start; the third is derived from the tiny file (below).
RUN <<'EOS'
set -eux
[ "$TARGETARCH" = "arm64" ] || exit 0
base=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
curl -fsSL -o /dist/models/yolox_tiny.onnx "$base/yolox_tiny.onnx"
echo "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7 /dist/models/yolox_tiny.onnx" | sha256sum -c -
curl -fsSL -o /dist/models/yolox_s.onnx "$base/yolox_s.onnx"
echo "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 /dist/models/yolox_s.onnx" | sha256sum -c -
# The twin of tiny at 640: the tiny release file with its input size
# rewritten (tools/rknn-export), published beside the rungs. No shipped
# rung names it: one context holds as many cameras on yolox_s
# (docs/evidence/rk3566-capacity-20261118.md). It stays baked because a
# second context may bring the rung back.
curl -fsSL -o /dist/models/yolox_tiny_640.onnx \
  https://github.com/bbangert/cairn-nvr/releases/download/models-rk3566-1/yolox_tiny_640.onnx
echo "6066fa79efe783a554a6f7e13792e0ee9246bc0c7b27d6eddad3f43bdd000a61 /dist/models/yolox_tiny_640.onnx" | sha256sum -c -
EOS

# ---------------------------------------------------------------------------
# The board directory, written by `mix cairn_stage.board` against the .rel of
# the base image this image is FROM. The task refuses an ERTS other than the
# release's, an application the release also ships at another version, and a
# release that ships the board, and writes nothing if anything fails; so the
# board's lock and the release cannot disagree past this stage.
FROM hexpm/elixir:${ELIXIR_VERSION}-erlang-${OTP_VERSION}-debian-${DEBIAN_TAG} AS board-build
ARG CAIRN_CONTRACT

# git for `cairn_stage`, a sparse git dependency at CORE_REF.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates git && rm -rf /var/lib/apt/lists/*

WORKDIR /src
ENV MIX_ENV=prod
RUN mix local.hex --force && mix local.rebar --force

# mix.exs reads CORE_REF. --check-locked: a lock deps.get would rewrite is
# not the one CI checked.
COPY CORE_REF mix.exs mix.lock ./
RUN mix deps.get --only prod --check-locked && mix deps.compile
COPY lib lib
COPY priv/profiles priv/profiles
# Into the `priv` the task ships and `Cairn.Rockchip.Native` loads from.
COPY --from=board-native /dist/rockchip-nif/ priv/native/
RUN mix compile --warnings-as-errors
COPY --from=base /app/releases /core/releases
# A NIF or profile missing from the directory would show only on the board,
# as rknn refused at preflight or an unknown-profile config error; a contract
# other than the label's, only at boot, as the loader's refusal.
RUN <<'EOS'
set -eux
test -n "$CAIRN_CONTRACT"
vsn=$(cut -d' ' -f2 /core/releases/start_erl.data)
mix cairn_stage.board --core-rel "/core/releases/$vsn/cairn.rel" --out /board
grep -qxF "{board, 'Elixir.Cairn.Rockchip.Board'}." /board/board.manifest
grep -qxF "{contract, $CAIRN_CONTRACT}." /board/board.manifest
test -f /board/lib/cairn_rockchip-*/priv/native/libcairn_rockchip.so
test -f /board/lib/cairn_rockchip-*/priv/profiles/rk3566-tier1.yml
EOS

# ---------------------------------------------------------------------------
# No CAIRN_QNN_* and no /opt/qnn: this board has no HTP, and runtime.exs
# merges no node qnn facts when the variables are unset.
FROM base AS runtime
ARG TARGETARCH

RUN [ "$TARGETARCH" = "arm64" ] \
  || { echo "this image is arm64 only (TARGETARCH=$TARGETARCH)" >&2; exit 1; }

# No extra packages: the libraries need only glibc, libstdc++ and libgcc_s,
# which the base image has. The ldd check proves it here, on the
# target platform, rather than at the first dlopen on the board.
COPY --from=rockchip-libs /opt/rockchip /opt/rockchip
RUN <<'EOS'
set -eux
echo /opt/ort > /etc/ld.so.conf.d/cairn.conf
echo /opt/rockchip/lib >> /etc/ld.so.conf.d/cairn.conf
ldconfig
for lib in /opt/rockchip/lib/librga.so.2 /opt/rockchip/lib/librockchip_mpp.so.1; do
  ldd "$lib"
  if ldd "$lib" | grep -q 'not found'; then exit 1; fi
done
EOS

COPY --from=rockchip-libs /dist/models/ /app/models/
COPY --from=board-native /dist/rockchip/ /usr/local/bin/
COPY --from=board-build /board /opt/cairn/board
ENV CAIRN_BOARD_DIR=/opt/cairn/board
# The host runs the probe as rknn's canary and CPU baseline and the loader
# loads the NIF at boot, both only on the board, so an unresolved library is
# caught here on the target platform instead.
RUN <<'EOS'
set -eux
for bin in /usr/local/bin/cairn-rockchip-probe /opt/cairn/board/lib/cairn_rockchip-*/priv/native/libcairn_rockchip.so; do
  ldd "$bin"
  if ldd "$bin" | grep -q 'not found'; then exit 1; fi
done
EOS

# After the last RUN, as core does: a RUN below an ARG re-runs when its
# value changes, and the revision changes with every commit. The base's
# cairn.* labels stay: they describe the release this image runs.
ARG CORE_VERSION
ARG CAIRN_REVISION
LABEL org.opencontainers.image.revision=$CAIRN_REVISION \
  cairn.core.version=$CORE_VERSION
