#!/usr/bin/env bash
# image-smoke.sh IMAGE: boots the built image (arm64; under qemu off an arm64
# host) and asserts what holds without a Rockchip SoC: the loader attached
# the board directory the image names, the board NIF loaded, a config naming
# the shipped profile rk3566-tier1 loads, the probe and the profile's CPU
# twins are where the profile and the host look, and the node is still up a
# while later. MPP, RGA and librknnrt are opened only by a decoder or an
# engine, so the missing hardware and librknnrt are not a failure. Needs docker; writes its
# log and the mounted config and data under WORK (default: a fresh temporary
# directory).
set -uo pipefail

image=${1:?usage: image-smoke.sh IMAGE}
work=${WORK:-$(mktemp -d)}
mkdir -p "$work/config" "$work/data"
log=$work/image-smoke.log
: >"$log"
name=cairn-rockchip-smoke

say() { echo "[image-smoke] $*" | tee -a "$log"; }
fail() {
  say "FAIL: $*"
  docker logs "$name" 2>&1 | tail -60 | tee -a "$log"
  docker rm -f "$name" >/dev/null 2>&1
  exit 1
}
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

cat >"$work/config/config.yml" <<'EOF'
plugins:
  det:
    profile: rk3566-tier1
    allow_experimental: true
EOF

# The artifact store off: it would download librknnrt and the profile's
# rungs, which nothing here can run.
docker rm -f "$name" >/dev/null 2>&1
docker run -d --name "$name" --platform linux/arm64 -p 4000:4000 \
  -v "$work/config:/config" -v "$work/data:/data" \
  -e ELIXIR_ERL_OPTIONS="-cairn artifacts false" \
  "$image" >>"$log" 2>&1 || fail "docker run"

# Emulated, the boot takes minutes rather than seconds.
code=""
for _ in $(seq 1 150); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:4000/ || true)
  [ "$code" = 200 ] && break
  [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ] || fail "the container exited"
  sleep 4
done
[ "$code" = 200 ] || fail "UI never served 200 (last: $code)"
say "PASS: UI serves 200"

expect() { # name expression expected
  # The last line: the VM without libsctp opens with an ESOCK warning.
  got=$(docker exec "$name" bin/cairn rpc "IO.puts(inspect(($2)))" 2>&1 | tail -n 1) ||
    fail "$1: rpc failed: $got"
  [ "$got" = "$3" ] || fail "$1: got $got, want $3"
  say "PASS: $1: $got"
}
shell() { docker exec "$name" sh -c "$1" >>"$log" 2>&1; }

expect "board" 'Cairn.Board.module()' 'Cairn.Rockchip.Board'
expect "board application, permanent" \
  ':application.info()[:started] |> List.keyfind(:cairn_rockchip, 0)' '{:cairn_rockchip, :permanent}'
expect "board priv from CAIRN_BOARD_DIR" \
  ':code.priv_dir(:cairn_rockchip) |> to_string() |> String.starts_with?(System.get_env("CAIRN_BOARD_DIR") <> "/")' \
  'true'
expect "board NIF loaded" 'Cairn.Rockchip.Native.load_error()' 'nil'
expect "the config runs rk3566-tier1" \
  'Enum.map(Cairn.Config.Server.get().plugin_groups, &{&1.name, &1.profile.name})' \
  '[{"det", "rk3566-tier1"}]'
shell 'command -v cairn-rockchip-probe' || fail "cairn-rockchip-probe is not on the PATH"
shell 'test -s /app/models/yolox_s.onnx && test -s /app/models/yolox_tiny.onnx' ||
  fail "the profile's CPU twins are not in /app/models"
say "PASS: the probe is on the PATH, the twins in /app/models"

# A board application that exits halts the node (it is permanent), so a
# late failure shows as a stopped container.
sleep 60
[ "$(docker inspect -f '{{.State.Running}} {{.RestartCount}}' "$name")" = "true 0" ] ||
  fail "the node did not stay up"
expect "still up a minute later" 'Application.started_applications() |> List.keymember?(:cairn_rockchip, 0)' 'true'

say "all passed; log: $log"
