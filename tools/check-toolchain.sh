#!/usr/bin/env bash
# check-toolchain.sh: fails unless mise.toml names the toolchain core built
# its release on at CORE_REF, and, for a tag, the base image's labels agree:
# cairn.elixir.version and cairn.otp.version with mise.toml (the directory
# task refuses another ERTS), cairn.contract with the `Cairn.Stage.contract/0`
# the board compiles against, and org.opencontainers.image.revision with the
# commit Cargo.lock locks core at (a tag moved since the lock was written).
#
# Prints key=value lines for $GITHUB_OUTPUT: ref, image (the hexpm image to
# build in), elixir, otp, debian, contract, and for a tag version and base.
# Reads core's Dockerfile and stage.ex from GitHub and the labels from ghcr,
# anonymously. Needs curl and jq.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
core=$("$root/tools/core-ref.sh")
key() { sed -n "s/^$1=//p" <<<"$core"; }
ref=$(key ref) version=$(key version) base=$(key base)
bad=0
fail() { echo "${GITHUB_ACTIONS:+::error::}check-toolchain: $*" >&2; bad=1; }
get() { curl -fsSL --retry 3 "$@"; }

raw=https://raw.githubusercontent.com/bbangert/cairn-nvr/$ref
dockerfile=$(get "$raw/Dockerfile")
arg() { sed -n "s/^ARG $1=//p" <<<"$dockerfile"; }
otp=$(arg OTP_VERSION) elixir=$(arg ELIXIR_VERSION) debian=$(arg DEBIAN_TAG)
contract=$(get "$raw/libs/cairn_stage/lib/cairn/stage.ex" | sed -n 's/^  def contract, do: \([0-9]*\)$/\1/p')
[ -n "$otp" ] && [ -n "$elixir" ] && [ -n "$debian" ] || fail "core's Dockerfile at $ref lacks a toolchain ARG"
[ -n "$contract" ] || fail "no contract/0 in core's stage.ex at $ref"

tool() { sed -n "s/^$1 = \"\(.*\)\"\$/\1/p" "$root/mise.toml"; }
mise_otp=$(tool erlang) mise_elixir=$(tool elixir)
# mise names the Elixir build by the OTP major it was compiled for.
[ "$mise_otp" = "$otp" ] && [ "$mise_elixir" = "$elixir-otp-${otp%%.*}" ] ||
  fail "mise.toml pins erlang $mise_otp, elixir $mise_elixir; core's Dockerfile at $ref has OTP '$otp', Elixir '$elixir'"

if [ -n "$base" ]; then
  repo=${base#ghcr.io/}
  repo=${repo%:*}
  token=$(get "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r .token)
  auth=(-H "Authorization: Bearer $token")
  index=$(get "${auth[@]}" -H "Accept: application/vnd.oci.image.index.v1+json" \
    "https://ghcr.io/v2/$repo/manifests/$version")
  digest=$(jq -r '.manifests[] | select(.platform.os == "linux" and .platform.architecture == "arm64") | .digest' <<<"$index")
  [ -n "$digest" ] || { echo "check-toolchain: $base has no linux/arm64 image" >&2; exit 1; }
  config=$(get "${auth[@]}" -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    "https://ghcr.io/v2/$repo/manifests/$digest" | jq -r .config.digest)
  labels=$(get "${auth[@]}" "https://ghcr.io/v2/$repo/blobs/$config" | jq -c .config.Labels)
  label() { jq -r --arg k "$1" '.[$k] // ""' <<<"$labels"; }
  locked=$(sed -n 's/^source = "git+[^"]*cairn-nvr\.git[^#]*#\([0-9a-f]*\)"$/\1/p' \
    "$root/native/cairn_rockchip/Cargo.lock" | sort -u)

  [ "$(label cairn.elixir.version)" = "${mise_elixir%-otp-*}" ] ||
    fail "$base has cairn.elixir.version '$(label cairn.elixir.version)', mise.toml elixir $mise_elixir"
  [ "$(label cairn.otp.version)" = "$mise_otp" ] ||
    fail "$base has cairn.otp.version '$(label cairn.otp.version)', mise.toml erlang $mise_otp"
  [ "$(label cairn.contract)" = "$contract" ] ||
    fail "$base has cairn.contract '$(label cairn.contract)', cairn_stage at $ref has contract $contract"
  [ "$(label org.opencontainers.image.revision)" = "$locked" ] ||
    fail "$base was built from '$(label org.opencontainers.image.revision)', Cargo.lock locks core at $locked"
fi

[ "$bad" = 0 ] || exit 1
echo "check-toolchain: OTP $otp, Elixir $elixir, contract $contract at $ref${base:+; $base agrees}" >&2
cat <<EOF
ref=$ref
image=hexpm/elixir:$elixir-erlang-$otp-debian-$debian
elixir=$elixir
otp=$otp
debian=$debian
contract=$contract
EOF
[ -z "$base" ] || printf 'version=%s\nbase=%s\n' "$version" "$base"
