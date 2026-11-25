#!/usr/bin/env bash
# check-core-ref.sh: fails unless the crate's Cargo.toml and Cargo.lock and
# the board's mix.lock all name CORE_REF, and resolve it to one commit. Cargo
# cannot read CORE_REF, and a ref spelled two ways (rev, short rev, tag)
# locks a second copy of cairn-detect whose types the NIF and cairn-ort then
# disagree about, so a tag is spelt `tag =` and a sha `rev =`, everywhere.
# mix.exs reads CORE_REF itself; its lock is checked because a stale one
# builds `cairn_stage` at another commit than the crates. Offline: a tag
# moved upstream is caught by tools/check-toolchain.sh, against the base
# image's revision label.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
crate=$root/native/cairn_rockchip
core=$("$root/tools/core-ref.sh")
ref=$(sed -n 's/^ref=//p' <<<"$core")
case $(sed -n 's/^kind=//p' <<<"$core") in
  tag) want="tag=$ref" ;;
  sha) want="rev=$ref" ;;
esac
bad=0
fail() { echo "check-core-ref: $*" >&2; bad=1; }

# Manifest: every entry naming the repository carries exactly that ref.
manifest=$(grep -n 'cairn-nvr\.git' "$crate/Cargo.toml" || true)
[ -n "$manifest" ] || fail "Cargo.toml names no cairn-nvr.git dependency"
while IFS= read -r line; do
  [ -n "$line" ] || continue
  got=$(sed -nE 's/.*\b(rev|tag|branch) = "([^"]*)".*/\1=\2/p' <<<"$line")
  [ "$got" = "$want" ] ||
    fail "Cargo.toml:${line%%:*}: ${got:-no ref}, want $want (CORE_REF)"
done <<<"$manifest"

# Lock: source = "git+<url>?<query>#<commit>"; the query is what the manifest
# asked for. One commit for every entry, and for a sha, that sha.
sources=$(sed -n 's/^source = "git+\([^"]*cairn-nvr\.git[^"]*\)"$/\1/p' "$crate/Cargo.lock")
[ -n "$sources" ] || fail "Cargo.lock locks nothing from cairn-nvr.git"
while IFS= read -r s; do
  [ -n "$s" ] || continue
  query=${s#*\?}
  query=${query%%#*}
  [ "$query" = "$want" ] || fail "Cargo.lock: source $s does not ask for $want (CORE_REF)"
done <<<"$sources"
commits=$(sed 's/.*#//' <<<"$sources" | sort -u)
[ "$(wc -l <<<"$commits")" = 1 ] || fail "Cargo.lock locks core at more than one commit: $(paste -sd' ' <<<"$commits")"
if [ "${want%%=*}" = rev ] && [ "$commits" != "$ref" ]; then
  fail "Cargo.lock locks core at $commits, not $ref"
fi

for c in cairn-detect cairn-ort; do
  n=$(grep -c "^name = \"$c\"$" "$crate/Cargo.lock" || true)
  [ "$n" = 1 ] || fail "Cargo.lock holds $n $c packages, want 1"
  # A local [patch] of one crate leaves the other on git: each on its own.
  src=$(awk -v n="name = \"$c\"" '$0 == n { f = 1; next } f && /^source = / { print; exit } f && /^$/ { exit }' "$crate/Cargo.lock")
  case $src in
    *cairn-nvr.git*) ;;
    *) fail "Cargo.lock: $c is not from cairn-nvr.git (a local patch?)" ;;
  esac
done

# mix.lock: "cairn_stage": {:git, <url>, "<commit>", [ref: "<ref>", ...]}.
stage=$(grep '^  "cairn_stage": {:git, "https://github.com/bbangert/cairn-nvr.git"' "$root/mix.lock" || true)
if [ -z "$stage" ]; then
  fail "mix.lock does not lock cairn_stage from cairn-nvr.git (CAIRN_CORE_PATH?)"
else
  stage_commit=$(sed -E 's/^[^,]*, [^,]*, "([0-9a-f]{40})".*/\1/' <<<"$stage")
  grep -qF "[ref: \"$ref\"," <<<"$stage" || fail "mix.lock: cairn_stage is not locked for ref $ref"
  [ "$stage_commit" = "$commits" ] ||
    fail "mix.lock locks cairn_stage at $stage_commit, Cargo.lock core at $commits"
fi

[ "$bad" = 0 ] || exit 1
echo "check-core-ref: Cargo.toml, Cargo.lock and mix.lock name $ref ($commits); one cairn-detect, one cairn-ort"
