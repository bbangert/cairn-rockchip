#!/usr/bin/env bash
# check-core-ref.sh: fails unless every cairn-nvr.git ref in the crate's
# Cargo.toml and Cargo.lock is CORE_REF, and the lock holds one cairn-detect
# and one cairn-ort. Cargo cannot read CORE_REF, and a ref spelled two ways
# (rev, short rev, tag) locks a second copy of cairn-detect whose types the
# NIF and cairn-ort then disagree about. mix.exs reads CORE_REF itself.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
crate=$root/native/cairn_rockchip
ref=$(tr -d '[:space:]' <"$root/CORE_REF")
bad=0
fail() { echo "check-core-ref: $*" >&2; bad=1; }

[ -n "$ref" ] || { echo "check-core-ref: CORE_REF is empty" >&2; exit 1; }

# Manifest: every entry naming the repository carries `rev` or `tag` = CORE_REF.
manifest=$(grep -n 'cairn-nvr\.git' "$crate/Cargo.toml" || true)
[ -n "$manifest" ] || fail "Cargo.toml names no cairn-nvr.git dependency"
while IFS= read -r line; do
  [ -n "$line" ] || continue
  got=$(sed -nE 's/.*\b(rev|tag|branch) = "([^"]*)".*/\1=\2/p' <<<"$line")
  [ "$got" = "rev=$ref" ] || [ "$got" = "tag=$ref" ] ||
    fail "Cargo.toml:${line%%:*}: ref ${got:-none}, CORE_REF is $ref"
done <<<"$manifest"

# Lock: source = "git+<url>?rev=<ref>#<commit>" (or ?tag=); the query is what
# the manifest asked for, so it must be CORE_REF too.
sources=$(sed -n 's/^source = "git+\([^"]*cairn-nvr\.git[^"]*\)"$/\1/p' "$crate/Cargo.lock")
[ -n "$sources" ] || fail "Cargo.lock locks nothing from cairn-nvr.git"
while IFS= read -r s; do
  [ -n "$s" ] || continue
  query=${s#*\?}
  query=${query%%#*}
  [ "$query" = "rev=$ref" ] || [ "$query" = "tag=$ref" ] ||
    fail "Cargo.lock: source $s does not name CORE_REF ($ref)"
  # With a sha for CORE_REF, the locked commit is that sha.
  if [[ $ref =~ ^[0-9a-f]{40}$ ]] && [ "${s##*#}" != "$ref" ]; then
    fail "Cargo.lock: source $s is locked at ${s##*#}, not $ref"
  fi
done <<<"$sources"

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

[ "$bad" = 0 ] || exit 1
echo "check-core-ref: Cargo.toml and Cargo.lock name $ref; one cairn-detect, one cairn-ort"
