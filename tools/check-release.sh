#!/bin/sh
# check-release.sh [vX.Y.Z]: the add-on's version is the application's, and
# the store finds exactly one add-on. With a tag, also that the tag is that
# version and that CORE_REF names a core release (a sha has no base image).
# Offline.
#
# The store offers whatever `version:` the default branch holds and pulls
# the image tagged with it, so the two files move together; the check cannot
# know whether that tag's image has been pushed yet.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
config=cairn_rockchip/config.yaml

app=$(sed -n 's/^ *version: "\([^"]*\)",$/\1/p' mix.exs | head -n 1)
addon=$(sed -n 's/^version: "\([^"]*\)"$/\1/p' "$config")
[ -n "$app" ] || { echo "check-release: no version in mix.exs" >&2; exit 1; }
[ "$addon" = "$app" ] ||
  { echo "check-release: $config version is '$addon', mix.exs version is '$app'" >&2; exit 1; }

# The store parses every config.{yaml,yml,json}, at any depth, as an add-on;
# a second one is a phantom add-on or a log line on every reload.
configs=$(git ls-files | grep -E '(^|/)config\.(yaml|yml|json)$' || true)
[ "$configs" = "$config" ] ||
  { echo "check-release: want only $config as an add-on config, tracked: $(echo "$configs" | paste -sd' ' -)" >&2; exit 1; }

if [ $# -ge 1 ]; then
  [ "$1" = "v$app" ] ||
    { echo "check-release: tag is '$1', mix.exs version is '$app' (want v$app)" >&2; exit 1; }
  kind=$("$root/tools/core-ref.sh" | sed -n 's/^kind=//p')
  [ "$kind" = tag ] ||
    { echo "check-release: CORE_REF is '$(tr -d '[:space:]' <CORE_REF)', not a core version tag; the image needs a base image" >&2; exit 1; }
  echo "check-release: tag $1 = mix.exs = $config; CORE_REF $(tr -d '[:space:]' <CORE_REF)"
else
  echo "check-release: $config and mix.exs are at $app; one add-on config"
fi
