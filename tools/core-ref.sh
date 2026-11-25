#!/usr/bin/env bash
# core-ref.sh: what CORE_REF names, as key=value lines (the format of
# $GITHUB_OUTPUT): `ref`, `kind` (tag or sha), and for a tag the base image
# `version` and `base`. The only place the image version is derived from the
# ref. A sha has no base image: core publishes one per version tag only.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
ref=$(tr -d '[:space:]' <"$root/CORE_REF")

if [[ $ref =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  version=${BASH_REMATCH[1]}
  echo "ref=$ref"
  echo "kind=tag"
  echo "version=$version"
  echo "base=ghcr.io/bbangert/cairn-nvr-base-aarch64:$version"
elif [[ $ref =~ ^[0-9a-f]{40}$ ]]; then
  echo "ref=$ref"
  echo "kind=sha"
else
  # A short sha or a branch would resolve, but to a commit nothing pins.
  echo "core-ref: CORE_REF is '$ref'; want a core tag vX.Y.Z or a full commit sha" >&2
  exit 1
fi
