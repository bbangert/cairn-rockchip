#!/usr/bin/env bash
# attached-test.sh: runs test/attached, core's application with this board
# attached, in the test environment of the core checkout CAIRN_CORE_PATH names.
# The file is passed to that checkout's `mix test` by path and not copied
# there; it leans on the checkout's test_helper.exs and test/support stubs
# (see the file's header).
#
# The board directory is BOARD_DIR, or else tools/board-test.sh builds one
# in WORK/board (with REL, CARGO and the rest as it takes them). Writes what
# board-test.sh does, plus the checkout's deps, _build/test, tmp/ and the
# test database, whose event and track rows its test_helper.exs deletes.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
core=${CAIRN_CORE_PATH:?set CAIRN_CORE_PATH to a core checkout}
core=$(cd "$core" && pwd)
dir=${BOARD_DIR:-}

if [ -z "$dir" ]; then
  work=${WORK:-$(mktemp -d)}
  WORK=$work CAIRN_CORE_PATH=$core "$root/tools/board-test.sh"
  dir=$work/board
fi
[ -f "$dir/board.manifest" ] || { echo "attached-test: $dir is not a board directory" >&2; exit 1; }
dir=$(cd "$dir" && pwd)

cd "$core"
export LC_ALL=${LC_ALL:-C.UTF-8}
mix deps.get
CAIRN_ROCKCHIP_BOARD_DIR=$dir exec mix test --only rockchip_board \
  "$root/test/attached/rockchip_release_test.exs"
