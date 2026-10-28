#!/usr/bin/env bash
# Copy the Jetstream segment decoder into DEST and install its pinned deps
# (package-lock.json, `npm ci --ignore-scripts`; the deps are pure JS + wasm, so a
# Mac-built node_modules also runs on the Linux VM). Prints the sha256 of each
# decoder file so a run receipt can name the exact decoder.
#   setup_decoder.sh DEST [SRC]
set -euo pipefail
DEST="${1:?usage: setup_decoder.sh DEST [SRC]}"
SRC="${2:-${DECODER_SRC:-$HOME/Developer/catbird-community-engine-20261018/tools/community-engine/backfill}}"
mkdir -p "$DEST"
for f in archive.mjs decode_segments.mjs snapshot.mjs package.json package-lock.json; do
  cp "$SRC/$f" "$DEST/$f"
done
(cd "$DEST" && npm ci --ignore-scripts --no-audit --no-fund >/dev/null)
if find "$DEST/node_modules" -name '*.node' | grep -q .; then
  echo "native addons found in node_modules; rebuild on the VM instead of shipping these" >&2
  exit 1
fi
(cd "$DEST" && shasum -a 256 archive.mjs decode_segments.mjs snapshot.mjs package-lock.json)
