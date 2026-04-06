#!/usr/bin/env bash
# Downloads h5wasm IIFE build into this directory.
# Run once: bash lib/setup_lib.sh

set -e
cd "$(dirname "$0")"

VERSION="0.10.1"
TGZ="h5wasm-${VERSION}.tgz"

echo "Downloading h5wasm v${VERSION}..."
npm pack "h5wasm@${VERSION}" --pack-destination . > /dev/null 2>&1
tar xzf "$TGZ" package/dist/iife/h5wasm.js --strip-components=3
rm -rf "$TGZ" package
echo "Done. h5wasm.js ($(du -h h5wasm.js | cut -f1)) ready."
