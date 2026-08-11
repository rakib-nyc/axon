#!/bin/zsh
# Build the AXON engine. Requires Xcode Command Line Tools (Metal is compiled at
# runtime from source embedded in axon.mm, so no offline metal compiler is needed).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/build"
clang++ -std=c++17 -fobjc-arc -O2 "$ROOT/src/axon.mm" \
  -framework Metal -framework Foundation -o "$ROOT/build/axon"
echo "built: $ROOT/build/axon"
