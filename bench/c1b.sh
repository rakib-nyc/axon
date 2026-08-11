#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
BASE=(nand=4 target=64 k=24 dthr=1 mode=1 T=20 clampsteps=10 biasup=1 inbr=4 stimcols=64)
run() {
  local label=$1; shift
  local out=$("$BIN" "${BASE[@]}" "$@" 2>&1)
  local sps=$(print -r -- "$out" | grep -o '\-> *[0-9]* steps/s' | head -1 | grep -o '[0-9]*')
  local wd=$(print -r -- "$out" | grep -o 'wdens [0-9.]*%' | tail -1 | grep -o '[0-9.]*')
  local jac=$(print -r -- "$out" | grep 'mean across cue pairs' | grep -o '[0-9.]*$')
  local core=$(print -r -- "$out" | grep -o 'cues) [0-9]* ([0-9.]*%' | grep -o '[0-9.]*%')
  local ratio=$(print -r -- "$out" | grep '^  ep  40' | grep -o 'LTD/LTP *[0-9.]*' | grep -o '[0-9.]*$')
  local r75=$(print -r -- "$out" | grep 'pattern completion with bias OFF' -A4 | grep '^   75%' | awk '{print $4"/"$5}')
  local r15=$(print -r -- "$out" | grep 'pattern completion with bias OFF' -A9 | grep '^   15%' | awk '{print $4"/"$5}')
  local r05=$(print -r -- "$out" | grep 'pattern completion with bias OFF' -A11 | grep '^    5%' | awk '{print $4"/"$5}')
  printf "%-26s %5s st/s wd=%-6s LTD/LTP=%-6s Jacc=%-6s core=%-6s  75%%:%-13s 15%%:%-13s 5%%:%s\n" \
         "$label" "$sps" "$wd" "${ratio:-0}" "$jac" "$core" "$r75" "$r15" "$r05"
}
echo "=== C1b sweep (self/other at each cue) ==="
run "baseline (fully clamped)"  epochs=40 contrast=0 margingate=0 trainclamp=20
run "gate only"                 epochs=40 contrast=0 margingate=1 trainclamp=20
run "C1b contrastive 1x"        epochs=40 contrast=1 margingate=1 ltdrep=1
run "C1b contrastive 2x LTD"    epochs=40 contrast=1 margingate=1 ltdrep=2
run "C1b contrastive 0.5x LTD"  epochs=40 contrast=1 margingate=1 ltdrep=1 ltdevery=2
run "C1b 1x, no margin gate"    epochs=40 contrast=1 margingate=0 ltdrep=1
run "UNTRAINED control"         epochs=0  contrast=0 margingate=0 trainclamp=20
