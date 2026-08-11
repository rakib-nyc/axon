#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# Explicit argv — no unquoted parameter expansion. zsh does NOT word-split those,
# which silently reverted every tunable to its default in the previous sweep.
BASE=(nand=4 target=64 k=24 dthr=1 mode=1 T=20 clampsteps=10 biasup=1 inbr=4 stimcols=64)

run() {
  local label=$1; shift
  local out
  out=$("$BIN" "${BASE[@]}" "$@" 2>&1)
  local ms=$(print -r -- "$out" | grep -o '[0-9.]* ms/timestep' | head -1)
  local sps=$(print -r -- "$out" | grep -o '\-> *[0-9]* steps/s' | head -1 | grep -o '[0-9]*')
  local wd=$(print -r -- "$out" | grep -o 'wdens [0-9.]*%' | head -1)
  local ct=$(print -r -- "$out" | grep 'cross-talk CLAMPED' | sed 's/.*cross-talk CLAMPED //;s/ .*FREE.*//')
  local asz=$(print -r -- "$out" | grep -o 'assembly size [0-9.]*' | head -1 | grep -o '[0-9.]*')
  local jac=$(print -r -- "$out" | grep 'mean across cue pairs' | grep -o '[0-9.]*$')
  local core=$(print -r -- "$out" | grep -o 'CORE (recalled at all 4 cues) [0-9]* ([0-9.]*%)' | grep -o '([0-9.]*%)')
  local r75=$(print -r -- "$out" | grep '^   75%' | awk '{print $6, $7, $8}')
  printf "%-30s %6s st/s  asm=%-6s xtalk=%-6s Jacc=%-6s core=%-8s  75%%cue self/other/margin: %s\n" \
         "$label" "$sps" "$asz" "$ct" "$jac" "$core" "$r75"
}

echo "=== baseline + C1 ablations (explicit argv) ==="
run "baseline legacy rule"        epochs=40 contrast=0 margingate=0
run "baseline + margin gate"      epochs=40 contrast=0 margingate=1
run "C1 full [5,11]"              epochs=40 contrast=1 margingate=1
run "C1 no margin gate"           epochs=40 contrast=1 margingate=0
run "C1 no clip [0,15]"           epochs=40 contrast=1 margingate=1 bandlo=0 bandhi=15
run "C1 wide band [2,13]"         epochs=40 contrast=1 margingate=1 bandlo=2 bandhi=13
run "C1 narrow band [6,10]"       epochs=40 contrast=1 margingate=1 bandlo=6 bandhi=10
run "UNTRAINED control"           epochs=0  contrast=0 margingate=0
