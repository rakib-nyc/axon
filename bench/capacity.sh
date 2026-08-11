#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# DEFAULT CONFIG: C1b 0.5x LTD, margin gate OFF
BASE=(nand=4 target=64 k=24 dthr=1 mode=1 T=20 clampsteps=10 biasup=1 inbr=4 stimcols=64 \
      contrast=1 margingate=0 ltdrep=1 ltdevery=2)
hdr() { printf "%-5s %-6s %6s %7s %7s %7s %8s %8s %9s %9s %7s %s\n" \
  "stim" "epochs" "st/s" "wdens" "Jacc" "core" "r75self" "r75oth" "asmJacc" "chance" "x/chance" "sum|union|excess"; }
row() {
  local ns=$1 ep=$2 extra=$3
  local out=$("$BIN" "${BASE[@]}" stim=$ns epochs=$ep ${=extra} 2>&1)
  local sps=$(print -r -- "$out"|grep -o '\-> *[0-9]* steps/s'|head -1|grep -o '[0-9]*')
  local wd=$(print -r -- "$out"|grep -o 'wdens [0-9.]*%'|tail -1|grep -o '[0-9.]*')
  local jac=$(print -r -- "$out"|grep 'mean across cue pairs'|grep -o '[0-9.]*$')
  local core=$(print -r -- "$out"|grep -o 'cues) [0-9]* ([0-9.]*%'|grep -o '[0-9.]*%')
  local r75=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $3}')
  local o75=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $4}')
  local aj=$(print -r -- "$out"|grep 'measured mean pairwise Jaccard'|grep -o '= [0-9.]*'|head -1|grep -o '[0-9.]*')
  local ch=$(print -r -- "$out"|grep 'E\[Jaccard\]'|grep -o '= [0-9.]*'|tail -1|grep -o '[0-9.]*')
  local rc=$(print -r -- "$out"|grep 'ratio measured/chance'|grep -o '[0-9.]*$')
  local su=$(print -r -- "$out"|grep 'sum of assembly sizes'|sed 's/.*sizes //;s/ -> .*//;s/ | /|/g;s/union //;s/excess //')
  printf "%-5s %-6s %6s %7s %7s %7s %8s %8s %9s %9s %7s %s\n" \
    "$ns" "$ep" "$sps" "$wd" "$jac" "$core" "$r75" "$o75" "$aj" "$ch" "$rc" "$su"
}
echo "=== CAPACITY SWEEP: default config (C1b 0.5x LTD, gate OFF) ==="
hdr
for n in 12 24 48 60 74 90; do row $n 40 ""; done
echo
echo "=== UNTRAINED CONTROLS ==="
hdr
for n in 12 48 90; do row $n 0 ""; done
