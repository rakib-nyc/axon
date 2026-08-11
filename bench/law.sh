#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
CORE=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 ltdrep=1 ltdevery=2)
probe() {
  local label=$1; shift
  local cfg=("$@")
  printf "%-34s" "$label"
  for n in 24 40 60 80; do
    local out=$("$BIN" "${CORE[@]}" "${cfg[@]}" stim=$n epochs=40 2>&1)
    local r=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $5}')
    local j=$(print -r -- "$out"|grep 'mean across cue pairs'|grep -o '[0-9.]*$')
    printf " %s:%-8s(J%-5s)" "$n" "$r" "$j"
  done
  echo
}
echo "margin at 75% cue (self - best-other), and cross-cue Jaccard, per NSTIM"
echo "wall = NSTIM where margin crosses zero"
echo
probe "base  bU=73  bD=1 T=20 [pred 37.5]"  biasup=73  biasdown=1 biasdownevery=1 T=20 clampsteps=10
probe "A     bU=146 bD=1 T=20 [pred 74]"    biasup=146 biasdown=1 biasdownevery=1 T=20 clampsteps=10
probe "B     bU=73  bD=.5 T=20 [pred 74]"   biasup=73  biasdown=1 biasdownevery=2 T=20 clampsteps=10
probe "C     bU=73  bD=1 T=10 [R1:74 R2:37]" biasup=73 biasdown=1 biasdownevery=1 T=10 clampsteps=5
probe "D     bU=146 bD=2 T=20 [ratio ctl 37.5]" biasup=146 biasdown=2 biasdownevery=1 T=20 clampsteps=10
