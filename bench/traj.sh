#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)
for n in 12 36 60; do for sd in 0 1 2 3 4; do
  out=$("$BIN" "${C[@]}" stim=$n epochs=40 seed=$sd split=1 wave=1 wavet=35 2>&1)
  d=$(print -r -- "$out"|grep 'free-phase drive  EXCLUSIVE'|sed 's/.*EXCLUSIVE://')
  o=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{printf "%s ",$5}')
  echo "D $n $sd $d"
  echo "O $n $sd $o"
done; done
