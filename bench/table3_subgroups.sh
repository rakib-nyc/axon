#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# Configuration C used throughout the paper.
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 \
   ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)

# Paper Table 3 (tab:subgroups): within-assembly density split by shared vs exclusive
# postsynaptic membership. Seed 0. Runtime ~4 min.
echo "=== Table 3: shared vs exclusive members ==="
for n in 12 24 36 48 60; do
  echo "--- patterns=$n ---"
  "$BIN" "${C[@]}" stim=$n epochs=40 split=1 2>&1 | sed -n '/SHARED vs EXCLUSIVE/,/^=====/p' | sed -n '2,4p'
done
echo "--- untrained control, patterns=48 ---"
"$BIN" "${C[@]}" stim=48 epochs=0 split=1 2>&1 | sed -n '/SHARED vs EXCLUSIVE/,/^=====/p' | sed -n '2,4p'
