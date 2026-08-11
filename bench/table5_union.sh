#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# Configuration C used throughout the paper.
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 \
   ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)

# Paper Table 5 (tab:union): characterisation of the population visited during free
# running. Five seeds per load. Runtime ~20 min.
echo "=== Table 5: visited-population characterisation ==="
printf "%-22s %8s %8s %9s %10s %11s %10s %14s\n" \
  "condition" "in-asm" "outside" "%outside" "dens(in)" "dens(OUT)" "dens(all)" "mult(OUT)/pop"
for n in 12 36 60; do
  for sd in 0 1 2 3 4; do
    "$BIN" "${C[@]}" stim=$n epochs=40 seed=$sd unionchar=1 2>&1 | grep '^UNIONCHAR'
  done | awk -v n=$n '{a+=$2;b+=$3;c+=$4;d+=$5;e+=$6;f+=$7;g+=$8;k++}
    END{printf "  trained patterns=%-3s %8.0f %8.0f %8.1f%% %9.2f%% %10.2f%% %9.2f%% %7.2f/%.2f\n",
        n,a/k,b/k,100*b/k/((a+b)/k),c/k,d/k,e/k,f/k,g/k}'
done
for sd in 0 1 2; do
  "$BIN" "${C[@]}" stim=36 epochs=0 seed=$sd unionchar=1 2>&1 | grep '^UNIONCHAR'
done | awk '{a+=$2;b+=$3;c+=$4;d+=$5;e+=$6;f+=$7;g+=$8;k++}
  END{printf "  UNTRAINED patterns=36  %8.0f %8.0f %8.1f%% %9.2f%% %10.2f%% %9.2f%% %7.2f/%.2f\n",
      a/k,b/k,100*b/k/((a+b)/k),c/k,d/k,e/k,f/k,g/k}'
