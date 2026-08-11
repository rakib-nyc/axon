#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# Configuration C used throughout the paper.
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 \
   ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)

# Paper Table 4 (tab:gaintraj): loop gain per two-step period across the free run.
# Five seeds per load. Runtime ~15 min.
TMP=$(mktemp)
for n in 12 36 60; do for sd in 0 1 2 3 4; do
  "$BIN" "${C[@]}" stim=$n epochs=40 seed=$sd split=1 wave=1 wavet=35 2>&1 \
   | grep 'free-phase drive  EXCLUSIVE' | sed "s|.*EXCLUSIVE:|D $n $sd |"
done; done > $TMP
python3 - "$TMP" <<'PY'
import sys, statistics as st
rows={}
for ln in open(sys.argv[1]):
    p=ln.split()
    if p and p[0]=='D': rows.setdefault(int(p[1]),[]).append([float(x) for x in p[3:]])
print("=== Table 4: loop gain per two-step window ===")
print(f"{'patterns':>9} " + " ".join(f"t{t}->{t+2}" for t in range(6,28,4)))
for n in sorted(rows):
    out=[]
    for t in range(6,28,4):
        g=[r[t-5+2]/r[t-5] for r in rows[n] if t-5+2 < len(r) and r[t-5]>0]
        out.append(f"{st.mean(g):.3f}" if g else "  -  ")
    print(f"{n:>9} " + "   ".join(out))
PY
rm -f $TMP
