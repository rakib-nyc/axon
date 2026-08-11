#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)
echo "=== SUSTAIN VARIANCE: 5 seeds x NSTIM {24,36,42,48,60} ==="
for n in 24 36 42 48 60; do
  vals=(); mars=(); dens=()
  for sd in 0 1 2 3 4; do
    out=$("$BIN" "${C[@]}" stim=$n epochs=40 seed=$sd wave=1 wavet=35 deep=1 2>&1)
    a=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{print $5; exit}')
    l=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{v=$5}END{print v}')
    m=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $5}')
    w=$(print -r -- "$out"|grep 'within-assembly :'|grep -o '= [0-9.]*%'|grep -o '[0-9.]*')
    vals+=($(python3 -c "print(f'{$l/$a:.3f}')")); mars+=($m); dens+=($w)
  done
  python3 -c "
import statistics as st
v=[float(x) for x in '${vals}'.split()]
m=[float(x) for x in '${mars}'.split()]
d=[float(x) for x in '${dens}'.split()]
print(f'  NSTIM=$n  sustain mean {st.mean(v):.3f} sd {st.stdev(v):.3f} range [{min(v):.3f},{max(v):.3f}]  |  margin mean {st.mean(m):+.3f} sd {st.stdev(m):.3f}  |  within% mean {st.mean(d):.2f} sd {st.stdev(d):.2f}')
print(f'     raw sustain: {v}')"
done
