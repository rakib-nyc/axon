#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# SINGLE CONFIG: C = 73/1/T10, contrast 0.5x LTD, margin gate off
C=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5)
echo "=== FULL CAPACITY SWEEP, single config C (73/1/T10) ==="
printf "%-6s %-5s %6s %7s %8s %8s %8s %9s %8s %8s %-9s %s\n" \
  "stim" "cond" "st/s" "within%" "r75self" "r75oth" "margin" "sustain" "drive6" "asmJacc" "precond" "sum|union|excess"
row(){ local n=$1 ep=$2
  local lbl=$([ $ep -eq 0 ] && echo untr || echo train)
  local clean=$("$BIN" "${C[@]}" stim=$n epochs=$ep 2>&1|grep -o '\-> *[0-9]* steps/s'|head -1|grep -o '[0-9]*')
  local out=$("$BIN" "${C[@]}" stim=$n epochs=$ep deep=1 wave=1 wavet=35 2>&1)
  local w=$(print -r -- "$out"|grep 'within-assembly :'|grep -o '= [0-9.]*%'|grep -o '[0-9.]*')
  local s=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $3}')
  local o=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $4}')
  local m=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $5}')
  local a=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{print $5; exit}')
  local l=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{v=$5}END{print v}')
  local d=$(print -r -- "$out"|grep -A1 'free-phase drive'|tail -1|tr ' ' '\n'|grep '^t6:'|cut -d: -f2)
  local aj=$(print -r -- "$out"|grep 'measured mean pairwise Jaccard'|grep -o '= [0-9.]*'|head -1|grep -o '[0-9.]*')
  local su=$(print -r -- "$out"|grep 'sum of assembly sizes'|sed 's/.*sizes //;s/ -> .*//;s/ | /|/g;s/union //;s/excess //')
  local sr=$(python3 -c "print(f'{$l/$a:.3f}')" 2>/dev/null)
  local pc=$(python3 -c "
d=${d:-0}; s=${s:-0}; o=${o:-1}
print('VALID' if (d>58 and s>o) else ('FAIL:drv' if d<=58 else 'FAIL:s<=o'))" 2>/dev/null)
  printf "%-6s %-5s %6s %7s %8s %8s %8s %9s %8s %8s %-9s %s\n" "$n" "$lbl" "$clean" "$w" "$s" "$o" "$m" "$sr" "${d:-na}" "${aj:-na}" "$pc" "$su"
}
for n in 12 24 36 48 60 74 90 140 200 300; do row $n 40; row $n 0; done
