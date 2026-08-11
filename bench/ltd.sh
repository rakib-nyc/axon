#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
B=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 biasup=73 biasdown=1 T=10 clampsteps=5)
echo "=== CONVERSE TEST: can within-assembly density be MOVED? (NSTIM=48) ==="
printf "%-10s %6s %8s %9s %8s %8s %9s %9s %9s\n" "LTD" "st/s" "within%" "between%" "wdens" "r75self" "r75oth" "margin" "sustain"
for cfg in "4 1 0.25x" "2 1 0.5x" "1 1 1x" "1 2 2x" "1 3 3x" "1 4 4x"; do
  set -- ${=cfg}
  clean=$("$BIN" "${B[@]}" ltdevery=$1 ltdrep=$2 stim=48 epochs=40 2>&1|grep -o '\-> *[0-9]* steps/s'|head -1|grep -o '[0-9]*')
  out=$("$BIN" "${B[@]}" ltdevery=$1 ltdrep=$2 stim=48 epochs=40 deep=1 wave=1 wavet=35 2>&1)
  w=$(print -r -- "$out"|grep 'within-assembly :'|grep -o '= [0-9.]*%'|grep -o '[0-9.]*')
  bt=$(print -r -- "$out"|grep 'between         :'|grep -o '= [0-9.]*%'|grep -o '[0-9.]*')
  wd=$(print -r -- "$out"|grep -o 'wdens [0-9.]*%'|tail -1|grep -o '[0-9.]*')
  r=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $3,$4,$5}')
  a=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{print $5; exit}')
  l=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{v=$5}END{print v}')
  su=$(python3 -c "print(f'{$l/$a:.3f}')" 2>/dev/null)
  printf "%-10s %6s %8s %9s %8s %s %9s\n" "$3" "$clean" "$w" "$bt" "$wd" "$r" "$su"
done
echo "--- untrained controls ---"
for cfg in "2 1 0.5x" "1 3 3x"; do
  set -- ${=cfg}
  out=$("$BIN" "${B[@]}" ltdevery=$1 ltdrep=$2 stim=48 epochs=0 deep=1 2>&1)
  w=$(print -r -- "$out"|grep 'within-assembly :'|grep -o '= [0-9.]*%'|grep -o '[0-9.]*')
  r=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $3,$4,$5}')
  printf "  %-8s within %-9s %s\n" "$3" "$w" "$r"
done
