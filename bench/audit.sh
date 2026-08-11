#!/bin/zsh
# Resolve repo root from this script location; no absolute paths.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/axon"
[ -x "$BIN" ] || { echo "build first: $ROOT/build.sh"; exit 1; }
# VALIDITY PRECONDITION for the sustain ratio (stated in methods, applied to all points):
#   (a) mean free-phase drive to assembly members > 2x the untrained floor (29)  => > 58
#   (b) r75 self > r75 other
#   (c) the matched untrained control does not sustain
B=(nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 biasup=73 biasdown=1 T=10 clampsteps=5)
printf "%-30s %8s %8s %8s %8s %9s  %s\n" "point" "sustain" "drive_t5" "r75self" "r75oth" "margin" "VERDICT"
audit(){ local lbl=$1; shift
  local out=$("$BIN" "${B[@]}" "$@" deep=1 wave=1 wavet=35 2>&1)
  local a=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{print $5; exit}')
  local l=$(print -r -- "$out"|sed -n '/WAVE:/,/^=====/p'|awk '$2=="FREE"{v=$5}END{print v}')
  local d=$(print -r -- "$out"|grep -A2 'free-phase drive'|tail -1|tr ' ' '\n'|grep '^t5:'|cut -d: -f2)
  local s=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $3}')
  local o=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $4}')
  local m=$(print -r -- "$out"|grep 'pattern completion with bias OFF' -A4|grep '^   75%'|awk '{print $5}')
  local su=$(python3 -c "print(f'{$l/$a:.3f}')" 2>/dev/null)
  local v=$(python3 -c "
d=$d; s=$s; o=$o
fa = d>58; fb = s>o
print('VALID' if (fa and fb) else ('FAIL:drive%.0f'%d if not fa else 'FAIL:self<=other'))" 2>/dev/null)
  printf "%-30s %8s %8s %8s %8s %9s  %s\n" "$lbl" "$su" "$d" "$s" "$o" "$m" "$v"; }
echo "--- capacity sweep (ltdrep=1 ltdevery=2) ---"
for n in 12 24 48 90 140; do audit "NSTIM=$n" ltdrep=1 ltdevery=2 stim=$n epochs=40; done
echo "--- LTD sweep at NSTIM=48 ---"
audit "LTD 0.25x" ltdevery=4 ltdrep=1 stim=48 epochs=40
audit "LTD 0.5x"  ltdevery=2 ltdrep=1 stim=48 epochs=40
audit "LTD 1x"    ltdevery=1 ltdrep=1 stim=48 epochs=40
audit "LTD 2x"    ltdevery=1 ltdrep=2 stim=48 epochs=40
audit "LTD 3x"    ltdevery=1 ltdrep=3 stim=48 epochs=40
audit "LTD 4x"    ltdevery=1 ltdrep=4 stim=48 epochs=40
echo "--- pool-sweep v1 anomaly (labelled SUSTAINED) ---"
audit "pool 65536 ncol=2560" ltdrep=1 ltdevery=2 ncol=2560 stim=140 epochs=40
echo "--- untrained reference floor ---"
audit "UNTRAINED NSTIM=48" ltdrep=1 ltdevery=2 stim=48 epochs=0
