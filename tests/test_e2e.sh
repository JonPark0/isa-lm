#!/bin/sh
# End-to-end: test_e2e.sh ISA BUILD_DIR [RUN-prefix]
# Compares ml_run logits to the numpy reference, and generated text to the host
# scalar build.
ISA=$1; B=$2; RUN=$3
M=build/models; TOK=$M/tok96.bin; SCALAR=build/scalar/ml_run
T=$B/e2e; mkdir -p $T
pass=0; fail=0
ok()  { pass=$((pass+1)); }
bad() { echo "FAIL: $*"; fail=$((fail+1)); }
TOKENS=1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61

# ref: logits vs numpy reference, both weight types
for wt in q8 q4; do
    $RUN $B/ml_run $M/rand_$wt.mlb --tokens $TOKENS --dump-logits $T/$wt.f32 \
        && python3 tools/ref_forward.py $M/rand_$wt.mlb --tokens $TOKENS --out $T/$wt.ref.f32 \
        && python3 tests/compare_logits.py $T/$wt.f32 $T/$wt.ref.f32 96 1e-3 \
        && ok || bad "ref $wt"
done

# greedy: same text as the host scalar build
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 16 -T 0 > $T/greedy.txt 2>/dev/null \
    && $SCALAR $M/rand_q8.mlb -z $TOK -i "ab" -n 16 -T 0 > $T/greedy.ref.txt 2>/dev/null \
    && [ -s $T/greedy.txt ] && cmp -s $T/greedy.txt $T/greedy.ref.txt && ok || bad "greedy text differs from scalar"

# sampled: same seed -> same text as scalar
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 20 -T 0.9 -p 0.9 -s 42 > $T/samp.txt 2>/dev/null \
    && $SCALAR $M/rand_q8.mlb -z $TOK -i "ab" -n 20 -T 0.9 -p 0.9 -s 42 > $T/samp.ref.txt 2>/dev/null \
    && cmp -s $T/samp.txt $T/samp.ref.txt && ok || bad "sampled text differs from scalar"

# overflow: steps beyond max_seq (32) stop at max_seq
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 1000 -T 0 2>&1 >/dev/null); rc=$?
pos=$(echo "$err" | sed -n 's/^positions: \([0-9]*\).*/\1/p')
[ $rc -eq 0 ] && [ -n "$pos" ] && [ "$pos" -le 32 ] && ok || bad "overflow rc=$rc positions=$pos"
seq 1 40 | paste -sd, > $T/long.txt
$RUN $B/ml_run $M/rand_q8.mlb --tokens $(cat $T/long.txt) --dump-logits $T/long.f32 2>/dev/null
[ $? -eq 2 ] && ok || bad "dump-logits with 40 tokens > max_seq should exit 2"

# threads: pthread builds (dir suffix -pt) split work but match -t 1 bit for bit
case $B in *-pt*) NT=4 ;; *) NT=1 ;; esac
for t in 3 4 7; do
    $RUN $B/ml_run $M/rand_q8.mlb --tokens $TOKENS --dump-logits $T/t1.f32 -t 1 \
        && $RUN $B/ml_run $M/rand_q8.mlb --tokens $TOKENS --dump-logits $T/tn.f32 -t $t \
        && cmp -s $T/t1.f32 $T/tn.f32 && ok || bad "logits with -t $t differ from -t 1"
done
err=$($RUN $B/ml_run $M/rand_q4.mlb --tokens 1,2 -n 8 -t 4 2>&1 >/dev/null)
echo "$err" | grep -q "threads: $NT," && ok || bad "expected 'threads: $NT' in: $err"

# stop tokens: no tokenizer -> ids 1/2 are ordinary tokens, run all -n positions;
# --eos ID stops as soon as ID is generated
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 20 -T 0 2>&1 >/dev/null)
echo "$err" | grep -q "^positions: 20," && ok || bad "no-tokenizer run should use all 20 positions: $err"
first=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 3 -T 0 2>/dev/null | awk '{print $2}')
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 20 -T 0 --eos "$first" 2>&1 >/dev/null)
echo "$err" | grep -q "^positions: 2," && ok || bad "--eos $first should stop right after the prompt (positions 2): $err"

# usage
err=$($RUN $B/ml_run 2>&1); rc=$?
[ $rc -eq 2 ] && echo "$err" | grep -q "usage:" && ok || bad "usage rc=$rc"
$RUN $B/ml_run $M/rand_q8.mlb --tokens 1,500 -n 4 >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "out-of-range token id should exit 2"

for bad_opt in "-T nan" "-T inf" "-T -1" "-p nan" "-p 1.5" "-p -0.1"; do
    $RUN $B/ml_run $M/rand_q8.mlb --tokens 1 -n 4 $bad_opt >/dev/null 2>&1
    [ $? -eq 2 ] && ok || bad "ml_run $bad_opt should exit 2"
done

for bad_arg in "-n 5x" "-n 99999999999" "-t 2junk" "-s -1" "--tokens 4294967297" "--tokens 1,,2" "--tokens 1," "-n"; do
    $RUN $B/ml_run $M/rand_q8.mlb --tokens 1 $bad_arg >/dev/null 2>&1
    [ $? -eq 2 ] && ok || bad "ml_run $bad_arg should exit 2"
done
out=$($RUN $B/ml_run --help 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q "usage:" && ok || bad "--help rc=$rc"

if [ $fail -eq 0 ]; then echo "e2e $ISA: $pass passed"; else echo "e2e $ISA: $pass passed, $fail failed"; exit 1; fi
