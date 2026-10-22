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

# usage
err=$($RUN $B/ml_run 2>&1); rc=$?
[ $rc -eq 2 ] && echo "$err" | grep -q "usage:" && ok || bad "usage rc=$rc"
$RUN $B/ml_run $M/rand_q8.mlb --tokens 1,500 -n 4 >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "out-of-range token id should exit 2"

if [ $fail -eq 0 ]; then echo "e2e $ISA: $pass passed"; else echo "e2e $ISA: $pass passed, $fail failed"; exit 1; fi
