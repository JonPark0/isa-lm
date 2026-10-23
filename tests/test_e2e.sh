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

# mixed Q4/Q8 model (q8_mask): reference, packed and row-major, batched prefill
$RUN $B/ml_run $M/mix_q4.mlb --tokens $TOKENS --dump-logits $T/mix.f32 \
    && python3 tools/ref_forward.py $M/mix_q4.mlb --tokens $TOKENS --out $T/mix.ref.f32 \
    && python3 tests/compare_logits.py $T/mix.f32 $T/mix.ref.f32 96 1e-3 >/dev/null \
    && $RUN $B/ml_run $M/mix_q4.mlb --tokens $TOKENS --dump-logits $T/mix.np.f32 --no-pack \
    && python3 tests/compare_logits.py $T/mix.np.f32 $T/mix.ref.f32 96 1e-3 >/dev/null \
    && $RUN $B/ml_run $M/mix_q4.mlb --tokens $TOKENS --dump-logits $T/mix.b1.f32 --batch 1 \
    && cmp -s $T/mix.f32 $T/mix.b1.f32 && ok || bad "mixed Q4/Q8 model"

# Q/K/V biases (qkv_bias, Qwen2): reference, batched prefill
$RUN $B/ml_run $M/bias_q4.mlb --tokens $TOKENS --dump-logits $T/bias.f32 \
    && python3 tools/ref_forward.py $M/bias_q4.mlb --tokens $TOKENS --out $T/bias.ref.f32 \
    && python3 tests/compare_logits.py $T/bias.f32 $T/bias.ref.f32 96 1e-3 >/dev/null \
    && $RUN $B/ml_run $M/bias_q4.mlb --tokens $TOKENS --dump-logits $T/bias.b1.f32 --batch 1 \
    && cmp -s $T/bias.f32 $T/bias.b1.f32 && ok || bad "Q/K/V bias model"

# Q8 KV cache (--kv q8) against the reference emulating it. K, V and q are
# quantized from float results that differ in the last bits between C and numpy,
# so a value near a rounding boundary can land one step away: compare
# distributions (argmax, top-5, KL), as for real models.
for wt in q8 q4; do
    $RUN $B/ml_run $M/rand_$wt.mlb --tokens $TOKENS --dump-logits $T/$wt.kv8.f32 --kv q8 \
        && python3 tools/ref_forward.py $M/rand_$wt.mlb --tokens $TOKENS --out $T/$wt.kv8.ref.f32 --kv q8 \
        && python3 tests/compare_logits.py $T/$wt.kv8.f32 $T/$wt.kv8.ref.f32 96 --kl 1e-3 >/dev/null \
        && ok || bad "ref --kv q8 $wt"
done

# 150 positions: attention crosses its 64-position blocks, both cache types.
# The host scalar build with an F32 cache must match the reference closely.
# Over 150 positions, builds with FMA (and any Q8 cache, which quantizes
# values that differ in their last bits) drift by up to ~2% as rounding
# boundaries fall differently; main behaved the same. They are held to 5%
# without the argmax check: enough to catch a wrong block offset or stride,
# which garbles the logits.
CTX=$(seq 1 150 | awk '{print ($1 * 37) % 95 + 1}' | paste -sd, -)
for kv in f32 q8; do
    if [ $kv = f32 ]; then strict="1e-3"; else strict="5e-2 --any-argmax"; fi
    python3 tools/ref_forward.py $M/ctx_q4.mlb --tokens $CTX --out $T/ctx.$kv.ref.f32 --kv $kv \
        && $SCALAR $M/ctx_q4.mlb --tokens $CTX --dump-logits $T/ctx.$kv.scalar.f32 --kv $kv \
        && python3 tests/compare_logits.py $T/ctx.$kv.scalar.f32 $T/ctx.$kv.ref.f32 96 $strict >/dev/null \
        && $RUN $B/ml_run $M/ctx_q4.mlb --tokens $CTX --dump-logits $T/ctx.$kv.f32 --kv $kv \
        && python3 tests/compare_logits.py $T/ctx.$kv.f32 $T/ctx.$kv.ref.f32 96 5e-2 --any-argmax >/dev/null \
        && ok || bad "150-position ref --kv $kv"
done

# builds with packed kernels must actually pack (Q4 is packed by every one)
G=$($RUN $B/ml_run --help 2>&1 | sed -n 's/^pack group: \([0-9]*\).*/\1/p')
err=$($RUN $B/ml_run $M/rand_q4.mlb --tokens 1,2 -n 4 2>&1 >/dev/null)
[ -n "$G" ] && echo "$err" | grep -q "packed: $G," && ok || bad "expected 'packed: $G' (help says pack group '$G'): $err"

# --pack-cache: written on the first run, mapped on the next, same logits;
# a cache for another model (or a damaged one) is replaced, never used
if [ "$G" != 0 ]; then
    cp $M/mix_q4.mlb $T/pc.mlb; rm -f $T/pc.mlb.pk$G
    $RUN $B/ml_run $M/mix_q4.mlb --tokens $TOKENS --dump-logits $T/pc.ref.f32
    a=$($RUN $B/ml_run $T/pc.mlb --tokens $TOKENS --dump-logits $T/pc1.f32 --pack-cache 2>&1)
    b=$($RUN $B/ml_run $T/pc.mlb --tokens $TOKENS --dump-logits $T/pc2.f32 --pack-cache 2>&1)
    echo "$a" | grep -q ": written" && echo "$b" | grep -q ": loaded" \
        && cmp -s $T/pc1.f32 $T/pc.ref.f32 && cmp -s $T/pc2.f32 $T/pc.ref.f32 && ok || bad "--pack-cache: $a / $b"
    python3 tools/gen_random.py --out $T/pc.mlb --wtype q4 --q8 emb,v,o,down --seed 4 >/dev/null
    c=$($RUN $B/ml_run $T/pc.mlb --tokens 1,2 -n 3 --pack-cache 2>&1)
    head -c 1000 $T/pc.mlb.pk$G > $T/pc.trunc && mv $T/pc.trunc $T/pc.mlb.pk$G
    d=$($RUN $B/ml_run $T/pc.mlb --tokens 1,2 -n 3 --pack-cache 2>&1)
    echo "$c" | grep -q ": written" && echo "$d" | grep -q ": written" && ok || bad "--pack-cache reused a stale cache: $c / $d"
    rm -f $T/pc.mlb $T/pc.mlb.pk$G
fi

# --no-pack keeps the row-major kernels: must match the reference too
for wt in q8 q4; do
    $RUN $B/ml_run $M/rand_$wt.mlb --tokens $TOKENS --dump-logits $T/$wt.np.f32 --no-pack \
        && python3 tests/compare_logits.py $T/$wt.np.f32 $T/$wt.ref.f32 96 1e-3 >/dev/null \
        && ok || bad "--no-pack ref $wt"
done

# batched prefill gives the same logits as token-by-token, byte for byte
for wt in q8 q4; do
    $RUN $B/ml_run $M/rand_$wt.mlb --tokens $TOKENS --dump-logits $T/$wt.b1.f32 --batch 1 \
        && $RUN $B/ml_run $M/rand_$wt.mlb --tokens $TOKENS --dump-logits $T/$wt.b5.f32 --batch 5 \
        && cmp -s $T/$wt.f32 $T/$wt.b1.f32 && cmp -s $T/$wt.f32 $T/$wt.b5.f32 \
        && ok || bad "batched prefill logits differ ($wt)"
done
$RUN $B/ml_run $M/ctx_q4.mlb --tokens $CTX --dump-logits $T/ctx.q8.b1.f32 --kv q8 --batch 1 \
    && cmp -s $T/ctx.q8.f32 $T/ctx.q8.b1.f32 && ok || bad "batched prefill logits differ (--kv q8)"
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "abcdefghij" -n 24 -T 0 --batch 1 > $T/gb1.txt 2>/dev/null \
    && $RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "abcdefghij" -n 24 -T 0 > $T/gb.txt 2>/dev/null \
    && cmp -s $T/gb1.txt $T/gb.txt && ok || bad "generation with batched prefill differs from --batch 1"

# greedy: same text as the host scalar build
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 16 -T 0 > $T/greedy.txt 2>/dev/null \
    && $SCALAR $M/rand_q8.mlb -z $TOK -i "ab" -n 16 -T 0 > $T/greedy.ref.txt 2>/dev/null \
    && [ -s $T/greedy.txt ] && cmp -s $T/greedy.txt $T/greedy.ref.txt && ok || bad "greedy text differs from scalar"

# sampled: same seed -> same text as scalar
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 20 -T 0.9 -p 0.9 -s 42 > $T/samp.txt 2>/dev/null \
    && $SCALAR $M/rand_q8.mlb -z $TOK -i "ab" -n 20 -T 0.9 -p 0.9 -s 42 > $T/samp.ref.txt 2>/dev/null \
    && cmp -s $T/samp.txt $T/samp.ref.txt && ok || bad "sampled text differs from scalar"

# --top-k 1 is greedy at any temperature; a repetition penalty changes the text
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 16 -T 1.3 --top-k 1 -s 5 > $T/topk1.txt 2>/dev/null \
    && cmp -s $T/topk1.txt $T/greedy.txt && ok || bad "--top-k 1 should give the greedy text"
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 24 -T 0 > $T/greedy24.txt 2>/dev/null
$RUN $B/ml_run $M/rand_q8.mlb -z $TOK -i "ab" -n 24 -T 0 --repeat-penalty 1.5 > $T/pen.txt 2>/dev/null \
    && $SCALAR $M/rand_q8.mlb -z $TOK -i "ab" -n 24 -T 0 --repeat-penalty 1.5 > $T/pen.ref.txt 2>/dev/null \
    && cmp -s $T/pen.txt $T/pen.ref.txt && ! cmp -s $T/pen.txt $T/greedy24.txt && ok \
    || bad "--repeat-penalty: should match scalar and differ from plain greedy"

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
$RUN $B/ml_run $M/ctx_q4.mlb --tokens $CTX --dump-logits $T/ctx.q8.t3.f32 --kv q8 -t 3 \
    && cmp -s $T/ctx.q8.f32 $T/ctx.q8.t3.f32 && ok || bad "logits with --kv q8 -t 3 differ from -t 1"
err=$($RUN $B/ml_run $M/rand_q4.mlb --tokens 1,2 -n 8 -t 4 2>&1 >/dev/null)
echo "$err" | grep -q "threads: $NT," && ok || bad "expected 'threads: $NT' in: $err"

# stop tokens: no tokenizer -> ids 1/2 are ordinary tokens, run all -n positions;
# --eos ID stops as soon as ID is generated
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 20 -T 0 2>&1 >/dev/null)
echo "$err" | grep -q "^positions: 20," && ok || bad "no-tokenizer run should use all 20 positions: $err"
first=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 3 -T 0 2>/dev/null | awk '{print $2}')
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2 -n 20 -T 0 --eos "$first" 2>&1 >/dev/null)
echo "$err" | grep -q "^positions: 2," && ok || bad "--eos $first should stop right after the prompt (positions 2): $err"

# --profile: per-stage time breakdown on stderr, output unchanged
$RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2,3 -n 12 -T 0 > $T/noprof.txt 2>/dev/null
err=$($RUN $B/ml_run $M/rand_q8.mlb --tokens 1,2,3 -n 12 -T 0 --profile 2>&1 > $T/prof.txt)
if cmp -s $T/noprof.txt $T/prof.txt && echo "$err" | grep -q "^profile:" \
    && for k in qkv attention wo ffn classifier other; do echo "$err" | grep -q " $k "; done; then ok
else bad "--profile output: $err"; fi

# ml_tok: tokenizers converted from tokenizer.json (ids from the tokenizers library)
out=$(printf 'hello world' | $RUN $B/ml_tok $M/spm_tiny.mlt)
[ "$out" = "1 313" ] && ok || bad "ml_tok spm encode: '$out'"
out=$(printf "it's what they'll do" | $RUN $B/ml_tok $M/bpe_tiny.mlt)
[ "$out" = "0 287 272 313 316 273 308" ] && ok || bad "ml_tok byte-level encode: '$out'"
out=$(printf '287 272 313 316 273 308' | $RUN $B/ml_tok $M/bpe_tiny.mlt --decode)
[ "$out" = "it's what they'll do" ] && ok || bad "ml_tok byte-level decode: '$out'"
printf 'x' | $RUN $B/ml_tok $M/bpe_tiny.mlt --bogus >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "ml_tok with a bad option should exit 2"
printf '99999' | $RUN $B/ml_tok $M/bpe_tiny.mlt --decode >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "ml_tok --decode with an id out of range should exit 2"

# --chat: deterministic replies (same as the scalar build), -i equals a
# one-line stdin, the conversation stops cleanly when the context is full
CHAT="$M/chat_q8.mlb -z $M/bpe_tiny.mlt --chat llama3 -T 0 -n 6"
printf 'hello world\nwhat now\n' | $RUN $B/ml_run $CHAT --system "be brief" > $T/chat.txt 2>/dev/null \
    && printf 'hello world\nwhat now\n' | $SCALAR $CHAT --system "be brief" > $T/chat.ref.txt 2>/dev/null \
    && [ -s $T/chat.txt ] && cmp -s $T/chat.txt $T/chat.ref.txt && ok || bad "--chat differs from scalar"
$RUN $B/ml_run $CHAT -i "hello world" > $T/chat1.txt 2>/dev/null \
    && printf 'hello world\n' | $RUN $B/ml_run $CHAT > $T/chat1s.txt 2>/dev/null \
    && cmp -s $T/chat1.txt $T/chat1s.txt && ok || bad "--chat -i differs from one stdin line"
err=$(seq 1 40 | sed 's/^/tell me more /' | $RUN $B/ml_run $CHAT 2>&1 >/dev/null); rc=$?
[ $rc -eq 0 ] && echo "$err" | grep -q "context full" && ok || bad "--chat past the context: rc=$rc $err"
for bad_chat in "--chat chatml" "--system x"; do
    $RUN $B/ml_run $M/chat_q8.mlb -z $M/bpe_tiny.mlt $bad_chat -i hi >/dev/null 2>&1
    [ $? -eq 2 ] && ok || bad "ml_run $bad_chat should exit 2"
done
$RUN $B/ml_run $M/chat_q8.mlb --chat llama3 -i hi >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "--chat without -z should exit 2"

# usage
err=$($RUN $B/ml_run 2>&1); rc=$?
[ $rc -eq 2 ] && echo "$err" | grep -q "usage:" && ok || bad "usage rc=$rc"
$RUN $B/ml_run $M/rand_q8.mlb --tokens 1,500 -n 4 >/dev/null 2>&1
[ $? -eq 2 ] && ok || bad "out-of-range token id should exit 2"

for bad_opt in "-T nan" "-T inf" "-T -1" "-p nan" "-p 1.5" "-p -0.1" "--top-k -1" "--repeat-penalty 0" "--repeat-penalty nan" "--repeat-last-n -2"; do
    $RUN $B/ml_run $M/rand_q8.mlb --tokens 1 -n 4 $bad_opt >/dev/null 2>&1
    [ $? -eq 2 ] && ok || bad "ml_run $bad_opt should exit 2"
done

for bad_arg in "--pack-cache --no-pack" "--kv q4" "--kv" "-n 5x" "-n 99999999999" "-t 2junk" "-s -1" "--tokens 4294967297" "--tokens 1,,2" "--tokens 1," "-n"; do
    $RUN $B/ml_run $M/rand_q8.mlb --tokens 1 $bad_arg >/dev/null 2>&1
    [ $? -eq 2 ] && ok || bad "ml_run $bad_arg should exit 2"
done
out=$($RUN $B/ml_run --help 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q "usage:" && ok || bad "--help rc=$rc"

if [ $fail -eq 0 ]; then echo "e2e $ISA: $pass passed"; else echo "e2e $ISA: $pass passed, $fail failed"; exit 1; fi
