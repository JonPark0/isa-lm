#!/bin/sh
# Compares isa-lm with llama.cpp on one Hugging Face model: speed per x86 ISA
# level and perplexity, with the same weight formats (Q8_0, and Q4_0 for every
# matrix). Results go to $OUT/results.md; docs/benchmarks.md describes a run.
#
#   scripts/compare_llamacpp.sh LLAMA_CPP_DIR HF_DIR F16_GGUF TEXT [OUT]
#
# LLAMA_CPP_DIR is a llama.cpp checkout with CMake build directories:
#   build-native  GGML_NATIVE=ON (llama-bench, llama-perplexity, llama-quantize)
#   build-avx2    GGML_NATIVE=OFF, AVX2 + FMA + F16C, no AVX-512   (llama-bench)
#   build-sse42   GGML_NATIVE=OFF, SSE4.2 only                      (llama-bench)
#   build-base    GGML_NATIVE=OFF, every x86 extension off (SSE2)  (llama-bench)
# A missing directory skips that row. F16_GGUF is the model converted by
# llama.cpp (for Qwen2.5, the official *-fp16.gguf); TEXT is the perplexity
# text (for example the wikitext-2 test set).
#
# Environment: THREADS (default "1 4"), TOKENS (prompt and generated tokens,
# default 128), REPS (default 3), CTX and CHUNKS for perplexity (512, 4),
# NATIVE_ISA (isa-lm ISA matching build-native; default: the best the host runs).
set -eu
[ $# -ge 4 ] || { sed -n '2,20p' "$0"; exit 2; }
LCPP=$1 HF=$2 F16=$3 TEXT=$4 OUT=${5:-build/compare}
THREADS=${THREADS:-"1 4"} TOKENS=${TOKENS:-128} REPS=${REPS:-3} CTX=${CTX:-512} CHUNKS=${CHUNKS:-4}
mkdir -p "$OUT"
log() { printf '%s\n' "$*" >&2; }

if [ -z "${NATIVE_ISA:-}" ]; then
    for isa in avx512vnni avxvnni avx2 sse41 sse2; do
        make -s ISA=$isa build/$isa/ml_tok >/dev/null 2>&1 || continue
        if build/$isa/ml_tok --help >/dev/null 2>&1; then NATIVE_ISA=$isa; break; fi
    done
fi

# isa-lm builds (threaded) and models
for isa in $NATIVE_ISA avx2 sse41 sse2; do make -s ISA=$isa THREADS=pthread -j4 >/dev/null; done
make -s ISA=$NATIVE_ISA -j4 >/dev/null
[ -f "$OUT/tok.mlt" ] || python3 tools/convert_tokenizer.py "$HF/tokenizer.json" "$OUT/tok.mlt" >/dev/null
for wt in q8 q4; do
    [ -f "$OUT/isa_$wt.mlb" ] || python3 tools/convert.py hf "$HF" "$OUT/isa_$wt.mlb" --wtype $wt --max-seq 4096 >/dev/null
done
# llama.cpp models: --pure keeps every tensor (embeddings and output too) in the
# one type, like isa-lm's plain Q8 and Q4 files
for wt in q8_0 q4_0; do
    [ -f "$OUT/lcpp_$wt.gguf" ] || "$LCPP/build-native/bin/llama-quantize" --pure "$F16" "$OUT/lcpp_$wt.gguf" \
        $(echo $wt | tr a-z A-Z) >/dev/null 2>&1
done

size_mb() { echo $(( $(wc -c < "$1") / 1048576 )); }

# mean of REPS runs: "prefill decode" in tok/s
isa_speed() {   # ISA MODEL THREADS
    for r in $(seq $REPS); do build/$1-pt/ml_bench "$2" -t $3 -n $TOKENS | tail -1; done |
        sed 's/.*prefill \([0-9.]*\) tok\/s, decode \([0-9.]*\).*/\1 \2/' |
        awk '{p += $1; d += $2} END {printf "%.1f %.1f\n", p / NR, d / NR}'
}
# llama-bench: prompt processing of TOKENS, then TOKENS generated after a
# TOKENS-long context (-d), as ml_bench does; mean of REPS
lcpp_speed() {   # BUILD MODEL THREADS
    b=$LCPP/$1/bin/llama-bench
    p=$("$b" -m "$2" -t $3 -p $TOKENS -n 0 -r $REPS -o csv 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,""); printf "%.1f", $(NF-1)}')
    d=$("$b" -m "$2" -t $3 -p 0 -n $TOKENS -d $TOKENS -r $REPS -o csv 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,""); printf "%.1f", $(NF-1)}')
    echo "$p $d"
}

{
echo "| ISA level | Runtime | Weights | Size | Threads | Prefill tok/s | Decode tok/s |"
echo "|-----------|---------|---------|-----:|--------:|--------------:|-------------:|"
for pair in "$NATIVE_ISA:build-native" avx2:build-avx2 sse41:build-sse42 sse2:build-base; do
    isa=${pair%%:*} dir=${pair#*:}
    for wt in q8 q4; do for t in $THREADS; do
        log "isa-lm $isa $wt -t $t"
        set -- $(isa_speed $isa "$OUT/isa_$wt.mlb" $t)
        echo "| $isa | isa-lm | ${wt}_0 | $(size_mb "$OUT/isa_$wt.mlb") MB | $t | $1 | $2 |"
        if [ -x "$LCPP/$dir/bin/llama-bench" ]; then
            log "llama.cpp $dir $wt -t $t"
            set -- $(lcpp_speed $dir "$OUT/lcpp_${wt}_0.gguf" $t)
            echo "| $isa | llama.cpp | ${wt}_0 | $(size_mb "$OUT/lcpp_${wt}_0.gguf") MB | $t | $1 | $2 |"
        fi
    done; done
done
echo
echo "Perplexity ($CHUNKS chunks of $CTX tokens, second half of each scored, as llama-perplexity does):"
echo
echo "| Runtime | Weights | Perplexity |"
echo "|---------|---------|-----------:|"
ppl_lcpp() {
    "$LCPP/build-native/bin/llama-perplexity" -m "$1" -f "$TEXT" -c $CTX --chunks $CHUNKS -t 4 2>&1 |
        sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p'
}
echo "| llama.cpp | f16 | $(ppl_lcpp "$F16") |"
for wt in q8 q4; do
    echo "| llama.cpp | ${wt}_0 | $(ppl_lcpp "$OUT/lcpp_${wt}_0.gguf") |"
    p=$(python3 tools/perplexity.py build/$NATIVE_ISA-pt/ml_run "$OUT/isa_$wt.mlb" "$OUT/tok.mlt" "$TEXT" \
        --ml-tok build/$NATIVE_ISA/ml_tok --ctx $CTX --chunks $CHUNKS -t 4 --llama-cpp | sed 's/.*perplexity \([0-9.]*\).*/\1/')
    echo "| isa-lm | ${wt}_0 | $p |"
done
} | tee "$OUT/results.md"
