#!/bin/sh
# Runs the whole build/test matrix and prints a summary table.
# Rows whose toolchain is missing are SKIP; a FAIL row makes the script fail.
LOG=build/test-all; mkdir -p $LOG
rows=""; failed=0
row() {   # row <name> <command...>
    name=$1; shift
    f=$LOG/$(echo "$name" | tr ' /=' '___').log
    if sh -c "$*" > "$f" 2>&1; then st=PASS; else st=FAIL; failed=1; fi
    rows="$rows$(printf '%-34s %s' "$name" "$st")\n"
    printf '%-34s %s\n' "$name" "$st"
}
skip() { rows="$rows$(printf '%-34s SKIP (%s)' "$1" "$2")\n"; printf '%-34s SKIP (%s)\n' "$1" "$2"; }
have() { command -v "$1" >/dev/null 2>&1; }

cpu_has() { grep -qw "$1" /proc/cpuinfo; }
flag_of() { case $1 in sse41) echo sse4_1 ;; avxvnni) echo avx_vnni ;; avx512vnni) echo avx512_vnni ;; *) echo $1 ;; esac; }
for isa in scalar sse2 sse41 avx2 avxvnni avx512vnni; do
    if [ $isa = scalar ] || cpu_has "$(flag_of $isa)"; then
        row "$isa test+e2e" "make -s ISA=$isa test e2e"
    else
        row "$isa build only" "make -s ISA=$isa"
        skip "$isa test+e2e" "host CPU lacks $(flag_of $isa)"
    fi
done
row "x86 isa guard (qemu cpus)"   "make -s guard-test"
row "avx2 THREADS=pthread e2e"    "make -s ISA=avx2 THREADS=pthread test e2e"
row "sse2 ASM=1 test+e2e"         "make -s ISA=sse2 ASM=1 test e2e"
row "scalar ML_NO_MMAP test"      "make -s ISA=scalar EXTRA_CFLAGS=-DML_NO_MMAP test"
if have cmake; then
    row "cmake avx2 ctest" "cmake -S . -B build/cmake-avx2 -DML_ISA=avx2 >/dev/null && cmake --build build/cmake-avx2 && ctest --test-dir build/cmake-avx2"
else skip "cmake avx2 ctest" "no cmake"; fi
if have x86_64-w64-mingw32-gcc && have cmake; then
    row "mingw cross-compile (windows)" "printf 'set(CMAKE_SYSTEM_NAME Windows)\nset(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)\n' > build/mingw.cmake && cmake -S . -B build/cmake-mingw -DML_ISA=avx2 -DCMAKE_TOOLCHAIN_FILE=\$PWD/build/mingw.cmake >/dev/null && cmake --build build/cmake-mingw"
else skip "mingw cross-compile (windows)" "no mingw"; fi

if have arm-linux-gnueabihf-gcc && have qemu-arm; then
    row "armv7 scalar test+e2e (qemu)" "make -s ISA=scalar CROSS=arm-linux-gnueabihf- test e2e"
    row "armv7 neon test+e2e (qemu)"   "make -s ISA=neon CROSS=arm-linux-gnueabihf- test e2e"
else skip "armv7 (qemu)" "no arm-linux-gnueabihf-gcc/qemu-arm"; fi
if have aarch64-linux-gnu-gcc && have qemu-aarch64; then
    row "aarch64 neon test+e2e (qemu)"         "make -s ISA=neon CROSS=aarch64-linux-gnu- test e2e"
    row "aarch64 neon_dotprod test+e2e (qemu)" "make -s ISA=neon_dotprod CROSS=aarch64-linux-gnu- test e2e"
    row "aarch64 sve VL128+VL512 (qemu)"       "make -s ISA=sve CROSS=aarch64-linux-gnu- test-sve"
    row "aarch64 isa guard (qemu cpus)"        "make -s guard-test-arm"
    row "aarch64 neon THREADS=pthread (qemu)"  "make -s ISA=neon CROSS=aarch64-linux-gnu- THREADS=pthread test e2e"
else skip "aarch64 (qemu)" "no aarch64-linux-gnu-gcc/qemu-aarch64"; fi

if python3 -c "import pytest" 2>/dev/null; then
    row "python tools (pytest)" "python3 -m pytest -q tests"
else skip "python tools (pytest)" "pytest not installed"; fi

# Real-model smoke test: needs network access to huggingface/github.
M=models
if [ ! -f $M/stories15M.mlb ]; then
    mkdir -p $M
    curl -sfL -o $M/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
        && curl -sfL -o $M/tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin \
        && python3 tools/convert.py llama2c $M/stories15M.bin $M/stories15M.mlb --wtype q8 >/dev/null
fi
if [ -f $M/stories15M.mlb ] && [ -f $M/tokenizer.bin ]; then
    row "stories15M generate (avx2)" "build/avx2/ml_run $M/stories15M.mlb -z $M/tokenizer.bin -i 'Once upon a time' -n 64 -T 0 | tee $LOG/stories15M.txt | grep -q -E '[a-z]+ [a-z]+ [a-z]+'"
    # logits vs the numpy reference: argmax, top-5 and KL (see compare_logits.py --kl)
    S=1,9038,2501,263,931,727,471,263,2217,7826
    python3 tools/ref_forward.py $M/stories15M.mlb --tokens $S --out $LOG/s_ref.f32
    for b in scalar avx2; do for p in "" --no-pack; do
        row "stories15M logits $b ${p:-packed}" "build/$b/ml_run $M/stories15M.mlb --tokens $S --dump-logits $LOG/s.f32 $p && python3 tests/compare_logits.py $LOG/s.f32 $LOG/s_ref.f32 32000 --kl 1e-2"
    done; done
else skip "stories15M generate" "download blocked"; rm -f $M/stories15M.bin; fi

echo; echo "== summary =="; printf "$rows"
exit $failed
