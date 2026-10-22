#!/bin/sh
# Builds must run on a CPU that has their ISA and, on one that lacks it, exit 3
# naming the missing feature instead of dying on SIGILL. Older CPUs are emulated
# with qemu-x86_64 -cpu <model>.
pass=0; fail=0
ok()  { pass=$((pass+1)); }
bad() { echo "FAIL: $*"; fail=$((fail+1)); }

for isa in scalar sse2 sse41 avx2 avxvnni avx512vnni; do
    if grep -qw "$(echo $isa | sed -e s/sse41/sse4_1/ -e s/avxvnni/avx_vnni/ -e s/avx512vnni/avx512_vnni/ -e s/scalar/sse2/)" /proc/cpuinfo; then
        build/$isa/test_quant >/dev/null 2>&1 && ok || bad "$isa build should run on host"
    else
        echo "SKIP: host lacks $isa"
    fi
done

# expect_reject <isa> <qemu cpu model> <feature named in error>
expect_reject() {
    err=$(qemu-x86_64 -cpu "$2" build/$1/test_quant 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 3 ] && echo "$err" | grep -q -- "lacks $3 "; then ok
    else bad "$1 on $2: rc=$rc stderr=$err"; fi
}
# expect_run <isa> <qemu cpu model>
expect_run() {
    qemu-x86_64 -cpu "$2" build/$1/test_quant >/dev/null 2>&1 && ok || bad "$1 should run on $2"
}

if command -v qemu-x86_64 >/dev/null; then
    expect_run    sse2       qemu64
    expect_reject sse41      core2duo    sse4.1
    expect_run    sse41      Nehalem
    expect_reject avx2       Nehalem     avx
    expect_reject avx2       SandyBridge fma
    expect_run    avx2       Haswell
    expect_reject avxvnni    Haswell     avx-vnni
    expect_reject avx512vnni Haswell     avx512f
else
    echo "SKIP: qemu-x86_64 not installed (negative cases)"
fi
if [ $fail -eq 0 ]; then echo "isa guard: $pass passed"; else echo "isa guard: $pass passed, $fail failed"; exit 1; fi
