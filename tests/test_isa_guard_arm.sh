#!/bin/sh
# AArch64 builds must run on cores with their extensions and exit 3, naming
# the missing one, on cores without (emulated with qemu-aarch64 -cpu <core>).
pass=0; fail=0
run() {   # run <build> <cpu> <expected rc> [feature]
    err=$(qemu-aarch64 -cpu "$2" build/$1/test_quant 2>&1 >/dev/null); rc=$?
    if [ $rc -eq $3 ] && { [ -z "$4" ] || echo "$err" | grep -q -- "lacks $4 "; }; then pass=$((pass+1))
    else echo "FAIL: $1 on $2: rc=$rc stderr=$err"; fail=$((fail+1)); fi
}
run neon_dotprod-aarch64 cortex-a53 3 asimddp
run neon_dotprod-aarch64 cortex-a76 0
run sve-aarch64          cortex-a76 3 sve
run sve-aarch64          max        0
if [ $fail -eq 0 ]; then echo "isa guard arm: $pass passed"; else echo "isa guard arm: $pass passed, $fail failed"; exit 1; fi
