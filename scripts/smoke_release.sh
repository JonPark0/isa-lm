#!/bin/sh
# Unpacks the archives in $DIST and runs them: every build prints its version
# (natively, under qemu, or under wine), the wrappers pick a build, and each
# platform's logits on a test model match the host scalar build's tokens.
# Usage: scripts/smoke_release.sh   (after scripts/package_release.sh and `make ISA=scalar`)
set -eu
VER=$(sed -n 's/^#define ML_VERSION "\(.*\)"/\1/p' core/ml.h)
DIST=${DIST:-dist}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
M=build/models/rand_q8.mlb
TOK=1,5,9,13,17,21,25,29
fail=0
check() { if "$@"; then :; else echo "FAIL: $*"; fail=1; fi; }
(cd "$DIST" && sha256sum -c --quiet SHA256SUMS)
for a in linux-x86_64 linux-aarch64 linux-armv7hf; do tar -C "$T" -xzf "$DIST/isa-lm-$VER-$a.tar.gz"; done
(cd "$T" && unzip -q "$OLDPWD/$DIST/isa-lm-$VER-windows-x86_64.zip")

version() {   # version RUNNER BINARY: prints the version line of --help
    $1 "$2" --help 2>&1 | grep -q "isa-lm $VER"
}
for b in "$T"/isa-lm-$VER-linux-x86_64/bin/*/ml_tok; do
    rc=0; "$b" --help >/dev/null 2>&1 || rc=$?   # not a bare call: set -e would exit with its 3
    [ $rc -eq 3 ] && { echo "skip $b: CPU lacks it"; continue; }
    check version "" "$b"
done
for b in "$T"/isa-lm-$VER-linux-aarch64/bin/*/ml_run; do check version "qemu-aarch64 -cpu max" "$b"; done
for b in "$T"/isa-lm-$VER-linux-armv7hf/bin/*/ml_run; do check version "qemu-arm -cpu max" "$b"; done
if command -v wine64 >/dev/null || [ -x /usr/lib/wine/wine64 ]; then
    W=$(command -v wine64 || echo /usr/lib/wine/wine64)
    export WINEDEBUG=-all WINEPREFIX="$T/wine"
    check version "$W" "$T/isa-lm-$VER-windows-x86_64/bin/sse2/ml_run.exe"
    $W "$T/isa-lm-$VER-windows-x86_64/bin/sse2/ml_run.exe" "$M" --tokens $TOK -n 12 -T 0 2>/dev/null | tr -d '\r' > "$T/win.txt"   # text-mode CRLF
else
    echo "skip windows: no wine"
fi

# the x86-64 wrapper picks a build that runs here, and greedy output matches scalar
"$T/isa-lm-$VER-linux-x86_64/ml_run" --help 2>&1 | grep "build ISA"
"$T/isa-lm-$VER-linux-x86_64/ml_run" "$M" --tokens $TOK -n 12 -T 0 > "$T/x86.txt" 2>/dev/null
ISA_LM_ISA=sse2 "$T/isa-lm-$VER-linux-x86_64/ml_run" "$M" --tokens $TOK -n 12 -T 0 > "$T/sse2.txt" 2>/dev/null
build/scalar/ml_run "$M" --tokens $TOK -n 12 -T 0 > "$T/ref.txt" 2>/dev/null
qemu-aarch64 -cpu max "$T/isa-lm-$VER-linux-aarch64/bin/neon_dotprod/ml_run" "$M" --tokens $TOK -n 12 -T 0 > "$T/a64.txt" 2>/dev/null
qemu-arm -cpu max "$T/isa-lm-$VER-linux-armv7hf/bin/neon/ml_run" "$M" --tokens $TOK -n 12 -T 0 > "$T/arm.txt" 2>/dev/null
for f in x86 sse2 a64 arm win; do
    [ -f "$T/$f.txt" ] || continue
    if cmp -s "$T/$f.txt" "$T/ref.txt"; then echo "$f: same greedy tokens as scalar"; else echo "FAIL: $f differs from scalar"; fail=1; fi
done
exit $fail
