#!/bin/sh
# Builds the release archives into $DIST (default dist/):
#   krafla-VER-linux-x86_64.tar.gz    sse2 sse41 avx2 avxvnni avx512vnni   (static)
#   krafla-VER-linux-aarch64.tar.gz   neon neon_dotprod neon_i8mm sve      (static)
#   krafla-VER-linux-armv7hf.tar.gz   neon scalar                          (static)
#   krafla-VER-windows-x86_64.zip     sse2 sse41 avx2 avxvnni avx512vnni   (MinGW, static)
# Each holds bin/<isa>/{ml_run,ml_bench,ml_tok}, wrappers that pick the best
# build the CPU supports, the model tools, and the license files.
# Usage: scripts/package_release.sh [TAG]   (TAG, e.g. v0.1.0, must match ML_VERSION)
set -eu
VER=$(sed -n 's/^#define ML_VERSION "\(.*\)"/\1/p' core/ml.h)
if [ -n "${1:-}" ] && [ "$1" != "v$VER" ]; then echo "tag $1 does not match ML_VERSION $VER" >&2; exit 1; fi
DIST=${DIST:-dist}
STAGE=build/release
rm -rf "$STAGE"
mkdir -p "$DIST" "$STAGE"
# EXTRA_CFLAGS only gives the release builds their own build directories
REL=-DML_RELEASE_BUILD

common() {   # common DIR: docs, licenses and model tools
    mkdir -p "$1/tools"
    cp README.md LICENSE THIRD_PARTY_NOTICES.md CITATION.cff "$1/"
    cp tools/convert.py tools/convert_tokenizer.py tools/mlb.py tools/perplexity.py "$1/tools/"
}

unix_wrappers() {   # unix_wrappers DIR "ISAS best first"
    for tool in ml_run ml_bench ml_tok; do
        cat > "$1/$tool" <<EOF
#!/bin/sh
# Runs $tool from the best build in bin/ that this CPU supports.
# KRAFLA_ISA=<isa> (or the older ISA_LM_ISA) picks one instead (see bin/).
here=\$(cd "\$(dirname "\$0")" && pwd)
for isa in \${KRAFLA_ISA:-\${ISA_LM_ISA:-$2}}; do
    [ -x "\$here/bin/\$isa/ml_tok" ] || continue
    "\$here/bin/\$isa/ml_tok" --help >/dev/null 2>&1
    [ \$? -ne 3 ] && exec "\$here/bin/\$isa/$tool" "\$@"   # 3: the CPU lacks this ISA
done
echo "krafla: no build in \$here/bin runs on this CPU" >&2
exit 3
EOF
        chmod +x "$1/$tool"
    done
}

linux() {   # linux NAME CROSS-PREFIX "ISAS best first"
    d="$STAGE/krafla-$VER-$1"
    common "$d"
    for isa in $3; do
        if [ -n "$2" ]; then set -- "$1" "$2" "$3" "CROSS=$2"; else set -- "$1" "$2" "$3" "EXTRA_LDFLAGS=-static"; fi
        make -s ISA=$isa THREADS=pthread EXTRA_CFLAGS=$REL "$4" apps
        b=$(make -s ISA=$isa THREADS=pthread EXTRA_CFLAGS=$REL "$4" print-build)
        mkdir -p "$d/bin/$isa"
        cp "$b/ml_run" "$b/ml_bench" "$b/ml_tok" "$d/bin/$isa/"
        "${2}strip" "$d/bin/$isa/"*
    done
    unix_wrappers "$d" "$3"
    tar -C "$STAGE" -czf "$DIST/krafla-$VER-$1.tar.gz" "krafla-$VER-$1"
}

windows() {   # windows "ISAS best first"
    d="$STAGE/krafla-$VER-windows-x86_64"
    common "$d"
    printf 'set(CMAKE_SYSTEM_NAME Windows)\nset(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)\n' > "$STAGE/mingw.cmake"
    for isa in $1; do
        b="$STAGE/win-$isa"
        # MinGW cannot align the stack to 32 bytes for AVX spills: assemble
        # aligned vector moves as unaligned ones
        cmake -S . -B "$b" -DML_ISA=$isa -DML_THREADS=ON -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_TOOLCHAIN_FILE="$PWD/$STAGE/mingw.cmake" -DCMAKE_EXE_LINKER_FLAGS=-static \
            -DCMAKE_C_FLAGS="-Wa,-muse-unaligned-vector-move" > /dev/null
        cmake --build "$b" --target ml_run ml_bench ml_tok -j4 > /dev/null
        mkdir -p "$d/bin/$isa"
        cp "$b/ml_run.exe" "$b/ml_bench.exe" "$b/ml_tok.exe" "$d/bin/$isa/"
        x86_64-w64-mingw32-strip "$d/bin/$isa/"*.exe
    done
    for tool in ml_run ml_bench ml_tok; do
        {
            printf '@echo off\r\n'
            printf 'rem Runs %s from the best build in bin\\ that this CPU supports.\r\n' "$tool"
            printf 'rem set KRAFLA_ISA=avx2 (for example) picks one instead.\r\n'
            printf 'setlocal\r\n'
            printf 'set "ISAS=%s"\r\n' "$1"
            printf 'if defined ISA_LM_ISA set "ISAS=%%ISA_LM_ISA%%"\r\n'
            printf 'if defined KRAFLA_ISA set "ISAS=%%KRAFLA_ISA%%"\r\n'
            printf 'for %%%%I in (%%ISAS%%) do (\r\n'
            printf '  if exist "%%~dp0bin\\%%%%I\\ml_tok.exe" (\r\n'
            printf '    "%%~dp0bin\\%%%%I\\ml_tok.exe" --help >nul 2>&1\r\n'
            printf '    if not errorlevel 3 ( "%%~dp0bin\\%%%%I\\%s.exe" %%* & exit /b )\r\n' "$tool"
            printf '  )\r\n'
            printf ')\r\n'
            printf 'echo krafla: no build in %%~dp0bin runs on this CPU 1>&2\r\n'
            printf 'exit /b 3\r\n'
        } > "$d/$tool.cmd"
    done
    (cd "$STAGE" && zip -qr "../../$DIST/krafla-$VER-windows-x86_64.zip" "krafla-$VER-windows-x86_64")
}

linux linux-x86_64 "" "avx512vnni avxvnni avx2 sse41 sse2"
linux linux-aarch64 aarch64-linux-gnu- "neon_i8mm sve neon_dotprod neon"
linux linux-armv7hf arm-linux-gnueabihf- "neon scalar"
windows "avx512vnni avxvnni avx2 sse41 sse2"
(cd "$DIST" && sha256sum krafla-$VER-* > SHA256SUMS)
ls -l "$DIST"
