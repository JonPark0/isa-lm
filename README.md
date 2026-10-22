# isa-lm

A dependency-free C11 inference runtime for Llama-style decoder Transformers with
INT8 and INT4 weights. It runs on x86-64 CPUs from SSE2 up to AVX-512 VNNI, and
on ARM CPUs from ARMv7 NEON up to SVE.

The code is written against a portable baseline. You pick the acceleration when
you compile, and each op is bound to the best kernel for that ISA, with no
runtime dispatch:

```sh
make ISA=avx2                                   # build/avx2/{ml_run,ml_bench,test_*}
make ISA=neon CROSS=aarch64-linux-gnu-          # cross build; tests run under qemu
make ISA=avx512vnni THREADS=pthread             # with the worker pool
```

| `ISA` | Minimum CPU | Accelerated |
|-------|-------------|-------------|
| `scalar` | any | reference implementation of all ops |
| `sse2` | any x86-64 | all ops |
| `sse41` | Penryn / Nehalem | int8 matvec (`pmaddubsw`) |
| `avx2` | Haswell / Zen | all ops, FMA |
| `avxvnni` | Alder Lake / Zen 5 | int8 matvec (`vpdpbusd` ymm) |
| `avx512vnni` | Cascade Lake / Ice Lake / Zen 4 | matvec, quantize, norms, dot, axpy |
| `neon` | ARMv7 + NEON, any AArch64 | all ops |
| `neon_dotprod` | Cortex-A55/A76, Apple M1 | int8 matvec (`sdot`) |
| `sve` | Neoverse V1/N2, Graviton 3 | matvec, quantize, norms, dot, axpy (any vector length) |

If you run a binary on a CPU that lacks its ISA, it exits with code 3 and names
the missing feature. It does not crash with SIGILL.

## Quick start

```sh
pip install numpy
make ISA=avx2

# llama2.c checkpoint + tokenizer
python3 tools/convert.py llama2c stories15M.bin stories15M.mlb --wtype q8
build/avx2/ml_run stories15M.mlb -z tokenizer.bin -i "Once upon a time" -n 128 -T 0.8

# Hugging Face Llama (safetensors), including Llama 3 RoPE scaling; token ids in, ids out
python3 tools/convert.py hf ./Llama-3.2-1B llama.mlb --wtype q4 --max-seq 4096
build/avx2/ml_run llama.mlb --tokens 128000,9906,1917 -n 32 --eos 128001
```

## Documentation

| Document | Contents |
|----------|----------|
| [User guide](docs/user-guide.md) | building, cross builds, converting models, CLI options, threads, limits |
| [Architecture](docs/architecture.md) | layers, compile-time kernel binding, ISA guard, data flow, memory, threading, numerics |
| [File format](docs/file-format.md) | the `.mlb` format: header offsets, tensor layout, quantization rule, RoPE scaling |
| [Writing kernels](docs/kernels.md) | adding a kernel or an ISA, kernel rules, per-ISA idioms, assembly overrides |
| [Testing](docs/testing.md) | test layers, qemu, sanitizers, CI, benchmarks |
| [Design spec](docs/design/2026-10-22-legacy-isa-ml-runtime-design.md) | original design and implementation plan |

## Status

| Area | Status |
|------|--------|
| x86 builds | tested natively for all six ISAs: kernel tests against scalar and double-precision references, end-to-end logits against a numpy reference |
| ARM builds | tested under qemu: ARMv7 NEON, AArch64 NEON, dotprod, and SVE at 128-bit and 512-bit vector length; CI also runs them on native Arm runners |
| Thread pool | bit-identical results for any thread count; ThreadSanitizer clean; `-t 4` about 1.8–3.2× faster than `-t 1` on 4 vCPUs |
| Windows | builds with MinGW; MSVC is supported in CMake but not verified |
| Not implemented | tokenizers other than llama2.c `tokenizer.bin`, RoPE scaling types other than llama3, non-Llama architectures |

## License

No license has been chosen yet.
