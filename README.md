# legacy-ml

A small inference runtime for Llama-style decoder Transformers with INT8 and INT4
weights. It is written in C11 and needs only libc. It runs on x86-64 CPUs from
SSE2 up to AVX-512 VNNI, and on ARM CPUs from ARMv7 NEON up to SVE.

All code is written against a portable baseline. You choose the ISA acceleration
when you build:

```sh
make ISA=avx2                                  # build/avx2/{ml_run,ml_bench,test_*}
make ISA=neon CROSS=aarch64-linux-gnu-         # cross build, run under qemu
make ISA=sse2 THREADS=pthread ASM=1            # worker pool + hand-written asm kernels
```

Design: `docs/design/2026-10-22-legacy-isa-ml-runtime-design.md`

## Build matrix

| `ISA`          | Compiler flags                                    | Minimum CPU                   | Accelerated ops |
|----------------|---------------------------------------------------|-------------------------------|-----------------|
| `scalar`       | none                                              | any                           | reference for all ops |
| `sse2`         | `-msse2`                                          | any x86-64                    | all 10 |
| `sse41`        | `-msse4.1`                                        | Core 2 Penryn / Nehalem       | int8 matvec (`pmaddubsw`) |
| `avx2`         | `-mavx2 -mfma`                                    | Haswell / Zen 1               | all 10 |
| `avxvnni`      | `+ -mavxvnni`                                     | Alder Lake / Zen 5            | int8 matvec (`vpdpbusd` ymm) |
| `avx512vnni`   | `-mavx512f -mavx512bw -mavx512vnni -mfma`         | Cascade Lake / Ice Lake / Zen 4 | matvec, quantize, rmsnorm, add, dot, axpy |
| `neon`         | ARMv7: `-march=armv7-a -mfpu=neon -mfloat-abi=hard`; AArch64: `-march=armv8-a` | Cortex-A7/A9 with NEON; any AArch64 | all 10 |
| `neon_dotprod` | `-march=armv8.2-a+dotprod`                        | Cortex-A55/A76, Apple M1      | int8 matvec (`sdot`) |
| `sve`          | `-march=armv8.2-a+dotprod+sve`                    | Neoverse V1/N2, A64FX         | matvec, quantize, rmsnorm, add, dot, axpy (any vector length) |

If the selected ISA has no kernel for an op, the build uses the next lower kernel
in its chain. The chains are `avx512vnni → avx2 → sse41 → sse2 → scalar`,
`avxvnni → avx2 → …` and `sve → neon_dotprod → neon → scalar`. This binding
happens at compile time in `config/ml_config.h`, so calls go directly to the
chosen kernel with no function pointers.

Every executable checks the CPU before it runs any ISA-specific code. It uses
CPUID/XGETBV on x86 and `AT_HWCAP` on ARM Linux. On an unsupported CPU it prints
`error: CPU lacks avx2 required by ISA=avx2` and exits with code 3 instead of
crashing with an illegal-instruction signal.

Other options: `THREADS=1|pthread`, `ASM=0|1`, `DEBUG=1`, `EXTRA_CFLAGS=-DML_NO_MMAP`
(use `fread` instead of `mmap`).
CMake (`-DML_ISA=…`) builds the x86-64 targets with GCC, Clang or MSVC.

## Quick start

```sh
# 1. Get a model: convert a checkpoint, or generate a random test model
python3 tools/convert.py llama2c stories15M.bin stories15M.mlb --wtype q8
python3 tools/convert.py hf ./Llama-dir model.mlb --wtype q4 --max-seq 2048
python3 tools/gen_random.py --out rand.mlb --wtype q4

# 2. Generate text
make ISA=avx2
build/avx2/ml_run stories15M.mlb -z tokenizer.bin -i "Once upon a time" -n 128 -T 0.8
build/avx2/ml_bench stories15M.mlb
```

## Model format and numerics

- **`.mlb` file:** a 256-byte header, then tensors. Each array starts on a 64-byte
  boundary, and the file is mapped with `mmap` without copying.
- **Weights:** stored as `Q8_0` (int8 values in [-127, 127]) or `Q4_0` (packed
  nibbles), in blocks of 32 values. Each block has an FP32 scale. The scales are
  FP32 because SSE2 and ARMv7 have no FP16 conversion instructions.
- **Activations:** quantized to Q8 before every matvec. Block dot products are
  computed in integers, then multiplied by the scales in FP32.
- **Rounding:** `quantize_q8` rounds half away from zero, and everything is
  built with `-ffp-contract=off`. As a result every ISA produces bit-identical
  quantized activations.

Limits checked at load time: `dim`, `hidden_dim` and `n_layers` up to 65536;
`vocab` and `max_seq` up to 2^24 (enough for Llama-3 vocabularies). All size
arithmetic is overflow-checked, so on 32-bit ARMv7 an oversized model fails with
`out of memory` instead of crashing. Plain RoPE only: the converter refuses HF
configs that set `rope_scaling`.

## Adding or replacing a kernel

1. Add `ml_k_<op>_<isa>` to `kernels/<isa>/k_<isa>.c`.
2. Declare it in `k_<isa>.h` together with `#define ML_HAS_<OP>_<ISA>`. Nothing
   else changes: `ml_config.h` picks the new kernel up automatically.
3. Run `make ISA=<isa> test e2e`. `test_kernels` compares the bound kernel with
   the scalar one and with a double-precision reference. `e2e` compares model
   logits with the numpy reference in `tools/ref_forward.py`.

To replace a kernel with assembly, list it in `kernels/<isa>/asm.mk` and add
`kernels/<isa>/<name>.S`, using the same symbol and the same C ABI. Building with
`ASM=1` assembles that file and compiles the C version out through
`-DML_ASM_<NAME>_<ISA>`. `kernels/sse2/add.S` is a sample (x86-64 System V).

## Verification status

`make test-all` runs the full matrix below and prints this summary:

| Target | How it was verified |
|---|---|
| scalar, sse2, sse41, avx2, avxvnni, avx512vnni | native: kernel tests, e2e against the numpy reference |
| ISA guard on x86 | `qemu-x86_64 -cpu core2duo/Nehalem/SandyBridge/Haswell` |
| ARMv7 scalar and neon | `arm-linux-gnueabihf-gcc` + `qemu-arm` |
| AArch64 neon, neon_dotprod | `aarch64-linux-gnu-gcc` + `qemu-aarch64` |
| sve | qemu at 128-bit and 512-bit vector length |
| ISA guard on ARM | `qemu-aarch64 -cpu cortex-a53/cortex-a76` |
| THREADS=pthread | `-t 3/4/7` give logits bit-identical to `-t 1`; ThreadSanitizer clean |
| ASM=1 | sse2 build with `add.S` |
| CMake | GCC + Ninja (`ctest`); MinGW cross-compile for Windows (built, not executed) |
| **Not verified** | MSVC; running on real ARM hardware; `stories15M` generation (download was blocked here) |

## Benchmarks (this host, single thread, noisy shared VM)

- **matvec GOPS:** the `ml_bench` kernel benchmark on a 4096×4096 matrix.
- **quantize ns:** time to quantize one 4096-value activation vector to Q8.
- **tok/s:** 128 greedy tokens on a random model (dim 512, 6 layers, hidden 1376,
  vocab 4096).

| ISA        | matvec Q8 GOPS | matvec Q4 GOPS | quantize ns | tok/s Q8 | tok/s Q4 |
|------------|---------------:|---------------:|------------:|---------:|---------:|
| scalar     |  8.8 |  7.3 | 40285 | 212 | 203 |
| sse2       | 17.5 | 14.7 |  2039 | 458 | 357 |
| sse41      | 27.2 | 16.8 |  1896 | 529 | 476 |
| avx2       | 36.8 | 27.0 |  1168 | 811 | 703 |
| avxvnni    | 39.7 | 26.2 |  1147 | 759 | 589 |
| avx512vnni | 43.0 | 28.9 |   854 | 854 | 659 |

The pthread pool synchronizes once per matvec. With matrices this small, that
synchronization costs more than it saves: `-t 4` measured 286 tok/s against
854 tok/s for `-t 1`. Use threads only for large models.
