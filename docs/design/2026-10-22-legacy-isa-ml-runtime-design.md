# Legacy-ISA ML Inference Runtime — Design Spec

Date: 2026-10-22
Status: Approved in conversation (sections 1–4), pending written-spec review

## 1. Goal & Scope

A dependency-free, low-level **inference-only** runtime for small Llama-style
decoder Transformers with INT8/INT4 quantized weights, running on legacy and
modern x64 / ARM CPUs. The code is written against a portable baseline; ISA
acceleration is selected **at compile time** via a build flag.

### Requirements (from user)
- Low-level, runs on legacy x64 and ARM instruction sets.
- Portable baseline code; ISA-specific acceleration enabled per build.
- Inference only. Small Transformer, INT8/INT4 weights.
- Baseline ISA floor: x86-64 v1 (SSE2), ARMv7 (NEON optional), AArch64.

### Assumptions
- Single-process CPU inference, little-endian targets only.
- Runtime dependencies: libc only (pthread optional).
- Language: C11 + compiler intrinsics; optional hand-written `.S` overrides.

### Success criteria
1. Every ISA build produces results matching the scalar reference within the
   tolerances in §7.
2. Accelerated builds are measurably faster than scalar on `ml_bench`.
3. A binary built for an unsupported ISA exits with a clear error, never SIGILL.

### Out of scope
Training/backprop, GPU, runtime multi-ISA dispatch, general graph (ONNX)
loading, big-endian targets, FP16/BF16 weights.

## 2. Architecture

```
legacy-ml/
  app/       ml_run.c        CLI: load model → generate tokens
             ml_bench.c      kernel + model benchmarks
  model/     transformer.c/h forward pass, KV cache, sampler
             tokenizer.c/h   llama2.c tokenizer.bin format (BPE)
  ops/       ops.c/h         thin dispatch layer over ml_k_* kernels + threading
  kernels/   scalar/ sse2/ sse41/ avx2/ avxvnni/ avx512vnni/
             neon/ neon_dotprod/ sve/
  core/      arena.c/h       single aligned allocation, bump allocator
             tensor.h        ml_qtensor, ml_qact
             quant.h         block format constants
             file.c/h        mmap / CreateFileMapping / fread fallback
             isa_check.c     CPUID / HWCAP guard (compiled without ISA flags)
             threads.c/h     worker pool + barrier (THREADS=pthread only)
  config/    ml_config.h     ISA macro → kernel symbol binding + fallback chain
  tools/     convert.py      checkpoint → .mlb
             gen_random.py   random-weight test model
  tests/     test_kernels.c  test_quant.c  test_e2e.sh  test_convert.py
  Makefile   CMakeLists.txt (MSVC/x64)
```

Dependency direction: `app → model → ops → kernels → core`. Kernels depend only
on `core/tensor.h`, `core/quant.h` and their ISA's intrinsics header.

## 3. Build System & Compile-Time Customization

```
make ISA=<isa> [THREADS=1|pthread] [ASM=0|1] [CROSS=<prefix>] [DEBUG=0|1]
```

| ISA            | Defines / flags                                              |
|----------------|--------------------------------------------------------------|
| `scalar`       | `-DML_ISA_SCALAR`                                            |
| `sse2`         | `-DML_ISA_SSE2 -msse2`                                       |
| `sse41`        | `-DML_ISA_SSE41 -msse4.1` (implies SSSE3)                    |
| `avx2`         | `-DML_ISA_AVX2 -mavx2 -mfma`                                 |
| `avxvnni`      | `-DML_ISA_AVXVNNI -mavx2 -mfma -mavxvnni`                    |
| `avx512vnni`   | `-DML_ISA_AVX512VNNI -mavx512f -mavx512bw -mavx512vnni -mfma`|
| `neon` (v7)    | `-DML_ISA_NEON -march=armv7-a -mfpu=neon -mfloat-abi=hard`   |
| `neon` (A64)   | `-DML_ISA_NEON -march=armv8-a`                               |
| `neon_dotprod` | `-DML_ISA_NEON_DOTPROD -march=armv8.2-a+dotprod`             |
| `sve`          | `-DML_ISA_SVE -march=armv8.2-a+sve`                          |

- `neon` selects v7 vs A64 flags from the target triple (`__aarch64__`).
- `ml_config.h` binds each generic name statically, e.g.
  `#define ml_k_matvec_q8 ml_k_matvec_q8_avx2`. No function pointers; LTO may
  inline.
- **Fallback chain** (single table in `ml_config.h`): a kernel missing for the
  selected ISA resolves to the next lower one:
  `avx512vnni → avxvnni → avx2 → sse41 → sse2 → scalar`,
  `sve → neon_dotprod → neon → scalar`.
  The chain is implemented by per-kernel `ML_HAS_<KERNEL>_<ISA>` macros that
  each ISA directory's header defines.
- The scalar kernels are always compiled (used as fallback and by tests).
- `ASM=1`: a kernel listed in `kernels/<isa>/asm.mk` is built from
  `kernels/<isa>/<kernel>.S` instead of `.c`; same symbol, same C ABI
  (SysV x64 / AAPCS / AAPCS64). Initial release ships one sample,
  `kernels/sse2/add.S` (`ml_k_add_sse2`, SysV x64), to prove the mechanism.
- Compilers: GCC and Clang (primary); MSVC for x64 via CMake.

## 4. Quantization Formats

Block size `QK = 32`, symmetric quantization. All `K` dimensions are multiples
of 32 (enforced by converter and loader).

| Format | Use                | Quants                                         | Scale        | bits/w |
|--------|--------------------|------------------------------------------------|--------------|--------|
| `Q8_0` | weights, activations | int8, clamped to [-127, 127]                | float32/block | 9     |
| `Q4_0` | weights            | 16 B/block; low nibble = elem 0–15, high nibble = elem 16–31; value = nibble − 8 ∈ [-8, 7] | float32/block | 5 |
| `F32`  | norm weights, KV cache, embedding (if not quantized) | — | — | 32 |

- **FP32 scales**: SSE2/ARMv7 have no FP16 conversion; avoids per-block
  conversion cost on legacy CPUs.
- **SoA per tensor**: `q` (all quant bytes, 64 B aligned) and `d` (all scales,
  64 B aligned) are separate arrays → aligned SIMD loads on every ISA.
- **[-127, 127] clamp**: enables the x86 `|x|`/`sign(y)` trick with
  `pmaddubsw` without int16 saturation (127·127·2 = 32258 < 32767).
- Q8 quantization: `d = max|x| / 127`, `q = round(x / d)`; `d = 0` → all-zero
  block.
- Integer dot per block: `sumi = Σ qw·qx` (int32, no overflow: 32·127·127 <
  2³¹); accumulate `sum += (float)sumi * dw * dx` in FP32.

```c
typedef struct { int8_t *q; float *d; int rows, cols; uint8_t type; } ml_qtensor;
typedef struct { int8_t *q; float *d; int n; } ml_qact;   /* Q8_0 activations */
```

## 5. Kernel Interface

All ISAs implement the same C ABI. Kernel-suffix naming: `ml_k_<op>_<isa>`.

```c
void ml_k_matvec_q8 (float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4 (float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_quantize_q8(ml_qact *out, const float *x, int k);
void ml_k_rmsnorm   (float *o, const float *x, const float *w, int n, float eps);
void ml_k_softmax   (float *x, int n);
void ml_k_rope      (float *q, float *k, const float *cos_t, const float *sin_t,
                     int head_dim, int n_heads, int n_kv_heads); /* tables: row `pos` of precomputed [max_seq][head_dim/2] */
void ml_k_silu_mul  (float *g, const float *u, int n);   /* g = silu(g) * u */
void ml_k_add       (float *y, const float *x, int n);   /* y += x */
float ml_k_dot_f32  (const float *a, const float *b, int n);           /* attention scores */
void ml_k_axpy_f32  (float *y, float a, const float *x, int n);        /* attention Σ p·V */
```

- `matvec` processes rows `[r0, r1)`; caller splits rows across threads.
  Kernels process **4 rows at a time** (one activation block load reused 4×),
  with a scalar-row tail for remainders.
- Rounding everywhere: `quantize_q8` uses `q = trunc(x*id + copysign(0.5, x))`
  with `id = 127/amax` (round half away from zero, expressible on every ISA
  including ARMv7). All code builds with `-ffp-contract=off` so scalar and
  SIMD quantization are bit-identical; SIMD kernels use FMA only via explicit
  intrinsics.
- RoPE uses interleaved pairs `(2i, 2i+1)` with `freq_i = theta^(-2i/head_dim)`
  (llama2.c convention); `cos/sin` tables are precomputed at load into the arena.
  The HF converter permutes `wq`/`wk` accordingly.
- SIMD `exp` for `softmax`/`silu`: polynomial approximation with range
  reduction, relative error ≤ 1e-6 over [-88, 88]. Scalar uses `expf`.

### Integer dot-product strategy per ISA

| ISA            | Q8·Q8 block dot                                                        |
|----------------|------------------------------------------------------------------------|
| `sse2`         | sign-extend int8→int16 (`punpck` + `psraw 8`) → `pmaddwd` → int32       |
| `sse41`        | `pabsb(x)`, `psignb(y,x)` → `pmaddubsw` → `pmaddwd(·,1)`               |
| `avx2`         | same as sse41 at 256-bit; FMA for scale accumulation                    |
| `avxvnni`      | `pabsb`/`psignb` → `vpdpbusd` (VEX, 256-bit)                            |
| `avx512vnni`   | `vpabsb`/sign via mask → `vpdpbusd` (512-bit, 2 blocks per op)          |
| `neon`         | `vmull_s8` + `vpadalq_s16` (works on ARMv7 and A64)                     |
| `neon_dotprod` | `vdotq_s32` (`sdot`)                                                    |
| `sve`          | `svdot_s32`, vector-length-agnostic loop over blocks                    |

Q4 kernels unpack nibbles (`and 0x0F`, `shift 4`, subtract 8) into int8 then
reuse the Q8 path.

## 6. Model File, Execution, Memory, Threading

### `.mlb` file (little-endian)
- Header (fixed 256 B): magic `"MLB1"`, `u32 version=1`, `u32 dim, n_layers,
  n_heads, n_kv_heads, hidden_dim, vocab, max_seq`, `f32 rope_theta, norm_eps`,
  `u32 wtype` (Q8_0 | Q4_0), `u32 shared_classifier`, zero padding.
- Tensors follow in fixed order, each array (`q`, then `d`) starting on a 64 B
  boundary: `tok_emb`; per layer `attn_norm, wq, wk, wv, wo, ffn_norm, w1, w2,
  w3`; `final_norm`; `wcls` (omitted if shared). Norms are F32; `tok_emb` and
  matrices use `wtype`.
- `tools/convert.py`: llama2.c checkpoint and HF safetensors (Llama) → `.mlb`.
  Tokenizer uses llama2.c `tokenizer.bin` as-is.
- Validation models: TinyStories `stories15M` (smoke), random-weight tiny model
  from `tools/gen_random.py` (tests).

### Loading & memory
- `mmap` (POSIX) / `CreateFileMapping` (Windows) / `fread` fallback
  (`ML_NO_MMAP`). Weights are zero-copy pointers into the mapping.
- Runtime buffers (KV cache F32 `n_layers × max_seq × kv_dim × 2`, activation
  buffers, Q8 activation buffers, logits) sized from the header and carved from
  **one** 64 B-aligned arena allocation. No allocation after load.

### Forward pass (per token)
```
x = dequant(tok_emb[token])
for each layer:
  h = rmsnorm(x); qa = quantize_q8(h)
  q,k,v = matvec(Wq|Wk|Wv, qa); rope(q,k,pos); KV[layer][pos] = k,v
  per head: s = dot_f32(q_h, K_t) / sqrt(hd); softmax(s); o_h = Σ axpy(s_t, V_t)
  x += matvec(Wo, quantize_q8(o))
  h = rmsnorm(x); qa = quantize_q8(h)
  g = matvec(W1, qa); u = matvec(W3, qa); silu_mul(g, u)
  x += matvec(W2, quantize_q8(g))
logits = matvec(Wcls, quantize_q8(rmsnorm(x)))
token = sample(logits)   /* greedy | temperature | top-p, seeded xorshift RNG */
```
GQA: KV head for query head `h` is `h / (n_heads / n_kv_heads)`.

### Threading
- `THREADS=1` (default): no pthread code compiled.
- `THREADS=pthread`: persistent pool of N workers (`-t N` at runtime) + barrier.
  `matvec` rows split statically in multiples of 4; attention split by head.

### Error handling
- Library functions return `ml_status` (`ML_OK, ML_E_IO, ML_E_FORMAT,
  ML_E_SHAPE, ML_E_NOMEM, ML_E_ISA`); no `abort()` in library code.
- Loader validates magic, version, sizes vs file length, `K % 32 == 0`,
  alignment, `n_heads % n_kv_heads == 0`.
- `ml_isa_check()` runs first in every executable. x86: CPUID (+ XGETBV for
  OS AVX/AVX-512 state). ARM: `getauxval(AT_HWCAP/AT_HWCAP2)` on Linux;
  compile-time-only assumption elsewhere. `isa_check.c` is compiled without ISA
  flags, and so is each executable's `main` (`app/main_guard.c`), which calls
  `ml_isa_check()` before entering the ISA-compiled `ml_main()` — no
  ISA-specific instruction can run before the check. Unsupported CPU →
  `ML_E_ISA` with the missing feature named.

## 7. Testing, Verification, Benchmarks

1. `tests/test_kernels.c` — every kernel of the built ISA vs scalar reference
   on random inputs and edge shapes (K = 32, rows not multiple of 4, single
   row, `r0 > 0`).
   - Integer path: per-block int32 sums bit-exact; final FP32 relative error
     ≤ 1e-5.
   - FP path (`softmax, silu_mul, rmsnorm, rope, dot_f32, axpy_f32`): relative
     error ≤ 1e-5.
2. `tests/test_quant.c` — quantize→dequantize error ≤ d/2 per element;
   zero block handling; clamp to ±127.
3. `tests/test_e2e.sh` — random tiny model: logits of each ISA build vs scalar
   build (max relative error ≤ 1e-3 over 16 steps) and identical greedy token
   sequences; `stories15M` smoke generation.
4. `tests/test_convert.py` — header fields, alignment, file size.
5. `ml_isa_check` negative test: run the `avx512vnni` build on this host and
   expect `ML_E_ISA`.

`make test-all` build matrix:

| ISA                                   | Verification in this environment                       |
|---------------------------------------|--------------------------------------------------------|
| scalar, sse2, sse41, avx2, avxvnni    | native execution                                       |
| avx512vnni                            | compile + ISA-guard rejection (host lacks AVX512-VNNI) |
| neon (ARMv7)                          | `arm-linux-gnueabihf-gcc` + `qemu-arm`                 |
| neon, neon_dotprod, sve (AArch64)     | `aarch64-linux-gnu-gcc` + `qemu-aarch64 -cpu max`      |

ARM rows require installing cross toolchains and `qemu-user`; if unavailable,
those rows are compile-only and reported as such.

`ml_bench`: per-kernel GB/s and GOPS at model-relevant shapes, model tok/s,
printed as a per-ISA table.
