# Legacy-ISA ML Inference Runtime Implementation Plan

**Goal:** A libc-only C11 runtime that runs Q8_0/Q4_0 Llama-style decoders on x86-64 (SSE2 → AVX-512 VNNI) and ARM (ARMv7 NEON → SVE), with the ISA chosen by `make ISA=<isa>`.

**Architecture:** `app → model → ops → kernels → core`. Every op has a scalar reference kernel plus optional per-ISA kernels; `config/ml_config.h` binds the generic `ml_k_*` name to the best kernel available in the selected ISA's fallback chain at compile time. A numpy reference (`tools/ref_forward.py`) validates the scalar build; every other build is validated against scalar.

**Tech Stack:** C11 (GCC 13 / Clang 18), compiler intrinsics, GNU Make (+ CMake for MSVC), Python 3 + numpy for tools/tests, qemu-user + cross GCC for ARM.

**Spec:** `docs/design/2026-10-22-legacy-isa-ml-runtime-design.md`

## Global Constraints

- C11, `-O2 -Wall -Wextra -Werror -ffp-contract=off` for every C file; no runtime dependency beyond libc (+ pthread only when `THREADS=pthread`).
- Little-endian only; `QK = 32`; every `K` dimension `% 32 == 0`.
- Q8_0 quants clamped to `[-127, 127]`; scales FP32; Q4 nibble low = elem 0–15, high = elem 16–31, value = nibble − 8.
- Rounding in `quantize_q8`: `id = amax ? 127.0f/amax : 0; d = amax/127.0f; q = (int)trunc(x*id + copysignf(0.5f, x))`, clamped to ±127.
- Tensor storage is SoA: `q` array and `d` array, each 64 B aligned.
- Kernel symbols: `ml_k_<op>_<isa>`; generic names `ml_k_<op>` are macros from `config/ml_config.h` only.
- Fallback chains: `avx512vnni → avxvnni → avx2 → sse41 → sse2 → scalar`; `sve → neon_dotprod → neon → scalar`.
- Library code returns `ml_status` (`ML_OK=0, ML_E_IO, ML_E_FORMAT, ML_E_SHAPE, ML_E_NOMEM, ML_E_ISA`); no `abort()`.
- No allocation after model load (one 64 B-aligned arena).
- Tolerances: kernel FP outputs rel. err ≤ 1e-5; integer matvec with unit scales bit-exact; e2e logits rel. err ≤ 1e-3 vs reference; quantize_q8 bit-exact across ISAs.
- `main` of every executable lives in `app/main_guard.c` (compiled without ISA flags) and calls `ml_isa_check()` before `ml_main()`.

## Review Focus

1. **Truncated/corrupt `.mlb`** (file shorter than header implies, bad magic, `K % 32 != 0`) → `ML_E_FORMAT`/`ML_E_SHAPE`, no crash. Test: Task 4 `test_load_rejects_truncated`, `test_load_rejects_bad_shape`.
2. **Zero activation block** (`amax == 0`) → `d = 0`, all-zero quants, no NaN in matvec or logits. Test: Task 2 `test_quantize_zero_block`.
3. **Softmax with large/−inf-like inputs** (values up to 1e4, entries at −1e30) → finite output summing to 1. Test: Task 2 `test_softmax_extreme`.
4. **Sequence reaching `max_seq`** (prompt + steps > max_seq) → generation stops at `max_seq` with exit 0, no KV overflow. Test: Task 6 e2e case `overflow`.
5. **Unknown ISA / bad CLI args** (`make ISA=foo`, `ml_run` missing model) → explicit error listing valid ISAs / usage, non-zero exit. Test: Task 1 step `make ISA=foo` check; Task 6 `ml_run` usage check.

## File Structure

```
Makefile                       ISA→flags map, chain, targets: all, test, test-all, bench, clean
mk/isa.mk                      per-ISA CFLAGS, chain lists, ARCH detection, RUN (qemu) prefix
CMakeLists.txt                 x64 build for MSVC/other generators (Task 13)
config/ml_config.h             ISA headers include + generic-name binding
core/ml.h                      ml_status, QK, type codes, ml_qtensor, ml_qact, alignment macros
core/arena.c/.h                aligned arena
core/quant_ref.c/.h            dequant_row, block helpers (non-kernel, scalar)
core/file.c/.h                 mmap / fread mapping
core/isa_check.c/.h            CPUID/HWCAP guard
core/threads.c/.h              pool (pthread) / inline (single)
kernels/scalar/k_scalar.c/.h   all kernels, reference
kernels/<isa>/k_<isa>.c/.h     per-ISA kernels; header defines ML_HAS_<OP>_<ISA>
kernels/<isa>/asm.mk           ASM_KERNELS list (sse2 only has entries)
kernels/sse2/add.S             sample asm kernel
ops/ops.c/.h                   threaded matvec, attention, wrappers
model/mlb.c/.h                 .mlb loader → ml_model
model/transformer.c/.h         forward pass, KV cache, rope tables
model/sampler.c/.h             greedy / temperature / top-p, xorshift64*
model/tokenizer.c/.h           llama2.c tokenizer.bin BPE
app/main_guard.c               guarded main
app/ml_run.c                   ml_main for generation CLI
app/ml_bench.c                 ml_main for benchmarks
tools/mlb.py                   .mlb read/write + quantization (shared by tools)
tools/gen_random.py            random tiny model
tools/ref_forward.py           numpy forward, same quantized semantics
tools/convert.py               llama2.c / HF safetensors → .mlb
tests/tt.h                     test macros
tests/test_quant.c  tests/test_kernels.c  tests/test_load.c
tests/test_e2e.sh  tests/test_convert.py  tests/test_isa_guard.sh
```

Build output: `build/<isa>[-<arch>]/{ml_run,ml_bench,test_quant,test_kernels,test_load}`.

---

### Task 1: Build skeleton, core types, test harness

**Files:**
- Create: `Makefile`, `mk/isa.mk`, `core/ml.h`, `core/arena.c`, `core/arena.h`, `tests/tt.h`, `tests/test_arena.c`, `.gitignore` (`build/`, `*.mlb`, `models/`)

**Interfaces:**
- Produces:
  - `core/ml.h`: `typedef enum { ML_OK=0, ML_E_IO, ML_E_FORMAT, ML_E_SHAPE, ML_E_NOMEM, ML_E_ISA } ml_status;` `#define QK 32`; `enum { ML_F32=0, ML_Q8_0=1, ML_Q4_0=2 };` `typedef struct { const int8_t *q; const float *d; int rows, cols; uint8_t type; } ml_qtensor;` (Q4: `q` holds `rows*cols/2` packed bytes; kernels cast to `const uint8_t*`) `typedef struct { int8_t *q; float *d; int n; } ml_qact;` `#define ML_ALIGN 64`, `const char *ml_status_str(ml_status)`.
  - `core/arena.h`: `typedef struct { uint8_t *base; size_t cap, off; } ml_arena;` `ml_status ml_arena_init(ml_arena*, size_t cap);` `void *ml_arena_alloc(ml_arena*, size_t n);` (64 B aligned, NULL when full) `void ml_arena_free(ml_arena*);`
  - `tests/tt.h`: `TT_ASSERT(cond)`, `TT_EQ_INT(a,b)`, `TT_NEAR(a,b,rtol)` (passes if `|a-b| <= rtol*max(|a|,|b|,1e-6)`), `TT_RUN(fn)`, `TT_MAIN_END()` returning non-zero if any failure; each failure prints `file:line`.
  - Makefile variables: `ISA` (default `scalar`), `THREADS` (`1`|`pthread`), `ASM` (`0`|`1`), `CROSS`, `DEBUG`, `EXTRA_CFLAGS`. Output dir `build/<isa>` (host) or `build/<isa>-<arch>` (cross). `ARCH := $(shell $(CC) -dumpmachine | cut -d- -f1)` normalized to `x86_64|aarch64|arm`. Each ISA in the chain gets `-DML_ENABLE_<ISA>`; target gets `-DML_ISA_<ISA>`. `RUN` = `qemu-aarch64 [-cpu max]` / `qemu-arm -cpu max` when `ARCH != host`, else empty. Cross builds link `-static`.

- [ ] **Step 1: Write `tests/test_arena.c`** — `test_arena_alignment` (three allocs of 1, 100, 7 bytes all `% 64 == 0`), `test_arena_full` (cap 128: alloc 64 ok, alloc 65 → NULL).
- [ ] **Step 2: Run** `make ISA=scalar test` → Expected: build fails (no arena).
- [ ] **Step 3: Implement** `core/ml.h`, `core/arena.c/.h`, `tests/tt.h`, `Makefile`, `mk/isa.mk` with the full flag table from spec §3 (neon picks v7 vs A64 flags by `ARCH`). Unknown `ISA` → `$(error Unknown ISA '$(ISA)'. Valid: scalar sse2 sse41 avx2 avxvnni avx512vnni neon neon_dotprod sve)`. `make test` builds and runs every `build/<isa>/test_*` via `$(RUN)`.
- [ ] **Step 4: Verify** `make ISA=scalar test` → `test_arena: 2 passed`; `make ISA=foo 2>&1 | grep "Valid: scalar"` → match, exit non-zero.
- [ ] **Step 5: Commit** `git commit -m "build: skeleton, core types, arena, test harness"`

### Task 2: Scalar kernels, binding header, kernel test harness

**Files:**
- Create: `kernels/scalar/k_scalar.c/.h`, `config/ml_config.h`, `core/quant_ref.c/.h`, `tests/test_quant.c`, `tests/test_kernels.c`

**Interfaces:**
- Consumes: `core/ml.h` (Task 1).
- Produces:
  - Scalar kernels with spec §5 signatures, suffix `_scalar`: `ml_k_matvec_q8_scalar`, `ml_k_matvec_q4_scalar`, `ml_k_quantize_q8_scalar`, `ml_k_rmsnorm_scalar`, `ml_k_softmax_scalar`, `ml_k_rope_scalar(float *q, float *k, const float *cos_t, const float *sin_t, int head_dim, int n_heads, int n_kv_heads)`, `ml_k_silu_mul_scalar`, `ml_k_add_scalar`, `ml_k_dot_f32_scalar`, `ml_k_axpy_f32_scalar`.
  - `core/quant_ref.h`: `void ml_dequant_row(float *out, const ml_qtensor *W, int row);` `void ml_quantize_row_q4(int8_t *q, float *d, const float *x, int k);` (q4: `id = amax ? 7.0f/amax : 0; d = amax/7.0f; v = clamp(trunc(x*id + copysignf(0.5f,x)), -8, 7)`, stored as `v + 8`; used by tests, must match `tools/mlb.py`).
  - `config/ml_config.h`: includes `k_<isa>.h` for every `ML_ENABLE_<ISA>`; for each of the 10 ops an `#if defined(ML_HAS_<OP>_<ISA>) … #elif … #else scalar` chain in fallback order (x86 chain and ARM chain); defines `ML_ISA_NAME` string.
  - `rmsnorm`: `o = x * w / sqrt(mean(x²) + eps)`; `softmax` subtracts max; `silu(g) = g/(1+exp(-g))`; rope pair `(2i,2i+1)` with `cos_t[i], sin_t[i]`, applied to all `n_heads` of `q` and `n_kv_heads` of `k`.
  - `matvec_q8/q4`: `y[r] = Σ_b (float)(Σ_j qw·qx) * dW[r,b] * dX[b]` for `r ∈ [r0,r1)`; scalar processes 1 row at a time.

- [ ] **Step 1: Write `tests/test_quant.c`**:
  - `test_quantize_roundtrip`: random x ∈ [−3,3], k=256 → `|x − q·d| ≤ d/2 + 1e-7` per element.
  - `test_quantize_zero_block`: 32 zeros then 32 randoms → `d[0] == 0`, `q[0..31] == 0`, no NaN anywhere.
  - `test_quantize_clamp`: block containing `amax` → that element quantizes to ±127, none to −128.
  - `test_quantize_ties`: x = {0.5·d', −0.5·d', 1.5·d'} with crafted amax → q = {1, −1, 2} (half away from zero).
- [ ] **Step 2: Write `tests/test_kernels.c`** — one function per op, each runs the bound `ml_k_<op>` and `ml_k_<op>_scalar`, and checks scalar against a naive double-precision `ref_<op>` in the test file:
  - `test_matvec_q8_exact`: rows ∈ {1, 3, 4, 7, 64}, cols ∈ {32, 256}, `r0 ∈ {0, 1}`; all scales = 1.0f → bound result `==` scalar `==` exact integer sum (bit-exact).
  - `test_matvec_q8_scaled`, `test_matvec_q4_exact`, `test_matvec_q4_scaled`: random scales → `TT_NEAR(..., 1e-5)`; rows outside `[r0,r1)` untouched (pre-filled with 12345.0f).
  - `test_quantize_q8_bitexact`: bound vs scalar, `memcmp` equal for q and d, k ∈ {32, 4096}.
  - `test_rmsnorm`, `test_silu_mul`, `test_add`, `test_dot_f32`, `test_axpy_f32`, `test_rope`: n ∈ {1, 7, 32, 288, 4097} (rope: head_dim 48, n_heads 6, n_kv_heads 2) → `TT_NEAR 1e-5`.
  - `test_softmax`: n ∈ {1, 5, 256} → `TT_NEAR 1e-5`, sum within 1e-5 of 1.
  - `test_softmax_extreme`: input {1e4, 9999, −1e30, 0} → all finite, sum ≈ 1, element 2 == 0.
  - Prints `ML_ISA_NAME` and which concrete symbol each op resolved to (stringified macro).
- [ ] **Step 3: Run** `make ISA=scalar test` → Expected: FAIL (link errors).
- [ ] **Step 4: Implement** scalar kernels, `quant_ref`, `ml_config.h` as specified.
- [ ] **Step 5: Verify** `make ISA=scalar test` → all tests pass, output lists `scalar` for every op.
- [ ] **Step 6: Commit** `git commit -m "kernels: scalar reference kernels and binding header"`

### Task 3: ISA guard and guarded main

**Files:**
- Create: `core/isa_check.c/.h`, `app/main_guard.c`, `tests/test_isa_guard.sh`
- Modify: `Makefile` (compile `isa_check.c` and `main_guard.c` with base CFLAGS only; link guard into every executable, tests included; tests' `main` renamed `ml_main`)

**Interfaces:**
- Produces: `ml_status ml_isa_check(const char **missing);` `int ml_main(int argc, char **argv);` (each executable defines it). `main` prints `error: CPU lacks <feature> required by ISA=<name>` to stderr and returns 3 on `ML_E_ISA`.
- Feature checks (spec §6), all requiring OS support:
  - x86: sse2 = CPUID.1:EDX[26]; sse41 = CPUID.1:ECX[9,19]; avx2 = CPUID.1:ECX[12,27,28] + XCR0&0x6==0x6 + CPUID.7.0:EBX[5]; avxvnni = avx2 + CPUID.(7,1):EAX[4]; avx512vnni = CPUID.7.0:EBX[16,30] + CPUID.7.0:ECX[11] + XCR0&0xE6==0xE6.
  - ARM Linux: A64 neon = HWCAP_ASIMD(1<<1); neon_dotprod = HWCAP_ASIMDDP(1<<20); sve = HWCAP_SVE(1<<22); ARMv7 neon = HWCAP_NEON(1<<12). Non-Linux ARM: return `ML_OK`.
  - Use `<cpuid.h>` `__get_cpuid_count` + inline `xgetbv` (GCC/Clang), `__cpuidex`/`_xgetbv` under `_MSC_VER`.
  - `missing` names: `"sse2" "ssse3" "sse4.1" "avx" "fma" "avx2" "os-avx-state" "avx-vnni" "avx512f" "avx512bw" "avx512-vnni" "os-avx512-state" "neon" "asimddp" "sve"`.

- [ ] **Step 1: Write `tests/test_isa_guard.sh`**: for each ISA in `scalar sse2 sse41 avx2 avxvnni`: `build/<isa>/test_quant` exits 0; for `avx512vnni`: exits 3 and stderr contains `avx512-vnni` (this host has avx512f but not VNNI-512 — if a host has it, the script prints `SKIP: host supports avx512vnni`).
- [ ] **Step 2: Run** it → FAIL (no guard, avx512vnni build missing).
- [ ] **Step 3: Implement** guard, `main_guard.c`, Makefile changes; add target `make guard-test` that builds the six x86 ISAs' `test_quant` and runs the script. (Other ISAs' builds only succeed fully after their kernel tasks, but `test_quant` uses only scalar `quant_ref` + bound `quantize_q8`, which falls back to scalar, so all six build now.)
- [ ] **Step 4: Verify** `make guard-test` → `isa guard: 6 passed`.
- [ ] **Step 5: Commit** `git commit -m "core: CPUID/HWCAP ISA guard with guarded main"`

### Task 4: `.mlb` format: Python writer, random model, C loader

**Files:**
- Create: `tools/mlb.py`, `tools/gen_random.py`, `core/file.c/.h`, `model/mlb.c/.h`, `tests/test_load.c`, `tests/test_convert.py` (mlb part)

**Interfaces:**
- Produces:
  - Header (256 B, little-endian, offsets in bytes): `0 magic "MLB1"`, `4 u32 version=1`, `8 dim`, `12 n_layers`, `16 n_heads`, `20 n_kv_heads`, `24 hidden_dim`, `28 vocab`, `32 max_seq` (u32 each), `36 f32 rope_theta`, `40 f32 norm_eps`, `44 u32 wtype`, `48 u32 shared_classifier`, rest zero.
  - Body from offset 256, order per spec §6. Quantized matrix `rows×cols`: `q` bytes (`rows*cols` Q8, `rows*cols/2` Q4) zero-padded to 64, then `d` (`rows*cols/32` f32) zero-padded to 64. F32 tensor: floats zero-padded to 64. Shapes: `tok_emb vocab×dim`, `wq dim×dim`, `wk,wv kv_dim×dim`, `wo dim×dim`, `w1,w3 hidden×dim`, `w2 dim×hidden`, `wcls vocab×dim`, norms `dim` F32.
  - `tools/mlb.py`: `quantize_q8(x) -> (q:int8, d:f32)` and `quantize_q4(x) -> (packed:uint8, d:f32)` implementing Global Constraints rounding in float32 numpy (`np.trunc(x*id + np.copysign(0.5, x))`); `write_mlb(path, cfg: dict, weights: dict, wtype)`; `read_mlb(path) -> (cfg, tensors)` returning dequantizable arrays.
  - `tools/gen_random.py --out PATH --wtype q8|q4 [--dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden 128 --vocab 96 --max-seq 32 --seed 0]`; weights `N(0, 0.02)`·√(1/fan_in)-scaled; norms `1 + N(0, 0.01)`.
  - `core/file.h`: `typedef struct { const uint8_t *data; size_t size; void *handle; } ml_file;` `ml_status ml_file_map(ml_file*, const char *path);` `void ml_file_unmap(ml_file*);` (POSIX `mmap`; `_WIN32` `CreateFileMapping`; `-DML_NO_MMAP` → malloc+fread).
  - `model/mlb.h`: `typedef struct { int dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, max_seq, head_dim, kv_dim; float rope_theta, norm_eps; int wtype; } ml_config;` `typedef struct { const float *attn_norm, *ffn_norm; ml_qtensor wq, wk, wv, wo, w1, w2, w3; } ml_layer;` `typedef struct { ml_config cfg; ml_file file; ml_qtensor tok_emb, wcls; const float *final_norm; ml_layer *layers; } ml_model;` `ml_status ml_model_load(ml_model*, const char *path);` `void ml_model_free(ml_model*);`

- [ ] **Step 1: Write `tests/test_convert.py::test_random_model_layout`**: generate q8 and q4 models → header fields round-trip via `read_mlb`; every array offset `% 64 == 0`; file size equals computed size; dequantized weights within `d/2` of originals.
- [ ] **Step 2: Write `tests/test_load.c`** (model paths from argv, Makefile `test` target generates them into `build/models/` first):
  - `test_load_ok`: q8 and q4 models load; `cfg` matches gen args; `wk.rows == kv_dim`; `ml_dequant_row(tok_emb, 5)` equals value written by a sidecar `build/models/rand_q8.emb5.f32` dumped by `gen_random.py --dump-emb-row 5`.
  - `test_load_rejects_truncated`: copy file truncated to 50% → `ML_E_FORMAT`.
  - `test_load_rejects_bad_magic` → `ML_E_FORMAT`; `test_load_rejects_bad_shape` (header `dim=48`, not % 32, crafted by `gen_random.py --corrupt dim48`) → `ML_E_SHAPE`; `n_heads % n_kv_heads != 0` → `ML_E_SHAPE`; nonexistent path → `ML_E_IO`.
- [ ] **Step 3: Run** `python3 -m pytest tests/test_convert.py -q` and `make ISA=scalar test` → FAIL.
- [ ] **Step 4: Implement** `mlb.py`, `gen_random.py`, `file.c`, `mlb.c`. Loader checks: magic/version, `dim,hidden_dim % 32 == 0`, `dim % n_heads == 0`, `n_heads % n_kv_heads == 0`, computed end offset ≤ file size, all values > 0.
- [ ] **Step 5: Verify** both commands pass, and `make ISA=scalar EXTRA_CFLAGS=-DML_NO_MMAP test` passes (fread fallback). Windows `CreateFileMapping` path is compile-unverified here.
- [ ] **Step 6: Commit** `git commit -m "model: .mlb format, random model generator, loader"`

### Task 5: Numpy reference forward

**Files:**
- Create: `tools/ref_forward.py`
- Test: `tests/test_convert.py::test_ref_forward_*`

**Interfaces:**
- Consumes: `tools/mlb.py` (Task 4).
- Produces: `forward(cfg, tensors, tokens: list[int]) -> np.ndarray[len(tokens), vocab]` (float32 logits per position, exact spec §6 data flow: activations re-quantized to Q8 via `mlb.quantize_q8` before each matvec, int32 block sums, FP32 accumulate, rope `theta^(-2i/head_dim)` interleaved, GQA mapping `h // (n_heads // n_kv_heads)`, attention scale `1/sqrt(head_dim)`). CLI: `ref_forward.py MODEL --tokens 1,2,3 --out logits.f32` writes raw float32 row-major.

- [ ] **Step 1: Write tests**: `test_ref_forward_shape` (3 tokens → shape (3, vocab), all finite); `test_ref_forward_causal` (logits for first 2 positions identical when a 3rd token is appended); `test_ref_forward_q4` (q4 model runs, finite).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_convert.py -q -k ref` → FAIL.
- [ ] **Step 3: Implement** `ref_forward.py` (loop form, clarity over speed).
- [ ] **Step 4: Verify** → PASS.
- [ ] **Step 5: Commit** `git commit -m "tools: numpy reference forward pass"`

### Task 6: Transformer forward, sampler, tokenizer, `ml_run` (single-thread)

**Files:**
- Create: `ops/ops.c/.h`, `core/threads.c/.h` (single-thread variant only here), `model/transformer.c/.h`, `model/sampler.c/.h`, `model/tokenizer.c/.h`, `app/ml_run.c`, `tests/test_e2e.sh`, `tools/gen_tokenizer.py`

**Interfaces:**
- Consumes: kernels via `ml_config.h` (Task 2), `ml_model` (Task 4), `ref_forward.py` CLI (Task 5).
- Produces:
  - `core/threads.h`: `typedef void (*ml_task_fn)(void *ctx, int tid, int nthreads);` `typedef struct ml_pool ml_pool;` `ml_status ml_pool_create(ml_pool **p, int n);` `void ml_pool_run(ml_pool*, ml_task_fn, void *ctx);` `void ml_pool_destroy(ml_pool*);` — here: single-thread implementation that calls `fn(ctx,0,1)`.
  - `ops/ops.h`: `void ml_matvec(ml_pool*, float *y, const ml_qtensor *W, const ml_qact *x);` (dispatch on `W->type`; rows split into `nthreads` chunks, chunk boundaries multiples of 4) `void ml_attention(ml_pool*, float *out, const float *q, const float *kcache, const float *vcache, float *att_scratch, const ml_config*, int pos);` (split by head).
  - `model/transformer.h`: `typedef struct { ml_model *m; ml_arena arena; ml_pool *pool; float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits, *kcache, *vcache, *rope_cos, *rope_sin; ml_qact qa_dim, qa_hidden; } ml_state;` `ml_status ml_state_init(ml_state*, ml_model*, int nthreads);` `float *ml_forward(ml_state*, int token, int pos);` `void ml_state_free(ml_state*);` KV layout `[layer][pos][kv_dim]`.
  - `model/sampler.h`: `typedef struct { float temperature, topp; uint64_t rng; float *probs; int *idx; } ml_sampler;` `int ml_sample(ml_sampler*, float *logits, int n);` greedy when `temperature == 0` (ties → lowest index); xorshift64* RNG; top-p as llama2.c (sort candidates with prob ≥ (1−topp)/(n−1)).
  - `model/tokenizer.h`: llama2.c `tokenizer.bin` (`int32 max_token_length`, then per token `f32 score, int32 len, bytes`). `ml_status ml_tokenizer_load(ml_tokenizer*, const char*, int vocab);` `int ml_encode(ml_tokenizer*, const char *text, int bos, int eos, int *out, int max);` `const char *ml_decode(ml_tokenizer*, int prev, int tok);` (BOS=1, EOS=2, `<0xXX>` byte tokens, strip leading space after BOS).
  - `ml_run` CLI: `ml_run MODEL [-z tokenizer.bin] [-i prompt | --tokens 1,2,3] [-n steps=64] [-T temp=0] [-p topp=0.9] [-s seed=1] [-t threads=1] [--dump-logits FILE]`. `--dump-logits` writes float32 logits for every position of the `--tokens` sequence (teacher forcing, no sampling) and exits. Usage error → message on stderr, exit 2. Generation stops at `max_seq` (exit 0) or EOS.
  - `tools/gen_tokenizer.py --vocab 96 --out PATH`: tiny valid tokenizer (`<unk>`, `<s>`, `</s>`, 93 single chars/bytes) for random models.
- [ ] **Step 1: Write `tests/test_e2e.sh ISA [RUN]`**:
  - `ref`: for q8 and q4 random models, `--tokens 1,5,9,13,…` (16 tokens) → `ml_run --dump-logits` vs `ref_forward.py --out` compared by a small inline Python: max rel err ≤ 1e-3 (scalar ISA compares to ref; other ISAs compare to `build/scalar/` output) and argmax per position identical.
  - `greedy`: `ml_run m.mlb -z tok.bin -i "ab" -n 16 -T 0` output identical to scalar build's.
  - `overflow`: `-n 1000` on `max_seq=32` model → exit 0, at most 32 positions.
  - `usage`: `ml_run` without args → exit 2, stderr contains `usage:`.
- [ ] **Step 2: Run** `make ISA=scalar e2e` → FAIL.
- [ ] **Step 3: Implement** threads (single), ops, transformer (rope tables computed in double then stored float), sampler, tokenizer, `ml_run`, `gen_tokenizer.py`; Makefile target `e2e` builds models into `build/models/` and runs the script with `$(RUN)`.
- [ ] **Step 4: Verify** `make ISA=scalar e2e` → `e2e scalar: 4 passed`.
- [ ] **Step 5: Commit** `git commit -m "model: transformer forward, sampler, tokenizer, ml_run"`

### Task 7: SSE2 and SSE4.1 kernels

**Files:**
- Create: `kernels/sse2/k_sse2.c/.h`, `kernels/sse41/k_sse41.c/.h`, `kernels/x86_common.h` (shared `exp_ps` 128-bit helper, horizontal sums)

**Interfaces:**
- Consumes: kernel signatures (Task 2).
- Produces: sse2 — all 10 ops (`ML_HAS_<OP>_SSE2`); sse41 — `matvec_q8`, `matvec_q4` only (dot strategy from spec §5); other ops fall back to sse2.
  - `exp_ps`: `n = round(x·log2e)`, `r = x − n·ln2_hi − n·ln2_lo`, degree-5 polynomial (Cephes `expf` coefficients), scale by `2^n` via exponent bits; clamp input to [−88, 88]. Rel. err ≤ 1e-6.
  - matvec: 4 rows per iteration, one activation block load reused 4×; scalar-row tail.
- [ ] **Step 1: Red check** `build/sse2/test_kernels | grep -c _sse2` → 0 (everything still resolves to scalar).
- [ ] **Step 2: Implement** sse2 kernels.
- [ ] **Step 3: Implement** sse41 matvecs.
- [ ] **Step 4: Verify** `make ISA=sse2 test e2e` and `make ISA=sse41 test e2e` pass; `build/sse2/test_kernels | grep -c _sse2` = 10, `build/sse41/test_kernels | grep -c _sse41` = 2.
- [ ] **Step 5: Commit** `git commit -m "kernels: SSE2 and SSE4.1"`

### Task 8: AVX2 and AVX-VNNI kernels

**Files:**
- Create: `kernels/avx2/k_avx2.c/.h`, `kernels/avxvnni/k_avxvnni.c/.h`

**Interfaces:**
- Produces: avx2 — all 10 ops (256-bit, `_mm256_fmadd_ps` for scale accumulation, 256-bit `exp_ps`); avxvnni — `matvec_q8`, `matvec_q4` via `_mm256_dpbusd_avx_epi32` with `|x|`/`sign` trick.
- [ ] **Step 1: Implement** avx2 kernels; **Step 2:** avxvnni matvecs.
- [ ] **Step 3: Verify** `make ISA=avx2 test e2e`, `make ISA=avxvnni test e2e` pass; resolution counts 10 and 2.
- [ ] **Step 4: Commit** `git commit -m "kernels: AVX2 and AVX-VNNI"`

### Task 9: AVX-512 VNNI kernels

**Files:**
- Create: `kernels/avx512vnni/k_avx512vnni.c/.h`

**Interfaces:**
- Produces: `matvec_q8`, `matvec_q4` (2 blocks per 512-bit op, `_mm512_dpbusd_epi32`; sign via `_mm512_movepi8_mask` + `_mm512_mask_sub_epi8`), `quantize_q8`, `rmsnorm`, `add`, `dot_f32`, `axpy_f32` (512-bit). Others fall back to avxvnni/avx2.
- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify** `make ISA=avx512vnni` compiles warning-free; `make guard-test` still passes (exit 3 on this host). Execution verification: if the host lacks AVX512-VNNI, record "compile-only" in the commit message.
- [ ] **Step 3: Commit** `git commit -m "kernels: AVX-512 VNNI (compile-verified)"`

### Task 10: ARM toolchain + NEON kernels (ARMv7 and AArch64)

**Files:**
- Create: `kernels/neon/k_neon.c/.h`, `kernels/arm_common.h` (`exp_ps` NEON, horizontal add helpers with ARMv7 fallbacks for `vaddvq_*`)
- Modify: `mk/isa.mk` (cross RUN prefixes)

**Interfaces:**
- Produces: all 10 ops for `neon`; same source builds for ARMv7 (`#ifndef __aarch64__` paths: no `vaddvq`, no `vcvtnq`, no `vdivq` → reciprocal estimate + 2 Newton steps, no `vfmaq` unless `__ARM_FEATURE_FMA`) and A64.
- [ ] **Step 1: Install** `apt-get install -y gcc-aarch64-linux-gnu gcc-arm-linux-gnueabihf qemu-user` → `aarch64-linux-gnu-gcc --version` and `qemu-aarch64 --version` succeed. If install fails: build with `clang --target=aarch64-linux-gnu -ffreestanding -c` compile-only and report ARM rows as compile-only.
- [ ] **Step 2: Verify scalar on ARM first**: `make ISA=scalar CROSS=aarch64-linux-gnu- test e2e` and `CROSS=arm-linux-gnueabihf-` pass under qemu (e2e compares against host `build/scalar/` outputs — the comparison uses the 1e-3 tolerance).
- [ ] **Step 3: Implement** neon kernels.
- [ ] **Step 4: Verify** `make ISA=neon CROSS=aarch64-linux-gnu- test e2e` and `make ISA=neon CROSS=arm-linux-gnueabihf- test e2e` pass; resolution count 10.
- [ ] **Step 5: Commit** `git commit -m "kernels: NEON for ARMv7 and AArch64"`

### Task 11: NEON dotprod and SVE kernels

**Files:**
- Create: `kernels/neon_dotprod/k_neon_dotprod.c/.h`, `kernels/sve/k_sve.c/.h`
- Modify: `Makefile` (`test-sve` runs tests at two vector lengths)

**Interfaces:**
- Produces: neon_dotprod — `matvec_q8`, `matvec_q4` via `vdotq_s32`; sve — `matvec_q8`, `matvec_q4` (`svdot_s32`, VLA loop with `svwhilelt`), `rmsnorm`, `add`, `dot_f32`, `axpy_f32`, `quantize_q8`. Others fall back down the ARM chain.
- [ ] **Step 1: Implement** neon_dotprod, then sve.
- [ ] **Step 2: Verify** `make ISA=neon_dotprod CROSS=aarch64-linux-gnu- test e2e` passes; `make ISA=sve CROSS=aarch64-linux-gnu- test-sve` runs `test_kernels` and e2e under `qemu-aarch64 -cpu max,sve128=on` and `-cpu max,sve512=on` → both pass.
- [ ] **Step 3: Commit** `git commit -m "kernels: NEON dotprod and SVE"`

### Task 12: pthread worker pool

**Files:**
- Modify: `core/threads.c` (add `#ifdef ML_THREADS_PTHREAD` pool), `Makefile` (`THREADS=pthread` → `-DML_THREADS_PTHREAD -pthread`)
- Test: `tests/test_e2e.sh` new case `threads`

**Interfaces:**
- Produces: same `threads.h` API; `ml_pool_create(&p, n)` spawns `n−1` workers (caller is tid 0); `ml_pool_run` blocks until all finish (mutex + condvar generation counter); `n` clamped to [1, 64].
- [ ] **Step 1: Add e2e case `threads`**: with `THREADS=pthread` build, `--dump-logits` with `-t 1` and `-t 4` produce byte-identical files (`cmp`); greedy output identical.
- [ ] **Step 2: Run** `make ISA=avx2 THREADS=pthread e2e` → FAIL (`-t 4` ignored or not built).
- [ ] **Step 3: Implement** pool.
- [ ] **Step 4: Verify** `make ISA=avx2 THREADS=pthread e2e` and `make ISA=scalar THREADS=pthread e2e` pass; `make ISA=avx2 THREADS=1` build has no `pthread_` symbols (`nm build/avx2/ml_run | grep -c pthread_create` = 0).
- [ ] **Step 5: Commit** `git commit -m "core: optional pthread worker pool"`

### Task 13: ASM override mechanism + CMake build

**Files:**
- Create: `kernels/sse2/asm.mk` (`ASM_KERNELS := add`), `kernels/sse2/add.S`, `CMakeLists.txt`
- Modify: `Makefile`, `kernels/sse2/k_sse2.c` (wrap `ml_k_add_sse2` in `#ifndef ML_ASM_ADD`)

**Interfaces:**
- Produces: `ASM=1` → for each `k` in `kernels/<isa>/asm.mk` within the chain: assemble `kernels/<isa>/<k>.S`, add `-DML_ASM_<K upper>`. `add.S`: SysV x64 `ml_k_add_sse2(float *y /*rdi*/, const float *x /*rsi*/, int n /*edx*/)`, `movups` loop of 4 + scalar tail, `.section .note.GNU-stack`.
  - CMake: `-DML_ISA=<isa> -DML_THREADS=ON|OFF` for x86 ISAs, same flag table (`/arch:AVX2` and `__AVX2__`-style defines under MSVC; `sse2`/`sse41` need no `/arch` on x64), builds `ml_run`, `ml_bench`, tests; `ctest` runs `test_quant`, `test_kernels`.
- [ ] **Step 1: Verify red**: `make ISA=sse2 ASM=1 test` → fails (no asm.mk handling / missing symbol).
- [ ] **Step 2: Implement** Makefile ASM handling and `add.S`.
- [ ] **Step 3: Verify** `make ISA=sse2 ASM=1 test e2e` and `make ISA=avx2 ASM=1 test` pass; `objdump -d build/sse2/test_kernels | grep -A3 '<ml_k_add_sse2>:'` shows the hand-written loop (`movups`).
- [ ] **Step 4: Implement** `CMakeLists.txt`; **verify** `cmake -S . -B build/cmake-avx2 -G Ninja -DML_ISA=avx2 && cmake --build build/cmake-avx2 && ctest --test-dir build/cmake-avx2` passes. MSVC is not available here: note "MSVC unverified" in commit message.
- [ ] **Step 5: Commit** `git commit -m "build: ASM kernel overrides, CMake build"`

### Task 14: Converter, benchmark, full matrix, README

**Files:**
- Create: `tools/convert.py`, `app/ml_bench.c`, `README.md`
- Modify: `tests/test_convert.py`, `Makefile` (`bench`, `test-all`)

**Interfaces:**
- Consumes: `tools/mlb.py`, all kernels, `ml_state`.
- Produces:
  - `convert.py llama2c IN.bin OUT.mlb --wtype q8|q4` (legacy v0 llama2.c format: 7×int32 header `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab(<0 ⇒ unshared), seq_len`, then F32 weights in llama2.c order; freq_cis tables skipped; `rope_theta=10000`, `norm_eps=1e-5`).
  - `convert.py hf DIR OUT.mlb --wtype q8|q4` (single or sharded `*.safetensors` with F32/F16/BF16; `config.json` keys `hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, vocab_size, max_position_embeddings, rope_theta, rms_norm_eps, tie_word_embeddings`; `wq`/`wk` permuted with `w.reshape(nh, 2, hd//2, dim).transpose(0,2,1,3).reshape(-1, dim)` to interleaved RoPE). Minimal safetensors reader with `json` + `numpy`.
  - `ml_bench MODEL? [--iters N]`: kernels `matvec_q8`, `matvec_q4` at 4096×4096 and 768×768, `quantize_q8` k=4096 → prints `op shape ns/iter GB/s GOPS`; with MODEL prints `tok/s` over 64 tokens. Header line includes `ML_ISA_NAME` and each op's resolved symbol.
  - `make test-all`: runs `test e2e` for `scalar sse2 sse41 avx2 avxvnni`, `guard-test`, compile of `avx512vnni`, ARM matrix (Tasks 10–11) when cross tools exist, `THREADS=pthread` for avx2, `ASM=1` for sse2, `pytest tests/` → prints a summary table `ISA | build | tests | e2e` with `PASS/FAIL/COMPILE-ONLY/SKIP`.
- [ ] **Step 1: Write tests** `test_convert.py::test_llama2c_roundtrip` (write a tiny synthetic llama2.c v0 file with numpy, convert, `ref_forward` on converted model equals `ref_forward` on the F32 weights quantized directly via `mlb.py`, max rel err 1e-6) and `test_hf_permutation` (synthetic safetensors with BF16 + tied embeddings; after conversion, numpy rotate-half RoPE on original weights == interleaved RoPE on converted weights for q·k scores, rel err 1e-5).
- [ ] **Step 2: Run** → FAIL; **Step 3: Implement** `convert.py`; **Step 4: Verify** pytest passes.
- [ ] **Step 5: Implement** `ml_bench`, `bench` target (`make ISA=x bench` runs it), `test-all`.
- [ ] **Step 6: Verify** `make test-all` → summary table, no FAIL; `for i in scalar sse2 avx2 avxvnni; do make ISA=$i bench; done` shows avx2/avxvnni matvec GOPS above scalar (success criterion 2). Smoke: try downloading `stories15M.bin`/`tokenizer.bin`; if blocked (CDN 403 seen during planning), record SKIP.
- [ ] **Step 7: Write `README.md`**: build matrix table (ISA, flags, min CPU), quick start (`gen_random`, `convert`, `ml_run`), adding a kernel (create `k_<isa>.c`, define `ML_HAS_*`, nothing else changes), ASM override, verification status per ISA. Include the bench table from Step 6.
- [ ] **Step 8: Commit** `git commit -m "tools: converter, benchmark, full test matrix, README"`
