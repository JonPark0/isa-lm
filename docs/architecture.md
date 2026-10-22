# Architecture

## Layers

```
app/       ml_run (generation CLI), ml_bench (benchmarks), main_guard.c (process entry)
model/     mlb.c (loader), transformer.c (forward pass, KV cache, RoPE tables),
           sampler.c, tokenizer.c
ops/       ops.c — matvec and attention, split across the worker pool
kernels/   scalar/ sse2/ sse41/ avx2/ avxvnni/ avx512vnni/ neon/ neon_dotprod/ sve/
           x86_common.h, x86_avx.h, arm_common.h — shared SIMD helpers
core/      ml.h (types, status codes), arena.c, file.c (mmap), quant_ref.c,
           threads.c, isa_check.c
config/    ml_config.h — compile-time kernel binding
tools/     mlb.py, gen_random.py, gen_tokenizer.py, ref_forward.py, convert.py
```

Dependencies point down only: `app → model → ops → kernels → core`. Kernels see
only `core/ml.h` and their own intrinsics headers.

## Compile-time kernel selection

There are ten kernels. Every ISA implements the same C ABI, named
`ml_k_<op>_<isa>`:

| Kernel | Purpose |
|--------|---------|
| `matvec_q8`, `matvec_q4` | `y[r0..r1) = W · x` for Q8/Q4 weights and Q8 activations |
| `quantize_q8` | float activations → Q8_0 |
| `rmsnorm`, `softmax`, `silu_mul`, `add`, `rope` | element-wise and normalization ops |
| `dot_f32`, `axpy_f32` | attention scores and the weighted sum over V |

`make ISA=X` passes `-DML_ISA_X`, plus `-DML_ENABLE_<I>` for every ISA in X's
fallback chain:

```
avx512vnni → avx2 → sse41 → sse2 → scalar
avxvnni    → avx2 → …
sve        → neon_dotprod → neon → scalar
```

Each enabled `kernels/<i>/k_<i>.h` defines `ML_HAS_<OP>_<I>` for the kernels it
implements. `config/ml_config.h` then binds each generic name to the
highest-priority implementation present:

```c
#if   defined(ML_HAS_MATVEC_Q8_AVX512VNNI)
#define ml_k_matvec_q8 ml_k_matvec_q8_avx512vnni
#elif defined(ML_HAS_MATVEC_Q8_AVXVNNI)
...
#else
#define ml_k_matvec_q8 ml_k_matvec_q8_scalar
#endif
```

The binding is static. There are no function pointers and no runtime dispatch,
and LTO can inline kernels into their callers.

The `avx512vnni` chain skips `avxvnni` on purpose. Some CPUs with AVX512-VNNI
(Ice Lake, Cascade Lake) lack the VEX-encoded AVX-VNNI, and `avxvnni` only
implements the matvecs, which `avx512vnni` covers itself.

Which kernel each op resolved to is printed by `build/<isa>/test_kernels` and
by the first line of `ml_bench`.

## ISA guard

`app/main_guard.c` holds every executable's `main`. It and `core/isa_check.c`
are compiled **without** ISA flags. `main` checks the CPU first and only then
calls `ml_main`, which is compiled with the ISA flags. As a result no AVX or SVE
instruction can execute on an unsupported CPU, not even in a function prologue.

| Platform | How the CPU is checked |
|----------|------------------------|
| x86 | `CPUID` leaves 1, 7.0 and 7.1, plus `XGETBV` to confirm the OS saves AVX and AVX-512 state |
| ARM Linux | `getauxval(AT_HWCAP)`: ASIMD, ASIMDDP, SVE, ARMv7 NEON |
| other | the build configuration is trusted |

## Data flow for one token

```
x = dequant(tok_emb[token])
for each layer:
    h  = rmsnorm(x) ; qa = quantize_q8(h)
    q, k, v = matvec(Wq|Wk|Wv, qa) ; rope(q, k)       k, v written straight into the KV cache
    o  = attention(q, K[0..pos], V[0..pos])           per head, GQA head h → kv head h / (n_heads / n_kv_heads)
    x += matvec(Wo, quantize_q8(o))
    h  = rmsnorm(x) ; qa = quantize_q8(h)
    g  = silu(matvec(W1, qa)) * matvec(W3, qa)
    x += matvec(W2, quantize_q8(g))
logits = matvec(Wcls, quantize_q8(rmsnorm(x)))
```

Every matvec runs on integers:

1. Activations are quantized to Q8_0.
2. Each 32-weight block's dot product is computed as an int32.
3. That int32 is scaled by `d_w · d_x` and accumulated in FP32.

The RoPE cos/sin tables `[max_seq][head_dim/2]` are computed once at load, in
double precision, with llama3 scaling folded in.

## Memory

- **Weights:** memory-mapped read-only (`mmap` on POSIX, `CreateFileMapping` on
  Windows, or `fread` with `-DML_NO_MMAP`).
- **Runtime buffers:** carved from **one** 64-byte-aligned arena, sized from the
  header. These are the activations, the Q8 activation buffers, the KV cache
  `[layer][max_seq][kv_dim]` in F32, the RoPE tables and the logits.
- **Size checks:** every size product is overflow-checked, so on a 32-bit target
  a too-large model fails with `ML_E_NOMEM`.
- **After load:** no allocation happens during generation.

## Threads

`core/threads.c` provides a pool built with `THREADS=pthread`. With
`THREADS=1`, all pthread code is compiled out.

- **Caller as thread 0:** the caller works as thread 0, and `n − 1` workers wait
  on an atomic generation counter.
- **Waiting:** a worker spins about 20 000 pause iterations, then sleeps on a
  condition variable. A sequentially consistent `sleepers` counter makes sure
  `ml_pool_run` never misses a sleeping worker.
- **Work split:** matvec splits rows into chunks that are multiples of 4;
  attention splits by head.
- **Serial cut-off:** ops below 2^18 multiply-adds run on the caller alone,
  because waking threads would cost more than the work.
- **Determinism:** each output element is computed by exactly one thread in a
  fixed order, so results are bit-identical for any thread count.

## Error handling

Library functions return `ml_status`: `ML_OK`, `ML_E_IO`, `ML_E_FORMAT`,
`ML_E_SHAPE`, `ML_E_NOMEM` or `ML_E_ISA`. No library code calls `abort()`. The
executables map these to exit codes; see the [user guide](user-guide.md#running).

## Numerics

- **No implicit FMA:** everything builds with `-ffp-contract=off`. SIMD kernels
  use FMA only through explicit intrinsics.
- **Bit-exact quantization:** `quantize_q8` gives the same bits on every ISA.
  This is why the [quantization rule](file-format.md#quantization-rule) is
  written as exact float32 operations.
- **Integer matvec:** block sums are exact. Float accumulation order differs
  between ISAs, so outputs agree within 1e-5 of the dot product's magnitude.
- **exp:** SIMD kernels approximate `exp` with the Cephes `expf` polynomial
  (relative error around 1e-7); scalar kernels use `expf`.
- **Masked softmax:** if every input is −inf, `softmax` returns a uniform
  distribution.
