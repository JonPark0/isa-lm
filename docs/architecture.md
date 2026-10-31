# Architecture

## Layers

```
app/       ml_run (generation CLI), ml_bench (benchmarks), ml_tok (encode/decode text),
           main_guard.c (process entry)
model/     mlb.c (loader), transformer.c (forward pass, KV cache, RoPE tables),
           sampler.c, tokenizer.c (llama2.c format), bpe.c + unicode.c (tokenizer.json BPE)
ops/       ops.c — matvec and attention, split across the worker pool
kernels/   scalar/ sse2/ sse41/ avx2/ avxvnni/ avx512vnni/ neon/ neon_dotprod/ neon_i8mm/ sve/
           x86_common.h, x86_avx.h, arm_common.h — shared SIMD helpers
core/      ml.h (types, status codes), arena.c, file.c (mmap), quant_ref.c,
           threads.c, isa_check.c
config/    ml_config.h — compile-time kernel binding
tools/     mlb.py, gen_random.py, gen_tokenizer.py, ref_forward.py, convert.py,
           convert_tokenizer.py, gen_unicode.py
```

Dependencies point down only: `app → model → ops → kernels → core`. Kernels see
only `core/ml.h` and their own intrinsics headers.

## Compile-time kernel selection

There are twelve kernels. Every ISA implements the same C ABI, named
`ml_k_<op>_<isa>`:

| Kernel | Purpose |
|--------|---------|
| `matvec_q8`, `matvec_q4` | `y[r0..r1) = W · x` for Q8/Q4 weights and Q8 activations |
| `quantize_q8` | float activations → Q8_0 |
| `rmsnorm`, `softmax`, `silu_mul`, `add`, `rope` | element-wise and normalization ops |
| `dot_f32`, `axpy_f32` | attention over an F32 KV cache: scores of one query against many strided K rows, and the weighted sum of V rows |
| `dot_q8`, `axpy_q8` | the same over a Q8 KV cache: integer dot products, one scale per row |

`make ISA=X` passes `-DML_ISA_X`, plus `-DML_ENABLE_<I>` for every ISA in X's
fallback chain:

```
avx512vnni → avx2 → sse41 → sse2 → scalar
avxvnni    → avx2 → …
neon_i8mm  → neon_dotprod → neon → scalar
sve        → neon_dotprod → …
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
| ARM Linux | `getauxval(AT_HWCAP)`: ASIMD, ASIMDDP, SVE, ARMv7 NEON; `AT_HWCAP2`: I8MM |
| other | the build configuration is trusted |

## Data flow for one token

```
x = dequant(tok_emb[token])
for each layer:
    h  = rmsnorm(x) ; qa = quantize_q8(h)
    q, k, v = matvec(Wq|Wk|Wv, qa) (+ biases) ; per-head q/k norm (Qwen3) ; rope(q, k)
                                                      k, v go into the KV cache (F32, or quantized for Q8)
    o  = attention(q, K[w..pos], V[w..pos])           per head, GQA head h → kv head h / (n_heads / n_kv_heads);
                                                      w = 0, or pos + 1 − sliding_window with a window
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

## Weight packing

The row-major matvec kernels spend most of their instructions on per-row
bookkeeping: every 32-weight block costs one integer dot product but several
float operations to apply its scale. At load, `ml_model_pack` therefore
repacks every weight matrix, except the embedding table, into a
**row-interleaved** layout ([`core/pack.h`](https://github.com/JonPark0/isa-lm/blob/main/core/pack.h)):

- **Row groups:** rows are grouped by G, the SIMD lane count: 16 for AVX-512
  VNNI, 8 for AVX2 and AVX-VNNI, 4 for SSE2, SSE4.1 and the NEON builds. One
  32-bit lane owns one row.
- **Step layout:** each block is stored as 4-byte steps, with G rows per step.
  One vector load plus one dot instruction against a 4-byte activation
  broadcast (`vpdpbusd`, `pmaddubsw`+`pmaddwd`, `sdot`) advances all G rows.
- **Scales:** the block's scales are stored contiguously for the group, so
  scaling is one vector multiply-add for G rows.
- **Signedness:** `vpdpbusd` and `pmaddubsw` need an unsigned operand. Q4
  nibbles are used raw (0..15); on VNNI and SSE2, Q8 weights are stored +128. The
  matching `offset × Σx` is subtracted per block in int32, so the result is
  exact. AVX2 and SSE4.1 keep Q8 signed and use `psignb`. On those two, a Q4
  block's 8 `pmaddubsw` results are summed in int16 (at most 30480) before a
  single `pmaddwd`.

After packing, the original mapped pages are released with
`madvise(MADV_DONTNEED)`, so resident memory stays about the size of the model.
`ml_run --no-pack` keeps the row-major kernels. Only the scalar build has no
packed kernels.

- **SSE2** has no 8-bit multiply at all. Each 16-bit lane of a weight step
  holds two bytes of one row; the kernel splits it into the even byte
  (`w & 0xFF`) and the odd byte (`w >> 8`) and multiplies each with `pmaddwd`
  against activations expanded once per call into matching broadcast pairs.
  Every `pmaddwd` then yields one partial sum per row, and the inner loop has
  no shuffles, which matters on CPUs with a single shuffle port.
- **NEON without a dot product** (ARMv7, Cortex-A53/A72) uses
  `vmull_s8`/`vpadalq_s16` on the same 4-row groups.
- **sve** builds use the NEON dotprod packed kernels.

### Pack cache

`ml_run --pack-cache` writes the packed arrays, exactly as laid out in memory,
to `MODEL.pk<G>` behind a 512-byte header. The header records the pack group,
the Q8 offset, the model's size and header, and a hash of bytes sampled across
the model file. A later run whose build and model match maps the file instead
of packing (`ml_model_pack_cached`); otherwise it packs and rewrites it,
through a temporary file and a rename.

### Prefetch

Decoding reads every weight once per token, a long sequential stream per
thread. Hardware prefetchers stop at 4 KB page boundaries, so the packed
matvecs request the weights `ML_PREFETCH_DIST` (4 KB) ahead of the block they
work on (`ml_prefetch_ahead` in `core/ml.h`). On the x86 benchmark VM this
made single-thread decoding 1.3–2.5× faster. On Arm Neoverse it made Q8
decoding 26% slower, so it is on by default for x86 only;
`-DML_USE_PREFETCH=0` or `=1` overrides that.

## Batched prompt prefill

Decoding one token reads every weight once, so on large models it is bound by
memory bandwidth. `ml_prefill` instead processes up to 128 prompt tokens per
step, and each matrix product becomes a matmul (`ml_k_matmul_*_pk`). Each
packed weight vector is loaded once and feeds a tile of 4 tokens, which makes
the kernel compute-bound.

- **Same arithmetic as decoding:** every per-token operation (norms,
  quantization, RoPE, attention, SiLU) is the one `ml_forward` performs, in the
  same order, and the matmul's per-token arithmetic is the matvec's. Batched and
  token-by-token runs produce bit-identical logits; the tests enforce this.
- **No classifier for prompt tokens:** prompt tokens other than the last skip
  the vocabulary-sized classifier.
- **i8mm:** `neon_i8mm` builds run the prefill on `smmla`, which multiplies a
  2×8 int8 matrix by the transpose of another: two weight rows against two
  tokens, twice the work of an `sdot`. Zipping the 32-bit lanes of two
  consecutive dotprod steps yields exactly its operands, so the packed layout
  is unchanged. Decoding keeps `sdot`, since one token would leave half of
  each `smmla` idle.
- **Fallback:** builds without packed kernels, or with `--no-pack`, still batch
  the prefill but use the matvec per token.

## Attention

For each position, every query head scores all cached positions, takes a
softmax, and sums the V rows. On long contexts this dominates: a head reads
`pos × head_dim` K values and as many V values per token.

- **Whole rows per call:** `dot_f32` and `axpy_f32` take a block of cached
  positions at once (strided rows of the cache), so the SIMD loops run over
  positions instead of paying a call and a horizontal sum per position.
  `axpy` keeps its outputs in registers across the rows.
- **GQA blocks:** heads that share a KV head walk the cache together in blocks
  of 64 positions. Each block of K (and later V) is loaded into L1 once and
  used by every head of the group.
- **Q8 cache (`--kv q8`):** each position's K and V are stored per KV head as
  int8 with one float scale, using the activations' quantization rule, and
  each query head is quantized the same way. Scores are integer dot products;
  the V sum converts int8 to float on the fly. The cache shrinks by about
  3.8× (`head_dim` 64: 68 bytes per head instead of 256). The logits change
  slightly, so F32 stays the default.

Per head, the arithmetic does not depend on the block size or the thread
count, so results stay bit-identical across `-t` and `--batch`.

## Tokenizers

`model/bpe.c` implements the two `tokenizer.json` kinds the converter accepts
(`.mlt` format in [File format](file-format.md#mlt-tokenizer-format)):

1. **Added tokens** (`<s>`, `<|eot_id|>`) are cut out of the text first.
2. **Pieces between them:**
   - SentencePiece-style: spaces become `▁`, a `▁` is prepended, and each
     character starts as one symbol (or as byte tokens if it is not in the
     vocabulary).
   - Byte-level: the Llama 3 regex splits the text, matched by hand on code
     points with letter and number tables generated from Unicode
     (`tools/gen_unicode.py`); each chunk starts as one symbol per byte, unless
     the whole chunk is a token.
3. **Merges:** the lowest-ranked merge among adjacent symbols is applied, the
   leftmost first, until none applies. Symbols form a linked list and candidate
   pairs sit in a heap, so a piece of n symbols takes O(n log n): 48 KB of text
   encodes in 30 ms.

## Memory

- **Weights:** memory-mapped read-only (`mmap` on POSIX, `CreateFileMapping` on
  Windows, or `fread` with `-DML_NO_MMAP`).
- **Runtime buffers:** carved from **one** 64-byte-aligned arena, sized from the
  header. These are the activations (one token, plus 32-token prefill
  batches), the Q8 activation buffers, the KV cache `[layer][max_seq][kv_dim]`
  (F32, or int8 plus one scale per position and KV head with `--kv q8`), the
  RoPE tables and the logits.
- **Packed weights:** one allocation the size of the matrices. The originals'
  pages are released after packing.
- **Size checks:** every size product is overflow-checked, so on a 32-bit target
  a too-large model fails with `ML_E_NOMEM`.
- **After load:** no allocation happens during generation.

## Profiling

`ml_run --profile` times each stage of the forward pass (qkv, attention, wo,
ffn, classifier, other) and prints the split. On every model measured, the
matrix products take more than 95% of the time.

## Threads

`core/threads.c` provides a pool built with `THREADS=pthread`. With
`THREADS=1`, all pthread code is compiled out.

- **Caller as thread 0:** the caller works as thread 0, and `n − 1` workers wait
  on an atomic generation counter.
- **Waiting:** a worker spins about 20 000 pause iterations, then sleeps on a
  condition variable. A sequentially consistent `sleepers` counter makes sure
  `ml_pool_run` never misses a sleeping worker.
- **Work split:** matvec and matmul split rows into chunks that are multiples
  of 4, or of the pack group G; attention splits by head.
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
