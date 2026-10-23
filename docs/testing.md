# Testing and benchmarking

## Test layers

| Command | What it checks |
|---------|----------------|
| `make ISA=X test` | C unit tests: `test_arena`, `test_quant`, `test_kernels` (including the packed and batched kernels), `test_load`, `test_sampler`, `test_threads` |
| `make ISA=X e2e` | `tests/test_e2e.sh` runs `ml_run` against the numpy reference, the scalar build, and CLI edge cases |
| `python3 -m pytest tests` | the Python tools: format round trips, the reference forward, both converters, RoPE scaling |
| `make guard-test` | x86 ISA guard, using `qemu-x86_64 -cpu core2duo/Nehalem/SandyBridge/Haswell` |
| `make guard-test-arm` | AArch64 ISA guard, using `qemu-aarch64 -cpu cortex-a53/cortex-a76/neoverse-n2/max` |
| `make ISA=sve CROSS=aarch64-linux-gnu- test-sve` | the SVE build at 128-bit and 512-bit vector length |
| `make test-all` | everything above for every ISA, then a summary table |

### `test_kernels`

`test_kernels` checks each kernel twice:

- the kernel the build bound (`ml_k_<op>`) against the scalar kernel;
- the scalar kernel against a naive double-precision reference in the test.

It runs both checks in every build, so one source tests every ISA.

| What | Tolerance |
|------|-----------|
| Integer matvec with all scales set to 1 | bit-exact |
| Scaled matvec, `rope`, `axpy` | relative 1e-5 of the sum of term magnitudes |
| `quantize_q8` (zero blocks, tiny blocks, ties) | bit-exact against scalar |
| Other float kernels | relative 1e-5 |
| `softmax` | also finite for ±1e4 and −1e30 inputs, and uniform when every input is −inf |

The first lines of the output name the kernel each op resolved to, for example
`matvec_q8 -> ml_k_matvec_q8_avx512vnni`.

### End-to-end tests

`tests/test_e2e.sh ISA BUILD [RUN]` checks the following:

- **Logits:** for random Q8 and Q4 models, the logits match
  `tools/ref_forward.py` within 1e-3 relative error, and the argmax is the same
  at every position.
- **Generated text:** greedy and seeded sampling produce text identical to the
  host scalar build.
- **Context limit:** generation stops at `max_seq`, and `--dump-logits` with
  more tokens than `max_seq` exits with code 2.
- **Threads:** `-t 3/4/7` give logits identical to `-t 1`, and `ml_run` reports
  the thread count it used.
- **Packing:** builds with packed kernels really pack (`packed: G` matches
  `--help`), and `--no-pack` also matches the reference.
- **Batched prefill:** `--batch 1`, `--batch 5` and the default produce
  byte-identical logits, and generated text is the same with and without
  batching. This also holds with `--kv q8`, and so does `-t 3` against `-t 1`.
- **Q8 KV cache:** `--kv q8` logits match `tools/ref_forward.py --kv q8`, which
  emulates the quantized cache, by the `--kl` criterion below.
- **Long context:** a 160-position model runs 150 positions, so attention
  crosses its 64-position blocks. The host scalar build must match the
  reference within 1e-3 (F32 cache) and 5% (Q8). The build under test is held
  to 5% without the argmax check (`compare_logits.py --any-argmax`): builds
  with FMA round some activations differently and drift by up to 2% over 150
  positions, while a wrong block offset or stride garbles the logits (85%
  error).
- **Real model (`make test-all`, needs network):** stories15M logits from the
  scalar and avx2 builds, packed and `--no-pack`, are compared with the numpy
  reference using `compare_logits.py --kl 1e-2`. That mode checks that the
  argmax matches, at least 4 of the top 5 tokens agree, and the KL divergence is
  small. The avx2 build with `--kv q8` is held to the same bound against
  the F32 reference (KL about 1e-3). A real model can have an activation sitting exactly on a quantization
  rounding tie. Two valid implementations can then round it differently, which
  moves logits by up to about 2% while the distribution stays the same (KL
  ~1e-3). The random-model tests keep the strict 1e-3 relative bound.
- **Bad input:** malformed options (`-n 5x`, ids out of range, NaN
  temperature, …) exit with code 2, and `--help` exits with code 0.

### Loader robustness

`test_load` patches a copy of a valid model and checks each failure in turn:

- truncated file;
- empty file;
- bad magic or version;
- `dim` of 48 (not a multiple of 32);
- bad head counts;
- unknown `wtype` or `rope_type`;
- a Q8 quant equal to −128;
- `vocab < 3`;
- dimensions whose sizes would overflow.

It also loads a model whose KV cache is larger than 4 GB. On 32-bit ARMv7 that
model must fail with `ML_E_NOMEM` instead of corrupting memory.

## Running ARM tests on x86

```sh
sudo apt-get install qemu-user gcc-aarch64-linux-gnu gcc-arm-linux-gnueabihf
make ISA=neon CROSS=arm-linux-gnueabihf- test e2e
make ISA=neon_dotprod CROSS=aarch64-linux-gnu- test e2e
make ISA=neon_i8mm CROSS=aarch64-linux-gnu- test e2e      # qemu -cpu max has i8mm
```

## Sanitizers

```sh
make ISA=avx2 EXTRA_CFLAGS="-fsanitize=address,undefined -g" EXTRA_LDFLAGS="-fsanitize=address,undefined" test
make ISA=avx2 THREADS=pthread EXTRA_CFLAGS="-fsanitize=thread -g" EXTRA_LDFLAGS=-fsanitize=thread test
```

## CI

`.github/workflows/ci.yml` has two jobs:

- **`matrix`** (ubuntu-24.04) runs `make test-all`. That covers:
  - x86 builds natively; ISAs the runner's CPU lacks are built and reported as
    SKIP;
  - ARM builds under qemu;
  - CMake and MinGW builds;
  - pytest;
  - a download of `stories15M` from llama2.c, followed by text generation.
- **`arm64`** (ubuntu-24.04-arm) runs scalar, neon, neon_dotprod, neon_i8mm
  and sve tests natively on Arm hardware (each only if the CPU has the
  extension), then prints a benchmark.

## Benchmarking

```sh
python3 tools/gen_random.py --out bench.mlb --dim 512 --layers 6 --heads 8 --kv-heads 4 \
    --hidden 1376 --vocab 4096 --max-seq 128
make ISA=avx2 BENCH_MODEL=bench.mlb bench
build/avx2-pt/ml_bench bench.mlb -n 128 -t 4        # THREADS=pthread build
```

`ml_bench` reports three kinds of numbers:

- matvec ns/iter, GB/s and GOPS, for 768² and 4096² matrices in Q8 and Q4;
- `quantize_q8` time for 4096 values;
- tokens/s on the model you pass, with greedy decoding.

`ml_bench` also times the packed kernels (`matvec_q8_pk`, `matvec_q4_pk`) when
the build has them.

All x86 results below are from a 4-vCPU VM with AVX-512 VNNI. The kernel and
prefill tables were measured in one session, as the best of several runs
pinned to one core. The VM is shared, so expect ±10% noise.

**Kernel throughput, single thread** (`ml_bench`, 4096×4096, GOPS):

| ISA        | Q8 row-major | Q8 packed | Q4 row-major | Q4 packed |
|------------|-------------:|----------:|-------------:|----------:|
| sse2       | 24.7 | 28.5 | 18.3 | 34.5 |
| sse41      | 35.6 | 33.5 | 22.0 | 47.6 |
| avx2       | 47.2 | 48.5 | 33.8 | 76.0 |
| avxvnni    | 47.1 | 51.3 | 34.8 | 77.7 |
| avx512vnni | 50.3 | 52.0 | 51.3 | 91.3 |

At this size Q8 is limited by memory bandwidth, so packing mostly helps Q4.
When the matrix fits in cache (768×768), the packed kernels reach 92 GOPS on
avx2 and 129–139 GOPS on avx512vnni.

**Decoding**, dim-2048 model (4 layers, hidden 5632, vocab 32000), packed
weights, tok/s, without → with the software prefetch of the weight stream
(`-DML_USE_PREFETCH=0` builds the former):

| ISA | Q8 `-t 1` | Q4 `-t 1` | Q8 `-t 4` | Q4 `-t 4` |
|-----|----------:|----------:|----------:|----------:|
| sse2       | 22.6 → 42.3 | 38.2 → 56.4 | | |
| sse41      | 23.3 → 49.3 | 40.9 → 81.7 | | |
| avx2       | 29.3 → 52.2 | 54.9 → 122.4 | 125 → 242 | 369 → 451 |
| avxvnni    | 34.6 → 53.3 | 56.1 → 141.6 | | |
| avx512vnni | 46.3 → 58.0 | 73.6 → 140.1 | 192 → 254 | 482 → 503 |

Decoding reads every weight once per token. The hardware prefetchers stop at
4 KB page boundaries, so one core alone could not keep the stream going; the
packed matvecs now request the weights 4 KB ahead. On the Arm runner
(Neoverse) the same prefetch made NEON dotprod Q8 decoding 26% slower (470
against 638 tok/s, interleaved best of 3) and left Q4 unchanged, so Arm builds
leave it off.

**Prompt prefill**, 64 tokens, dim-2048 model (4 layers, hidden 5632, vocab
32000), packed weights, `-t 1` (tok/s):

| ISA        | Q8 `--batch 1` | Q8 batched | Q4 `--batch 1` | Q4 batched |
|------------|---------------:|-----------:|---------------:|-----------:|
| sse2       | 34.9 | 100.8 |  73.3 |  92.9 |
| sse41      | 30.4 | 114.4 |  92.9 | 178.2 |
| avx2       | 41.4 | 202.7 | 156.4 | 394.9 |
| avxvnni    | 44.5 | 311.9 | 152.7 | 484.2 |
| avx512vnni | 67.4 | 339.3 | 211.1 | 558.3 |

Packing the SSE2 build raised its Q8 prefill from 51 to 93 tok/s and its Q8
decode from 16 to 24 tok/s; Q4 decode went from 23 to about 40 tok/s.

The batched kernels run each tile through a body instantiated for a constant
token count (`PK_TILES`, `ML_UNROLL`), so the per-token accumulators stay in
registers. Before that, GCC kept them on the stack, and batched prefill was
1.2–1.6× slower (avx2 Q4 292, avxvnni Q8 197, avx512vnni Q4 384 tok/s).

**Long context**, dim-2048 Q4 model (4 layers, 32 heads, 8 KV heads, hidden
5632), avx2, `-t 1`, 4096 positions decoded from a 1-token prompt:

| Attention version | attention time | share of forward | tok/s over 4096 positions |
|-------------------|---------------:|-----------------:|--------------------------:|
| one call per position (before) | 45.2 s | 35% | 31.7 |
| row blocks, GQA-shared (`--kv f32`) | 19.0 s | 19% | 40.9 |
| Q8 cache (`--kv q8`) | 13.4 s | 14% | 43.8 |

**Arm, single thread** (GitHub `ubuntu-24.04-arm` runner, Neoverse-class
cores). These numbers come from the CI benchmark step, which publishes them as
annotations. The model is dim 1024, 4 layers, hidden 2816.

| ISA | Q4 4096² GOPS (row-major → packed) | Q4 decode tok/s | Q4 prefill tok/s | Q8 prefill tok/s |
|-----|-----------------------------------:|----------------:|-----------------:|-----------------:|
| neon (no dot product) | 16.7 → 30.6 | 152 → 267 | 184 → 326 | 224 → 290 |
| neon_dotprod | 29.5 → 67.6 | 261 → 550 | 320 → 948 | 436 → 1144 |
| neon_i8mm | 29.5 → 67.6 | 261 → 550 | 320 → 1167 | 433 → 1418 |

The neon_dotprod packed kernels use `vdotq_laneq_s32`, so each step reads its 4
activation bytes from a vector lane instead of broadcasting them. neon_i8mm
decodes with the same kernels and prefills on `smmla`, 23–24% faster than
`sdot`.

## Wiki

The GitHub wiki is generated from `README.md` and `docs/`; edit those, never
the wiki. `.github/workflows/wiki.yml` runs `tools/sync_wiki.py` on every push
to `main` that touches them and pushes the result with `GITHUB_TOKEN`; it can
also be started by hand from the Actions tab. To sync without Actions:

```sh
git clone https://github.com/JonPark0/isa-lm.wiki.git wiki
python3 tools/sync_wiki.py wiki --repo JonPark0/isa-lm
cd wiki && git add -A && git commit -m "Sync from docs" && git push
```
