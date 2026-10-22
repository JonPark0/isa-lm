# Testing and benchmarking

## Test layers

| Command | What it checks |
|---------|----------------|
| `make ISA=X test` | C unit tests: `test_arena`, `test_quant`, `test_kernels` (including the packed and batched kernels), `test_load`, `test_sampler`, `test_threads` |
| `make ISA=X e2e` | `tests/test_e2e.sh` runs `ml_run` against the numpy reference, the scalar build, and CLI edge cases |
| `python3 -m pytest tests` | the Python tools: format round trips, the reference forward, both converters, RoPE scaling |
| `make guard-test` | x86 ISA guard, using `qemu-x86_64 -cpu core2duo/Nehalem/SandyBridge/Haswell` |
| `make guard-test-arm` | AArch64 ISA guard, using `qemu-aarch64 -cpu cortex-a53/cortex-a76/max` |
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
  batching.
- **Real model (`make test-all`, needs network):** stories15M logits from the
  scalar and avx2 builds, packed and `--no-pack`, are compared with the numpy
  reference using `compare_logits.py --kl 1e-2`. That mode checks that the
  argmax matches, at least 4 of the top 5 tokens agree, and the KL divergence is
  small. A real model can have an activation sitting exactly on a quantization
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
- **`arm64`** (ubuntu-24.04-arm) runs scalar, neon, neon_dotprod and sve tests
  natively on Arm hardware, then prints a benchmark.

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

All results below are from a 4-vCPU x86 VM with AVX-512 VNNI. The VM is shared,
so expect ±10% noise.

**Kernel throughput, single thread** (`ml_bench`, Q4 4096×4096, GOPS):

| ISA        | row-major | packed |
|------------|----------:|-------:|
| sse41      | 17.6 | 30.5 |
| avx2       | 25.1 | 63.5 |
| avxvnni    | 22.8 | 56.0 |
| avx512vnni | 45.9 | 75.7 |

When the matrix fits in cache (768×768), the packed kernels reach 92 GOPS on
avx2 and 129–139 GOPS on avx512vnni.

**Decoding**, dim-2048 model, avx512vnni (tok/s):

| Format | `-t 1` row-major | `-t 1` packed | `-t 4` row-major | `-t 4` packed |
|--------|-----------------:|--------------:|-----------------:|--------------:|
| Q8 | 26 | 39 | 81 | 116 |
| Q4 | 24 | 53 | 98 | 181 |

**Prompt prefill**, 64 tokens, dim-2048 model, `-t 1` (tok/s):

| ISA        | Q8 `--batch 1` | Q8 batched | Q4 `--batch 1` | Q4 batched |
|------------|---------------:|-----------:|---------------:|-----------:|
| sse41      | 23.5 |  48.9 | 40.9 |  95.0 |
| avx2       | 31.2 |  89.9 | 52.5 | 180.9 |
| avxvnni    | 34.2 | 118.5 | 56.6 | 224.8 |
| avx512vnni | 43.1 | 197.8 | 57.2 | 311.1 |

With `-t 4`, avx512vnni prefills 665 tok/s for Q8 and 906 tok/s for Q4.

## Wiki

The GitHub wiki is generated from `README.md` and `docs/`; edit those, then:

```sh
git clone https://github.com/JonPark0/isa-lm.wiki.git wiki
python3 tools/sync_wiki.py wiki --repo JonPark0/isa-lm
cd wiki && git add -A && git commit -m "Sync from docs" && git push
```
