# Testing and benchmarking

## Test layers

| Command | What it checks |
|---------|----------------|
| `make ISA=X test` | C unit tests: `test_arena`, `test_quant`, `test_kernels`, `test_load`, `test_sampler`, `test_threads` |
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

Results from a 4-vCPU x86 VM with AVX-512 VNNI, single-threaded (the VM is
shared, so expect ±10% noise):

| ISA        | matvec Q8 GOPS | matvec Q4 GOPS | tok/s Q8 | tok/s Q4 |
|------------|---------------:|---------------:|---------:|---------:|
| scalar     |  8.8 |  7.3 | 212 | 203 |
| sse2       | 17.5 | 14.7 | 458 | 357 |
| sse41      | 27.2 | 16.8 | 529 | 476 |
| avx2       | 36.8 | 27.0 | 811 | 703 |
| avxvnni    | 39.7 | 26.2 | 759 | 589 |
| avx512vnni | 43.0 | 28.9 | 854 | 659 |

Thread scaling for the avx2 build with `THREADS=pthread`:

| Model | `-t 1` | `-t 4` |
|-------|-------:|-------:|
| dim 512 | about 750 tok/s | 1376 tok/s |
| dim 2048, Q4 | 24 tok/s | 76 tok/s |
