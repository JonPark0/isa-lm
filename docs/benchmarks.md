# Comparison with llama.cpp

isa-lm and [llama.cpp](https://github.com/ggml-org/llama.cpp) run the same
model with the same weight formats on one machine, at four x86 instruction-set
levels; the [GitHub runners](#github-runners) section repeats the comparison
on an x86-64 machine without AVX-512 and on AArch64 at three levels.
`scripts/compare_llamacpp.sh` reproduces each run.

## Setup

- **Machine:** 4 vCPUs of an Intel Xeon at 2.10 GHz (Emerald Rapids) with
  AVX-512 VNNI and AMX, 15 GB of RAM; GCC 13.3 for both runtimes. llama.cpp's
  native build uses AMX tiles for prompt processing on this CPU (see below);
  isa-lm has no AMX kernels.
- **llama.cpp:** release b11835 (commit `f64a498`), CPU backend, built four
  times:

  | Build | Options |
  |-------|---------|
  | AVX-512 VNNI | `GGML_NATIVE=ON` |
  | AVX2 | `GGML_NATIVE=OFF`, AVX, AVX2, BMI2, FMA, F16C; no AVX-512 |
  | SSE4.2 | `GGML_NATIVE=OFF`, SSE4.2 only |
  | SSE2 | `GGML_NATIVE=OFF`, every x86 extension off |

  isa-lm builds for `avx512vnni`, `avx2`, `sse41` and `sse2`, with
  `THREADS=pthread`. The SSE4.2 llama.cpp build sits against isa-lm's SSE4.1
  build: SSE4.2 adds only string and CRC instructions.
- **Model:** Qwen2.5 0.5B Instruct. llama.cpp quantizes the official
  `qwen2.5-0.5b-instruct-fp16.gguf` with `llama-quantize --pure`, so that
  every matrix, embeddings and classifier included, is Q8_0 or Q4_0, as in
  isa-lm's `--wtype q8` and `--wtype q4` files. The GGUF files are larger
  because they store the tied classifier a second time; each runtime reads one
  copy per token.
- **Speed:** prefill of 128 tokens, then 128 tokens decoded after them
  (`ml_bench -n 128`; `llama-bench -p 128 -n 0` and `-p 0 -n 128 -d 128`),
  mean of 3 runs, tokens per second.
- **Quality:** wikitext-2 test set, 4 chunks of 512 tokens, the second half of
  each chunk scored: `llama-perplexity -c 512 --chunks 4` and
  `tools/perplexity.py --llama-cpp`, which chunks and scores the same way.

## Speed

Tokens per second; the ratio is isa-lm's decode speed over llama.cpp's.

| ISA level | Weights | Threads | isa-lm prefill | llama.cpp prefill | isa-lm decode | llama.cpp decode | Decode ratio |
|-----------|---------|--------:|---------------:|------------------:|--------------:|-----------------:|-------------:|
| AVX-512 VNNI | Q8_0 | 1 | 221.2 | 361.4 | 24.3 | 21.8 | 1.11× |
| AVX-512 VNNI | Q8_0 | 4 | 674.0 | 1222.8 | 79.0 | 71.0 | 1.11× |
| AVX-512 VNNI | Q4_0 | 1 | 288.7 | 353.7 | 41.6 | 31.0 | 1.34× |
| AVX-512 VNNI | Q4_0 | 4 | 890.2 | 857.3 | 156.4 | 105.2 | 1.49× |
| AVX2 | Q8_0 | 1 | 91.4 | 72.0 | 22.8 | 12.4 | 1.84× |
| AVX2 | Q8_0 | 4 | 295.8 | 217.3 | 78.0 | 41.3 | 1.89× |
| AVX2 | Q4_0 | 1 | 167.4 | 83.5 | 43.1 | 18.6 | 2.32× |
| AVX2 | Q4_0 | 4 | 519.1 | 298.7 | 140.7 | 64.6 | 2.18× |
| SSE4.1 / SSE4.2 | Q8_0 | 1 | 48.5 | 14.4 | 21.0 | 6.7 | 3.13× |
| SSE4.1 / SSE4.2 | Q8_0 | 4 | 161.3 | 53.8 | 57.0 | 23.0 | 2.48× |
| SSE4.1 / SSE4.2 | Q4_0 | 1 | 75.3 | 28.3 | 35.0 | 13.2 | 2.65× |
| SSE4.1 / SSE4.2 | Q4_0 | 4 | 240.1 | 108.2 | 96.3 | 43.3 | 2.22× |
| SSE2 | Q8_0 | 1 | 40.9 | 13.9 | 19.1 | 6.5 | 2.94× |
| SSE2 | Q8_0 | 4 | 130.9 | 53.3 | 56.7 | 21.8 | 2.60× |
| SSE2 | Q4_0 | 1 | 38.6 | 5.7 | 25.4 | 3.7 | 6.86× |
| SSE2 | Q4_0 | 4 | 113.6 | 21.9 | 68.8 | 12.6 | 5.46× |

- **Decode** is faster with isa-lm at every level: 1.1–1.5× with AVX-512
  VNNI, about 2× with AVX2, and 2.2–7× on the SSE builds.
- **Prefill** of Q8_0 is 1.6–1.8× faster with llama.cpp on the native build;
  Q4_0 is 1.2× faster with llama.cpp on one thread and level on four. From
  AVX2 down, isa-lm prefills faster (1.3–6.8×).

The native llama.cpp build owes its prefill to AMX, not to AVX-512 VNNI: its
weights are repacked for AMX tiles by default. With that off
(`llama-bench --repack 0`, same machine, later run) it prefills Q8_0 at 93
tok/s on one thread and 318 on four, and Q4_0 at 83 and 320, about a third of
isa-lm's AVX-512 VNNI rows. An AMX kernel for isa-lm's packed layout was tried
(branch `exp/amx`); because every 32-weight block has its own scale, each tile
product has to be stored and rescaled with AVX-512, and that made it only
10–20% faster for Q8_0 and slower for Q4_0, so it is not part of the build.

Measured with isa-lm at commit `0cb4d16`. Against the previous run (isa-lm
before the faster prefill), AVX-512 VNNI prefill went from 145 to 221 tok/s
(Q8_0, 1 thread) and from 484 to 890 (Q4_0, 4 threads); llama.cpp's own numbers
moved by -3% to +28% between the two runs on the same kind of machine, so
compare rows of one run.

## GitHub runners

The `llama.cpp comparison` workflow (`.github/workflows/compare.yml`, run by
hand) builds both runtimes on GitHub's `ubuntu-24.04` (x86-64, 4 vCPUs with
AVX2 but no AVX-512, so its native level is AVX2) and `ubuntu-24.04-arm`
(AArch64, 4 vCPUs with dotprod and i8mm) runners and runs
`scripts/compare_llamacpp.sh`. On AArch64 the llama.cpp builds are the native
one (against isa-lm's best build, `neon_i8mm`), `armv8.2-a+dotprod` and
`armv8-a`. Results of one run (isa-lm `d54e2dd`, whose kernels are the
main branch's, and llama.cpp b11835):

### x86-64 (AVX2)

| ISA level | Weights | Threads | isa-lm prefill | llama.cpp prefill | isa-lm decode | llama.cpp decode | Decode ratio |
|-----------|---------|--------:|---------------:|------------------:|--------------:|-----------------:|-------------:|
| AVX2 | Q8_0 | 1 | 130.8 | 94.1 | 35.1 | 25.8 | 1.36× |
| AVX2 | Q8_0 | 4 | 258.4 | 179.6 | 59.1 | 59.3 | 1.00× |
| AVX2 | Q4_0 | 1 | 212.9 | 114.7 | 66.8 | 40.0 | 1.67× |
| AVX2 | Q4_0 | 4 | 426.9 | 238.3 | 97.4 | 83.8 | 1.16× |
| SSE4.1 / SSE4.2 | Q8_0 | 1 | 60.8 | 16.7 | 33.4 | 9.5 | 3.52× |
| SSE4.1 / SSE4.2 | Q8_0 | 4 | 120.5 | 34.9 | 57.2 | 19.5 | 2.93× |
| SSE4.1 / SSE4.2 | Q4_0 | 1 | 89.3 | 32.0 | 40.9 | 15.8 | 2.59× |
| SSE4.1 / SSE4.2 | Q4_0 | 4 | 179.8 | 66.8 | 78.1 | 31.9 | 2.45× |
| SSE2 | Q8_0 | 1 | 45.6 | 16.1 | 29.7 | 9.4 | 3.16× |
| SSE2 | Q8_0 | 4 | 90.5 | 34.4 | 50.1 | 19.7 | 2.54× |
| SSE2 | Q4_0 | 1 | 49.5 | 5.0 | 33.0 | 3.3 | 10.00× |
| SSE2 | Q4_0 | 4 | 98.7 | 10.2 | 63.0 | 6.7 | 9.40× |

### AArch64

| ISA level | Weights | Threads | isa-lm prefill | llama.cpp prefill | isa-lm decode | llama.cpp decode | Decode ratio |
|-----------|---------|--------:|---------------:|------------------:|--------------:|-----------------:|-------------:|
| i8mm (native) | Q8_0 | 1 | 156.0 | 125.9 | 31.6 | 32.1 | 0.98× |
| i8mm (native) | Q8_0 | 4 | 591.2 | 429.4 | 105.9 | 104.9 | 1.01× |
| i8mm (native) | Q4_0 | 1 | 142.3 | 147.9 | 45.8 | 38.1 | 1.20× |
| i8mm (native) | Q4_0 | 4 | 518.9 | 482.6 | 125.9 | 109.1 | 1.15× |
| dotprod | Q8_0 | 1 | 132.8 | 129.7 | 33.4 | 32.4 | 1.03× |
| dotprod | Q8_0 | 4 | 490.3 | 494.1 | 104.3 | 99.8 | 1.05× |
| dotprod | Q4_0 | 1 | 117.3 | 156.8 | 46.3 | 40.4 | 1.15× |
| dotprod | Q4_0 | 4 | 431.6 | 589.2 | 129.3 | 114.7 | 1.13× |
| NEON (Armv8.0) | Q8_0 | 1 | 36.4 | 26.6 | 20.9 | 15.4 | 1.36× |
| NEON (Armv8.0) | Q8_0 | 4 | 140.4 | 105.6 | 65.2 | 49.7 | 1.31× |
| NEON (Armv8.0) | Q4_0 | 1 | 41.1 | 24.4 | 26.5 | 15.8 | 1.68× |
| NEON (Armv8.0) | Q4_0 | 4 | 153.8 | 96.1 | 81.7 | 53.7 | 1.52× |

- **x86-64 without AVX-512:** isa-lm is ahead or level everywhere: decode 1.0–1.7×
  with AVX2 and 2.5–10× on SSE builds, prefill 1.4–1.9× with AVX2 and 2.6–10×
  on SSE builds.
- **AArch64:** decode is level with llama.cpp for Q8_0 at dotprod and i8mm and
  1.1–1.2× faster for Q4_0; on plain Armv8.0 NEON isa-lm is 1.3–1.7× faster
  in both prefill and decode. Prefill with i8mm is 1.2–1.4× faster for Q8_0
  and level for Q4_0, but with dotprod llama.cpp prefills Q4_0 1.3–1.4× faster
  (its repacked Q4_0 dotprod kernels): isa-lm's Q4_0 dotprod matmul has room
  to grow.
- **Quality** is the same on all three machines: isa-lm's Q8_0 scores 13.33–13.37
  against 13.34 for llama.cpp's F16, and its Q4_0 15.13–15.17 against
  15.82–15.93 for llama.cpp's.

## Quality

| Runtime | Weights | Perplexity |
|---------|---------|-----------:|
| llama.cpp | F16 | 13.34 |
| llama.cpp | Q8_0 | 13.44 |
| isa-lm | Q8_0 | 13.33 |
| llama.cpp | Q4_0 | 15.86 |
| isa-lm | Q4_0 | 15.13 |

The two runtimes tokenize this text identically (no BOS for Qwen2.5). isa-lm's
Q8_0 matches F16. Its Q4_0 is closer to F16 because `tools/convert.py` fits
each block's scale (see [Q8, Q4, or mixed?](user-guide.md#q8-q4-or-mixed)), while
`llama-quantize` without an importance matrix maps the largest weight to -8.
llama.cpp's default Q4_0 (without `--pure`) keeps the classifier in a higher
precision type, which isa-lm can do with `--q8 cls`. Neither row uses an
importance matrix; with one from wikitext-2 train, isa-lm's Q4_0 scores 14.52
([importance matrix](user-guide.md#q8-q4-or-mixed)).

## Reproducing

```sh
git clone --depth 1 --branch b11835 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build-native -j --target llama-bench llama-perplexity llama-quantize
# build-avx2, build-sse42, build-base: -DGGML_NATIVE=OFF with the options above
cd -
scripts/compare_llamacpp.sh ../llama.cpp ./Qwen2.5-0.5B-Instruct \
    qwen2.5-0.5b-instruct-fp16.gguf wikitext-2-test.txt build/compare
```

The script converts the model for isa-lm, quantizes the GGUF, runs both
benchmarks and writes `build/compare/results.md`. It picks the ISA levels
from the host: on AArch64 the llama.cpp build directories are `build-native`,
`build-dotprod` (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`) and
`build-base` (`armv8-a`). A missing llama.cpp build directory skips that row;
`THREADS`, `TOKENS`, `REPS`, `CTX` and `CHUNKS` change the settings.

On GitHub, run the `llama.cpp comparison` workflow (Actions tab, Run
workflow); it reports each machine's table in the run summary, as an artifact
and as an annotation.
