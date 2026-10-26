# Comparison with llama.cpp

isa-lm and [llama.cpp](https://github.com/ggml-org/llama.cpp) run the same
model with the same weight formats on one machine, at four x86 instruction-set
levels. `scripts/compare_llamacpp.sh` reproduces the run.

## Setup

- **Machine:** 4 vCPUs of an Intel Xeon at 2.10 GHz with AVX-512 VNNI, 15 GB
  of RAM; GCC 13.3 for both runtimes.
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
- **Prefill** of Q8_0 is 1.6–1.8× faster with llama.cpp on AVX-512 VNNI; Q4_0
  is 1.2× faster with llama.cpp on one thread and level on four. From AVX2
  down, isa-lm prefills faster (1.3–6.8×).

Measured with isa-lm at commit `0cb4d16`. Against the previous run (isa-lm
before the faster prefill), AVX-512 VNNI prefill went from 145 to 221 tok/s
(Q8_0, 1 thread) and from 484 to 890 (Q4_0, 4 threads); llama.cpp's own numbers
moved by -3% to +28% between the two runs on the same kind of machine, so
compare rows of one run.

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
importance matrix; with one from wikitext-2 train, isa-lm's Q4_0 scores 14.71
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
benchmarks and writes `build/compare/results.md`. A missing llama.cpp build
directory skips that row; `THREADS`, `TOKENS`, `REPS`, `CTX` and `CHUNKS`
change the settings.
