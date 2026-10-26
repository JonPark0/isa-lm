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
| AVX-512 VNNI | Q8_0 | 1 | 144.9 | 335.4 | 21.9 | 20.0 | 1.09× |
| AVX-512 VNNI | Q8_0 | 4 | 401.8 | 953.3 | 65.1 | 56.8 | 1.15× |
| AVX-512 VNNI | Q4_0 | 1 | 209.6 | 365.4 | 37.5 | 27.1 | 1.38× |
| AVX-512 VNNI | Q4_0 | 4 | 483.7 | 828.8 | 103.0 | 80.8 | 1.27× |
| AVX2 | Q8_0 | 1 | 81.6 | 69.5 | 20.2 | 10.6 | 1.91× |
| AVX2 | Q8_0 | 4 | 210.3 | 198.3 | 60.2 | 36.7 | 1.64× |
| AVX2 | Q4_0 | 1 | 146.2 | 77.8 | 35.2 | 16.9 | 2.08× |
| AVX2 | Q4_0 | 4 | 453.9 | 269.3 | 110.1 | 56.0 | 1.97× |
| SSE4.1 / SSE4.2 | Q8_0 | 1 | 43.2 | 13.2 | 18.4 | 6.1 | 3.02× |
| SSE4.1 / SSE4.2 | Q8_0 | 4 | 96.5 | 47.6 | 45.5 | 20.1 | 2.26× |
| SSE4.1 / SSE4.2 | Q4_0 | 1 | 56.2 | 25.1 | 23.8 | 11.2 | 2.12× |
| SSE4.1 / SSE4.2 | Q4_0 | 4 | 141.0 | 92.5 | 64.6 | 38.3 | 1.69× |
| SSE2 | Q8_0 | 1 | 36.1 | 12.1 | 17.0 | 5.8 | 2.93× |
| SSE2 | Q8_0 | 4 | 77.0 | 46.9 | 47.2 | 18.7 | 2.52× |
| SSE2 | Q4_0 | 1 | 32.4 | 5.6 | 23.5 | 3.3 | 7.12× |
| SSE2 | Q4_0 | 4 | 86.4 | 19.4 | 55.9 | 11.4 | 4.90× |

- **Decode** is faster with isa-lm at every level: 1.1–1.4× with AVX-512
  VNNI, about 2× with AVX2, and 1.7–7× on the SSE builds.
- **Prefill** is 1.7–2.4× faster with llama.cpp on AVX-512 VNNI, whose batched
  matrix multiplication beats isa-lm's batches of 32 there. From AVX2 down,
  isa-lm prefills faster.

## Quality

| Runtime | Weights | Perplexity |
|---------|---------|-----------:|
| llama.cpp | F16 | 13.34 |
| llama.cpp | Q8_0 | 13.44 |
| isa-lm | Q8_0 | 13.33 |
| llama.cpp | Q4_0 | 15.86 |
| isa-lm | Q4_0 | 15.10 |

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
