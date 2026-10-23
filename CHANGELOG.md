# Changelog

## Unreleased

### Added

- Qwen2 / Qwen2.5 models: q/k/v projection biases (`.mlb` header flag at
  offset 76), the Qwen2 tokenizer (NFC normalization, single-digit numbers;
  MLT1 flags 8 and 16), and the `qwen` and `chatml` chat templates.
  Qwen2.5 0.5B Instruct Q8: perplexity 19.05 against 19.00 for an F32
  reference.
- `scripts/compare_llamacpp.sh` and [the comparison with llama.cpp](docs/benchmarks.md):
  decode is 1.1–1.4× faster than llama.cpp with AVX-512 VNNI, about 2× with
  AVX2 and 1.7–7× on SSE builds; llama.cpp prefills faster with AVX-512.
- `tools/perplexity.py --llama-cpp` chunks and scores as `llama-perplexity`.
- `tools/convert.py hf` reads Mistral checkpoints and caps the context of
  sliding-window models at the window.

### Changed

- `tools/convert.py hf` refuses architectures it cannot run correctly
  (`model_type` other than llama, mistral or qwen2; activations other than
  SiLU; explicit `head_dim`; biases other than q/k/v) instead of writing a
  wrong model.
- `tools/perplexity.py` works with tokenizers that add no BOS.
- Q4 scales are fitted per block (weighted least squares over candidate
  mappings, all 16 levels) instead of `amax / 7`: Llama 3.2 1B Q4 perplexity
  20.46 → 19.91, Qwen2.5 0.5B 23.58 → 21.73. The format is unchanged;
  convert again to benefit.

## 0.1.0 — 2026-10-23

First release.

### Runtime

- C11 inference for Llama-style decoders (RMSNorm, SwiGLU, GQA, RoPE with
  Llama 3 scaling) with Q8_0 and Q4_0 weights, and mixed Q4 models that keep
  chosen matrix classes in Q8 (`--q8 v,o,down`).
- Compile-time kernel binding for scalar, SSE2, SSE4.1, AVX2, AVX-VNNI,
  AVX-512 VNNI, ARMv7/AArch64 NEON, NEON dotprod, i8mm and SVE; a CPU guard
  exits with code 3 instead of crashing on processors without the build's ISA.
- Row-interleaved weight packing for each ISA's integer dot products,
  batched prompt prefill (on `smmla` with i8mm), software prefetch on x86,
  blocked GQA attention, an optional Q8 KV cache (`--kv q8`), a thread pool
  with bit-identical results for any thread count, and a pack cache
  (`--pack-cache`) for fast start-up.

### Models and tokenizers

- Converters for llama2.c checkpoints and Hugging Face safetensors (Llama 2,
  Llama 3/3.1/3.2, TinyLlama, Mistral).
- Hugging Face `tokenizer.json` support for SentencePiece-style BPE and Llama
  3 byte-level BPE (`tools/convert_tokenizer.py`), identical to the
  `tokenizers` library on every text tested; `ml_tok` encodes and decodes.

### Tools

- `ml_run`: generation with top-k, top-p and repetition penalty, and a chat
  mode (`--chat llama3|zephyr|llama2`).
- `ml_bench`, `tools/perplexity.py`, `--profile`.

### Verification

- Kernel tests against scalar and double-precision references, end-to-end
  logits against a numpy reference, qemu for ARM and older x86 CPUs, native
  Arm CI, and real-model checks (stories15M, TinyLlama 1.1B, Llama 3.2 1B).
- Release binaries for Linux x86-64, AArch64 and ARMv7 (static) and Windows
  x86-64 (MinGW, checked under Wine), each run before publishing.
