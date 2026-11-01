---
title: "Changelog: what changed in each isa-lm release"
description: "Release notes of isa-lm: new models, kernels, formats, speed-ups and fixes in each version."
nav_order: 9
nav_title: "Changelog"
---
<!-- Generated from CHANGELOG.md by scripts/sync_docs.py; edit CHANGELOG.md instead. -->

# Changelog

## Unreleased

### Added

- Documentation site on Just the Docs: a sidebar, search, dark mode (following
  the system, with a switch), heading links, copy buttons on code, "Edit this
  page" links, a 404 page and icons. The home page gains download and model
  buttons, a chart of decode speed against llama.cpp
  (`docs/assets/decode-ratio.svg`, drawn from `docs/benchmarks.md` by
  `scripts/sync_docs.py` and checked in CI) and a table of supported CPUs.
- A Korean home page (`/ko/`, linked both ways with hreflang), a changelog
  page generated from `CHANGELOG.md` (`scripts/sync_docs.py`), and Mermaid
  diagrams of compile-time kernel binding and the CPU guard in the
  architecture page.

- `.github/workflows/hf-models.yml` and `scripts/hf_publish.py`: convert the
  models of `scripts/hf_models.json` (Qwen2.5, Qwen3, SmolLM2, TinyLlama;
  Llama 3.2 and Gemma 3 on request), measure each file (perplexity, speed, a
  chat reply) and publish them on Hugging Face with a model card. Converted
  SmolLM2 135M/360M, Qwen2.5 0.5B/1.5B, Qwen3 0.6B and TinyLlama 1.1B files are
  at `jonpark0/<model>-isa-lm` (README, user guide and FAQ link them).

## 0.5.0 — 2026-11-01

### Added

- Gemma 3 4B, 12B and 27B: their image-text checkpoints (`model_type`
  `gemma3`) convert to the text model, leaving the vision tower out. Gemma 3
  4B Instruct Q8 logits match an F32 numpy forward (KL 0.002); Q4 takes 2083
  MiB and decodes at 14.8 tok/s on 4 vCPUs.
- Gemma 2: logit soft-capping of the attention scores and the logits (`.mlb`
  header offsets 124 and 128), even layers behind the sliding window.
- F16 KV cache (`--kv f16`): K and V in half precision, half the memory of
  the F32 cache at the same speed (Qwen2.5 0.5B Q4, 1500 tokens: prefill 722
  against 718 tok/s), with perplexity within the rounding (21.71 against
  21.74; the Q8 cache gives 22.82).

### Changed

- `tools/convert.py` converts checkpoint tensors to float32 one at a time, so
  a 4B model converts in about the memory of its largest tensor instead of
  16 GB.
- AVX-512 VNNI prompt prefill feeds each weight vector to 6 tokens instead
  of 4: matmuls 1.05-1.08× and Qwen2.5 0.5B prefill 1.05× faster on one
  thread (GitHub runner; four threads within noise).
- AVX2 Q8 prompt prefill takes |x| from a buffer made once per tile, saving a
  `psignb` per token and step: Q8 matmuls 1.10-1.16× and Qwen2.5 0.5B Q8
  prefill 1.14× faster on one thread (four threads within noise).

## 0.4.0 — 2026-10-31

### Added

- Gemma 3 text models (Gemma 3 1B): GELU, block output norms, embedding and
  attention scales (`.mlb` header offsets 104-116), local and global layers;
  the Gemma tokenizer (no U+2581 before the text, MLT1 flag 128) and
  `--chat gemma`. Gemma 3 1B Q8 perplexity 42.20 against 42.32 for an F32
  reference.
- Per-layer sliding windows and RoPE bases: a flag byte per layer (header
  offset 96) puts a layer behind the window and on the local RoPE base
  (offset 100). The converter writes them when only some layers slide
  (`layer_types`, Qwen2's `max_window_layers`) instead of capping the context.
- f16 block scales (`.mlb` header offset 120): Q4_0 takes 4.5 bits per
  weight instead of 5, Q8_0 8.5 instead of 9. Both kinds of file load in every
  build.
- `tools/convert.py --q4-rule round`: llama.cpp's Q4_0 rounding, now the
  default for Gemma 3 (`--q4-rule fit` for the fitted rule). Gemma 3 1B Q4
  perplexity 49.83 against 53.25 fitted; its quantization-aware checkpoint
  (`gemma-3-1b-it-qat-q4_0-unquantized`) reaches 40.50, below its Q8 (42.53).
  The fit stays the default for other models (Qwen2.5 0.5B 21.73 against
  22.45).

### Changed

- `tools/convert.py` writes f16 scales by default (`--scales f32` keeps the
  older files): perplexity changes by at most 0.15% (Qwen2.5 0.5B, Llama 3.2
  1B), and Qwen2.5 0.5B Q4 shrinks from 309 to 278 MB. Versions before 0.4.0
  cannot read files with f16 scales.
- AVX2, AVX-VNNI and AVX-512 VNNI builds keep f16 scales as f16 in the packed
  weights, so decoding reads fewer bytes: on GitHub's x86-64 runner
  (Qwen2.5 0.5B) Q4 decode is 1.04-1.12x faster and Q8 1.05-1.11x. These
  builds now require F16C, which every AVX2 CPU has.
- Prompt prefill takes up to 128 tokens per batched step (was 32; `--batch`
  1-128), so each step's pass over the weights serves 4x the tokens: prefill
  up to 1.4x faster on the AVX-512 reference VM (Qwen2.5 0.5B, 4 threads) and
  1.06-1.15x on GitHub's Neoverse N2 runner; level where the model fits the
  cache (GitHub's EPYC runner). Results stay identical to `--batch 1`.
- AArch64 dotprod builds (neon_dotprod, neon_i8mm, sve): Q4 nibbles are packed
  in signed form, so no per-block activation sums are subtracted; the dotprod
  batched kernels take 8 tokens per weight block, and the i8mm ones
  accumulate in smmla's layout. On GitHub's Neoverse N2 runner (Qwen2.5 0.5B)
  Q4 prefill is 1.16-1.24x faster, Q4 decode 1.07-1.12x, and Q8 prefill on
  neon_dotprod and sve 1.04x. Pack cache files from earlier builds are
  rewritten.
- [The benchmarks](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md)
  explain why AMX stays out of the build: on the reference CPU the tile
  products and the AVX-512 rescaling of every 32-weight block do not overlap,
  and a second AMX kernel, though 1.2-1.4x the AVX-512 VNNI kernel for Q8,
  prefills no faster on a whole model.

## 0.3.0 — 2026-10-31

### Added

- Qwen3: a `head_dim` other than `dim / n_heads` and per-head q/k RMSNorm
  (`.mlb` header offsets 80 and 84). Qwen3 0.6B Q8 perplexity 27.91 against
  28.10 for an F32 reference.
- SmolLM2 and GPT-2 tokenizers: the GPT-2 pre-tokenizer regex (MLT1 flag 32),
  after a digit split if the tokenizer has one (flag 64); identical to the
  tokenizers library on every checked text. `--chat smollm`.
- RoPE scaling of the linear and yarn kinds (header `rope_type` 2 and 3, yarn's
  attention factor at offset 88), with transformers' formulas.
- Sliding-window attention (offset 92): a window on every layer, as Mistral
  v0.1 has, is kept instead of capping the context.
- Windows: CI builds the scalar, SSE2 and AVX2 configurations with MSVC and
  checks them against the numpy reference.
- `scripts/compare_llamacpp.sh` runs on AArch64 too (native, dotprod and Armv8.0
  levels), and the manual `llama.cpp comparison` workflow runs it on GitHub's
  x86-64 and AArch64 runners. [The comparison](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md)
  now covers both: on x86-64 without AVX-512 isa-lm is ahead or level
  everywhere; on AArch64 decode is level to 1.2× faster with dotprod and i8mm
  and 1.3–1.7× faster on Armv8.0, while llama.cpp prefills Q4_0 faster with
  dotprod. On the reference machine llama.cpp's AVX-512 prefill lead comes
  from AMX; without it llama.cpp prefills at about a third of isa-lm's speed.

### Changed

- `tools/convert.py hf` refuses partial rotary embeddings, and q/k projections
  that do not match the configured heads.
- Importance matrix files record the attention width when it differs from
  `dim` (Qwen3); files for other models are unchanged.

## 0.2.0 — 2026-10-27

### Added

- Qwen2 / Qwen2.5 models: q/k/v projection biases (`.mlb` header flag at
  offset 76), the Qwen2 tokenizer (NFC normalization, single-digit numbers;
  MLT1 flags 8 and 16), and the `qwen` and `chatml` chat templates.
  Qwen2.5 0.5B Instruct Q8: perplexity 19.05 against 19.00 for an F32
  reference.
- `scripts/compare_llamacpp.sh` and [the comparison with llama.cpp](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md):
  decode is 1.1–1.5× faster than llama.cpp with AVX-512 VNNI, about 2× with
  AVX2 and 2.2–7× on SSE builds; llama.cpp still prefills Q8_0 faster with
  AVX-512.
- `tools/perplexity.py --llama-cpp` chunks and scores as `llama-perplexity`.
- `tools/convert.py hf` reads Mistral checkpoints and caps the context of
  sliding-window models at the window.
- Importance matrix: `ml_run --imatrix FILE` sums the squares of every matrix
  input per channel (over runs), and `tools/convert.py --imatrix FILE`
  weights the Q4 fits by it as llama.cpp does. Qwen2.5 0.5B Q4 with one from
  32K tokens of wikitext-2 train: perplexity 21.72 → 20.92 (llama.cpp
  scoring: 15.13 → 14.52, against 13.33 for Q8). A tied embedding takes the
  classifier's importance. The file is replaced only once fully written.
- Releases from main: merging a version bump (`ML_VERSION` and its
  CHANGELOG.md section) builds, tests and publishes the release
  (`.github/workflows/auto-release.yml`); CI fails a bump without notes.
- `scripts/bench.py` and the Benchmark workflow: A/B speed comparison of two
  versions, run alternately on one machine (GitHub's x86-64 and AArch64
  runners for the workflow).

### Changed

- Faster prompt prefill: the packed matmul works through L2-sized blocks of
  rows, the AVX-512 VNNI tiles share each activation broadcast between two
  row groups, and a batched step's attention and per-token work run across
  threads in one pass each. 128-token prefill of a Qwen2.5-0.5B-shaped model
  on a 4-vCPU AVX-512 VNNI Xeon: Q8 180 → 260 tok/s on 1 thread and 600 →
  925 on 4; Q4 250 → 290 and about 660 → 890. Logits are unchanged.
- Batched kernels sum activation blocks with `psadbw` (Qwen2.5 0.5B, one
  AVX-512 VNNI thread: Q4 prefill 240 → 287 tok/s, Q8 231 → 246), and the
  AVX-512 VNNI and AVX-VNNI tiles are unrolled so that GCC no longer copies
  every accumulator (kernel throughput: AVX-512 VNNI Q8 +6–18%, AVX-VNNI Q8
  +12–51%).
- Q4 blocks of values below 1e-6 are fitted rescaled: their float32 fit
  underflowed (zero scale below about 1e-12). Other blocks convert to the
  same bytes.
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
