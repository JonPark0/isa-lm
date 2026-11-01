---
title: isa-lm — portable INT8/INT4 LLM inference for legacy x86-64 and ARM CPUs
description: Dependency-free C11 runtime that runs quantized Llama-style language models on CPUs from SSE2 and ARMv7 NEON up to AVX-512 VNNI and SVE, with each operator bound at compile time to the best kernel for the instruction set.
---

# isa-lm

**isa-lm** is a dependency-free **C11 inference runtime** for Llama-style **large language models** with
**INT8 and INT4 weights**. It runs on **x86-64 CPUs from SSE2 up to AVX-512 VNNI** and on **ARM CPUs from ARMv7
NEON up to i8mm and SVE** — including the legacy and low-power processors that GPU-first runtimes leave out.
The code is written against a portable baseline; you choose the instruction set when you compile, and every
operator is bound to the best kernel for it, with no runtime dispatch.

- **Faster decoding than llama.cpp** at every ISA level measured, on the same
  model and formats: 1.2–1.5× with AVX-512 VNNI, 1.0–2.3× with AVX2, 2.1–10×
  with SSE2 and SSE4.1, 1.04–1.7× on AArch64 ([comparison](benchmarks.md)).
- **No AVX, GPU or libraries needed:** any x86-64 CPU (SSE2) and 32-bit ARMv7
  NEON boards, up to AVX-512 VNNI, i8mm and SVE; static binaries for each ISA.
- **Accurate Q4:** fitted block scales give Qwen2.5 0.5B Q4_0 perplexity 15.1
  against 15.8–15.9 for llama.cpp's Q4_0; Q8 matches an F32 reference, and the
  output is bit-identical for any thread count.
- **Models:** Llama 2/3, Mistral, Qwen2.5, Qwen3, SmolLM2, Gemma 2 and Gemma 3
  from Hugging Face safetensors, with their tokenizers and chat templates.

- **Instruction sets:** scalar, SSE2, SSE4.1, AVX2, AVX-VNNI, AVX-512 VNNI, ARMv7/AArch64 NEON, NEON dotprod,
  i8mm (`smmla`), SVE (any vector length); a CPU guard refuses unsupported processors instead of crashing
- **Quantization:** Q8_0, Q4_0, and mixed models that keep chosen matrices in Q8; optional Q8 KV cache
- **Speed:** row-interleaved weight packing for each ISA's integer dot products, batched prompt prefill,
  software prefetch, blocked GQA attention, a thread pool with bit-identical results, a pack cache for fast start-up
- **Models:** llama2.c checkpoints and Hugging Face safetensors (Llama 2, Llama 3/3.1/3.2, TinyLlama, SmolLM2,
  Mistral, Qwen2/2.5, Qwen3; llama3, linear and yarn RoPE scaling; sliding window), Hugging Face `tokenizer.json`
  (SentencePiece-style, Llama 3, Qwen and GPT-2 byte-level BPE, identical to the `tokenizers` library)
- **Use:** text generation, chat mode (`llama3`, `zephyr`, `llama2`, `qwen`, `smollm`, `chatml` templates), top-k /
  top-p / repetition penalty, perplexity measurement, importance matrices for Q4
- **Verification:** kernel tests against scalar and double-precision references, end-to-end logits against a
  numpy reference, qemu for ARM and old x86 CPUs, native Arm CI

```sh
make ISA=avx2                                    # or sse2, neon, neon_dotprod, sve, ...
python3 tools/convert.py hf ./Llama-3.2-1B-Instruct llama.mlb --wtype q4 --q8 v,o,down
python3 tools/convert_tokenizer.py ./Llama-3.2-1B-Instruct/tokenizer.json llama.mlt
build/avx2/ml_run llama.mlb -z llama.mlt --chat llama3 -t 4
```

## Documentation

[FAQ](faq.md) · [User guide](user-guide.md) · [Architecture](architecture.md) · [File formats](file-format.md) ·
[Writing kernels](kernels.md) · [Testing and benchmarks](testing.md) ·
[Comparison with llama.cpp](benchmarks.md) ·
[Design spec](design/2026-10-22-legacy-isa-ml-runtime-design.md)

## Citing

```bibtex
@software{isalm2026,
  author  = {Park, Yeochoon},
  title   = {isa-lm: a portable C11 runtime for INT8/INT4 Llama-style language model inference on legacy x86-64 and ARM instruction sets},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/JonPark0/isa-lm}
}
```

Source code: <https://github.com/JonPark0/isa-lm> · License: MIT
