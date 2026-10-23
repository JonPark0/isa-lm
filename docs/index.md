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

- **Instruction sets:** scalar, SSE2, SSE4.1, AVX2, AVX-VNNI, AVX-512 VNNI, ARMv7/AArch64 NEON, NEON dotprod,
  i8mm (`smmla`), SVE (any vector length); a CPU guard refuses unsupported processors instead of crashing
- **Quantization:** Q8_0, Q4_0, and mixed models that keep chosen matrices in Q8; optional Q8 KV cache
- **Speed:** row-interleaved weight packing for each ISA's integer dot products, batched prompt prefill,
  software prefetch, blocked GQA attention, a thread pool with bit-identical results, a pack cache for fast start-up
- **Models:** llama2.c checkpoints and Hugging Face safetensors (Llama 2, Llama 3/3.1/3.2 with RoPE scaling,
  TinyLlama, Mistral), Hugging Face `tokenizer.json` (SentencePiece-style and Llama 3 byte-level BPE, identical to
  the `tokenizers` library)
- **Use:** text generation, chat mode (`llama3`, `zephyr`, `llama2` templates), top-k / top-p / repetition
  penalty, perplexity measurement
- **Verification:** kernel tests against scalar and double-precision references, end-to-end logits against a
  numpy reference, qemu for ARM and old x86 CPUs, native Arm CI

```sh
make ISA=avx2                                    # or sse2, neon, neon_dotprod, sve, ...
python3 tools/convert.py hf ./Llama-3.2-1B-Instruct llama.mlb --wtype q4 --q8 v,o,down
python3 tools/convert_tokenizer.py ./Llama-3.2-1B-Instruct/tokenizer.json llama.mlt
build/avx2/ml_run llama.mlb -z llama.mlt --chat llama3 -t 4
```

## Documentation

[User guide](user-guide.md) · [Architecture](architecture.md) · [File formats](file-format.md) ·
[Writing kernels](kernels.md) · [Testing and benchmarks](testing.md) ·
[Design spec](design/2026-10-22-legacy-isa-ml-runtime-design.md)

## Citing

```bibtex
@software{isalm2026,
  author  = {Park, Yeochoon},
  title   = {isa-lm: a portable C11 runtime for INT8/INT4 Llama-style language model inference on legacy x86-64 and ARM instruction sets},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/JonPark0/isa-lm}
}
```

Source code: <https://github.com/JonPark0/isa-lm> · License: MIT
