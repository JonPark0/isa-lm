---
title: "FAQ: running LLMs on old and low-power CPUs"
description: "Questions and answers about Krafla, a C11 runtime for INT8/INT4 language models on CPUs: running without AVX, Raspberry Pi-class ARM boards, speed against llama.cpp, models, memory and limits."
nav_order: 2
nav_title: "FAQ"
---

# FAQ

## What is Krafla?

Krafla is a dependency-free C11 inference runtime for Llama-style language
models with INT8 (Q8_0) and INT4 (Q4_0) weights. It runs on the CPU only, on
x86-64 from SSE2 up to AVX-512 VNNI and on ARM from 32-bit ARMv7 NEON up to
i8mm and SVE. You pick the instruction set (ISA) when you compile, and every
operator is bound to the kernel written for that ISA, with no runtime
dispatch.

The project was called isa-lm until version 0.5.0. Krafla is a volcanic
caldera in north Iceland with a geothermal power station, and the name also
reads as **K**ernels **R**etargeted **A**cross **F**amilies of **L**egacy
**A**rchitectures.

## Can I run an LLM on a CPU without AVX or AVX2?

Yes. The `sse2` build needs nothing beyond SSE2, which every x86-64 CPU has
(Athlon 64, Core 2 and later), and the `sse41` build adds SSE4.1 (Penryn,
Nehalem and later). Both have hand-written int8 kernels rather than a scalar
fallback. On the [comparison machine](benchmarks.md) the SSE2 build decodes
Qwen2.5 0.5B Q4_0 at 25.7 tokens/s on one core and 77.3 on four. That machine
is a modern Xeon running the SSE2 code, so an old CPU will be slower; the
ratio to llama.cpp's SSE2 build on the same machine was 6–7×.

## How does Krafla compare with llama.cpp?

On Qwen2.5 0.5B with the same Q8_0 and Q4_0 formats on the same machines
([details](benchmarks.md)):

- **Decode** is at least as fast with Krafla at every ISA level measured:
  1.2–1.5× with AVX-512 VNNI, 1.0–2.3× with AVX2, 2.1–10× on the SSE2 and
  SSE4.1 builds, and 1.04–1.7× on AArch64.
- **Prefill** (prompt processing) is faster with Krafla from AVX2 down and on
  AArch64, except dotprod Q4_0, where llama.cpp leads by 1.10–1.13×. On a CPU
  with AMX, llama.cpp's AMX kernels prefill 1.03–1.17× faster than Krafla's
  AVX-512 VNNI build.
- **Perplexity** is lower with Krafla's Q4_0: 15.09–15.14 against 15.82–15.93
  for llama.cpp's Q4_0, because each block's scale is fitted rather than
  rounded from its maximum. Q8_0 scores 13.32–13.35 against 13.34 for
  llama.cpp's F16.

llama.cpp does much more: GPU back ends, K-quant and i-quant formats, many
more architectures (mixture of experts, image input) and an HTTP server.
Krafla covers Q8_0 and Q4_0 on the CPU, for the ISAs listed above.

## Does it run on a Raspberry Pi?

The ARM builds cover the Raspberry Pi's CPUs by ISA. Raspberry Pi 2 (ARMv7
NEON) takes the `neon` build of the `armv7hf` release; Raspberry Pi 3,
Zero 2 W and 4 (Cortex-A53, Cortex-A72) take `neon` in either the `armv7hf`
or the `aarch64` release, depending on the operating system; Raspberry Pi 5
(Cortex-A76) takes `neon_dotprod`. The original Raspberry Pi 1 and Zero
(ARMv6) have no NEON. These builds are tested under qemu and natively on
GitHub's Arm runners (Neoverse), and the CPU guard is checked on emulated
Cortex-A53 and Cortex-A76 cores; none of this has run on an actual Raspberry
Pi yet, so there are no Raspberry Pi speed numbers.

## Which models does it run?

Llama 2, Llama 3/3.1/3.2, TinyLlama, SmolLM2, Mistral, Qwen2/Qwen2.5, Qwen3,
Gemma 2 and Gemma 3 (the text model, 1B to 27B), from their Hugging Face
safetensors files, plus llama2.c checkpoints. `tools/convert.py` converts a
checkpoint to the `.mlb` format and `tools/convert_tokenizer.py` its
`tokenizer.json`. TinyLlama 1.1B, Llama 3.2 1B, Qwen2.5 0.5B, Qwen3 0.6B,
SmolLM2 135M and Gemma 3 1B and 4B have been run end to end; see
[Getting a model](user-guide.md#getting-a-model).

## Where do I get a model?

Converted files of SmolLM2 135M and 360M Instruct, Qwen2.5 0.5B and 1.5B
Instruct, Qwen3 0.6B and TinyLlama 1.1B Chat are on Hugging Face, in Q4_0 and
Q8_0 and, for all but the two SmolLM2 models, a mixed Q4 file, each with its
tokenizer and a model card with perplexity and speed: see
[Converted models on Hugging Face](user-guide.md#converted-models-on-hugging-face).
Download a `.mlb` file and `tokenizer.mlt` next to a release and run it; no
Python is needed. For other models, convert the Hugging Face safetensors with
`tools/convert.py`.

## Which build should I use?

In a release archive, the `ml_run` wrapper tries the builds from fastest to
slowest and starts the first one your CPU supports. When you build from
source, `make ISA=...` picks one (see [Choosing the ISA](user-guide.md#choosing-the-isa)).
A binary run on a CPU that lacks its ISA exits with code 3 and names the
missing feature instead of crashing.

## How much memory does a model need?

About the size of its `.mlb` file plus the KV cache. With the default f16
block scales, Qwen2.5 0.5B takes 265 MiB in Q4 and 500 MiB in Q8, Llama 3.2 1B
663 MiB in Q4 and 1252 MiB in Q8, Gemma 3 4B 2083 MiB in Q4. The KV cache is
`n_layers × max_seq × kv_dim × 8` bytes; `--kv f16` halves it and `--kv q8`
makes it about 3.8× smaller. `--max-seq` at conversion sets `max_seq`.

## Does it need a GPU, Python or other libraries?

No GPU. The runtime is C11 with no dependencies beyond the C library (and
pthreads for the threaded builds); the release binaries are static. Python 3
with numpy is needed only to convert models and tokenizers.

## Are the results deterministic?

Yes: the logits are bit-identical for any thread count and any prompt batch
size, and greedy decoding (`-T 0`, the default) gives the same text every
run. The tests check this, and check every kernel against scalar and
double-precision references and the whole forward pass against a numpy
reference.

## Which operating systems are supported?

Linux on x86-64, AArch64 and ARMv7, and Windows on x86-64 (MinGW release
binaries, MSVC through CMake). Other systems are untested.

## What is not supported?

Other tokenizers (WordPiece, Unigram), RoPE scaling of the dynamic and
longrope kinds, partial rotary embeddings (Phi), mixture-of-experts models,
image input, K-quant formats, GPUs and AMX. The full list is under
[Status](https://github.com/JonPark0/krafla#status) in the README.
