# isa-lm — Portable INT8/INT4 LLM Inference for Legacy x86-64 and ARM CPUs (C11)

[![CI](https://github.com/JonPark0/isa-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/JonPark0/isa-lm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C11](https://img.shields.io/badge/C-11-blue.svg)](docs/user-guide.md)
[![Docs](https://img.shields.io/badge/docs-site-green.svg)](https://jonpark0.github.io/isa-lm/)

A dependency-free C11 inference runtime for Llama-style decoder Transformers with
INT8 and INT4 weights. It runs on x86-64 CPUs from SSE2 up to AVX-512 VNNI, and
on ARM CPUs from ARMv7 NEON up to i8mm and SVE.

The code is written against a portable baseline. You pick the acceleration when
you compile, and each op is bound to the best kernel for that ISA, with no
runtime dispatch:

```sh
make ISA=avx2                                   # build/avx2/{ml_run,ml_bench,test_*}
make ISA=neon CROSS=aarch64-linux-gnu-          # cross build; tests run under qemu
make ISA=avx512vnni THREADS=pthread             # with the worker pool
```

| `ISA` | Minimum CPU | Accelerated |
|-------|-------------|-------------|
| `scalar` | any | reference implementation of all ops |
| `sse2` | any x86-64 | all ops, packed int8 matvec (`pmaddwd`) |
| `sse41` | Penryn / Nehalem | int8 matvec (`pmaddubsw`) |
| `avx2` | Haswell / Zen | all ops, FMA; f16 packed scales (F16C) |
| `avxvnni` | Alder Lake / Zen 5 | int8 matvec (`vpdpbusd` ymm) |
| `avx512vnni` | Cascade Lake / Ice Lake / Zen 4 | matvec, quantize, norms, dot, axpy |
| `neon` | ARMv7 + NEON, any AArch64 | all ops |
| `neon_dotprod` | Cortex-A55/A76, Apple M1 | int8 matvec (`sdot`) |
| `neon_i8mm` | Cortex-A710, Neoverse N2/V1, Apple M2 | prompt prefill (`smmla`) |
| `sve` | Neoverse V1/N2, Graviton 3 | matvec, quantize, norms, dot, axpy (any vector length) |

If you run a binary on a CPU that lacks its ISA, it exits with code 3 and names
the missing feature. It does not crash with SIGILL.

## Quick start

Prebuilt binaries for Linux (x86-64, AArch64, ARMv7) and Windows (x86-64) are on the
[releases page](https://github.com/JonPark0/isa-lm/releases); their `ml_run` wrapper starts the
fastest build your CPU supports. To build from source:

```sh
pip install numpy
make ISA=avx2

# llama2.c checkpoint + tokenizer
python3 tools/convert.py llama2c stories15M.bin stories15M.mlb --wtype q8
build/avx2/ml_run stories15M.mlb -z tokenizer.bin -i "Once upon a time" -n 128 -T 0.8

# Hugging Face Llama, Mistral, Qwen2 or Qwen3 (safetensors), and its tokenizer.json
python3 tools/convert.py hf ./Llama-3.2-1B-Instruct llama.mlb --wtype q4 --q8 v,o,down --max-seq 4096
python3 tools/convert_tokenizer.py ./Llama-3.2-1B-Instruct/tokenizer.json llama.mlt
build/avx2/ml_run llama.mlb -z llama.mlt --chat llama3 -t 4     # one message per line
build/avx2/ml_run qwen.mlb -z qwen.mlt --chat qwen -t 4         # Qwen2.5 Instruct
```

## Documentation

Site: <https://jonpark0.github.io/isa-lm/> · Wiki: <https://github.com/JonPark0/isa-lm/wiki> (both generated from [`docs/`](docs))

| Document | Contents |
|----------|----------|
| [User guide](docs/user-guide.md) | building, cross builds, converting models, CLI options, threads, limits |
| [Architecture](docs/architecture.md) | layers, compile-time kernel binding, ISA guard, data flow, memory, threading, numerics |
| [File format](docs/file-format.md) | the `.mlb` model format (header, tensor layout, quantization rule, RoPE scaling) and the `.mlt` tokenizer format |
| [Writing kernels](docs/kernels.md) | adding a kernel or an ISA, kernel rules, per-ISA idioms, assembly overrides |
| [Testing](docs/testing.md) | test layers, qemu, sanitizers, CI, benchmarks |
| [Comparison with llama.cpp](docs/benchmarks.md) | speed per ISA level (x86-64 and AArch64) and perplexity against llama.cpp on the same model and formats |
| [Design spec](docs/design/2026-10-22-legacy-isa-ml-runtime-design.md) | original design and implementation plan |

## Status

| Area | Status |
|------|--------|
| x86 builds | all six ISAs tested natively on an AVX-512 VNNI machine: kernel tests against scalar and double-precision references, end-to-end logits against a numpy reference. CI tests scalar through avx2 natively, and builds avxvnni/avx512vnni (SKIP when the runner CPU lacks them) |
| ARM builds | tested under qemu (ARMv7 NEON, AArch64 NEON, dotprod, i8mm, SVE at 128-bit and 512-bit vector length) and on real Arm hardware in CI (GitHub `ubuntu-24.04-arm`: scalar, neon, neon with threads, dotprod, i8mm, sve) |
| Real model | CI downloads llama2.c `stories15M`, converts it to Q8, generates text, and checks scalar and avx2 logits (packed and row-major) against the numpy reference: same argmax, top-5 agreement, KL ≤ 1e-2 |
| ISA guard | rejects unsupported CPUs with exit code 3, checked under qemu on core2duo/Nehalem/SandyBridge/Haswell and cortex-a53/a76/neoverse-n2 |
| Performance | weights repacked into row-interleaved layouts for every SIMD build from SSE2 and ARMv7 NEON up (Q4 decoding up to 2.2× faster); prompts prefilled in batches of up to 128 tokens (2–6× faster), on `smmla` with i8mm; software prefetch of the weight stream on x86 (decoding 1.3–2.5× faster on one core); attention 2.4× faster on long contexts, 3.4× with the optional Q8 KV cache (`--kv q8`); numbers in [Testing](docs/testing.md#benchmarking) |
| Thread pool | bit-identical results for any thread count; ThreadSanitizer clean; a 4-layer dim-2048 Q4 model decodes at 140 tok/s with `-t 1` and 503 tok/s with `-t 4` (avx512vnni, 4 vCPUs) |
| Windows | MSVC builds (scalar, SSE2, AVX2) pass the unit tests and match the numpy reference in CI; the MinGW-built release binaries are checked under Wine before publishing |
| Quantization | Q8_0, Q4_0, and Q4 with chosen matrix classes in Q8 (`--q8 v,o,down`: Llama 3.2 1B perplexity 17.89 at 831 MB, against 17.27 for Q8 at 1252 MB and 19.91 for Q4 at 663 MB); f16 block scales by default (Q4 at 4.5 bits per weight); each Q4 block's scale is fitted rather than rounded from its maximum (Qwen2.5 0.5B: 15.14 against 15.86 for llama.cpp's Q4_0, see the [comparison](docs/benchmarks.md)), except for Gemma 3, whose checkpoints do better with llama.cpp's rounding (its quantization-aware 1B checkpoint: Q4 perplexity 40.50, below its Q8); `tools/perplexity.py` measures it |
| Models | Llama 2/3/3.1/3.2, TinyLlama, SmolLM2, Mistral (with its sliding window), Qwen2/2.5 (q/k/v biases), Qwen3 (its own head size, per-head q/k norm) and Gemma 3 text (GELU, block output norms, local and global layers); RoPE scaling of the llama3, linear and yarn kinds. TinyLlama 1.1B, Llama 3.2 1B, Qwen2.5 0.5B, Qwen3 0.6B, SmolLM2 135M and Gemma 3 1B were run end to end from their Hugging Face files; Q8 perplexity matches an F32 reference (Qwen2.5 18.98 against 19.00, Qwen3 27.90 against 28.10, Gemma 3 41.97 against 42.32) |
| Tokenizers | llama2.c `tokenizer.bin`; Hugging Face `tokenizer.json` for SentencePiece-style BPE (Llama 2, TinyLlama, Mistral), Llama 3 byte-level BPE, Qwen2/Qwen3 byte-level BPE with NFC, and GPT-2 byte-level BPE (SmolLM2), converted by `tools/convert_tokenizer.py` and identical to the `tokenizers` library on every text tested |
| Not implemented | other tokenizers (WordPiece, Unigram), RoPE scaling of the dynamic and longrope kinds, partial rotary embeddings, image input (of Gemma 3's image-text checkpoints only the text model is converted), architectures other than Llama, Mistral, Qwen2, Qwen3, Gemma 2 and Gemma 3 (Phi, mixture of experts) |

## Citing

If isa-lm helps your research, please cite it (GitHub's *Cite this repository* uses [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{isalm2026,
  author  = {Park, Yeochoon},
  title   = {isa-lm: a portable C11 runtime for INT8/INT4 Llama-style language model inference on legacy x86-64 and ARM instruction sets},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/JonPark0/isa-lm}
}
```

Please also cite the models you run; their licenses are separate from this one.

## 한국어 요약

**isa-lm**은 INT8/INT4로 양자화한 Llama 계열 언어 모델을 CPU에서 실행하는 C11 추론 런타임입니다.
외부 라이브러리 없이 구형·저전력 프로세서까지 지원하는 것이 목표입니다.

- **지원 명령어 집합:** x86-64는 SSE2부터 SSE4.1, AVX2, AVX-VNNI, AVX-512 VNNI까지, ARM은 ARMv7 NEON부터
  NEON dotprod, i8mm, SVE까지 지원합니다. 빌드할 때 명령어 집합을 고르면 연산마다 그에 맞는 커널이 컴파일 단계에서
  묶입니다. 지원하지 않는 CPU에서 실행하면 비정상 종료 대신 오류 코드 3과 누락된 기능을 알려 줍니다.
- **양자화:** Q8_0, Q4_0, 그리고 일부 행렬만 Q8로 두는 혼합 정밀도를 지원합니다. KV 캐시도 Q8로 줄일 수 있습니다.
- **속도:** 명령어 집합별 정수 내적에 맞춘 가중치 재배치, 프롬프트 일괄 처리, 소프트웨어 프리페치, GQA 블록
  어텐션을 씁니다. 스레드 수를 바꿔도 결과는 비트 단위로 같습니다.
- **모델과 토크나이저:** llama2.c 체크포인트와 Hugging Face safetensors(Llama 2, Llama 3/3.1/3.2, TinyLlama,
  SmolLM2, Mistral, Qwen2/Qwen2.5, Qwen3)를 변환합니다. Hugging Face `tokenizer.json`은 `tokenizers` 라이브러리와 같은 결과로 인코딩합니다.
- **사용:** 텍스트 생성, 대화 모드(`--chat`), top-k·top-p·반복 패널티, 퍼플렉서티 측정을 제공합니다.
- **검증:** 커널은 스칼라·배정밀도 기준과, 전체 결과는 numpy 참조 구현과 비교합니다. ARM과 구형 x86은 qemu로,
  Arm 실기는 CI에서 확인합니다.

자세한 문서는 [문서 사이트](https://jonpark0.github.io/isa-lm/)와 [위키](https://github.com/JonPark0/isa-lm/wiki)를 참고하세요.

## License

[MIT](LICENSE). Third-party notices (llama2.c, Cephes, Unicode data) are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
