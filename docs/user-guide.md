# User guide

This guide covers building isa-lm, getting a model into the `.mlb` format, and
running it.

## Requirements

- A C11 compiler: GCC or Clang. For x86-64 you can also use MSVC through CMake.
- GNU Make.
- Python 3 with numpy, for the model tools. Install pytest as well if you want to
  run the test suite.
- For ARM cross builds: `gcc-aarch64-linux-gnu` or `gcc-arm-linux-gnueabihf`,
  plus `qemu-user` to run the results on an x86 host.

## Building

```sh
make ISA=<isa> [THREADS=1|pthread] [ASM=0|1] [CROSS=<prefix>] [DEBUG=1] [EXTRA_CFLAGS=...]
```

The build writes `ml_run`, `ml_bench` and the test binaries to
`build/<isa>[-<arch>][-pt][-asm][-dbg]/`. Each set of options gets its own
directory, so object files built with different flags never mix.

### Choosing the ISA

| `ISA`          | Runs on                                    |
|----------------|--------------------------------------------|
| `scalar`       | anything with a C compiler                 |
| `sse2`         | every x86-64 CPU                           |
| `sse41`        | Intel Penryn/Nehalem and later, AMD Bulldozer and later |
| `avx2`         | Intel Haswell and later, AMD Zen and later |
| `avxvnni`      | Intel Alder Lake and later, AMD Zen 5      |
| `avx512vnni`   | Intel Cascade Lake, Ice Lake, Sapphire Rapids; AMD Zen 4 and later |
| `neon`         | ARMv7 with NEON (Cortex-A7/A9/A15…), any AArch64 |
| `neon_dotprod` | ARMv8.2 dotprod (Cortex-A55/A76 and later, Apple M1 and later, Graviton 2 and later) |
| `neon_i8mm`    | dotprod + int8 matrix multiply (Cortex-A510/A710/X2 and later, Neoverse N2/V1, Apple M2 and later, Graviton 3 and later) |
| `sve`          | SVE CPUs (Neoverse V1/V2/N2, A64FX, Graviton 3 and later) |

To find what a Linux machine supports, run
`grep -o -w -E 'sse4_1|avx2|avx_vnni|avx512_vnni|asimddp|i8mm|sve' /proc/cpuinfo | sort -u`.

If you start a binary on a CPU that lacks its ISA, it exits with code 3 and
names the missing feature. It does not crash:

```
error: CPU lacks avx512-vnni required by ISA=avx512vnni
```

### Cross builds

```sh
make ISA=neon CROSS=arm-linux-gnueabihf-                     # ARMv7 hard-float
make ISA=sve CROSS=aarch64-linux-gnu- test e2e               # runs under qemu-aarch64 -cpu max
make ISA=sve CROSS=aarch64-linux-gnu- QEMU_CPU=max,sve-max-vq=1 test   # SVE at 128 bits
```

Cross builds are linked statically, so qemu does not need a sysroot. When
`CROSS` targets another architecture, `make test` and `make e2e` run the
binaries through qemu automatically.

### CMake (x86-64, including MSVC)

```sh
cmake -S . -B build/cmake -DML_ISA=avx2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake && ctest --test-dir build/cmake
```

Set `-DCMAKE_BUILD_TYPE=Release` yourself; without it CMake builds without
optimization. The pthread pool (`-DML_THREADS=ON`) is not available with MSVC.

## Getting a model

### From a llama2.c checkpoint

```sh
python3 tools/convert.py llama2c stories15M.bin stories15M.mlb --wtype q8
```

Use the matching llama2.c `tokenizer.bin` as it is.

### From a Hugging Face Llama checkpoint

The converter needs a directory containing `config.json` and one or more
`*.safetensors` files:

```sh
python3 tools/convert.py hf ./TinyLlama-1.1B model.mlb --wtype q4 --max-seq 2048
```

- **Weight types:** F32, F16 and BF16 safetensors are supported.
- **Tied embeddings:** if the checkpoint has no `lm_head`, the file is written
  with a shared classifier.
- **Rotary layout:** the q/k projections are permuted from the Hugging Face
  rotate-half layout to the interleaved pairs the runtime uses.
- **RoPE scaling:** `rope_scaling` of type `llama3` is supported (Llama 3.1 and
  3.2). The converter refuses other types (linear, dynamic, yarn) instead of
  writing a model that would compute wrong outputs.
- **Context length:** `--max-seq` caps the context, and with it the KV cache.
  Without it the converter uses `max_position_embeddings`, which is often
  131072 for Llama 3.
- **Tokenizer:** convert the checkpoint's `tokenizer.json` separately, see
  [Tokenizers](#tokenizers).

### Tokenizers

`ml_run -z` reads either a llama2.c `tokenizer.bin` or an `.mlt` file
converted from a Hugging Face `tokenizer.json`:

```sh
python3 tools/convert_tokenizer.py ./Llama-3.2-1B-Instruct/tokenizer.json llama3.mlt
printf "Hello, world" | build/avx2/ml_tok llama3.mlt        # 128000 9906 11 1917
echo "9906 11 1917" | build/avx2/ml_tok llama3.mlt --decode          # Hello, world
```

Two kinds are supported, and they encode exactly as the Hugging Face
`tokenizers` library does (the tests compare them on hundreds of texts):

| Kind | Models | Notes |
|------|--------|-------|
| SentencePiece-style BPE with byte fallback | Llama 2, TinyLlama, Mistral, Vicuna | spaces become `▁`, characters outside the vocabulary become `<0xXX>` byte tokens |
| byte-level BPE with the Llama 3 pre-tokenizer | Llama 3, 3.1, 3.2 | text is split by the Llama 3 regex (letters, numbers, contractions, white space) before merging |

The converter refuses anything else (WordPiece, Unigram, GPT-2 or Qwen
pre-tokenizers, extra normalizers) rather than write a tokenizer that would
encode differently.

- **BOS:** added when the tokenizer's template adds it (`<s>`,
  `<|begin_of_text|>`).
- **Special tokens** written in the prompt, such as `<|eot_id|>`, are encoded
  as the special tokens.
- **Stopping:** generation stops at the tokenizer's EOS (`</s>`,
  `<|end_of_text|>`). Instruct models often end a turn with another token:
  pass it with `--eos`, for example `--eos 128009` (`<|eot_id|>`) for
  Llama 3 Instruct.

A Llama 3.2 Instruct prompt, for example:

```sh
build/avx2/ml_run llama3.mlb -z llama3.mlt --eos 128009 -T 0 -n 200 -i "<|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```

### A random test model

```sh
python3 tools/gen_random.py --out rand.mlb --wtype q4 --dim 256 --layers 4 --vocab 512
python3 tools/gen_tokenizer.py --vocab 512 --out tok.bin
```

### Q8 or Q4?

Choose **Q8_0** for accuracy and **Q4_0** for size.

| Format | Bits per weight | Notes |
|--------|-----------------|-------|
| `Q8_0` | 9 | close to FP32 quality; best throughput on VNNI and dotprod CPUs |
| `Q4_0` | 5 | about half the memory traffic; costs some quality and needs extra unpack work |

## Running

```sh
build/avx2/ml_run model.mlb -z tokenizer.bin -i "Once upon a time" -n 128 -T 0.8 -p 0.9
build/avx2/ml_run model.mlb --tokens 1,450,3681 -n 32            # ids in, ids out
build/avx2/ml_run model.mlb --tokens 1,450,3681 --dump-logits l.f32
```

| Option | Meaning |
|--------|---------|
| `-z FILE` | Tokenizer: llama2.c `tokenizer.bin`, or `.mlt` from `tools/convert_tokenizer.py`. Needed for `-i` and for text output. |
| `-i TEXT` | Prompt text. Encoded with BOS. |
| `--tokens LIST` | Prompt as comma-separated token ids. Each id must be below `vocab`. |
| `-n N` | Positions to run, prompt included. Default 64; capped at the model's `max_seq`. |
| `-T F` | Temperature. `0` means greedy decoding (ties go to the lowest id). Default 0. |
| `-p F` | Top-p in [0, 1]. Values of 0 or 1 sample from the full distribution. Default 0.9. |
| `--top-k N` | Sample only from the `N` most likely tokens, then apply top-p among them. `0` (default) turns it off; `1` is greedy at any temperature. |
| `--repeat-penalty F` | Repetition penalty: divide the positive logit of each token seen in the last `--repeat-last-n` tokens by `F` and multiply a negative one, as Hugging Face does. Applies to greedy decoding too. `1` (default) turns it off; 1.1–1.3 is typical. |
| `--repeat-last-n N` | How many recent tokens, prompt included, the penalty looks at. Default 64. |
| `-s N` | Seed for the xorshift64* RNG. Default 1. |
| `-t N` | Threads, 1 to 64. Only builds made with `THREADS=pthread` use more than one. |
| `--dump-logits F` | Write float32 logits `[n_tokens][vocab]` for every `--tokens` position, then exit. |
| `--batch N` | Prompt tokens per batched prefill step, 1 to 32. Default 32; `1` processes the prompt token by token. The output is identical either way. |
| `--kv TYPE` | KV cache type: `f32` (default) or `q8`. `q8` stores K and V as int8 with one scale per position and KV head: about 3.8× less memory and faster attention on long contexts, with slightly different logits. |
| `--no-pack` | Keep weights row-major instead of repacking them for the packed kernels. Uses less memory during load, but runs slower. |
| `--profile` | Print the share of forward-pass time per stage (qkv, attention, wo, ffn, classifier, other). |
| `--eos ID` | Stop as soon as `ID` is generated. Without it, a run with `-z` stops at the tokenizer's BOS or EOS, and a run without `-z` uses every `-n` position. |

When it finishes, `ml_run` prints two lines to stderr:

- `prefill: P tokens in M ms (R tok/s)`, when the prompt has more than one
  token;
- `positions: N, tok/s: X, threads: T, isa: I, packed: G, kv: K`, where `G` is
  the pack group the weights were repacked to, or 0, and `K` the KV cache type.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | load or runtime error (bad file, out of memory) |
| 2 | usage error |
| 3 | the CPU lacks the ISA the binary was built for |

### Threads

Build with `THREADS=pthread` and pass `-t N`. The pool waits by spinning
briefly, then sleeps, so consecutive operations within a token cost no system
calls. Matrix-vector products smaller than 2^18 weights, and small attention
steps, run on the calling thread only.

On a 4-vCPU VM with the avx512vnni build, `-t 4` decodes a 4-layer dim-2048 Q4 model at
503 tok/s, against 140 tok/s for `-t 1`. Threading never changes the output:
`-t N` produces logits bit-identical to `-t 1`.

### Performance notes

- **Q4 or Q8:** prefer Q4 on large models. Decoding is limited by memory
  bandwidth, and Q4 moves 45% fewer bytes. With packed kernels, Q4 is the
  faster format on every ISA that has them.
- **Prompts:** long prompts are processed in batches of 32 tokens. That is 2–6×
  faster than token by token, depending on the ISA and the model size.
- **Long contexts:** attention grows with the position and, past a few
  thousand tokens, takes a third or more of the time. `--kv q8` cuts the KV
  cache's memory traffic and size by about 3.8×.
- **Memory:** packing makes one copy of the weights at load and then releases
  the originals' pages. If a device cannot hold two copies of the model even
  briefly, use `--no-pack`.

## Limits

- **Dimensions:** `dim`, `hidden_dim`, `n_layers` and head counts can be at most
  65536. `vocab` and `max_seq` can be at most 2^24.
- **Shapes:** `dim` and `hidden_dim` must be multiples of 32, `head_dim` must be
  even, and `n_heads` must be a multiple of `n_kv_heads`.
- **Memory:** the KV cache uses `n_layers × max_seq × kv_dim × 8` bytes, or
  `n_layers × max_seq × (kv_dim + 4 × n_kv_heads) × 2` with `--kv q8`. All size
  arithmetic is overflow-checked, so on 32-bit ARMv7 an oversized model fails
  with `out of memory` instead of crashing.
- **Architecture:** only Llama-family decoders are supported: RMSNorm, SwiGLU,
  GQA, no biases.
