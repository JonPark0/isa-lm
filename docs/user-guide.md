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

## Prebuilt binaries

Each [release](https://github.com/JonPark0/isa-lm/releases) has static
binaries for every ISA:

| Archive | Builds |
|---------|--------|
| `isa-lm-VER-linux-x86_64.tar.gz` | `sse2`, `sse41`, `avx2`, `avxvnni`, `avx512vnni` |
| `isa-lm-VER-linux-aarch64.tar.gz` | `neon`, `neon_dotprod`, `neon_i8mm`, `sve` |
| `isa-lm-VER-linux-armv7hf.tar.gz` | `neon`, `scalar` |
| `isa-lm-VER-windows-x86_64.zip` | `sse2`, `sse41`, `avx2`, `avxvnni`, `avx512vnni` |

The top-level `ml_run`, `ml_bench` and `ml_tok` (`.cmd` on Windows) try the
builds in `bin/` from the fastest down and start the first one the CPU
supports. `ISA_LM_ISA=avx2` (for example) picks one yourself. The archives
also hold the model tools (`tools/convert.py`, `tools/convert_tokenizer.py`,
`tools/perplexity.py`; they need Python 3 with numpy).

`scripts/package_release.sh` builds the same archives locally, and
`scripts/smoke_release.sh` runs them. A release is cut by merging a version
bump into main: `ML_VERSION` in `core/ml.h` raised to X.Y.Z and the
`## Unreleased` heading of `CHANGELOG.md` renamed to `## X.Y.Z — date`.
`.github/workflows/auto-release.yml` sees the new version, and
`release.yml` builds and runs the archives and publishes release vX.Y.Z,
creating the tag on that commit. Pushing a tag vX.Y.Z still works too; CI
fails a pull request that raises `ML_VERSION` without its CHANGELOG section.

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

### From a Hugging Face checkpoint

The converter reads Llama (also SmolLM2), Mistral, Qwen2 (Qwen2.5) and Qwen3
checkpoints: a directory containing `config.json` and one or more
`*.safetensors` files:

```sh
python3 tools/convert.py hf ./TinyLlama-1.1B model.mlb --wtype q4 --max-seq 2048
```

- **Weight types:** F32, F16 and BF16 safetensors are supported.
- **Tied embeddings:** if the checkpoint has no `lm_head`, the file is written
  with a shared classifier.
- **Rotary layout:** the q/k projections are permuted from the Hugging Face
  rotate-half layout to the interleaved pairs the runtime uses.
- **RoPE scaling:** `rope_scaling` of type `llama3` (Llama 3.1 and 3.2),
  `linear` and `yarn` (for example Qwen2.5 and Qwen3 beyond 32K tokens, as
  their model cards describe) are supported, with transformers' formulas. The
  converter refuses other types (dynamic, longrope) instead of writing a model
  that would compute wrong outputs.
- **Architectures:** `model_type` must be `llama`, `mistral`, `qwen2`, `qwen3`,
  `gemma2`, `gemma3_text` (Gemma 3 1B) or `gemma3` (Gemma 3 4B, 12B and 27B,
  image-text checkpoints: the converter takes the text model and leaves the
  vision tower out), with no biases other than Qwen2's q/k/v biases. A
  `head_dim` other than `hidden_size / num_attention_heads`, Qwen3's per-head
  q/k RMSNorm, Gemma's GELU, block output norms, embedding and attention
  scales and local/global layers, and Gemma 2's soft-capped attention scores
  and logits are supported; Gemma's `(1 + w)` norm weights are stored with the
  1 added. Others (Phi, mixture of experts, …) are refused. Checkpoint tensors
  are converted to float32 one at a time, so converting a 4B model needs about
  as much memory as its largest tensor.
- **Sliding window:** each position of a sliding layer attends to the last
  `sliding_window` positions: every layer for Mistral v0.1, the layers
  `layer_types` or Qwen2's `max_window_layers` select otherwise.
- **Context length:** `--max-seq` caps the context, and with it the KV cache.
  Without it the converter uses `max_position_embeddings`, which is often
  131072 for Llama 3.
- **Scales:** each block's scale is stored as `f16` by default (Q4 at 4.5 bits
  per weight, Q8 at 8.5). `--scales f32` keeps the older 4-byte scales (5 and
  9 bits per weight). Files of both kinds load in every build.
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

These kinds are supported, and they encode exactly as the Hugging Face
`tokenizers` library does (the tests compare them on hundreds of texts):

| Kind | Models | Notes |
|------|--------|-------|
| SentencePiece-style BPE with byte fallback | Llama 2, TinyLlama, Mistral, Vicuna, Gemma 3 | spaces become `▁` (Gemma adds none before the text), characters outside the vocabulary become `<0xXX>` byte tokens |
| byte-level BPE with the Llama 3 pre-tokenizer | Llama 3, 3.1, 3.2 | text is split by the Llama 3 regex (letters, numbers, contractions, white space) before merging |
| byte-level BPE with NFC and the Qwen2 pre-tokenizer | Qwen2, Qwen2.5, Qwen3 | text is normalized to NFC, then split as Llama 3 but with one digit per number chunk |
| byte-level BPE with the GPT-2 pre-tokenizer | SmolLM, SmolLM2, GPT-2 | text is split by the GPT-2 regex, after splitting off every digit if the tokenizer does (SmolLM2) |

The converter refuses anything else (WordPiece, Unigram, other normalizers or
pre-tokenizers) rather than write a tokenizer that would encode differently.

- **BOS:** added when the tokenizer's template adds it (`<s>`,
  `<|begin_of_text|>`); Qwen2, Qwen3 and SmolLM2 add none.
- **Special tokens** written in the prompt, such as `<|eot_id|>`, are encoded
  as the special tokens.
- **Stopping:** generation stops at the tokenizer's EOS (`</s>`,
  `<|end_of_text|>`). Instruct models often end a turn with another token:
  pass it with `--eos`, for example `--eos 128009` (`<|eot_id|>`) for
  Llama 3 Instruct.

### Chat

`--chat TEMPLATE` wraps each message in the model's chat template, keeps the
conversation in the KV cache, and stops each reply at the template's end
token:

```sh
build/avx2/ml_run llama3.mlb -z llama3.mlt --chat llama3 --system "You are a helpful assistant." -t 4
> What is the capital of France?
The capital of France is Paris.
> And of Germany?
The capital of Germany is Berlin.
```

| Template | Models | Reply ends at |
|----------|--------|---------------|
| `llama3` | Llama 3, 3.1, 3.2 Instruct | `<|eot_id|>` |
| `zephyr` | TinyLlama Chat, Zephyr | `</s>` |
| `llama2` | Llama 2 Chat, Mistral Instruct (without `--system`) | `</s>` |
| `qwen` | Qwen2.5 Instruct; without `--system` it uses Qwen's default system prompt | `<|im_end|>` |
| `smollm` | SmolLM2 Instruct; without `--system` it uses SmolLM2's default system prompt | `<|im_end|>` |
| `gemma` | Gemma 3 Instruct; `--system` goes at the start of the first user turn | `<end_of_turn>` |
| `chatml` | Qwen3 and other ChatML models; a system turn only with `--system` | `<|im_end|>` |

Qwen3 answers after a `<think>…</think>` section; ending a message with
`/no_think` asks it to skip the reasoning. The KV cache keeps earlier replies
with their reasoning, where Qwen3's own template drops it from the history.

- **Input:** one turn per line on stdin (a `> ` prompt appears on a terminal),
  or a single turn with `-i`. End with Ctrl-D.
- **Length:** `-n` caps each reply; without it a reply may use the rest of the
  context. The conversation ends with a message when the context is full.
- **Sampling:** `-T`, `-p`, `--top-k` and `--repeat-penalty` apply as usual.

Without `--chat`, write the template yourself with `-i` and stop with `--eos`
(`128009` is `<|eot_id|>` in Llama 3).

### A random test model

```sh
python3 tools/gen_random.py --out rand.mlb --wtype q4 --dim 256 --layers 4 --vocab 512
python3 tools/gen_tokenizer.py --vocab 512 --out tok.bin
```

### Q8, Q4, or mixed?

Choose **Q8_0** for accuracy and **Q4_0** for size. In between, `--q8`
keeps some matrix classes of a Q4 model in Q8 (`emb`, `cls`, `qk`, `v`, `o`,
`up`, `down`):

```sh
python3 tools/convert.py hf ./Llama-3.2-1B-Instruct llama.mlb --wtype q4 --q8 v,o,down
```

Llama 3.2 1B on wikitext-2 (2044 tokens, `tools/perplexity.py`; f16 scales,
the converter's default; sizes in MiB; decode on 4 vCPUs, avx512vnni):

| Weights | Size | Perplexity | Decode `-t 4` |
|---------|-----:|-----------:|--------------:|
| Q8 | 1252 MB | 17.27 | 35.0 tok/s |
| Q4 + Q8 `v,o,down` | 831 MB | 17.89 | 47.7 tok/s |
| Q4 + Q8 `v,down` | 799 MB | 18.10 | |
| Q4 | 663 MB | 19.91 | 64.1 tok/s |

The down projection (`down`) and the value and output projections matter
most. The Q8 KV cache costs nothing measurable here: `--kv q8` on the Q8
model gives 17.27. The embedding and classifier (`emb`, `cls`) are large for a 128K
vocabulary and help little (Q4 + `emb,cls`: 788 MB, 19.87).

Qwen2.5 0.5B Instruct, same text and machine (no BOS; the F32 numpy
reference scores 19.00):

| Weights | Size | Perplexity | Decode `-t 4` |
|---------|-----:|-----------:|--------------:|
| Q8 | 500 MB | 18.98 | 87.3 tok/s |
| Q4 + Q8 `emb,v,o,down` | 390 MB | 20.06 | 107.2 tok/s |
| Q4 | 265 MB | 21.74 | 141.5 tok/s |

With f32 scales (`--scales f32`) every file is 5–10% larger and perplexity
changes by at most 0.15%.

Small models lose more to Q4_0: here `up` and `down` matter most, then `v`.

Other models on the same text (Q8 against the F32 numpy reference; f16 scales,
Q4 with the default rule):

| Model | F32 | Q8 | Q4 |
|-------|----:|---:|---:|
| Qwen3 0.6B | 28.10 | 27.90 | 31.69 |
| SmolLM2 135M Instruct | 23.44 | 23.62 | 31.60 |
| Gemma 3 1B Instruct | 42.32 | 41.97 | 49.83 |
| Gemma 3 4B Instruct | — | 24.30 | 23.31 |

The F32 reference was not run over the whole text for Gemma 3 4B; on its
first 96 tokens the Q8 logits are within KL 0.0021 of it (same argmax at 97%
of the positions). Its Q8 file takes 3932 MiB, Q4 2083 MiB.

Gemma 3 1B loses the most to Q4_0. With the fitted rule it reaches 53.25,
and keeping `emb,cls,v,down` in Q8 gives 47.97 at 783 MB. llama.cpp's plain
Q4_0 rounding (`--q4-rule round`) does better on Gemma 3, 49.83 at 536 MB,
though not on the other models below. Google's
quantization-aware checkpoint (`gemma-3-1b-it-qat-q4_0-unquantized`) was
trained for exactly that rounding, and with it reaches 40.50 at 536 MB. That
is below its own Q8 (42.53) and below the regular checkpoint's F32 reference;
the fit gives 47.43 on it. The converter therefore rounds Gemma 3 checkpoints
by default and fits the others; `--q4-rule fit|round` overrides it.

| Q4 rule (f16 scales) | Qwen2.5 0.5B | Llama 3.2 1B | Qwen3 0.6B | SmolLM2 135M | Gemma 3 1B | Gemma 3 1B QAT |
|---|---:|---:|---:|---:|---:|---:|
| fit | **21.73** | **19.90** | **31.69** | **31.60** | 53.25 | 47.43 |
| round | 22.45 | 19.94 | 31.75 | 32.31 | **49.83** | **40.50** |

The converter fits each Q4 block's scale by weighted least squares over a few
candidate mappings, rather than dividing the block's maximum by 7 (Llama 3.2
1B Q4: 19.91 instead of 20.46; Qwen2.5 0.5B: 21.73 instead of 23.58). The
file format and the kernels are unchanged, so older Q4 files still load;
convert again to get the better scales. See
[the quantization rule](file-format.md#quantization-rule).

An importance matrix weights that fit by how large each input channel's
activations are on sample text, so that the weights meeting large activations
are kept closer. Collect it with a Q8 conversion of the same model (each run
adds to the file), then convert to Q4 with it:

```sh
python3 tools/convert.py hf ./Qwen2.5-0.5B-Instruct q8.mlb --wtype q8
python3 tools/perplexity.py build/avx2/ml_run q8.mlb tok.mlt sample.txt --ml-tok build/avx2/ml_tok \
    --ctx 512 --chunks 32 --extra "--imatrix qwen.imat"
python3 tools/convert.py hf ./Qwen2.5-0.5B-Instruct q4.mlb --wtype q4 --imatrix qwen.imat
```

A tied embedding (Qwen2.5, Llama 3.2) is also the classifier and takes the
classifier's importance; an untied one is only a lookup and keeps the plain
fit. Q8 matrices ignore the importance matrix. Qwen2.5 0.5B Instruct Q4 on the wikitext-2 test set
(`tools/perplexity.py --ctx 512`; importance matrix from the first 32K tokens
of the wikitext-2 training set, `--chunks 64`):

| Weights | 4 chunks | 16 chunks | `--llama-cpp`, 4 chunks |
|---------|---------:|----------:|------------------------:|
| Q8 | 19.01 | 22.10 | 13.33 |
| Q4 | 21.72 | 26.47 | 15.13 |
| Q4 `--imatrix` | 20.92 | 25.30 | 14.52 |

It closes about a quarter of the gap to Q8. The gain depends on the sample
text: one from TinyStories gives 25.74 on 16 chunks. Collect it on text like
the model's use.

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
| `--batch N` | Prompt tokens per batched prefill step, 1 to 128. Default 128; `1` processes the prompt token by token. The output is identical either way. |
| `--kv TYPE` | KV cache type: `f32` (default), `f16` or `q8`. `f16` rounds K and V to half precision: half the memory at the same speed, with logits within that rounding (Qwen2.5 0.5B Q4 perplexity 21.71 against 21.74). `q8` stores K and V as int8 with one scale per position and KV head: about 3.8× less memory and faster attention on long contexts, with slightly different logits (22.82 on the same test). |
| `--pack-cache` | Save the packed weights next to the model (`MODEL.pk<G>`, G = pack group) and map them from there on later runs instead of packing again. The cache is checked against the model and the build's layout and rewritten when stale. |
| `--no-pack` | Keep weights row-major instead of repacking them for the packed kernels. Uses less memory during load, but runs slower. |
| `--profile` | Print the share of forward-pass time per stage (qkv, attention, wo, ffn, classifier, other). |
| `--chat TEMPLATE` | Chat with the model: `llama3`, `zephyr`, `llama2`, `qwen`, `smollm`, `chatml` or `gemma`, see [Chat](#chat). Needs `-z`. |
| `--system TEXT` | System prompt for `--chat`. |
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
- **Prompts:** long prompts are processed in batches of 128 tokens. That is 2–6×
  faster than token by token, depending on the ISA and the model size.
- **Long contexts:** attention grows with the position and, past a few
  thousand tokens, takes a third or more of the time. `--kv f16` halves the
  KV cache at no measurable cost; `--kv q8` cuts its memory traffic and size
  by about 3.8×.
- **Memory:** packing makes one copy of the weights at load and then releases
  the originals' pages. If a device cannot hold two copies of the model even
  briefly, use `--no-pack`, or `--pack-cache` once on a bigger machine with the
  same build.
- **Start-up:** packing takes time at every start: 1.8 s for Llama 3.2 1B Q4
  on one core (0.7 s with `-t 4`). With `--pack-cache` later starts map the
  packed weights directly and take 0.13 s, at the cost of a second copy on
  disk.

## Limits

- **Dimensions:** `dim`, `hidden_dim`, `n_layers` and head counts can be at most
  65536. `vocab` and `max_seq` can be at most 2^24.
- **Shapes:** `dim` and `hidden_dim` must be multiples of 32, `head_dim` must be
  even, and `n_heads` must be a multiple of `n_kv_heads`.
- **Memory:** the KV cache uses `n_layers × max_seq × kv_dim × 8` bytes,
  `n_layers × max_seq × kv_dim × 4` with `--kv f16`, or
  `n_layers × max_seq × (kv_dim + 4 × n_kv_heads) × 2` with `--kv q8`. All size
  arithmetic is overflow-checked, so on 32-bit ARMv7 an oversized model fails
  with `out of memory` instead of crashing.
- **Architecture:** Llama-family decoders (Llama, Mistral, Qwen2, Qwen3):
  RMSNorm, SwiGLU, GQA, RoPE, biases only on the q, k and v projections
  (Qwen2), optional per-head q/k RMSNorm (Qwen3), and a sliding window and a
  second RoPE base for chosen layers. `tools/convert.py` refuses other architectures.
