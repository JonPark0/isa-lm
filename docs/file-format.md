# File formats: `.mlb` model, version 1, and `.mlt` tokenizer

An `.mlb` file is a fixed 256-byte header followed by tensor arrays. Every array
starts at a 64-byte boundary, so the runtime can `mmap` the file and point
straight into it without copying. Everything is little-endian.

`tools/mlb.py` is the reference writer and reader. `model/mlb.c` is the loader.

## Header

| Offset | Type | Field | Notes |
|-------:|------|-------|-------|
| 0  | `char[4]` | magic | `"MLB1"` |
| 4  | `u32` | version | `1` |
| 8  | `u32` | `dim` | model width; multiple of 32 |
| 12 | `u32` | `n_layers` | |
| 16 | `u32` | `n_heads` | must divide `dim`; `head_dim = dim / n_heads` must be even |
| 20 | `u32` | `n_kv_heads` | must divide `n_heads` (GQA) |
| 24 | `u32` | `hidden_dim` | FFN width; multiple of 32 |
| 28 | `u32` | `vocab` | at least 3 (ids 1 and 2 are BOS and EOS) |
| 32 | `u32` | `max_seq` | context length; sets the KV cache size |
| 36 | `f32` | `rope_theta` | > 0 |
| 40 | `f32` | `norm_eps` | ≥ 0 |
| 44 | `u32` | `wtype` | `1` = Q8_0, `2` = Q4_0 |
| 48 | `u32` | `shared_classifier` | non-zero: `wcls` is omitted and `tok_emb` is used |
| 52 | `u32` | `rope_type` | `0` = plain RoPE, `1` = llama3, `2` = linear, `3` = yarn scaling |
| 56 | `f32` | `rope_factor` | llama3, linear, yarn: > 0 |
| 60 | `f32` | `rope_low_freq_factor` | llama3: > 0; yarn: `beta_fast` |
| 64 | `f32` | `rope_high_freq_factor` | llama3: > `rope_low_freq_factor`; yarn: `beta_slow`, > 0 and < `beta_fast` |
| 68 | `u32` | `rope_orig_max_seq` | llama3, yarn: > 0 |
| 72 | `u32` | `q8_mask` | `wtype` Q4_0 only: matrix classes stored as Q8_0 instead (below); 0 for Q8_0 models |
| 76 | `u32` | `qkv_bias` | 0 or 1: the q, k and v projections have biases (Qwen2) |
| 80 | `u32` | `head_dim` | 0: `dim / n_heads`; else the head size (even; Qwen3) |
| 84 | `u32` | `qk_norm` | 0 or 1: per-head RMSNorm of q and k before RoPE (Qwen3) |
| 88 | `f32` | `rope_attn_factor` | yarn only: > 0, multiplies RoPE's cos and sin; 0 otherwise |
| 92 | `u32` | `sliding_window` | 0: full attention; else each position attends to the last this many |
| 96 | `u32` | `layer_flags` | 0 or 1: a byte of flags per layer follows the header (below) |
| 100 | `f32` | `rope_local_theta` | ≥ 0: RoPE base of the layers flagged for it, used unscaled (Gemma 3) |
| 104 | `u32` | `act` | FFN activation: `0` = SiLU, `1` = GELU with the tanh approximation (Gemma) |
| 108 | `u32` | `post_norms` | 0 or 1: the attention and FFN outputs are RMS-normalized before the residual (Gemma 3) |
| 112 | `f32` | `embed_scale` | ≥ 0: factor on the embedding rows; 0 means 1 (Gemma: `sqrt(dim)`) |
| 116 | `f32` | `attn_scale` | ≥ 0: attention score factor; 0 means `1/sqrt(head_dim)` |
| 120 | | padding | zero up to offset 256 |

`dim`, `n_layers`, `n_heads`, `n_kv_heads` and `hidden_dim` are limited to
65536. `vocab` and `max_seq` are limited to 2^24. Files written before RoPE
scaling existed have zeros at offsets 52–71, which means plain RoPE.

Derived values: `head_dim` (offset 80, or `dim / n_heads` when 0),
`q_dim = head_dim × n_heads` (a multiple of 32) and
`kv_dim = head_dim × n_kv_heads`. Files written before these fields existed
have zeros at offsets 80–95: `dim / n_heads`, no q/k norm, full attention.

`q8_mask` bits, one per matrix class: 1 `tok_emb`, 2 `wcls`, 4 `wq` and
`wk`, 8 `wv`, 16 `wo`, 32 `w1` and `w3`, 64 `w2`. With a shared classifier,
`tok_emb` is Q8_0 if bit 1 or 2 is set. Files written before mixed models
existed have 0 there.

With `qkv_bias` set, each layer stores `bq`, `bk` and `bv` after `wo`; they
are added to the projections before RoPE. Files written before biases existed
have 0 at offset 76. With `qk_norm` set, each layer then stores `q_norm` and
`k_norm` (`head_dim` floats each), applied to every query and key head after
the biases and before RoPE, with `norm_eps`.

With `layer_flags` set, `n_layers` bytes (padded to 64 like every array)
come first in the body, one per layer: bit 0, the layer attends through the
sliding window; bit 1, it rotates with `rope_local_theta` instead of
`rope_theta` and its scaling; other bits must be 0. Without them the window,
if any, applies to every layer.

## Body

The arrays start at offset 256 and come in this order:

```
layer flags                 n_layers            (u8; only if layer_flags)
tok_emb                     vocab × dim         (wtype, or Q8_0 per q8_mask; same below)
for each layer:
  attn_norm                 dim                 (F32)
  wq                        q_dim × dim         (wtype)
  wk                        kv_dim × dim        (wtype)
  wv                        kv_dim × dim        (wtype)
  wo                        dim × q_dim         (wtype)
  bq, bk, bv                q_dim, kv_dim, kv_dim (F32; only if qkv_bias)
  q_norm, k_norm            head_dim, head_dim  (F32; only if qk_norm)
  ffn_norm                  dim                 (F32)
  w1   (gate)               hidden_dim × dim    (wtype)
  w2   (down)               dim × hidden_dim    (wtype)
  w3   (up)                 hidden_dim × dim    (wtype)
  post_attn_norm, post_ffn_norm  dim, dim       (F32; only if post_norms)
final_norm                  dim                 (F32)
wcls                        vocab × dim         (wtype; omitted if shared_classifier)
```

Matrices are stored row by row, one row per output: `y[r] = Σ_c W[r][c] · x[c]`.

An F32 tensor is one array of floats. A quantized matrix is two arrays, written
one after the other:

1. **`q`**: the quants.
   - Q8_0: `rows × cols` bytes of `int8`.
   - Q4_0: `rows × cols / 2` bytes.
2. **`d`**: `rows × cols / 32` `f32` scales, one per block of 32 consecutive
   weights in a row.

Each array is zero-padded to a multiple of 64 bytes, so the next array starts
aligned. The padding after the last array is optional.

### Q8_0

Weight `w = q × d`. Quants must be in **[-127, 127]**. The loader rejects any
byte equal to `-128`, because the x86 kernels negate weights by the
activation's sign and `-128` has no int8 negation.

### Q4_0

Each block of 32 weights takes 16 bytes. Byte `j` of a block holds element `j`
in its low nibble and element `j + 16` in its high nibble. Weight
`w = (nibble − 8) × d`, which gives values in [-8, 7].

### Quantization rule

Both the writer and the runtime's activation quantizer use these exact float32
operations:

```
amax = max |x| over the block
if amax < 1e-30:   d = 0, q = 0                      # 127/amax would overflow
else:              d = amax / 127,  q = trunc(x * (127/amax) + copysign(0.5, x)),  clamp to ±127
```

Every SIMD implementation is bit-identical to the scalar one.

Q4_0 weights are written once, so `tools/mlb.py` fits each block instead: the
element of largest magnitude goes to a level between -6 and -8.4 (clamped to
[-8, 7]) in steps of 0.2, the scale `d` is the least-squares fit for those
levels with each element weighted by x², and the candidate with the least
weighted error is kept. A block whose largest magnitude is below 1e-6 is
fitted scaled up to 1 and its `d` scaled back, so that float32 squares and
cubes do not underflow. `d` can be negative. Any `d` and nibbles are valid
input to the runtime.

With `tools/convert.py --imatrix`, each element's weight is instead, as in
llama.cpp, `imp × sqrt(s² + x²)`: `s²` is the mean x² of the element's row,
and `imp` the importance of its input channel, the channel's mean squared
input (`ml_run --imatrix`) divided by its matrix's mean over channels, and at
least 0.001.

### Importance matrix file

`ml_run --imatrix FILE` writes, little-endian: `"MLI1"`, then `u32` version
(1), `n_layers`, `dim`, `hidden_dim`, `0`; `u64` tokens summed into the layers
and tokens summed into the classifier (prompt tokens prefilled without logits
skip it); then `float64` sums of squares, per channel: for each layer the
input of `wq`/`wk`/`wv` (`dim`), of `wo` (`dim`), of `w1`/`w3` (`dim`) and of
`w2` (`hidden_dim`), then the classifier input (`dim`).

## RoPE

Rotary embeddings act on interleaved pairs `(2i, 2i+1)` of each head. Pair `i`
uses the inverse frequency `inv_i = rope_theta^(−2i / head_dim)`, as in
llama2.c. Hugging Face checkpoints use the rotate-half layout instead; the
converter permutes the rows of `wq` and `wk` to match.

With `rope_type = 1` (llama3), each frequency is rescaled the same way
`transformers` does it. Here `wavelen = 2π / inv_i`, `f` is `rope_factor`,
`lo` and `hi` are the low and high frequency factors, and `orig` is
`rope_orig_max_seq`:

```
wavelen < orig / hi     → inv_i                                  (high frequency: unchanged)
wavelen > orig / lo     → inv_i / f                              (low frequency: scaled)
otherwise               → (1 − s) · inv_i / f + s · inv_i,  s = (orig / wavelen − lo) / (hi − lo)
```

With `rope_type = 2` (linear), every frequency is divided by `f`. With
`rope_type = 3` (yarn), as `transformers` computes it, with `d(r)` the
dimension at which a wavelength fits `r` rotations into `orig`:

```
d(r)  = head_dim · ln(orig / (2π r)) / (2 ln rope_theta)
lo    = max(floor(d(beta_fast)), 0),  hi = min(ceil(d(beta_slow)), head_dim − 1)
ramp  = clamp((i − lo) / (hi − lo), 0, 1)
inv_i → inv_i / f · ramp + inv_i · (1 − ramp)
```

and the cos and sin tables are multiplied by `rope_attn_factor` (the
converter takes `attention_factor` from the config, or computes
`0.1 ln f + 1`, or the `mscale` / `mscale_all_dim` ratio, as transformers does).

## Validation done by the loader

| Problem | Status |
|---------|--------|
| file cannot be opened | `ML_E_IO` |
| empty file, file shorter than the header, bad magic or version, unknown `wtype` or `rope_type`, invalid llama3, linear or yarn parameters, a `rope_attn_factor` without yarn, `qkv_bias`, `qk_norm` or `layer_flags` other than 0 or 1, unknown layer flag bits, a negative `rope_local_theta` or a local-RoPE layer without one, `sliding_window` above 2^24, file shorter than the arrays the header implies, a Q8 quant equal to −128 | `ML_E_FORMAT` |
| zero or over-limit dimension, `dim` or `hidden_dim` not a multiple of 32, `dim % n_heads ≠ 0` without `head_dim`, `n_heads % n_kv_heads ≠ 0`, odd `head_dim`, `q_dim` not a multiple of 32 or above 65536, `vocab < 3` | `ML_E_SHAPE` |

All size arithmetic is overflow-checked, which matters on 32-bit targets.

## `.mlt` tokenizer format

`tools/convert_tokenizer.py` writes this from a Hugging Face `tokenizer.json`;
`model/bpe.c` reads it. All integers are little-endian.

| Offset | Type | Field | Meaning |
|-------:|------|-------|---------|
| 0 | `char[4]` | magic | `MLT1` |
| 4 | `u32` | `version` | 1 |
| 8 | `u32` | `kind` | 1 = SentencePiece-style BPE, 2 = byte-level BPE with the Llama 3 pre-tokenizer |
| 12 | `u32` | `n_vocab` | tokens, at most the model's `vocab` |
| 16 | `u32` | `n_merges` | merge rules |
| 20 | `i32` | `bos` | BOS id, or −1 |
| 24 | `i32` | `eos` | EOS id, or −1 |
| 28 | `u32` | `flags` | 1: prepend BOS when encoding; 2: `ignore_merges` (a whole pre-token found in the vocabulary is one token); 4: SentencePiece prefix `▁` only at the start of the text, not after each added token; 8: byte-level, NFC-normalize the text between added tokens; 16: byte-level, numbers are split into single digits (`\p{N}` instead of `\p{N}{1,3}`); 32: byte-level, the GPT-2 regex instead of the Llama 3 one; 64: byte-level, every `\p{N}` is split off before the regex (a `Digits(individual_digits)` pre-tokenizer); other bits must be 0 |

Then `n_vocab` tokens, each `u8 attr`, `u32 len` and `len` bytes:

- **attr 1 (added):** matched literally in the text before anything else
  (`<s>`, `<|eot_id|>`); the longest match wins.
- **attr 2 (byte):** a SentencePiece byte-fallback token `<0xXX>`, stored as
  its single byte.
- **Bytes:** byte-level pieces are stored as the raw bytes they stand for (not
  the GPT-2 printable characters of `tokenizer.json`); SentencePiece pieces as
  UTF-8 with `▁` (U+2581) for spaces.

Then `n_merges` rules of three `u32`: left, right and merged token, in rank
order. Encoding applies the lowest-ranked applicable merge first, the
leftmost among equal ranks, which is what the `tokenizers` library does.
