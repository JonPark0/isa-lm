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
| 52 | `u32` | `rope_type` | `0` = plain RoPE, `1` = llama3 scaling |
| 56 | `f32` | `rope_factor` | llama3 only: > 0 |
| 60 | `f32` | `rope_low_freq_factor` | llama3 only: > 0 |
| 64 | `f32` | `rope_high_freq_factor` | llama3 only: > `rope_low_freq_factor` |
| 68 | `u32` | `rope_orig_max_seq` | llama3 only: > 0 |
| 72 | `u32` | `q8_mask` | `wtype` Q4_0 only: matrix classes stored as Q8_0 instead (below); 0 for Q8_0 models |
| 76 | `u32` | `qkv_bias` | 0 or 1: the q, k and v projections have biases (Qwen2) |
| 80 | | padding | zero up to offset 256 |

`dim`, `n_layers`, `n_heads`, `n_kv_heads` and `hidden_dim` are limited to
65536. `vocab` and `max_seq` are limited to 2^24. Files written before RoPE
scaling existed have zeros at offsets 52–71, which means plain RoPE.

Derived values: `head_dim = dim / n_heads` and `kv_dim = head_dim × n_kv_heads`.

`q8_mask` bits, one per matrix class: 1 `tok_emb`, 2 `wcls`, 4 `wq` and
`wk`, 8 `wv`, 16 `wo`, 32 `w1` and `w3`, 64 `w2`. With a shared classifier,
`tok_emb` is Q8_0 if bit 1 or 2 is set. Files written before mixed models
existed have 0 there.

With `qkv_bias` set, each layer stores `bq`, `bk` and `bv` after `wo`; they
are added to the projections before RoPE. Files written before biases existed
have 0 at offset 76.

## Body

The arrays start at offset 256 and come in this order:

```
tok_emb                     vocab × dim         (wtype, or Q8_0 per q8_mask; same below)
for each layer:
  attn_norm                 dim                 (F32)
  wq                        dim × dim           (wtype)
  wk                        kv_dim × dim        (wtype)
  wv                        kv_dim × dim        (wtype)
  wo                        dim × dim           (wtype)
  bq, bk, bv                dim, kv_dim, kv_dim (F32; only if qkv_bias)
  ffn_norm                  dim                 (F32)
  w1   (gate)               hidden_dim × dim    (wtype)
  w2   (down)               dim × hidden_dim    (wtype)
  w3   (up)                 hidden_dim × dim    (wtype)
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

Q4 uses 7 instead of 127 and clamps to [-8, 7]. Every SIMD implementation is
bit-identical to the scalar one.

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

## Validation done by the loader

| Problem | Status |
|---------|--------|
| file cannot be opened | `ML_E_IO` |
| empty file, file shorter than the header, bad magic or version, unknown `wtype` or `rope_type`, invalid llama3 parameters, file shorter than the arrays the header implies, a Q8 quant equal to −128 | `ML_E_FORMAT` |
| zero or over-limit dimension, `dim` or `hidden_dim` not a multiple of 32, `dim % n_heads ≠ 0`, `n_heads % n_kv_heads ≠ 0`, odd `head_dim`, `vocab < 3` | `ML_E_SHAPE` |

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
| 28 | `u32` | `flags` | 1: prepend BOS when encoding; 2: `ignore_merges` (a whole pre-token found in the vocabulary is one token); 4: SentencePiece prefix `▁` only at the start of the text, not after each added token; 8: byte-level, NFC-normalize the text between added tokens; 16: byte-level, numbers are split into single digits (`\p{N}` instead of `\p{N}{1,3}`); other bits must be 0 |

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
