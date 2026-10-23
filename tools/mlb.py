"""The .mlb model format: quantization, writer and reader.

Quantization mirrors the C kernels bit for bit (float32 arithmetic, round half
away from zero), so files written here dequantize identically in C.
"""
import struct
from collections import namedtuple

import numpy as np

QK = 32
TINY = np.float32(1e-30)   # blocks with max |x| below this quantize to zero (ML_QUANT_TINY)
ALIGN = 64
HEADER_SIZE = 256
WTYPES = {"f32": 0, "q8": 1, "q4": 2}
_HDR = struct.Struct("<4sIIIIIIIIffIIIfffIII")   # magic .. rope scaling, q8 mask, qkv bias: 80 bytes
ROPE_TYPES = {None: 0, "llama3": 1}
LAYER_KEYS = ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")
BIAS_KEYS = ("bq", "bk", "bv")   # F32, after wo, when the header's qkv_bias is set (Qwen2)

QT = namedtuple("QT", "type q d shape")   # quantized tensor as stored

# Q4 models may keep these matrix classes in Q8 (header q8_mask, offset 72).
Q8_CLASSES = {"emb": 1, "cls": 2, "qk": 4, "v": 8, "o": 16, "up": 32, "down": 64}
_CLASS_OF = {"tok_emb": "emb", "wcls": "cls", "wq": "qk", "wk": "qk", "wv": "v", "wo": "o",
             "w1": "up", "w3": "up", "w2": "down"}


def q8_mask(classes):
    """Names (e.g. ["emb", "v"]) -> header bitmask."""
    m = 0
    for c in classes:
        if c not in Q8_CLASSES:
            raise ValueError(f"unknown matrix class {c!r}; choose from {', '.join(Q8_CLASSES)}")
        m |= Q8_CLASSES[c]
    return m


def matrix_type(key, wtype, mask, shared=False):
    """Storage type (WTYPES value) of matrix `key` (tok_emb, wq, ..., wcls)."""
    bits = Q8_CLASSES[_CLASS_OF[key]]
    if shared and key == "tok_emb":
        bits |= Q8_CLASSES["cls"]   # the shared table is also the classifier
    return WTYPES["q8"] if wtype == WTYPES["q8"] or mask & bits else WTYPES["q4"]


def rope_inv_freq(cfg):
    """float64[head_dim/2] RoPE inverse frequencies, with llama3 scaling if
    cfg["rope_scaling"] is set (same formula as transformers and the C loader)."""
    hd = cfg["dim"] // cfg["n_heads"]
    inv = 1.0 / float(cfg["rope_theta"]) ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    rs = cfg.get("rope_scaling")
    if not rs:
        return inv
    f, lo, hi = float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"])
    old = float(rs["original_max_position_embeddings"])
    wavelen = 2 * np.pi / inv
    out = np.where(wavelen > old / lo, inv / f, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    medium = (wavelen >= old / hi) & (wavelen <= old / lo)
    return np.where(medium, (1 - smooth) * out / f + smooth * out, out)


def _round_half_away(v):
    return np.trunc(v + np.copysign(np.float32(0.5), v).astype(np.float32))


def quantize_q8(x):
    """x[..., K] float32 -> (q int8[..., K], d float32[..., K/32])."""
    x = np.asarray(x, np.float32)
    b = x.reshape(*x.shape[:-1], -1, QK)
    amax = np.abs(b).max(axis=-1)
    amax = np.where(amax < TINY, np.float32(0), amax).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
    d = (amax / np.float32(127)).astype(np.float32)
    q = np.clip(_round_half_away((b * idv[..., None]).astype(np.float32)), -127, 127)
    return q.astype(np.int8).reshape(x.shape), d


def quantize_q4(x):
    """x[..., K] float32 -> (packed uint8[..., K/2], d float32[..., K/32]).
    Byte j of a block holds element j (low nibble) and j+16 (high nibble)."""
    x = np.asarray(x, np.float32)
    b = x.reshape(*x.shape[:-1], -1, QK)
    amax = np.abs(b).max(axis=-1)
    amax = np.where(amax < TINY, np.float32(0), amax).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax != 0, np.float32(7) / amax, np.float32(0)).astype(np.float32)
    d = (amax / np.float32(7)).astype(np.float32)
    v = np.clip(_round_half_away((b * idv[..., None]).astype(np.float32)), -8, 7).astype(np.int32) + 8
    packed = (v[..., :QK // 2] | (v[..., QK // 2:] << 4)).astype(np.uint8)
    return packed.reshape(*x.shape[:-1], -1), d


def dequant(t):
    """QT or float array -> float32 array of the logical shape."""
    if not isinstance(t, QT):
        return t
    if t.type == WTYPES["q8"]:
        b = t.q.reshape(*t.shape[:-1], -1, QK).astype(np.float32)
    else:
        p = t.q.reshape(*t.shape[:-1], -1, QK // 2).astype(np.int32)
        b = np.concatenate([(p & 0xF) - 8, (p >> 4) - 8], axis=-1).astype(np.float32)
    return (b * t.d[..., None]).astype(np.float32).reshape(t.shape)


def _order(cfg, w):
    """Yields (name, kind, array) in file order; kind 'q' = matrix in wtype, 'f' = F32."""
    yield "tok_emb", "q", w["tok_emb"]
    for i, layer in enumerate(w["layers"]):
        for k in LAYER_KEYS:
            yield f"layers.{i}.{k}", ("f" if k.endswith("norm") else "q"), layer[k]
            if k == "wo" and cfg.get("qkv_bias"):
                for b in BIAS_KEYS:
                    yield f"layers.{i}.{b}", "f", layer[b]
    yield "final_norm", "f", w["final_norm"]
    if not cfg.get("shared_classifier", 0):
        yield "wcls", "q", w["wcls"]


def _check_cfg(cfg):
    for k in ("dim", "hidden_dim"):
        if cfg[k] % QK:
            raise ValueError(f"{k}={cfg[k]} must be a multiple of {QK}")


def write_mlb(path, cfg, w, wtype, q8=()):
    """Writes the model; returns [(array name, offset, nbytes)] for every array.
    q8: matrix classes (Q8_CLASSES) kept in Q8 when wtype is q4."""
    _check_cfg(cfg)
    mask = q8_mask(q8) if wtype == "q4" else 0
    shared = bool(cfg.get("shared_classifier", 0))
    rs = cfg.get("rope_scaling") or {}
    hdr = _HDR.pack(b"MLB1", 1, cfg["dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"],
                    cfg["hidden_dim"], cfg["vocab"], cfg["max_seq"], cfg["rope_theta"],
                    cfg["norm_eps"], WTYPES[wtype], int(cfg.get("shared_classifier", 0)),
                    ROPE_TYPES[rs.get("type")], rs.get("factor", 0.0), rs.get("low_freq_factor", 0.0),
                    rs.get("high_freq_factor", 0.0), rs.get("original_max_position_embeddings", 0),
                    mask, int(bool(cfg.get("qkv_bias"))))
    layout = []
    with open(path, "wb") as f:
        f.write(hdr.ljust(HEADER_SIZE, b"\0"))

        def put(name, arr):
            raw = np.ascontiguousarray(arr).tobytes()
            layout.append((name, f.tell(), len(raw)))
            f.write(raw)
            f.write(b"\0" * (-len(raw) % ALIGN))

        for name, kind, arr in _order(cfg, w):
            arr = np.asarray(arr, np.float32)
            if kind == "f":
                put(name, arr)
                continue
            key = name.split(".")[-1]
            t = matrix_type(key, WTYPES[wtype], mask, shared)
            q, d = quantize_q8(arr) if t == WTYPES["q8"] else quantize_q4(arr)
            put(name + ".q", q)
            put(name + ".d", d)
    return layout


def read_mlb(path):
    """-> (cfg dict, tensors) with tensors shaped like write_mlb's input; matrices are QT."""
    data = np.fromfile(path, np.uint8)
    f = _HDR.unpack(data[:_HDR.size].tobytes())
    if f[0] != b"MLB1":
        raise ValueError("bad magic")
    keys = ("version", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim", "vocab",
            "max_seq", "rope_theta", "norm_eps", "wtype", "shared_classifier")
    cfg = dict(zip(keys, f[1:13]))
    rtype, rf, rlo, rhi, rorig, cfg["q8_mask"], cfg["qkv_bias"] = f[13:]
    cfg["rope_scaling"] = None if rtype == 0 else {
        "type": {v: k for k, v in ROPE_TYPES.items()}[rtype], "factor": rf, "low_freq_factor": rlo,
        "high_freq_factor": rhi, "original_max_position_embeddings": rorig}
    cfg["rope_theta"], cfg["norm_eps"] = np.float32(cfg["rope_theta"]), np.float32(cfg["norm_eps"])
    dim, hid, voc = cfg["dim"], cfg["hidden_dim"], cfg["vocab"]
    kv = dim // cfg["n_heads"] * cfg["n_kv_heads"]
    shapes = {"tok_emb": (voc, dim), "wq": (dim, dim), "wk": (kv, dim), "wv": (kv, dim),
              "wo": (dim, dim), "w1": (hid, dim), "w2": (dim, hid), "w3": (hid, dim),
              "wcls": (voc, dim)}
    off = HEADER_SIZE

    def take(dtype, count):
        nonlocal off
        nbytes = count * np.dtype(dtype).itemsize
        a = data[off:off + nbytes].view(dtype)
        off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        return a

    def mat(key):
        r, c = shapes[key]
        t = matrix_type(key, cfg["wtype"], cfg["q8_mask"], bool(cfg["shared_classifier"]))
        nq = r * c if t == 1 else r * c // 2
        q = take(np.int8 if t == 1 else np.uint8, nq)
        d = take(np.float32, r * c // QK).reshape(r, c // QK)
        return QT(t, q.reshape(r, -1), d, (r, c))

    t = {"tok_emb": mat("tok_emb"), "layers": []}
    for _ in range(cfg["n_layers"]):
        layer = {}
        for k in LAYER_KEYS:
            layer[k] = take(np.float32, dim) if k.endswith("norm") else mat(k)
            if k == "wo" and cfg["qkv_bias"]:
                layer["bq"], layer["bk"], layer["bv"] = take(np.float32, dim), take(np.float32, kv), take(np.float32, kv)
        t["layers"].append(layer)
    t["final_norm"] = take(np.float32, dim)
    t["wcls"] = t["tok_emb"] if cfg["shared_classifier"] else mat("wcls")
    return cfg, t
