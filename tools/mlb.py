"""The .mlb model format: quantization, writer and reader.

Quantization mirrors the C kernels bit for bit (float32 arithmetic, round half
away from zero), so files written here dequantize identically in C.
"""
import math
import os
import struct
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

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


def _pack_q4(v):
    """Levels -8..7 [..., blocks, 32] -> packed uint8 [..., blocks * 16]."""
    v = (v + 8).astype(np.int32)
    packed = (v[..., :QK // 2] | (v[..., QK // 2:] << 4)).astype(np.uint8)
    return packed.reshape(*v.shape[:-2], -1)


Q4_STEPS = np.arange(-10, 3, dtype=np.float32) * np.float32(0.2)   # largest element -> level -(8 + step)


def _sum32(a):
    """Sum over the last axis of 32 in a fixed order: numpy's own reductions
    order float32 additions by the CPU's SIMD width, which would make the
    converted file depend on the machine."""
    while a.shape[-1] > 1:
        h = a.shape[-1] // 2
        a = a[..., :h] + a[..., h:]
    return a[..., 0]


def _q4_search(b, imp=None):
    """Blocks b[n, 32] -> (levels, scales) minimizing the x^2-weighted error,
    times imp[n, 32] (channel importance) when given."""
    mx = np.take_along_axis(b, np.abs(b).argmax(axis=-1)[:, None], axis=-1)[:, 0]
    zero = np.abs(mx) < TINY
    y = b * (np.float32(1) / np.where(zero, np.float32(1), mx))[:, None]   # largest element at 1
    w = b * b if imp is None else b * b * imp
    wb = w * b
    best = step = None
    for s in Q4_STEPS:
        lv = np.clip(np.rint(y * -(8 + s)), -8, 7)
        num, den = _sum32(wb * lv), _sum32(w * lv * lv)
        # with the least-squares scale num/den the error is sum(w x^2) - num^2/den
        score = num * num / np.where(den > 0, den, 1)
        if best is None:
            best, step = score, np.full(len(b), s, np.float32)
        else:
            better = score > best
            best, step = np.where(better, score, best), np.where(better, s, step)
    lv = np.clip(np.rint(y * -(8 + step)[:, None]), -8, 7)
    den = _sum32(w * lv * lv)
    d = np.where(den > 0, _sum32(wb * lv) / np.where(den > 0, den, 1), 0)
    return np.where(zero[:, None], 0, lv), np.where(zero, 0, d).astype(np.float32)


IMP_FLOOR = np.float32(1e-3)   # least channel importance, relative to the matrix mean


def _importance(imp):
    """Mean squared inputs [K] -> float32 weights with mean 1, or None if all zero.
    The mean is summed exactly: the result must not depend on the machine."""
    imp = np.asarray(imp, np.float64)
    mean = math.fsum(imp.tolist()) / len(imp)
    if not mean > 0:
        return None
    return np.maximum((imp / mean).astype(np.float32), IMP_FLOOR)


def quantize_q4(x, imp=None):
    """x[..., K] float32 -> (packed uint8[..., K/2], d float32[..., K/32]).
    Byte j of a block holds element j (low nibble) and j+16 (high nibble).

    Per block, the element of largest magnitude is mapped to level -8 - step
    for each of Q4_STEPS, the scale for those levels is fitted by least
    squares weighted by x^2, and the fit with the least weighted error is
    kept. Against amax/7 rounding this uses all 16 levels and lowers the error
    of the large weights (Qwen2.5 0.5B Q4 perplexity 23.58 -> 21.73). Chunks
    of blocks run on a thread pool (numpy releases the GIL).

    imp[K]: importance of each input channel, the mean square of the matrix
    input over sample text (read_imatrix). The error is then weighted by
    x^2 * imp, so that weights meeting large activations are kept closer."""
    x = np.asarray(x, np.float32)
    b = x.reshape(-1, QK)
    imp = None if imp is None else _importance(imp)
    nb = x.shape[-1] // QK

    def search(i):
        c = b[i:i + 4096]
        if imp is None:
            return _q4_search(c)
        return _q4_search(c, imp.reshape(nb, QK)[np.arange(i, i + len(c)) % nb])

    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        parts = list(ex.map(search, range(0, len(b), 4096)))
    lv = np.concatenate([p[0] for p in parts]).reshape(*x.shape[:-1], -1, QK)
    d = np.concatenate([p[1] for p in parts]).reshape(*x.shape[:-1], -1)
    return _pack_q4(lv), d


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


def write_mlb(path, cfg, w, wtype, q8=(), imatrix=None):
    """Writes the model; returns [(array name, offset, nbytes)] for every array.
    q8: matrix classes (Q8_CLASSES) kept in Q8 when wtype is q4.
    imatrix: read_imatrix's result, weighting the Q4 fits."""
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
            imp = imatrix_for(imatrix, name) if imatrix else None
            q, d = quantize_q8(arr) if t == WTYPES["q8"] else quantize_q4(arr, imp)
            put(name + ".q", q)
            put(name + ".d", d)
    return layout


_IMAT = struct.Struct("<4sIIIIIQQ")   # magic, version, n_layers, dim, hidden_dim, 0, tokens, classifier tokens
_IMAT_INPUT = {"wq": "qkv", "wk": "qkv", "wv": "qkv", "wo": "wo", "w1": "ffn", "w3": "ffn", "w2": "down"}


def read_imatrix(path, cfg):
    """ml_run --imatrix file -> {"layers": [{input: mean square [K]}], "cls": [dim]},
    inputs qkv, wo, ffn (dim) and down (hidden_dim)."""
    data = open(path, "rb").read()
    magic, ver, nl, dim, hid, _, ntok, ncls = _IMAT.unpack_from(data)
    if magic != b"MLI1" or ver != 1:
        raise ValueError(f"{path}: not an importance matrix")
    if (nl, dim, hid) != (cfg["n_layers"], cfg["dim"], cfg["hidden_dim"]):
        raise ValueError(f"{path}: for a model with {nl} layers, dim {dim}, hidden_dim {hid}")
    if not ntok:
        raise ValueError(f"{path}: no tokens")
    s = np.frombuffer(data, np.float64, offset=_IMAT.size)
    if len(s) != nl * (3 * dim + hid) + dim:
        raise ValueError(f"{path}: truncated")
    layers = []
    for i in range(nl):
        o = i * (3 * dim + hid)
        layers.append({"qkv": s[o:o + dim] / ntok, "wo": s[o + dim:o + 2 * dim] / ntok,
                       "ffn": s[o + 2 * dim:o + 3 * dim] / ntok, "down": s[o + 3 * dim:o + 3 * dim + hid] / ntok})
    return {"layers": layers, "cls": s[-dim:] / ncls if ncls else None, "tokens": ntok}


def imatrix_for(imatrix, name):
    """Importance of the input channels of matrix `name` (write_mlb's names), or None."""
    if name == "wcls":
        return imatrix["cls"]
    parts = name.split(".")
    if parts[0] == "layers":
        return imatrix["layers"][int(parts[1])][_IMAT_INPUT[parts[2]]]
    return None   # the embedding is a lookup, not a product


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
