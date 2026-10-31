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
# magic .. rope scaling, q8 mask, qkv bias (80 bytes), head_dim, qk_norm,
# rope attention factor, sliding window (96), per-layer flags present, local
# RoPE theta: 104 bytes
_HDR = struct.Struct("<4sIIIIIIIIffIIIfffIIIIIfIIf")
LAYER_SLIDING, LAYER_LOCAL_ROPE = 1, 2   # per-layer flag bits (Gemma 3, Qwen2 with some sliding layers)
ROPE_TYPES = {None: 0, "llama3": 1, "linear": 2, "yarn": 3}
LAYER_KEYS = ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")
BIAS_KEYS = ("bq", "bk", "bv")   # F32, after wo, when the header's qkv_bias is set (Qwen2)
QK_NORM_KEYS = ("q_norm", "k_norm")   # F32 [head_dim], after the biases, when qk_norm is set (Qwen3)


def head_dim(cfg):
    """Size of one attention head: cfg["head_dim"], or dim / n_heads when unset."""
    return cfg.get("head_dim") or cfg["dim"] // cfg["n_heads"]


def layer_window(cfg, layer):
    """Positions layer `layer` attends to at most (0: all): the sliding window
    for every layer, or for the layers whose flags say so."""
    win = cfg.get("sliding_window") or 0
    flags = cfg.get("layer_flags")
    return win if flags is None or flags[layer] & LAYER_SLIDING else 0


def layer_local_rope(cfg, layer):
    """Whether layer `layer` uses the local RoPE base (rope_local_theta, unscaled)."""
    flags = cfg.get("layer_flags")
    return flags is not None and bool(flags[layer] & LAYER_LOCAL_ROPE)

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
    """float64[head_dim/2] RoPE inverse frequencies, scaled as cfg["rope_scaling"]
    says (llama3, linear or yarn; the formulas of transformers and the C loader)."""
    hd = head_dim(cfg)
    inv = 1.0 / float(cfg["rope_theta"]) ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    rs = cfg.get("rope_scaling")
    if not rs:
        return inv
    if rs["type"] == "linear":
        return inv / float(rs["factor"])
    if rs["type"] == "yarn":
        f, base, orig = float(rs["factor"]), float(cfg["rope_theta"]), float(rs["original_max_position_embeddings"])

        def corr(rot):   # dimension at which a wavelength fits `rot` rotations into orig
            return hd * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        lo = max(math.floor(corr(float(rs["beta_fast"]))), 0)
        hi = min(math.ceil(corr(float(rs["beta_slow"]))), hd - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(hd // 2, dtype=np.float64) - lo) / (hi - lo), 0, 1)
        return inv / f * ramp + inv * (1 - ramp)
    f, lo, hi = float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"])
    old = float(rs["original_max_position_embeddings"])
    wavelen = 2 * np.pi / inv
    out = np.where(wavelen > old / lo, inv / f, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    medium = (wavelen >= old / hi) & (wavelen <= old / lo)
    return np.where(medium, (1 - smooth) * out / f + smooth * out, out)


def rope_attention_factor(cfg):
    """Factor on RoPE's cos and sin (yarn's attention scaling), else 1."""
    rs = cfg.get("rope_scaling") or {}
    return float(rs.get("attention_factor", 1.0)) if rs.get("type") == "yarn" else 1.0


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


SMALL = np.float32(1e-6)   # Q4 blocks with max |x| below this are fitted rescaled (_q4_search)
Q4_STEPS = np.arange(-10, 3, dtype=np.float32) * np.float32(0.2)   # largest element -> level -(8 + step)


def _sum32(a):
    """Sum over the last axis of 32 in a fixed order: numpy's own reductions
    order float32 additions by the CPU's SIMD width, which would make the
    converted file depend on the machine."""
    while a.shape[-1] > 1:
        h = a.shape[-1] // 2
        a = a[..., :h] + a[..., h:]
    return a[..., 0]


def _q4_search(b, w=None):
    """Blocks b[n, 32] -> (levels, scales) minimizing the error weighted by
    w[n, 32] (default x^2)."""
    mx = np.take_along_axis(b, np.abs(b).argmax(axis=-1)[:, None], axis=-1)[:, 0]
    zero = np.abs(mx) < TINY
    y = b * (np.float32(1) / np.where(zero, np.float32(1), mx))[:, None]   # largest element at 1
    # Blocks of small values are fitted scaled to a largest magnitude of 1 (and
    # their weights to a largest weight of 1): the cubes and squares of the fit
    # would underflow float32 below about 1e-7. The others stay as they are.
    small = (np.abs(mx) < SMALL) & ~zero
    up = np.where(small, np.float32(1) / np.where(small, np.abs(mx), 1), np.float32(1)).astype(np.float32)
    b = b * up[:, None]
    if w is None:
        w = b * b
    else:
        wmax = w.max(axis=-1)
        w = w / np.where(small & (wmax > 0), wmax, np.float32(1))[:, None]   # a division: 1/wmax could overflow
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
    d = np.where(den > 0, _sum32(wb * lv) / np.where(den > 0, den, 1), 0) / up
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
    input over sample text (read_imatrix). The error is then weighted as
    llama.cpp does, by imp * sqrt(s2 + x^2) with s2 the row's mean x^2, so
    that weights meeting large activations are kept closer (Qwen2.5 0.5B Q4,
    imatrix from 32K tokens of wikitext-2 train: perplexity 26.47 -> 25.30 on
    16 chunks of the test set; weighting by imp * x^2 did less)."""
    x = np.asarray(x, np.float32)
    b = x.reshape(-1, QK)
    imp = None if imp is None else _importance(imp)
    nb = x.shape[-1] // QK
    if imp is not None:
        # row mean of x^2, summed in a fixed order (block trees, then blocks in turn)
        sq = b.astype(np.float64) ** 2
        s2_64 = np.cumsum(_sum32(sq).reshape(-1, nb), axis=1)[:, -1] / x.shape[-1]
        s2 = s2_64.astype(np.float32)
        impb = imp.reshape(nb, QK)

    def search(i):
        c = b[i:i + 4096]
        if imp is None:
            return _q4_search(c)
        idx = np.arange(i, i + len(c))
        w = impb[idx % nb] * np.sqrt(s2[idx // nb][:, None] + c * c)
        # tiny rows: s2 and c^2 underflow float32 below about 4e-23; take them in float64
        tiny = np.abs(c).max(axis=-1) < SMALL
        if tiny.any():
            c64 = c[tiny].astype(np.float64)
            w[tiny] = (impb[idx[tiny] % nb] * np.sqrt(s2_64[idx[tiny] // nb][:, None] + c64 * c64)).astype(np.float32)
        return _q4_search(c, w)

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
            if k == "wo" and cfg.get("qk_norm"):
                for b in QK_NORM_KEYS:
                    yield f"layers.{i}.{b}", "f", layer[b]
    yield "final_norm", "f", w["final_norm"]
    if not cfg.get("shared_classifier", 0):
        yield "wcls", "q", w["wcls"]


def _check_cfg(cfg):
    for k in ("dim", "hidden_dim"):
        if cfg[k] % QK:
            raise ValueError(f"{k}={cfg[k]} must be a multiple of {QK}")
    flags = cfg.get("layer_flags")
    if flags is not None and any(f & LAYER_LOCAL_ROPE for f in flags) and not cfg.get("rope_local_theta"):
        raise ValueError("layers with the local RoPE base need rope_local_theta")
    if cfg["n_heads"] * head_dim(cfg) % QK:
        raise ValueError(f"n_heads * head_dim = {cfg['n_heads'] * head_dim(cfg)} must be a multiple of {QK}")


def write_mlb(path, cfg, w, wtype, q8=(), imatrix=None):
    """Writes the model; returns [(array name, offset, nbytes)] for every array.
    q8: matrix classes (Q8_CLASSES) kept in Q8 when wtype is q4.
    imatrix: read_imatrix's result, weighting the Q4 fits."""
    _check_cfg(cfg)
    mask = q8_mask(q8) if wtype == "q4" else 0
    shared = bool(cfg.get("shared_classifier", 0))
    rs = cfg.get("rope_scaling") or {}
    yarn = rs.get("type") == "yarn"   # beta_fast and beta_slow take llama3's frequency factor slots
    explicit_hd = head_dim(cfg) if head_dim(cfg) != cfg["dim"] // cfg["n_heads"] or cfg["dim"] % cfg["n_heads"] else 0
    hdr = _HDR.pack(b"MLB1", 1, cfg["dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"],
                    cfg["hidden_dim"], cfg["vocab"], cfg["max_seq"], cfg["rope_theta"],
                    cfg["norm_eps"], WTYPES[wtype], int(cfg.get("shared_classifier", 0)),
                    ROPE_TYPES[rs.get("type")], rs.get("factor", 0.0),
                    rs.get("beta_fast" if yarn else "low_freq_factor", 0.0),
                    rs.get("beta_slow" if yarn else "high_freq_factor", 0.0),
                    rs.get("original_max_position_embeddings", 0),
                    mask, int(bool(cfg.get("qkv_bias"))), explicit_hd, int(bool(cfg.get("qk_norm"))),
                    rs.get("attention_factor", 0.0) if yarn else 0.0, cfg.get("sliding_window") or 0,
                    int(cfg.get("layer_flags") is not None), float(cfg.get("rope_local_theta") or 0.0))
    layout = []
    with open(path, "wb") as f:
        f.write(hdr.ljust(HEADER_SIZE, b"\0"))
        if cfg.get("layer_flags") is not None:   # n_layers bytes, padded like every array
            flags = bytes(int(v) for v in cfg["layer_flags"])
            assert len(flags) == cfg["n_layers"]
            f.write(flags + b"\0" * (-len(flags) % ALIGN))

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
            imp = imatrix_for(imatrix, name, shared) if imatrix else None
            q, d = quantize_q8(arr) if t == WTYPES["q8"] else quantize_q4(arr, imp)
            put(name + ".q", q)
            put(name + ".d", d)
    return layout


# magic, version, n_layers, dim, hidden_dim, wo input size if not dim (else 0),
# tokens, classifier tokens
_IMAT = struct.Struct("<4sIIIIIQQ")
_IMAT_INPUT = {"wq": "qkv", "wk": "qkv", "wv": "qkv", "wo": "wo", "w1": "ffn", "w3": "ffn", "w2": "down"}


def read_imatrix(path, cfg):
    """ml_run --imatrix file -> {"layers": [{input: mean square [K]}], "cls": [dim]},
    inputs qkv, ffn (dim), wo (n_heads * head_dim) and down (hidden_dim)."""
    data = open(path, "rb").read()
    if len(data) < _IMAT.size:
        raise ValueError(f"{path}: not an importance matrix")
    magic, ver, nl, dim, hid, qd, ntok, ncls = _IMAT.unpack_from(data)
    if magic != b"MLI1" or ver != 1:
        raise ValueError(f"{path}: not an importance matrix")
    qd = qd or dim
    want_qd = cfg["n_heads"] * head_dim(cfg)
    if (nl, dim, hid, qd) != (cfg["n_layers"], cfg["dim"], cfg["hidden_dim"], want_qd):
        raise ValueError(f"{path}: for a model with {nl} layers, dim {dim}, hidden_dim {hid}, attention width {qd}")
    if not ntok:
        raise ValueError(f"{path}: no tokens")
    s = np.frombuffer(data, np.float64, offset=_IMAT.size)
    per = 2 * dim + qd + hid
    if len(s) != nl * per + dim:
        raise ValueError(f"{path}: wrong size for this model")
    if ncls > ntok or not np.all(np.isfinite(s) & (s >= 0)):
        raise ValueError(f"{path}: damaged (negative or non-finite sums)")
    layers = []
    for i in range(nl):
        o = i * per
        layers.append({"qkv": s[o:o + dim] / ntok, "wo": s[o + dim:o + dim + qd] / ntok,
                       "ffn": s[o + dim + qd:o + 2 * dim + qd] / ntok,
                       "down": s[o + 2 * dim + qd:o + per] / ntok})
    return {"layers": layers, "cls": s[-dim:] / ncls if ncls else None, "tokens": ntok}


def imatrix_for(imatrix, name, shared=False):
    """Importance of the input channels of matrix `name` (write_mlb's names), or None.
    A tied embedding (shared) is the classifier too and takes its importance;
    otherwise the embedding is only a lookup and keeps the plain fit."""
    if name == "wcls" or (name == "tok_emb" and shared):
        return imatrix["cls"]
    parts = name.split(".")
    if parts[0] == "layers":
        return imatrix["layers"][int(parts[1])][_IMAT_INPUT[parts[2]]]
    return None


def read_mlb(path):
    """-> (cfg dict, tensors) with tensors shaped like write_mlb's input; matrices are QT."""
    data = np.fromfile(path, np.uint8)
    f = _HDR.unpack(data[:_HDR.size].tobytes())
    if f[0] != b"MLB1":
        raise ValueError("bad magic")
    keys = ("version", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim", "vocab",
            "max_seq", "rope_theta", "norm_eps", "wtype", "shared_classifier")
    cfg = dict(zip(keys, f[1:13]))
    (rtype, rf, rlo, rhi, rorig, cfg["q8_mask"], cfg["qkv_bias"], cfg["head_dim"], cfg["qk_norm"],
     rattn, cfg["sliding_window"], has_flags, cfg["rope_local_theta"]) = f[13:]
    rname = {v: k for k, v in ROPE_TYPES.items()}[rtype]
    cfg["rope_scaling"] = None if rtype == 0 else (
        {"type": "linear", "factor": rf} if rname == "linear" else
        {"type": "yarn", "factor": rf, "beta_fast": rlo, "beta_slow": rhi,
         "original_max_position_embeddings": rorig, "attention_factor": rattn} if rname == "yarn" else
        {"type": rname, "factor": rf, "low_freq_factor": rlo, "high_freq_factor": rhi,
         "original_max_position_embeddings": rorig})
    cfg["rope_theta"], cfg["norm_eps"] = np.float32(cfg["rope_theta"]), np.float32(cfg["norm_eps"])
    dim, hid, voc = cfg["dim"], cfg["hidden_dim"], cfg["vocab"]
    hd = head_dim(cfg)
    qd, kv = hd * cfg["n_heads"], hd * cfg["n_kv_heads"]
    shapes = {"tok_emb": (voc, dim), "wq": (qd, dim), "wk": (kv, dim), "wv": (kv, dim),
              "wo": (dim, qd), "w1": (hid, dim), "w2": (dim, hid), "w3": (hid, dim),
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

    cfg["layer_flags"] = [int(v) for v in take(np.uint8, cfg["n_layers"])] if has_flags else None
    t = {"tok_emb": mat("tok_emb"), "layers": []}
    for _ in range(cfg["n_layers"]):
        layer = {}
        for k in LAYER_KEYS:
            layer[k] = take(np.float32, dim) if k.endswith("norm") else mat(k)
            if k == "wo" and cfg["qkv_bias"]:
                layer["bq"], layer["bk"], layer["bv"] = take(np.float32, qd), take(np.float32, kv), take(np.float32, kv)
            if k == "wo" and cfg["qk_norm"]:
                layer["q_norm"], layer["k_norm"] = take(np.float32, hd), take(np.float32, hd)
        t["layers"].append(layer)
    t["final_norm"] = take(np.float32, dim)
    t["wcls"] = t["tok_emb"] if cfg["shared_classifier"] else mat("wcls")
    return cfg, t
