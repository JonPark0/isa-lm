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
_HDR = struct.Struct("<4sIIIIIIIIffII")   # magic .. shared_classifier, 52 bytes
LAYER_KEYS = ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")

QT = namedtuple("QT", "type q d shape")   # quantized tensor as stored


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
    yield "final_norm", "f", w["final_norm"]
    if not cfg.get("shared_classifier", 0):
        yield "wcls", "q", w["wcls"]


def _check_cfg(cfg):
    for k in ("dim", "hidden_dim"):
        if cfg[k] % QK:
            raise ValueError(f"{k}={cfg[k]} must be a multiple of {QK}")


def write_mlb(path, cfg, w, wtype):
    """Writes the model; returns [(array name, offset, nbytes)] for every array."""
    _check_cfg(cfg)
    hdr = _HDR.pack(b"MLB1", 1, cfg["dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"],
                    cfg["hidden_dim"], cfg["vocab"], cfg["max_seq"], cfg["rope_theta"],
                    cfg["norm_eps"], WTYPES[wtype], int(cfg.get("shared_classifier", 0)))
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
            q, d = quantize_q8(arr) if wtype == "q8" else quantize_q4(arr)
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
    cfg = dict(zip(keys, f[1:]))
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
        nq = r * c if cfg["wtype"] == 1 else r * c // 2
        q = take(np.int8 if cfg["wtype"] == 1 else np.uint8, nq)
        d = take(np.float32, r * c // QK).reshape(r, c // QK)
        return QT(cfg["wtype"], q.reshape(r, -1), d, (r, c))

    t = {"tok_emb": mat("tok_emb"), "layers": []}
    for _ in range(cfg["n_layers"]):
        layer = {}
        for k in LAYER_KEYS:
            layer[k] = take(np.float32, dim) if k.endswith("norm") else mat(k)
        t["layers"].append(layer)
    t["final_norm"] = take(np.float32, dim)
    t["wcls"] = t["tok_emb"] if cfg["shared_classifier"] else mat("wcls")
    return cfg, t
