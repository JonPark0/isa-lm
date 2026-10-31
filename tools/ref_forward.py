"""Numpy reference forward pass with the runtime's exact quantized semantics.

Validates the scalar C build: activations are re-quantized to Q8_0 before every
matvec, block dot products are integer, scales are applied in float.
Written for clarity, not speed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402

QK = mlb.QK


def _int_weights(t):
    """QT -> int32 array [rows, nblocks, 32]."""
    rows, cols = t.shape
    if t.type == mlb.WTYPES["q8"]:
        return t.q.reshape(rows, cols // QK, QK).astype(np.int32)
    p = t.q.reshape(rows, cols // QK, QK // 2).astype(np.int32)
    return np.concatenate([(p & 0xF) - 8, (p >> 4) - 8], axis=-1)


class Model:
    def __init__(self, cfg, t):
        self.cfg, self.t = cfg, t
        self._cache = {}

    def matvec(self, w, x):
        """w: QT, x: float32[cols] -> float32[rows]."""
        key = id(w)
        if key not in self._cache:
            self._cache[key] = (_int_weights(w), w.d.astype(np.float64))
        wi, wd = self._cache[key]
        q, d = mlb.quantize_q8(x[None, :])
        xi = q[0].reshape(-1, QK).astype(np.int32)
        sums = np.einsum("rbj,bj->rb", wi, xi)
        return (sums * wd * d[0].astype(np.float64)).sum(axis=1).astype(np.float32)


def quantize_rows(x, n):
    """x[..., k] float32 -> (int8 q, float32 d) per row of n, the quantize_q8 rule
    with one block of n (the runtime's Q8 KV cache and quantized query heads)."""
    x = np.asarray(x, np.float32)
    b = x.reshape(-1, n)
    amax = np.abs(b).max(axis=-1)
    amax = np.where(amax < mlb.TINY, np.float32(0), amax).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
    d = (amax / np.float32(127)).astype(np.float32)
    q = np.clip(mlb._round_half_away((b * idv[:, None]).astype(np.float32)), -127, 127)
    return q.astype(np.int8), d


def rmsnorm(x, w, eps):
    x64 = x.astype(np.float64)
    return (x64 / np.sqrt(np.mean(x64 * x64) + eps) * w).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def rope_tables(cfg, local=False):
    if local:   # the local RoPE base, without scaling (Gemma 3's sliding layers)
        cfg = dict(cfg, rope_theta=cfg["rope_local_theta"], rope_scaling=None)
    freq, af = mlb.rope_inv_freq(cfg), mlb.rope_attention_factor(cfg)
    ang = np.arange(cfg["max_seq"], dtype=np.float64)[:, None] * freq[None, :]
    return (np.cos(ang) * af).astype(np.float32), (np.sin(ang) * af).astype(np.float32)


def rope(v, c, s, hd):
    v = v.reshape(-1, hd // 2, 2).astype(np.float64)
    a, b = v[..., 0], v[..., 1]
    out = np.stack([a * c - b * s, a * s + b * c], axis=-1)
    return out.reshape(-1).astype(np.float32)


def forward(cfg, t, tokens, kv="f32"):
    """-> float32 logits [len(tokens), vocab], position p attending to 0..p
    (to the last sliding_window positions, if set).
    kv="q8" emulates the runtime's Q8 KV cache (--kv q8)."""
    m = Model(cfg, t)
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd, eps = mlb.head_dim(cfg), float(cfg["norm_eps"])
    local = rope_tables(cfg, True) if cfg.get("rope_local_theta") else None
    group = nh // nkv
    cos_t, sin_t = rope_tables(cfg)
    emb = mlb.dequant(t["tok_emb"])
    escale = np.float32(cfg.get("embed_scale") or 1.0)
    ascale = float(cfg.get("attn_scale") or 0.0) or 1.0 / np.sqrt(hd)
    acap, fcap = float(cfg.get("attn_softcap") or 0.0), float(cfg.get("final_softcap") or 0.0)
    gelu = cfg.get("act") == "gelu_tanh"
    kcache = [[] for _ in t["layers"]]
    vcache = [[] for _ in t["layers"]]
    out = []
    for pos, tok in enumerate(tokens):
        x = (emb[tok].astype(np.float32) * escale).astype(np.float32)
        for li, L in enumerate(t["layers"]):
            win = mlb.layer_window(cfg, li)
            ct, st = local if mlb.layer_local_rope(cfg, li) else (cos_t, sin_t)
            h = rmsnorm(x, L["attn_norm"], eps)
            q = m.matvec(L["wq"], h)
            k = m.matvec(L["wk"], h)
            v = m.matvec(L["wv"], h)
            if "bq" in L:   # float32 adds, as the runtime does
                q, k, v = q + L["bq"], k + L["bk"], v + L["bv"]
            if "q_norm" in L:   # per head, before RoPE (Qwen3)
                q = np.concatenate([rmsnorm(q[i:i + hd], L["q_norm"], eps) for i in range(0, len(q), hd)])
                k = np.concatenate([rmsnorm(k[i:i + hd], L["k_norm"], eps) for i in range(0, len(k), hd)])
            q = rope(q, ct[pos], st[pos], hd)
            k = rope(k, ct[pos], st[pos], hd)
            if kv == "q8":   # per position and KV head: int8 quants, one scale
                kq, kd = quantize_rows(k, hd)
                vq, vd = quantize_rows(v, hd)
                k = (kq.astype(np.float64) * kd[:, None].astype(np.float64)).reshape(-1)
                v = (vq.astype(np.float64) * vd[:, None].astype(np.float64)).reshape(-1)
                qq, qd = quantize_rows(q, hd)
                q = (qq.astype(np.float64) * qd[:, None].astype(np.float64)).reshape(-1)
            kcache[li].append(k)
            vcache[li].append(v)
            first = max(0, pos + 1 - win) if win else 0
            K = np.stack(kcache[li][first:]).astype(np.float64)   # [positions attended, kv_dim]
            V = np.stack(vcache[li][first:]).astype(np.float64)
            o = np.zeros(nh * hd, np.float32)
            for hh in range(nh):
                kh = hh // group
                qh = np.asarray(q[hh * hd:(hh + 1) * hd], np.float64)
                sc = K[:, kh * hd:(kh + 1) * hd] @ qh * ascale
                if acap:   # Gemma 2: scores soft-capped at acap
                    sc = acap * np.tanh(sc / acap)
                p = softmax(sc)
                o[hh * hd:(hh + 1) * hd] = p @ V[:, kh * hd:(kh + 1) * hd]
            a = m.matvec(L["wo"], o)
            if "post_attn_norm" in L:   # Gemma 3: the block's output is normalized before the residual
                a = rmsnorm(a, L["post_attn_norm"], eps)
            x = (x + a).astype(np.float32)
            h = rmsnorm(x, L["ffn_norm"], eps)
            g = m.matvec(L["w1"], h).astype(np.float64)
            u = m.matvec(L["w3"], h).astype(np.float64)
            if gelu:   # tanh approximation (gelu_pytorch_tanh)
                g = (0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u).astype(np.float32)
            else:
                g = (g / (1.0 + np.exp(-g)) * u).astype(np.float32)
            f = m.matvec(L["w2"], g)
            if "post_ffn_norm" in L:
                f = rmsnorm(f, L["post_ffn_norm"], eps)
            x = (x + f).astype(np.float32)
        lg = m.matvec(t["wcls"], rmsnorm(x, t["final_norm"], eps))
        if fcap:   # Gemma 2: logits soft-capped at fcap
            lg = (fcap * np.tanh(lg.astype(np.float64) / fcap)).astype(np.float32)
        out.append(lg)
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True, help="raw float32 [n_tokens, vocab]")
    ap.add_argument("--kv", choices=["f32", "q8"], default="f32", help="KV cache type to emulate")
    a = ap.parse_args()
    cfg, t = mlb.read_mlb(a.model)
    forward(cfg, t, [int(s) for s in a.tokens.split(",")], a.kv).tofile(a.out)


if __name__ == "__main__":
    main()
