"""Generates a random-weight .mlb model for tests."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402


def generate(cfg, seed=0):
    """cfg: dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, max_seq, rope_theta,
    norm_eps -> (cfg, weights) with float32 weights scaled by 1/sqrt(fan_in)."""
    rng = np.random.default_rng(seed)
    dim, hid = cfg["dim"], cfg["hidden_dim"]
    kv = dim // cfg["n_heads"] * cfg["n_kv_heads"]

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm():
        return (1.0 + 0.01 * rng.standard_normal(dim)).astype(np.float32)

    w = {"tok_emb": (0.5 * rng.standard_normal((cfg["vocab"], dim))).astype(np.float32), "layers": []}
    for _ in range(cfg["n_layers"]):
        w["layers"].append({"attn_norm": norm(), "wq": mat(dim, dim), "wk": mat(kv, dim),
                            "wv": mat(kv, dim), "wo": mat(dim, dim), "ffn_norm": norm(),
                            "w1": mat(hid, dim), "w2": mat(dim, hid), "w3": mat(hid, dim)})
    w["final_norm"] = norm()
    w["wcls"] = mat(cfg["vocab"], dim)
    cfg = dict(cfg, shared_classifier=0)
    return cfg, w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wtype", choices=("q8", "q4"), default="q8")
    ap.add_argument("--q8", default="", metavar="LIST",
                    help="q4 only: matrix classes kept in Q8, comma-separated from "
                         "emb, cls, qk, v, o, up, down (e.g. emb,cls,v,down)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rope-scaling", default=None, metavar="F,LO,HI,ORIG",
                    help="llama3 RoPE scaling: factor, low/high freq factors, original context")
    ap.add_argument("--dump-emb-row", type=int, default=None,
                    help="also write <out>.emb<N>.f32: dequantized embedding row N")
    a = ap.parse_args()
    cfg = dict(dim=a.dim, n_layers=a.layers, n_heads=a.heads, n_kv_heads=a.kv_heads,
               hidden_dim=a.hidden, vocab=a.vocab, max_seq=a.max_seq,
               rope_theta=10000.0, norm_eps=1e-5)
    if a.rope_scaling:
        f, lo, hi, orig = a.rope_scaling.split(",")
        cfg["rope_scaling"] = {"type": "llama3", "factor": float(f), "low_freq_factor": float(lo),
                               "high_freq_factor": float(hi), "original_max_position_embeddings": int(orig)}
    cfg, w = generate(cfg, a.seed)
    mlb.write_mlb(a.out, cfg, w, a.wtype, [c for c in a.q8.split(",") if c])
    if a.dump_emb_row is not None:
        _, t = mlb.read_mlb(a.out)
        row = mlb.dequant(t["tok_emb"])[a.dump_emb_row].astype(np.float32)
        row.tofile(f"{a.out}.emb{a.dump_emb_row}.f32")


if __name__ == "__main__":
    main()
