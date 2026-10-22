"""Converts checkpoints to .mlb.

  convert.py llama2c IN.bin OUT.mlb [--wtype q8|q4]   llama2.c legacy (v0) export
  convert.py hf DIR OUT.mlb [--wtype q8|q4] [--max-seq N]
                                                      HF Llama: config.json + *.safetensors
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402


def load_llama2c(path):
    """-> (cfg, float32 weights) from a llama2.c v0 checkpoint."""
    data = np.fromfile(path, np.uint8)
    dim, hid, nl, nh, nkv, vocab, seq = struct.unpack("<7i", data[:28].tobytes())
    shared = vocab > 0
    vocab = abs(vocab)
    hd = dim // nh
    kv = hd * nkv
    f = data[28:].view(np.float32)
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        if off + n > f.size:
            raise ValueError("checkpoint truncated")
        a = f[off:off + n].reshape(shape)
        off += n
        return a

    tok = take(vocab, dim)
    stacks = {"attn_norm": take(nl, dim), "wq": take(nl, dim, dim), "wk": take(nl, kv, dim),
              "wv": take(nl, kv, dim), "wo": take(nl, dim, dim), "ffn_norm": take(nl, dim),
              "w1": take(nl, hid, dim), "w2": take(nl, dim, hid), "w3": take(nl, hid, dim)}
    final = take(dim)
    take(seq * hd // 2)   # freq_cis_real (recomputed at load)
    take(seq * hd // 2)   # freq_cis_imag
    w = {"tok_emb": tok, "final_norm": final,
         "layers": [{k: v[i] for k, v in stacks.items()} for i in range(nl)]}
    if not shared:
        w["wcls"] = take(vocab, dim)
    cfg = dict(dim=dim, n_layers=nl, n_heads=nh, n_kv_heads=nkv, hidden_dim=hid, vocab=vocab,
               max_seq=seq, rope_theta=10000.0, norm_eps=1e-5, shared_classifier=int(shared))
    return cfg, w


_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def read_safetensors(path):
    """-> {name: float32 array}. Supports F32, F16, BF16."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
    raw = np.memmap(path, np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        a, b = info["data_offsets"]
        arr = np.frombuffer(raw[base + a:base + b], _DTYPES[info["dtype"]]).reshape(info["shape"])
        if info["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        out[name] = arr.astype(np.float32)
    return out


def _unrotate(w, n_heads):
    """HF rotate_half q/k rows -> interleaved-pair rows (llama2.c convention)."""
    rows, dim = w.shape
    hd = rows // n_heads
    return w.reshape(n_heads, 2, hd // 2, dim).transpose(0, 2, 1, 3).reshape(rows, dim)


def load_hf(d, max_seq=None):
    """-> (cfg, float32 weights) from a HF Llama directory."""
    with open(os.path.join(d, "config.json")) as fh:
        c = json.load(fh)
    rope_scaling = None
    rs = c.get("rope_scaling")
    if rs:
        kind = rs.get("rope_type", rs.get("type"))
        if kind != "llama3":
            raise ValueError(f"rope_scaling {rs} is not supported (only llama3): "
                             "the converted model would compute wrong outputs")
        rope_scaling = {"type": "llama3", "factor": float(rs["factor"]),
                        "low_freq_factor": float(rs["low_freq_factor"]),
                        "high_freq_factor": float(rs["high_freq_factor"]),
                        "original_max_position_embeddings": int(rs["original_max_position_embeddings"])}
    t = {}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    for f in files:
        t.update(read_safetensors(f))
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh)
    tied = c.get("tie_word_embeddings", False) or "lm_head.weight" not in t
    cfg = dict(dim=c["hidden_size"], n_layers=c["num_hidden_layers"], n_heads=nh, n_kv_heads=nkv,
               hidden_dim=c["intermediate_size"], vocab=c["vocab_size"],
               max_seq=min(c["max_position_embeddings"], max_seq or c["max_position_embeddings"]),
               rope_theta=float(c.get("rope_theta", 10000.0)), norm_eps=float(c.get("rms_norm_eps", 1e-5)),
               shared_classifier=int(tied), rope_scaling=rope_scaling)
    w = {"tok_emb": t["model.embed_tokens.weight"], "final_norm": t["model.norm.weight"], "layers": []}
    for i in range(cfg["n_layers"]):
        p = f"model.layers.{i}."
        w["layers"].append({
            "attn_norm": t[p + "input_layernorm.weight"],
            "wq": _unrotate(t[p + "self_attn.q_proj.weight"], nh),
            "wk": _unrotate(t[p + "self_attn.k_proj.weight"], nkv),
            "wv": t[p + "self_attn.v_proj.weight"],
            "wo": t[p + "self_attn.o_proj.weight"],
            "ffn_norm": t[p + "post_attention_layernorm.weight"],
            "w1": t[p + "mlp.gate_proj.weight"],
            "w2": t[p + "mlp.down_proj.weight"],
            "w3": t[p + "mlp.up_proj.weight"],
        })
    if not tied:
        w["wcls"] = t["lm_head.weight"]
    return cfg, w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("llama2c", "hf"))
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--wtype", choices=("q8", "q4"), default="q8")
    ap.add_argument("--max-seq", type=int, default=None, help="hf: cap the context length (KV cache size)")
    a = ap.parse_args(argv)
    cfg, w = load_llama2c(a.src) if a.kind == "llama2c" else load_hf(a.src, a.max_seq)
    mlb.write_mlb(a.out, cfg, w, a.wtype)
    print(f"wrote {a.out}: dim={cfg['dim']} layers={cfg['n_layers']} vocab={cfg['vocab']} "
          f"max_seq={cfg['max_seq']} wtype={a.wtype} shared_classifier={cfg['shared_classifier']}")


if __name__ == "__main__":
    main()
