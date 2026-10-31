"""Converts checkpoints to .mlb.

  convert.py llama2c IN.bin OUT.mlb [--wtype q8|q4] [--q8 LIST]   llama2.c legacy (v0) export
  convert.py hf DIR OUT.mlb [--wtype q8|q4] [--q8 LIST] [--max-seq N] [--scales f16|f32]
                                                      HF Llama, Mistral, Qwen2, Qwen3, Gemma 3 (text): config.json + *.safetensors
  --imatrix FILE (q4): weight the Q4 fits by an importance matrix from
                       ml_run --imatrix, run on a Q8 conversion of the same model
  --q4-rule round (q4): llama.cpp's Q4_0 rounding instead of the fit, for
                       quantization-aware checkpoints trained for Q4_0 (Gemma 3 QAT)
"""
import argparse
import glob
import json
import math
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


HF_TYPES = ("llama", "mistral", "qwen2", "qwen3", "gemma3_text")


def _window(c):
    """-> (window, per-layer sliding flags or None when every layer slides); (None, None) without a window."""
    w = c.get("sliding_window")
    if not w:
        return None, None
    n = c["num_hidden_layers"]
    if c.get("layer_types"):
        sliding = [t == "sliding_attention" for t in c["layer_types"]]
    elif c.get("model_type") == "gemma3_text":   # every sliding_window_pattern-th layer attends to all
        pat = c.get("sliding_window_pattern", 6)
        sliding = [(i + 1) % pat != 0 for i in range(n)]
    elif c.get("model_type") in ("qwen2", "qwen3"):   # layers from max_window_layers on, if enabled
        sliding = [bool(c.get("use_sliding_window")) and i >= c.get("max_window_layers", n) for i in range(n)]
    else:
        sliding = [True] * n
    if not any(sliding):
        return None, None
    return int(w), (None if all(sliding) else sliding)


def _rope_scaling(c, hd):
    """config.json rope_scaling -> write_mlb's rope_scaling, or None; refuses unknown types."""
    rs = c.get("rope_scaling")
    if not rs:
        return None
    kind = rs.get("rope_type", rs.get("type"))
    if rs.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    if kind == "llama3":
        return {"type": "llama3", "factor": float(rs["factor"]),
                "low_freq_factor": float(rs["low_freq_factor"]),
                "high_freq_factor": float(rs["high_freq_factor"]),
                "original_max_position_embeddings": int(rs["original_max_position_embeddings"])}
    if kind == "linear":
        return {"type": "linear", "factor": float(rs["factor"])}
    if kind == "yarn":   # as transformers' _compute_yarn_parameters
        if not rs.get("truncate", True):
            raise ValueError("yarn with truncate=false is not supported")
        orig = int(rs.get("original_max_position_embeddings") or c["max_position_embeddings"])
        f = float(rs.get("factor") or c["max_position_embeddings"] / orig)

        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        af = rs.get("attention_factor")
        if af is None:
            ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
            af = mscale(f, ms) / mscale(f, msa) if ms and msa else mscale(f)
        return {"type": "yarn", "factor": f, "beta_fast": float(rs.get("beta_fast") or 32),
                "beta_slow": float(rs.get("beta_slow") or 1), "original_max_position_embeddings": orig,
                "attention_factor": float(af)}
    raise ValueError(f"rope_scaling {rs} is not supported (llama3, linear and yarn are): "
                     "the converted model would compute wrong outputs")


def load_hf(d, max_seq=None):
    """-> (cfg, float32 weights) from a HF Llama, Mistral, Qwen2, Qwen3 or Gemma 3 (text) directory."""
    with open(os.path.join(d, "config.json")) as fh:
        c = json.load(fh)
    mtype = c.get("model_type", "llama")
    if mtype not in HF_TYPES:
        raise ValueError(f"model_type {mtype!r} is not supported (only {', '.join(HF_TYPES)})")
    gemma = mtype == "gemma3_text"
    act = c.get("hidden_activation", c.get("hidden_act", "silu"))
    if act not in ("silu", "gelu_pytorch_tanh"):
        raise ValueError(f"activation {act!r} is not supported (silu and gelu_pytorch_tanh are)")
    if c.get("final_logit_softcapping") or c.get("attn_logit_softcapping"):
        raise ValueError("logit soft-capping is not supported")
    if c.get("mlp_bias"):
        raise ValueError("MLP biases are not supported")
    if c.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    nh = c["num_attention_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // nh
    rope_scaling = _rope_scaling(c, hd)
    t = {}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    for f in files:
        t.update(read_safetensors(f))
    nkv = c.get("num_key_value_heads", nh)
    tied = c.get("tie_word_embeddings", False) or "lm_head.weight" not in t
    seq = min(c["max_position_embeddings"], max_seq or c["max_position_embeddings"])
    window, sliding = _window(c)
    if window and window >= seq:   # the window never cuts anything off
        window, sliding = None, None
    qk_norm = "model.layers.0.self_attn.q_norm.weight" in t
    if (t["model.layers.0.self_attn.q_proj.weight"].shape[0] != nh * hd or
            t["model.layers.0.self_attn.k_proj.weight"].shape[0] != nkv * hd):
        raise ValueError(f"q/k projections do not match {nh} and {nkv} heads of {hd}")
    if any(k.endswith(("o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"))
           for k in t):
        raise ValueError("biases other than on the q/k/v projections are not supported")
    bias = "model.layers.0.self_attn.q_proj.bias" in t
    cfg = dict(dim=c["hidden_size"], n_layers=c["num_hidden_layers"], n_heads=nh, n_kv_heads=nkv,
               hidden_dim=c["intermediate_size"], vocab=c["vocab_size"], max_seq=seq, qkv_bias=int(bias),
               rope_theta=float(c.get("rope_theta", 10000.0)), norm_eps=float(c.get("rms_norm_eps", 1e-5)),
               shared_classifier=int(tied), rope_scaling=rope_scaling, head_dim=hd, qk_norm=int(qk_norm),
               sliding_window=window or 0,
               layer_flags=[mlb.LAYER_SLIDING if sl else 0 for sl in sliding] if sliding else None,
               act="gelu_tanh" if act == "gelu_pytorch_tanh" else "silu")
    if gemma:
        # sliding layers rotate with the local base; embeddings scaled by
        # sqrt(dim); scores by query_pre_attn_scalar^-0.5; norms of each block's
        # output. Gemma's RMSNorm multiplies by (1 + w): the +1 goes into the
        # stored weights.
        n = cfg["n_layers"]
        local = ([t == "sliding_attention" for t in c["layer_types"]] if c.get("layer_types") else
                 [(i + 1) % c.get("sliding_window_pattern", 6) != 0 for i in range(n)])
        # the local layers keep their RoPE base even when the context fits the window
        cfg.update(layer_flags=[(mlb.LAYER_LOCAL_ROPE | (mlb.LAYER_SLIDING if window else 0)) if x else 0
                                for x in local],
                   rope_local_theta=float(c.get("rope_local_base_freq", 10000.0)),
                   embed_scale=float(np.float32(math.sqrt(c["hidden_size"]))),
                   attn_scale=float(c.get("query_pre_attn_scalar", hd)) ** -0.5, post_norms=1)
    norm = (lambda v: v + 1.0) if gemma else (lambda v: v)
    w = {"tok_emb": t["model.embed_tokens.weight"], "final_norm": norm(t["model.norm.weight"]), "layers": []}
    for i in range(cfg["n_layers"]):
        p = f"model.layers.{i}."
        w["layers"].append({
            "attn_norm": norm(t[p + "input_layernorm.weight"]),
            "wq": _unrotate(t[p + "self_attn.q_proj.weight"], nh),
            "wk": _unrotate(t[p + "self_attn.k_proj.weight"], nkv),
            "wv": t[p + "self_attn.v_proj.weight"],
            "wo": t[p + "self_attn.o_proj.weight"],
            "ffn_norm": norm(t[p + ("pre_feedforward_layernorm" if gemma else "post_attention_layernorm") + ".weight"]),
            "w1": t[p + "mlp.gate_proj.weight"],
            "w2": t[p + "mlp.down_proj.weight"],
            "w3": t[p + "mlp.up_proj.weight"],
        })
        if bias:   # q/k biases follow their rows through the RoPE permutation
            w["layers"][-1].update(bq=_unrotate(t[p + "self_attn.q_proj.bias"][:, None], nh)[:, 0],
                                   bk=_unrotate(t[p + "self_attn.k_proj.bias"][:, None], nkv)[:, 0],
                                   bv=t[p + "self_attn.v_proj.bias"])
        if qk_norm:   # per-head weights, permuted as each head's q/k rows are
            w["layers"][-1].update(q_norm=norm(_unrotate(t[p + "self_attn.q_norm.weight"][:, None], 1)[:, 0]),
                                   k_norm=norm(_unrotate(t[p + "self_attn.k_norm.weight"][:, None], 1)[:, 0]))
        if gemma:   # Gemma's post_attention_layernorm normalizes the attention output
            w["layers"][-1].update(post_attn_norm=norm(t[p + "post_attention_layernorm.weight"]),
                                   post_ffn_norm=norm(t[p + "post_feedforward_layernorm.weight"]))
    if not tied:
        w["wcls"] = t["lm_head.weight"]
    return cfg, w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("llama2c", "hf"))
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--wtype", choices=("q8", "q4"), default="q8")
    ap.add_argument("--q8", default="", metavar="LIST",
                    help="q4 only: matrix classes kept in Q8, comma-separated from "
                         "emb, cls, qk, v, o, up, down (e.g. emb,cls,v,down)")
    ap.add_argument("--max-seq", type=int, default=None, help="hf: cap the context length (KV cache size)")
    ap.add_argument("--imatrix", metavar="FILE", help="q4 only: importance matrix from ml_run --imatrix")
    ap.add_argument("--q4-rule", choices=("fit", "round"), default="fit",
                    help="q4 only: fit each block's scale (default), or round as llama.cpp's Q4_0 does, "
                         "which quantization-aware checkpoints trained for Q4_0 expect")
    ap.add_argument("--scales", choices=("f16", "f32"), default="f16",
                    help="storage of the block scales (f16: Q4 4.5 bits per weight, Q8 8.5; f32: 5 and 9)")
    a = ap.parse_args(argv)
    if a.imatrix and a.wtype != "q4":
        ap.error("--imatrix needs --wtype q4")
    if a.imatrix and a.q4_rule == "round":
        ap.error("--imatrix weights the fitted Q4 rule only, not --q4-rule round")
    cfg, w = load_llama2c(a.src) if a.kind == "llama2c" else load_hf(a.src, a.max_seq)
    imat = mlb.read_imatrix(a.imatrix, cfg) if a.imatrix else None
    mlb.write_mlb(a.out, cfg, w, a.wtype, [c for c in a.q8.split(",") if c], imat, a.scales, a.q4_rule)
    print(f"wrote {a.out}: dim={cfg['dim']} layers={cfg['n_layers']} vocab={cfg['vocab']} "
          f"max_seq={cfg['max_seq']} wtype={a.wtype}{' (q8: ' + a.q8 + ')' if a.q8 and a.wtype == 'q4' else ''} "
          f"shared_classifier={cfg['shared_classifier']}")
    if imat:
        print(f"imatrix {a.imatrix}: {imat['tokens']} tokens")


if __name__ == "__main__":
    main()
