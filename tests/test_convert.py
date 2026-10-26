"""Tests for the Python tools: .mlb writer/reader, reference forward, converter."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import gen_random  # noqa: E402
import mlb  # noqa: E402

SMALL = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=128,
             vocab=96, max_seq=32, rope_theta=10000.0, norm_eps=1e-5)


def _gen(tmp_path, wtype, seed=0):
    cfg, w = gen_random.generate(dict(SMALL), seed)
    path = str(tmp_path / f"m_{wtype}.mlb")
    layout = mlb.write_mlb(path, cfg, w, wtype)
    return path, cfg, w, layout


def test_random_model_layout(tmp_path):
    for wtype in ("q8", "q4"):
        path, cfg, w, layout = _gen(tmp_path, wtype)
        rcfg, t = mlb.read_mlb(path)
        for k, v in SMALL.items():
            assert rcfg[k] == (np.float32(v) if isinstance(v, float) else v), k
        assert rcfg["wtype"] == mlb.WTYPES[wtype]
        assert rcfg["shared_classifier"] == 0
        for name, off, nbytes in layout:
            assert off % 64 == 0, name
        end = layout[-1][1] + layout[-1][2]
        assert os.path.getsize(path) == (end + 63) // 64 * 64
        with open(path, "rb") as f:
            assert f.read(4) == b"MLB1"
        # dequantized weights within half a quantization step of the originals
        # (Q8); Q4 clips the opposite extreme of a block at level 7, about two
        # steps from where it would round
        for name, orig in (("tok_emb", w["tok_emb"]), ("wq0", w["layers"][0]["wq"]),
                           ("w2_1", w["layers"][1]["w2"])):
            qt = t["tok_emb"] if name == "tok_emb" else t["layers"][int(name[-1])][name[:2]]
            deq = mlb.dequant(qt)
            step = np.abs(np.repeat(qt.d, 32, axis=-1)) * (0.5 if wtype == "q8" else 2.1)
            assert deq.shape == orig.shape
            assert np.all(np.abs(deq - orig) <= step + 1e-6), name
        np.testing.assert_array_equal(t["layers"][0]["attn_norm"], w["layers"][0]["attn_norm"])


def test_quantize_q8_rounding():
    x = np.zeros((1, 32), np.float32)
    x[0, :6] = [127.0, 0.5, -0.5, 1.5, 2.5, -2.5]
    q, d = mlb.quantize_q8(x)
    assert d[0, 0] == 1.0
    assert list(q[0, :6]) == [127, 1, -1, 2, 3, -3]
    z, dz = mlb.quantize_q8(np.zeros((1, 32), np.float32))
    assert dz[0, 0] == 0 and not z.any()


def test_quantize_q4_layout():
    x = np.zeros((1, 32), np.float32)
    x[0, 0], x[0, 16] = 7.0, -7.0
    p, d = mlb.quantize_q4(x)
    lo, hi = (int(p[0, 0]) & 15) - 8, (int(p[0, 0]) >> 4) - 8   # elements 0 and 16
    # levels -7/7 with d = -1 and -6/6 with d = -7/6 fit exactly: either may win
    np.testing.assert_allclose([lo * d[0, 0], hi * d[0, 0]], [7.0, -7.0], rtol=1e-6)
    np.testing.assert_allclose(mlb.dequant(mlb.QT(mlb.WTYPES["q4"], p, d, x.shape)), x, atol=1e-5)


def test_quantize_q4_error():
    """All 16 levels and a fitted scale: lower error than amax/7 rounding."""
    x = np.random.default_rng(3).standard_normal((64, 256)).astype(np.float32)
    p, d = mlb.quantize_q4(x)
    err = mlb.dequant(mlb.QT(mlb.WTYPES["q4"], p, d, x.shape)) - x
    b = x.reshape(64, -1, 32)
    s = np.abs(b).max(-1, keepdims=True) / 7
    naive = (np.round(b / s) * s).reshape(x.shape) - x
    assert (err ** 2 * x ** 2).sum() < 0.8 * (naive ** 2 * x ** 2).sum()
    assert (err ** 2).sum() < (naive ** 2).sum()
    z = mlb.quantize_q4(np.zeros((2, 64), np.float32))
    assert (z[0] == 0x88).all()   # level 0 is nibble 8
    assert not z[1].any()


import ref_forward  # noqa: E402


def test_ref_forward_shape(tmp_path):
    path, *_ = _gen(tmp_path, "q8")
    cfg, t = mlb.read_mlb(path)
    logits = ref_forward.forward(cfg, t, [1, 5, 9])
    assert logits.shape == (3, SMALL["vocab"])
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))
    assert logits.std() > 1e-3   # not degenerate


def test_ref_forward_causal(tmp_path):
    path, *_ = _gen(tmp_path, "q8")
    cfg, t = mlb.read_mlb(path)
    a = ref_forward.forward(cfg, t, [3, 7])
    b = ref_forward.forward(cfg, t, [3, 7, 11])
    np.testing.assert_array_equal(a, b[:2])
    assert not np.array_equal(b[1], b[2])


def test_ref_forward_q4(tmp_path):
    path, *_ = _gen(tmp_path, "q4")
    cfg, t = mlb.read_mlb(path)
    logits = ref_forward.forward(cfg, t, [2, 4])
    assert logits.shape == (2, SMALL["vocab"]) and np.all(np.isfinite(logits))


def test_ref_forward_cli(tmp_path):
    path, *_ = _gen(tmp_path, "q8")
    out = tmp_path / "l.f32"
    rc = os.system(f"{sys.executable} tools/ref_forward.py {path} --tokens 1,2,3 --out {out}")
    assert rc == 0
    got = np.fromfile(out, np.float32).reshape(3, -1)
    cfg, t = mlb.read_mlb(path)
    np.testing.assert_array_equal(got, ref_forward.forward(cfg, t, [1, 2, 3]))


import json  # noqa: E402
import subprocess  # noqa: E402

import convert  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _write_llama2c(path, cfg, w, shared):
    dim, kv = cfg["dim"], cfg["dim"] // cfg["n_heads"] * cfg["n_kv_heads"]
    hd = dim // cfg["n_heads"]
    vocab = cfg["vocab"] if shared else -cfg["vocab"]
    with open(path, "wb") as f:
        f.write(struct.pack("<7i", dim, cfg["hidden_dim"], cfg["n_layers"], cfg["n_heads"],
                            cfg["n_kv_heads"], vocab, cfg["max_seq"]))
        L = w["layers"]
        parts = [w["tok_emb"]] + [np.stack([l[k] for l in L]) for k in
                                  ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")]
        parts += [w["final_norm"], np.zeros(cfg["max_seq"] * hd, np.float32)]   # freq_cis_real + imag
        if not shared:
            parts.append(w["wcls"])
        for p in parts:
            f.write(np.ascontiguousarray(p, np.float32).tobytes())
    assert kv  # shapes come from the generator


def test_llama2c_roundtrip(tmp_path):
    cfg, w = gen_random.generate(dict(SMALL), seed=3)
    src = str(tmp_path / "m.bin")
    _write_llama2c(src, cfg, w, shared=False)
    out = str(tmp_path / "m.mlb")
    convert.main(["llama2c", src, out, "--wtype", "q8"])
    direct = str(tmp_path / "d.mlb")
    mlb.write_mlb(direct, cfg, w, "q8")
    with open(out, "rb") as a, open(direct, "rb") as b:
        assert a.read() == b.read()   # same weights, same order -> identical file


def _write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _bf16(a):
    """float32 -> (bf16 bits as uint16, the float32 value those bits represent)."""
    bits = (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view(np.float32)


def _hf_rope(v, pos, hd, theta):   # rotate_half convention, per head
    v = v.reshape(-1, hd).astype(np.float64)
    inv = theta ** (-np.arange(0, hd, 2) / hd)
    ang = np.concatenate([pos * inv, pos * inv])
    rot = np.concatenate([-v[:, hd // 2:], v[:, :hd // 2]], axis=1)
    return v * np.cos(ang) + rot * np.sin(ang)


def _interleaved_rope(v, pos, hd, theta):
    v = v.reshape(-1, hd // 2, 2).astype(np.float64)
    ang = pos * theta ** (-2.0 * np.arange(hd // 2) / hd)
    a, b = v[..., 0], v[..., 1]
    return np.stack([a * np.cos(ang) - b * np.sin(ang), a * np.sin(ang) + b * np.cos(ang)], -1).reshape(-1, hd)


def _make_hf(tmp_path, bias=False, **extra):
    cfg, w = gen_random.generate(dict(SMALL, qkv_bias=bias), seed=5)
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd = dim // nh

    def to_hf(m, n_heads):   # interleaved rows -> rotate_half rows (inverse of the converter)
        return m.reshape(n_heads, hd // 2, 2, -1).transpose(0, 2, 1, 3).reshape(-1, m.shape[1])

    t, orig = {}, {}

    def put(name, arr):
        bits, val = _bf16(arr)
        t[name] = ("BF16", bits)
        orig[name] = val

    put("model.embed_tokens.weight", w["tok_emb"])
    for i, L in enumerate(w["layers"]):
        p = f"model.layers.{i}."
        put(p + "input_layernorm.weight", L["attn_norm"])
        put(p + "self_attn.q_proj.weight", to_hf(L["wq"], nh))
        put(p + "self_attn.k_proj.weight", to_hf(L["wk"], nkv))
        put(p + "self_attn.v_proj.weight", L["wv"])
        if bias:
            put(p + "self_attn.q_proj.bias", to_hf(L["bq"][:, None], nh)[:, 0])
            put(p + "self_attn.k_proj.bias", to_hf(L["bk"][:, None], nkv)[:, 0])
            put(p + "self_attn.v_proj.bias", L["bv"])
        put(p + "self_attn.o_proj.weight", L["wo"])
        put(p + "post_attention_layernorm.weight", L["ffn_norm"])
        put(p + "mlp.gate_proj.weight", L["w1"])
        put(p + "mlp.down_proj.weight", L["w2"])
        put(p + "mlp.up_proj.weight", L["w3"])
    put("model.norm.weight", w["final_norm"])
    d = tmp_path / "hf"
    d.mkdir(parents=True)
    _write_safetensors(str(d / "model.safetensors"), t)
    (d / "config.json").write_text(json.dumps({
        "hidden_size": dim, "intermediate_size": cfg["hidden_dim"], "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": nh, "num_key_value_heads": nkv, "vocab_size": cfg["vocab"],
        "max_position_embeddings": cfg["max_seq"], "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, **extra}))
    return str(d), cfg, orig


def test_hf_permutation(tmp_path):
    d, cfg, orig = _make_hf(tmp_path)
    ccfg, w = convert.load_hf(d)
    assert ccfg["shared_classifier"] == 1 and ccfg["vocab"] == cfg["vocab"]
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd = dim // nh
    rng = np.random.default_rng(0)
    x = rng.standard_normal(dim)
    for pos in (0, 3, 17):
        qh = _hf_rope(orig["model.layers.1.self_attn.q_proj.weight"] @ x, pos, hd, 10000.0)
        kh = _hf_rope(orig["model.layers.1.self_attn.k_proj.weight"] @ x, pos + 2, hd, 10000.0)
        qi = _interleaved_rope(w["layers"][1]["wq"] @ x, pos, hd, 10000.0)
        ki = _interleaved_rope(w["layers"][1]["wk"] @ x, pos + 2, hd, 10000.0)
        for h in range(nh):
            np.testing.assert_allclose(qh[h] @ kh[h // (nh // nkv)], qi[h] @ ki[h // (nh // nkv)],
                                       rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(w["layers"][0]["w2"], orig["model.layers.0.mlp.down_proj.weight"])


def _assert_close_to_ref(got, ref):
    """C logits against ref_forward. A Q8 activation that lands across a
    rounding boundary moves later logits by ~1% (tests/test_e2e.sh holds fixed
    models to 1e-3); a wrong conversion garbles them, so 5% is enough here."""
    assert np.abs(got - ref).max() <= 5e-2 * np.abs(ref).max()


def test_hf_tied_model_runs_in_c(tmp_path):
    exe = os.path.join(ROOT, "build", "scalar", "ml_run")
    if not os.path.exists(exe):
        import pytest
        pytest.skip("build/scalar/ml_run not built")
    d, cfg, _ = _make_hf(tmp_path)
    out = str(tmp_path / "hf.mlb")
    convert.main(["hf", d, out, "--wtype", "q4"])
    rcfg, t = mlb.read_mlb(out)
    assert rcfg["shared_classifier"] == 1
    subprocess.run([exe, out, "--tokens", "1,7,3", "--dump-logits", str(tmp_path / "c.f32")], check=True)
    got = np.fromfile(tmp_path / "c.f32", np.float32).reshape(3, -1)
    ref = ref_forward.forward(rcfg, t, [1, 7, 3])
    _assert_close_to_ref(got, ref)


def test_hf_qwen2_biases(tmp_path):
    """Qwen2 q/k biases are permuted with their rows: RoPE'd scores match HF."""
    d, cfg, orig = _make_hf(tmp_path, bias=True, model_type="qwen2", sliding_window=4096,
                            use_sliding_window=False)
    ccfg, w = convert.load_hf(d)
    assert ccfg["qkv_bias"] == 1 and ccfg["max_seq"] == cfg["max_seq"]
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd = dim // nh
    p = "model.layers.1.self_attn."
    L = w["layers"][1]
    x = np.random.default_rng(1).standard_normal(dim)
    for pos in (0, 5):
        qh = _hf_rope(orig[p + "q_proj.weight"] @ x + orig[p + "q_proj.bias"], pos, hd, 10000.0)
        kh = _hf_rope(orig[p + "k_proj.weight"] @ x + orig[p + "k_proj.bias"], pos + 1, hd, 10000.0)
        qi = _interleaved_rope(L["wq"] @ x + L["bq"], pos, hd, 10000.0)
        ki = _interleaved_rope(L["wk"] @ x + L["bk"], pos + 1, hd, 10000.0)
        for h in range(nh):
            np.testing.assert_allclose(qh[h] @ kh[h // (nh // nkv)], qi[h] @ ki[h // (nh // nkv)],
                                       rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(L["bv"], orig[p + "v_proj.bias"])


def test_hf_qwen2_runs_in_c(tmp_path):
    exe = os.path.join(ROOT, "build", "scalar", "ml_run")
    if not os.path.exists(exe):
        import pytest
        pytest.skip("build/scalar/ml_run not built")
    d, _, _ = _make_hf(tmp_path, bias=True, model_type="qwen2")
    out = str(tmp_path / "qwen.mlb")
    convert.main(["hf", d, out])
    rcfg, t = mlb.read_mlb(out)
    assert rcfg["qkv_bias"] == 1
    subprocess.run([exe, out, "--tokens", "1,7,3", "--dump-logits", str(tmp_path / "c.f32")], check=True)
    got = np.fromfile(tmp_path / "c.f32", np.float32).reshape(3, -1)
    ref = ref_forward.forward(rcfg, t, [1, 7, 3])
    _assert_close_to_ref(got, ref)


def test_hf_sliding_window_caps_context(tmp_path):
    d, cfg, _ = _make_hf(tmp_path, model_type="mistral", sliding_window=16)
    assert convert.load_hf(d)[0]["max_seq"] == 16
    d2, _, _ = _make_hf(tmp_path / "q", model_type="qwen2", sliding_window=16, use_sliding_window=True)
    assert convert.load_hf(d2)[0]["max_seq"] == 16


def test_hf_unsupported_models_refused(tmp_path):
    import pytest
    for i, extra in enumerate(({"model_type": "gemma"}, {"hidden_act": "gelu"}, {"head_dim": 4},
                               {"mlp_bias": True})):
        d, _, _ = _make_hf(tmp_path / str(i), **extra)
        with pytest.raises(ValueError):
            convert.load_hf(d)


def test_large_vocab_loads_in_c(tmp_path):
    """Llama-3-sized vocabularies (> 65536) must load and run."""
    exe = os.path.join(ROOT, "build", "scalar", "ml_run")
    if not os.path.exists(exe):
        import pytest
        pytest.skip("build/scalar/ml_run not built")
    cfg, w = gen_random.generate(dict(SMALL, dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
                                      hidden_dim=32, vocab=128256), seed=1)
    path = str(tmp_path / "bigvocab.mlb")
    mlb.write_mlb(path, cfg, w, "q4")
    out = str(tmp_path / "l.f32")
    subprocess.run([exe, path, "--tokens", "1,128255", "--dump-logits", out], check=True)
    got = np.fromfile(out, np.float32).reshape(2, -1)
    rcfg, t = mlb.read_mlb(path)
    ref = ref_forward.forward(rcfg, t, [1, 128255])
    _assert_close_to_ref(got, ref)


def test_hf_rope_scaling_refused(tmp_path):
    d, _, _ = _make_hf(tmp_path)
    c = json.loads((tmp_path / "hf" / "config.json").read_text())
    c["rope_scaling"] = {"rope_type": "yarn", "factor": 32.0}
    (tmp_path / "hf" / "config.json").write_text(json.dumps(c))
    import pytest
    with pytest.raises(ValueError, match="rope_scaling"):
        convert.load_hf(d)


LLAMA3 = {"type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def _hf_llama3_inv_freq(theta, hd, rs):
    """transformers' _compute_llama3_parameters, transcribed."""
    import math
    inv = 1.0 / theta ** (np.arange(0, hd, 2) / hd)
    old, lo, hi, f = rs["original_max_position_embeddings"], rs["low_freq_factor"], rs["high_freq_factor"], rs["factor"]
    wavelen = 2 * math.pi / inv
    out = np.where(wavelen > old / lo, inv / f, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / f + smooth * out
    medium = ~(wavelen < old / hi) & ~(wavelen > old / lo)
    return np.where(medium, smoothed, out)


def test_rope_llama3_inv_freq_matches_hf():
    cfg = dict(SMALL, rope_scaling=LLAMA3)
    hd = cfg["dim"] // cfg["n_heads"]
    got = mlb.rope_inv_freq(cfg)
    want = _hf_llama3_inv_freq(10000.0, hd, LLAMA3)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # all three regimes are exercised: unchanged, smoothed, divided by factor
    plain = 1.0 / 10000.0 ** (np.arange(0, hd, 2) / hd)
    assert got[0] == plain[0] and np.isclose(got[-1], plain[-1] / 8.0)
    assert np.any((got < plain) & (got > plain / 8.0))


def test_rope_scaling_header_roundtrip(tmp_path):
    cfg, w = gen_random.generate(dict(SMALL, rope_scaling=LLAMA3), seed=2)
    path = str(tmp_path / "rs.mlb")
    mlb.write_mlb(path, cfg, w, "q8")
    rcfg, _ = mlb.read_mlb(path)
    for k, v in LLAMA3.items():
        assert rcfg["rope_scaling"][k] == v, k
    path2 = str(tmp_path / "plain.mlb")
    mlb.write_mlb(path2, *gen_random.generate(dict(SMALL), seed=2), "q8")
    assert mlb.read_mlb(path2)[0]["rope_scaling"] is None


def test_hf_rope_scaling_llama3_converted(tmp_path):
    d, _, _ = _make_hf(tmp_path)
    c = json.loads((tmp_path / "hf" / "config.json").read_text())
    c["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
    (tmp_path / "hf" / "config.json").write_text(json.dumps(c))
    cfg, _ = convert.load_hf(d)
    assert cfg["rope_scaling"] == LLAMA3


def test_rope_scaled_model_runs_in_c(tmp_path):
    exe = os.path.join(ROOT, "build", "scalar", "ml_run")
    if not os.path.exists(exe):
        import pytest
        pytest.skip("build/scalar/ml_run not built")
    cfg, w = gen_random.generate(dict(SMALL, rope_scaling=LLAMA3), seed=4)
    path = str(tmp_path / "rs.mlb")
    mlb.write_mlb(path, cfg, w, "q8")
    toks = list(range(1, 30, 2))
    out = str(tmp_path / "l.f32")
    subprocess.run([exe, path, "--tokens", ",".join(map(str, toks)), "--dump-logits", out], check=True)
    got = np.fromfile(out, np.float32).reshape(len(toks), -1)
    rcfg, t = mlb.read_mlb(path)
    ref = ref_forward.forward(rcfg, t, toks)
    err = np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)
    assert err.max() <= 1e-3
    # and scaling changes the result vs. plain RoPE
    plain = ref_forward.forward(dict(rcfg, rope_scaling=None), t, toks)
    assert np.abs(plain[-1] - ref[-1]).max() > 1e-3 * np.abs(ref[-1]).max()


def test_sync_wiki_links(tmp_path):
    import sync_wiki
    out = sync_wiki.convert_links(
        "[a](user-guide.md#running) [b](design/x.md) [c](https://e.com) [d](#local) [e](../Makefile)",
        "docs/architecture.md", "o/r", "main")
    assert out == ("[a](User-Guide#running) [b](https://github.com/o/r/blob/main/docs/design/x.md) "
                   "[c](https://e.com) [d](#local) [e](https://github.com/o/r/blob/main/Makefile)")
    assert sync_wiki.convert_links("[g](docs/testing.md)", "README.md", "o/r", "main") == "[g](Testing)"


def test_mixed_q4_q8_classes(tmp_path):
    """--q8 classes: those matrices are stored (and read back) as Q8."""
    cfg, w = gen_random.generate(dict(SMALL), 0)
    path = str(tmp_path / "mix.mlb")
    mlb.write_mlb(path, cfg, w, "q4", ["emb", "v", "down"])
    rcfg, t = mlb.read_mlb(path)
    assert rcfg["q8_mask"] == mlb.Q8_CLASSES["emb"] | mlb.Q8_CLASSES["v"] | mlb.Q8_CLASSES["down"]
    q8, q4 = mlb.WTYPES["q8"], mlb.WTYPES["q4"]
    assert t["tok_emb"].type == q8 and t["wcls"].type == q4
    for layer in t["layers"]:
        assert layer["wv"].type == q8 and layer["w2"].type == q8
        assert layer["wq"].type == q4 and layer["wo"].type == q4 and layer["w1"].type == q4
    # a Q8 model ignores the classes and writes no mask
    mlb.write_mlb(path, cfg, w, "q8", ["emb"])
    assert mlb.read_mlb(path)[0]["q8_mask"] == 0
    try:
        mlb.q8_mask(["bogus"])
        assert False
    except ValueError:
        pass


def test_quantize_q4_imatrix():
    """Importance is relative (a constant factor changes nothing), skewed
    importance lowers the error where it is weighted, and an empty matrix
    falls back to the plain fit."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 256)).astype(np.float32)
    plain = mlb.quantize_q4(x)
    one, three = mlb.quantize_q4(x, np.ones(256)), mlb.quantize_q4(x, np.full(256, 3.0))
    assert (one[0] == three[0]).all() and (one[1] == three[1]).all()
    imp = np.where(np.arange(256) % 32 < 4, 50.0, 1.0)   # 4 channels of every block matter

    def err(p, d):
        return ((mlb.dequant(mlb.QT(mlb.WTYPES["q4"], p, d, x.shape)) - x) ** 2 * imp).sum()
    assert err(*mlb.quantize_q4(x, imp)) < 0.9 * err(*plain)
    z = mlb.quantize_q4(x, np.zeros(256))   # nothing recorded: plain fit
    assert (z[0] == plain[0]).all()


def test_imatrix_from_c(tmp_path):
    exe = os.path.join(ROOT, "build", "scalar", "ml_run")
    if not os.path.exists(exe):
        import pytest
        pytest.skip("build/scalar/ml_run not built")
    path, cfg, w, _ = _gen(tmp_path, "q8", seed=2)
    toks = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
    run = [exe, path, "--tokens", ",".join(map(str, toks)), "--dump-logits", str(tmp_path / "l.f32")]
    a, b = str(tmp_path / "a.imat"), str(tmp_path / "b.imat")
    subprocess.run(run + ["--imatrix", a], check=True)
    subprocess.run(run + ["--imatrix", b, "--batch", "1"], check=True)
    assert open(a, "rb").read() == open(b, "rb").read()   # batched prefill sums the same values
    ia = mlb.read_imatrix(a, cfg)
    assert ia["tokens"] == len(toks)
    # layer 0's q/k/v input is the normed embedding
    rcfg, t = mlb.read_mlb(path)
    x = mlb.dequant(t["tok_emb"])[toks]
    h = np.array([ref_forward.rmsnorm(r, t["layers"][0]["attn_norm"], float(rcfg["norm_eps"])) for r in x])
    np.testing.assert_allclose(ia["layers"][0]["qkv"], (h.astype(np.float64) ** 2).mean(0), rtol=1e-5)
    for L in ia["layers"]:
        assert all((v > 0).all() for v in L.values())
    assert (ia["cls"] > 0).all()
    subprocess.run(run + ["--imatrix", a], check=True)   # a second run adds to the file
    ia2 = mlb.read_imatrix(a, cfg)
    assert ia2["tokens"] == 2 * len(toks)
    np.testing.assert_array_equal(ia2["layers"][1]["down"], ia["layers"][1]["down"])
    # generation: the prompt is prefilled without the classifier
    g = str(tmp_path / "g.imat")
    subprocess.run([exe, path, "--tokens", "1,5,9", "-n", "6", "--imatrix", g], check=True, capture_output=True)
    hdr = mlb._IMAT.unpack_from(open(g, "rb").read())
    assert hdr[6:] == (6, 4)
    # another model's file is refused and left as it was
    cfg2, w2 = gen_random.generate(dict(SMALL, dim=96), 3)
    p2 = str(tmp_path / "other.mlb")
    mlb.write_mlb(p2, cfg2, w2, "q8")
    before = open(a, "rb").read()
    r = subprocess.run([exe, p2, "--tokens", "1,2", "--dump-logits", str(tmp_path / "o.f32"), "--imatrix", a],
                       capture_output=True)
    assert r.returncode != 0 and open(a, "rb").read() == before
    # a Q4 model fitted with it runs and matches the reference
    q4 = str(tmp_path / "imat_q4.mlb")
    mlb.write_mlb(q4, cfg, w, "q4", imatrix=ia2)
    subprocess.run([exe, q4, "--tokens", "1,7,3", "--dump-logits", str(tmp_path / "c.f32")], check=True)
    got = np.fromfile(tmp_path / "c.f32", np.float32).reshape(3, -1)
    rcfg, t = mlb.read_mlb(q4)
    _assert_close_to_ref(got, ref_forward.forward(rcfg, t, [1, 7, 3]))
    plain = str(tmp_path / "plain_q4.mlb")
    mlb.write_mlb(plain, cfg, w, "q4")
    assert open(q4, "rb").read() != open(plain, "rb").read()


def test_imatrix_weights_tied_classifier(tmp_path):
    """A tied embedding is the classifier: it takes the classifier's importance."""
    cfg, w = gen_random.generate(dict(SMALL), seed=1)
    D, H, L = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"]
    flat = {"layers": [dict(qkv=np.ones(D), wo=np.ones(D), ffn=np.ones(D), down=np.ones(H))] * L,
            "cls": np.ones(D), "tokens": 1}
    skew = dict(flat, cls=np.where(np.arange(D) % 32 < 2, 1e4, 1.0))
    for shared in (0, 1):
        c = dict(cfg, shared_classifier=shared)
        mlb.write_mlb(str(tmp_path / "a.mlb"), c, w, "q4", imatrix=flat)
        mlb.write_mlb(str(tmp_path / "b.mlb"), c, w, "q4", imatrix=skew)
        assert (tmp_path / "a.mlb").read_bytes() != (tmp_path / "b.mlb").read_bytes()


def test_quantize_q4_small_blocks():
    """Blocks of tiny values fit as if scaled up: float32 must not underflow."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    for imp in (None, np.ones(64)):
        p, d = mlb.quantize_q4(x, imp)
        for sc in (1e-9, 1e-16, 1e-20):
            ps, ds = mlb.quantize_q4((x * np.float32(sc)).astype(np.float32), imp)
            assert (ps == p).all()
            np.testing.assert_allclose(ds / np.float32(sc), d, rtol=1e-6)


def test_read_imatrix_refuses_bad_files(tmp_path):
    cfg = dict(SMALL)
    n = cfg["n_layers"] * (3 * cfg["dim"] + cfg["hidden_dim"]) + cfg["dim"]
    good = mlb._IMAT.pack(b"MLI1", 1, cfg["n_layers"], cfg["dim"], cfg["hidden_dim"], 0, 3, 3) + np.ones(n).tobytes()
    bad = {"short": good[:4], "reserved": good[:20] + b"\1" + good[21:], "trailing": good + b"\0" * 8,
           "truncated": good[:-8], "no tokens": good[:24] + bytes(16) + good[40:]}
    p = tmp_path / "x.imat"
    p.write_bytes(good)
    assert mlb.read_imatrix(str(p), cfg)["tokens"] == 3
    for name, data in bad.items():
        p.write_bytes(data)
        try:
            mlb.read_imatrix(str(p), cfg)
        except ValueError:
            continue
        raise AssertionError(f"{name} accepted")

