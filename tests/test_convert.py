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
        for name, orig in (("tok_emb", w["tok_emb"]), ("wq0", w["layers"][0]["wq"]),
                           ("w2_1", w["layers"][1]["w2"])):
            qt = t["tok_emb"] if name == "tok_emb" else t["layers"][int(name[-1])][name[:2]]
            deq = mlb.dequant(qt)
            step = np.repeat(qt.d, 32, axis=-1)
            assert deq.shape == orig.shape
            assert np.all(np.abs(deq - orig) <= step / 2 + 1e-6), name
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
    assert d[0, 0] == 1.0
    assert p[0, 0] == (7 + 8) | ((-7 + 8) << 4)


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


def _make_hf(tmp_path):
    cfg, w = gen_random.generate(dict(SMALL), seed=5)
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd = dim // nh

    def to_hf(m, n_heads):   # interleaved rows -> rotate_half rows (inverse of the converter)
        return m.reshape(n_heads, hd // 2, 2, dim).transpose(0, 2, 1, 3).reshape(-1, dim)

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
        put(p + "self_attn.o_proj.weight", L["wo"])
        put(p + "post_attention_layernorm.weight", L["ffn_norm"])
        put(p + "mlp.gate_proj.weight", L["w1"])
        put(p + "mlp.down_proj.weight", L["w2"])
        put(p + "mlp.up_proj.weight", L["w3"])
    put("model.norm.weight", w["final_norm"])
    d = tmp_path / "hf"
    d.mkdir()
    _write_safetensors(str(d / "model.safetensors"), t)
    (d / "config.json").write_text(json.dumps({
        "hidden_size": dim, "intermediate_size": cfg["hidden_dim"], "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": nh, "num_key_value_heads": nkv, "vocab_size": cfg["vocab"],
        "max_position_embeddings": cfg["max_seq"], "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True}))
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
    assert np.abs(got - ref).max() <= 1e-3 * np.abs(ref).max()
