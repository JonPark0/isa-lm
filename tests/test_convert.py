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
