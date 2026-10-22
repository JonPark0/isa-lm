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
