"""Tests for tools/convert_tokenizer.py: the files it writes, and the
tokenizer.json variants it must refuse."""
import copy
import json
import os
import struct
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import convert_tokenizer as ct  # noqa: E402


def _load(name):
    with open(os.path.join(HERE, "data", name), encoding="utf-8") as f:
        return json.load(f)


def _read(path):
    """-> (header fields, pieces [(attr, bytes)], merges [(l, r, m)])."""
    b = open(path, "rb").read()
    assert b[:4] == b"MLT1"
    ver, kind, nv, nm, bos, eos, flags = struct.unpack_from("<IIIIiiI", b, 4)
    off, pieces = 32, []
    for _ in range(nv):
        attr, n = struct.unpack_from("<BI", b, off)
        pieces.append((attr, b[off + 5:off + 5 + n]))
        off += 5 + n
    merges = [struct.unpack_from("<III", b, off + 12 * i) for i in range(nm)]
    assert off + 12 * nm == len(b)
    return (ver, kind, nv, nm, bos, eos, flags), pieces, merges


def test_spm_file(tmp_path):
    out = str(tmp_path / "spm.mlt")
    ct.write(out, *ct.convert(_load("spm_tiny.json")))
    (ver, kind, nv, nm, bos, eos, flags), pieces, merges = _read(out)
    assert (ver, kind, bos, eos) == (1, ct.SPM, 1, 2)
    assert flags & ct.ADD_BOS and not flags & ct.PREPEND_FIRST
    assert pieces[1] == (ct.ADDED, b"<s>")
    byte_a = [p for p in pieces if p[0] & ct.BYTE and p[1] == b"A"]
    assert len(byte_a) == 1                       # <0x41> stored as its byte
    for l, r, m in merges:                        # every merge concatenates its pieces
        assert pieces[l][1] + pieces[r][1] == pieces[m][1]


def test_byte_bpe_file(tmp_path):
    out = str(tmp_path / "bpe.mlt")
    ct.write(out, *ct.convert(_load("bpe_tiny.json")))
    (ver, kind, nv, nm, bos, eos, flags), pieces, merges = _read(out)
    assert (kind, bos, eos) == (ct.BYTE_BPE, 0, 1)
    assert flags & ct.IGNORE_MERGES
    singles = {p[1] for p in pieces if len(p[1]) == 1}
    assert singles == {bytes([i]) for i in range(256)}   # raw bytes, not GPT-2 characters
    assert any(p[1] == b" hello" for p in pieces)
    for l, r, m in merges:
        assert pieces[l][1] + pieces[r][1] == pieces[m][1]


def test_metaspace_first(tmp_path):
    j = _load("spm_tiny.json")
    j["normalizer"] = None
    j["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": False}
    kind, flags, *_ = ct.convert(j)
    assert kind == ct.SPM and flags & ct.PREPEND_FIRST


@pytest.mark.parametrize("change", [
    lambda j: j["model"].update(type="WordPiece"),
    lambda j: j["model"].update(byte_fallback=False),
    lambda j: j["normalizer"]["normalizers"].pop(),
    lambda j: j.update(pre_tokenizer={"type": "Metaspace", "replacement": "▁", "split": True}),
])
def test_refuses_spm_variants(change):
    j = copy.deepcopy(_load("spm_tiny.json"))
    change(j)
    with pytest.raises(ct.Unsupported):
        ct.convert(j)


@pytest.mark.parametrize("change", [
    lambda j: j["pre_tokenizer"]["pretokenizers"][0]["pattern"].update(Regex=r"'s|\w+| ?\d+"),   # GPT-2 style
    lambda j: j["pre_tokenizer"]["pretokenizers"][1].update(add_prefix_space=True),
    lambda j: j.update(normalizer={"type": "NFC"}),
])
def test_refuses_byte_level_variants(change):
    j = copy.deepcopy(_load("bpe_tiny.json"))
    change(j)
    with pytest.raises(ct.Unsupported):
        ct.convert(j)
