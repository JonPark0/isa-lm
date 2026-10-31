"""Converts a Hugging Face tokenizer.json to the MLT1 tokenizer format that
ml_run -z and ml_tok read (model/bpe.c).

    python3 tools/convert_tokenizer.py ./Llama-3.2-1B/tokenizer.json llama3.mlt

Supported, and checked before writing:
  - SentencePiece-style BPE with byte fallback (Llama 2, TinyLlama, Mistral):
    a Prepend/Replace normalizer, or a Metaspace pre-tokenizer without split;
    or a Replace normalizer alone, with no U+2581 before the text (Gemma 3);
  - byte-level BPE with the Llama 3 pre-tokenizer regex (Llama 3, 3.1, 3.2),
    or its single-digit variant after an NFC normalizer (Qwen2, Qwen2.5);
  - byte-level BPE with the GPT-2 regex (ByteLevel use_regex), optionally after
    a Digits(individual_digits) pre-tokenizer (SmolLM, SmolLM2).
Anything else is refused rather than converted into a tokenizer that would
encode differently.
"""
import argparse
import json
import struct
import sys

SPM, BYTE_BPE = 1, 2
ADDED, BYTE = 1, 2
ADD_BOS, IGNORE_MERGES, PREPEND_FIRST, NFC, DIGITS1, GPT2, ISOLATE_DIGITS, NO_PREPEND = 1, 2, 4, 8, 16, 32, 64, 128
SP = "▁"
LLAMA3_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
                r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN2_REGEX = LLAMA3_REGEX.replace(r"\p{N}{1,3}", r"\p{N}")


class Unsupported(Exception):
    pass


def bytes_to_unicode():
    """GPT-2's byte -> printable character map used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def parts(node, key):
    """The steps of a Sequence normalizer / pre-tokenizer / processor, or [node]."""
    if node is None:
        return []
    if node.get("type") == "Sequence":
        return node[key]
    return [node]


def detect(j):
    """-> (kind, flags) for a supported tokenizer.json, else raises Unsupported."""
    m = j["model"]
    mtype = m.get("type", "BPE" if "merges" in m else None)   # older files leave BPE implicit
    if mtype != "BPE":
        raise Unsupported(f"model type {mtype!r} (only BPE is supported)")
    if m.get("dropout") or m.get("continuing_subword_prefix") or m.get("end_of_word_suffix"):
        raise Unsupported("BPE dropout or subword prefixes/suffixes")
    norms = parts(j.get("normalizer"), "normalizers")
    pres = parts(j.get("pre_tokenizer"), "pretokenizers")
    flags = IGNORE_MERGES if m.get("ignore_merges") else 0
    split = [p for p in pres if p["type"] == "Split"]
    if any(p["type"] == "ByteLevel" for p in pres):
        if [n["type"] for n in norms] == ["NFC"]:
            flags |= NFC
        elif norms:
            raise Unsupported("a normalizer other than NFC before byte-level BPE")
        types = [p["type"] for p in pres]
        if types in (["ByteLevel"], ["Digits", "ByteLevel"]) and pres[-1].get("use_regex", True):
            # the GPT-2 regex inside ByteLevel, after an optional digit split
            if pres[-1].get("add_prefix_space"):
                raise Unsupported("ByteLevel with add_prefix_space")
            if types[0] == "Digits":
                if not pres[0].get("individual_digits"):
                    raise Unsupported("Digits without individual_digits")
                flags |= ISOLATE_DIGITS
            return BYTE_BPE, flags | GPT2
        if len(pres) != 2 or len(split) != 1 or pres[0] is not split[0]:
            raise Unsupported("byte-level pre-tokenizer other than Split(regex) + ByteLevel, "
                              "or ByteLevel with its GPT-2 regex")
        s, bl = pres
        rx = s["pattern"].get("Regex")
        if rx not in (LLAMA3_REGEX, QWEN2_REGEX) or s.get("behavior") != "Isolated" or s.get("invert"):
            raise Unsupported("pre-tokenizer regex is not the Llama 3 or Qwen2 one")
        if rx == QWEN2_REGEX:
            flags |= DIGITS1
        if bl.get("add_prefix_space") or bl.get("use_regex"):
            raise Unsupported("ByteLevel with add_prefix_space or its own regex")
        return BYTE_BPE, flags
    if not m.get("byte_fallback"):
        raise Unsupported("SentencePiece-style BPE without byte_fallback")
    types = [n["type"] for n in norms]
    if types == ["Prepend", "Replace"] and not pres:
        if norms[0]["prepend"] != SP or norms[1]["pattern"].get("String") != " " or norms[1]["content"] != SP:
            raise Unsupported("normalizer other than Prepend(U+2581) + Replace(' ' -> U+2581)")
        return SPM, flags
    # Gemma: spaces become U+2581 with no prefix; the Split on " " that follows
    # never finds a space, so it changes nothing
    if types == ["Replace"] and all(p["type"] == "Split" and p["pattern"].get("String") == " " for p in pres):
        if norms[0]["pattern"].get("String") != " " or norms[0]["content"] != SP:
            raise Unsupported("normalizer other than Replace(' ' -> U+2581)")
        return SPM, flags | NO_PREPEND
    if not norms and len(pres) == 1 and pres[0]["type"] == "Metaspace":
        p = pres[0]
        if p.get("replacement") != SP or p.get("split", False):
            raise Unsupported("Metaspace with split or another replacement")
        scheme = p.get("prepend_scheme", "always" if p.get("add_prefix_space", True) else "never")
        if scheme == "never":
            raise Unsupported("Metaspace without a prefix space")
        return SPM, flags | (PREPEND_FIRST if scheme == "first" else 0)
    raise Unsupported(f"normalizer {types} / pre-tokenizer {[p['type'] for p in pres]}")


def special_ids(j, added):
    """BOS from the post-processor template (and whether it is added), EOS by name."""
    bos, add_bos = -1, False
    for p in parts(j.get("post_processor"), "processors"):
        if p["type"] == "TemplateProcessing":
            first = p["single"][0]
            if "SpecialToken" in first:
                name = first["SpecialToken"]["id"]
                bos = p["special_tokens"][name]["ids"][0]
                add_bos = True
    eos = -1
    special = {a["content"]: a["id"] for a in j.get("added_tokens", []) if a.get("special", True)}
    for name in ("</s>", "<|end_of_text|>", "<|endoftext|>", "<eos>"):   # special ones only (Gemma's "</s>" is text)
        if name in special:
            eos = special[name]
            break
    return bos, add_bos, eos


def convert(j):
    """-> (kind, flags, bos, eos, pieces [(attr, bytes)], merges [(l, r, merged)])."""
    kind, flags = detect(j)
    m = j["model"]
    vocab = m["vocab"]
    added = {a["content"]: a["id"] for a in j.get("added_tokens", [])}
    n = max(list(vocab.values()) + list(added.values())) + 1
    pieces = [(0, b"")] * n
    dec = {c: b for b, c in bytes_to_unicode().items()}
    for tok, i in vocab.items():
        if kind == BYTE_BPE:
            try:
                pieces[i] = (0, bytes(dec[c] for c in tok))
            except KeyError:
                raise Unsupported(f"token {tok!r} is not byte-level encoded")
        elif len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            pieces[i] = (BYTE, bytes([int(tok[3:5], 16)]))
        else:
            pieces[i] = (0, tok.encode("utf-8"))
    for tok, i in added.items():
        if not pieces[i][0] & BYTE:   # byte-fallback pieces stay bytes
            pieces[i] = (ADDED, tok.encode("utf-8"))
    merges = []
    for r, mg in enumerate(m.get("merges", [])):
        a, b = mg.split(" ") if isinstance(mg, str) else mg
        if a not in vocab or b not in vocab or a + b not in vocab:
            raise Unsupported(f"merge {r} ({a!r}, {b!r}) has tokens missing from the vocabulary")
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    bos, add_bos, eos = special_ids(j, added)
    if add_bos:
        flags |= ADD_BOS
    return kind, flags, bos, eos, pieces, merges


def write(path, kind, flags, bos, eos, pieces, merges):
    with open(path, "wb") as f:
        f.write(b"MLT1" + struct.pack("<IIIIiiI", 1, kind, len(pieces), len(merges), bos, eos, flags))
        for attr, p in pieces:
            f.write(struct.pack("<BI", attr, len(p)) + p)
        for m in merges:
            f.write(struct.pack("<III", *m))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer_json")
    ap.add_argument("out")
    ap.add_argument("--bos", type=int, help="override the BOS id (-1: none)")
    ap.add_argument("--eos", type=int, help="override the EOS id (-1: none)")
    a = ap.parse_args()
    with open(a.tokenizer_json, encoding="utf-8") as f:
        j = json.load(f)
    try:
        kind, flags, bos, eos, pieces, merges = convert(j)
    except Unsupported as e:
        sys.exit(f"error: unsupported tokenizer: {e}")
    bos = bos if a.bos is None else a.bos
    eos = eos if a.eos is None else a.eos
    write(a.out, kind, flags, bos, eos, pieces, merges)
    print(f"{a.out}: {'SentencePiece BPE' if kind == SPM else 'byte-level BPE'}, {len(pieces)} tokens, "
          f"{len(merges)} merges, bos {bos}{' (added)' if flags & ADD_BOS else ''}, eos {eos}")


if __name__ == "__main__":
    main()
