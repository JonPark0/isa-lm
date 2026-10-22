"""Writes a tiny llama2.c-format tokenizer.bin: <unk>, <s>, </s>, then single
characters ' '..'|' (printable ASCII) up to the requested vocab size."""
import argparse
import struct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pieces = [b"<unk>", b"\n<s>\n", b"\n</s>\n"]
    pieces += [bytes([c]) for c in range(32, 32 + a.vocab - 3)]
    with open(a.out, "wb") as f:
        f.write(struct.pack("<i", max(len(p) for p in pieces)))
        for p in pieces:
            f.write(struct.pack("<fi", 0.0, len(p)) + p)


if __name__ == "__main__":
    main()
