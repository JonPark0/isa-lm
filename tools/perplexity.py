"""Perplexity of a model on a text, through ml_run --dump-logits.

    python3 tools/perplexity.py build/avx2/ml_run model.mlb tok.mlt text.txt \\
        --ml-tok build/avx2/ml_tok --ctx 512 --chunks 8 -t 4

The text is encoded with ml_tok, cut into chunks of --ctx tokens (each led by
BOS if the tokenizer adds one), and every chunk's tokens after the first are
scored. Lower is better;
compare models on the same text, tokenizer and --ctx.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ml_run")
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("text")
    ap.add_argument("--ml-tok", required=True, help="path to the ml_tok binary")
    ap.add_argument("--ctx", type=int, default=512, help="tokens per chunk, BOS included")
    ap.add_argument("--chunks", type=int, default=4, help="chunks to score (from the start of the text)")
    ap.add_argument("-t", "--threads", type=int, default=1)
    ap.add_argument("--extra", default="", help="extra ml_run options, e.g. '--kv q8'")
    a = ap.parse_args()
    with open(a.text, "rb") as f:
        text = f.read()
    enc = subprocess.run([a.ml_tok, a.tokenizer, "--no-bos"], input=text, capture_output=True, check=True)
    ids = [int(x) for x in enc.stdout.split()]
    lead = [int(x) for x in subprocess.run([a.ml_tok, a.tokenizer], input=b"", capture_output=True,
                                           check=True).stdout.split()]   # [BOS], or [] (Qwen2)
    body = a.ctx - len(lead)
    nll, count = 0.0, 0
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "logits.f32")
        for c in range(a.chunks):
            chunk = ids[c * body:(c + 1) * body]
            if len(chunk) < 2:
                break
            toks = lead + chunk
            cmd = [a.ml_run, a.model, "--tokens", ",".join(map(str, toks)), "--dump-logits", out, "-t", str(a.threads)]
            subprocess.run(cmd + a.extra.split(), check=True, capture_output=True)
            logits = np.fromfile(out, np.float32).reshape(len(toks), -1).astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            tgt = np.array(toks[1:])
            nll -= logp[np.arange(len(tgt)), tgt].sum()
            count += len(tgt)
    print(f"{os.path.basename(a.model)}: perplexity {math.exp(nll / count):.4f} over {count} tokens")


if __name__ == "__main__":
    sys.exit(main())
