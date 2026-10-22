"""compare_logits.py A B VOCAB [RTOL]: raw float32 logits files [n, VOCAB].
Per position: max|a-b| <= RTOL * max|b|, and argmax equal."""
import sys

import numpy as np

a = np.fromfile(sys.argv[1], np.float32)
b = np.fromfile(sys.argv[2], np.float32)
vocab = int(sys.argv[3])
rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
if a.size != b.size or a.size == 0 or a.size % vocab:
    sys.exit(f"size mismatch: {a.size} vs {b.size}")
a, b = a.reshape(-1, vocab), b.reshape(-1, vocab)
err = np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)
bad = np.nonzero((err > rtol) | (a.argmax(1) != b.argmax(1)))[0]
print(f"max rel err {err.max():.2e} over {len(a)} positions")
if len(bad):
    sys.exit(f"positions out of tolerance: {bad.tolist()}")
