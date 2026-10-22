"""compare_logits.py A B VOCAB [RTOL] [--kl MAX]: raw float32 logits files [n, VOCAB].

Default (random test models): per position max|a-b| <= RTOL * max|b| and argmax equal.
--kl MAX (real models, where an activation sitting on a rounding tie can flip
one quantization step between two valid implementations): per position the
argmax is equal, at least 4 of the top 5 tokens agree, and KL(softmax(b) ||
softmax(a)) <= MAX."""
import sys

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
kl_max = float(sys.argv[sys.argv.index("--kl") + 1]) if "--kl" in sys.argv else None
if kl_max is not None:
    args.remove(sys.argv[sys.argv.index("--kl") + 1])
a = np.fromfile(args[0], np.float32)
b = np.fromfile(args[1], np.float32)
vocab = int(args[2])
rtol = float(args[3]) if len(args) > 3 else 1e-3
if a.size != b.size or a.size == 0 or a.size % vocab:
    sys.exit(f"size mismatch: {a.size} vs {b.size}")
a, b = a.reshape(-1, vocab).astype(np.float64), b.reshape(-1, vocab).astype(np.float64)

if kl_max is None:
    err = np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)
    bad = np.nonzero((err > rtol) | (a.argmax(1) != b.argmax(1)))[0]
    print(f"max rel err {err.max():.2e} over {len(a)} positions")
else:
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    pa, pb = softmax(a), softmax(b)
    kl = (pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum(axis=1)
    top = [len(set(np.argsort(-a[i])[:5]) & set(np.argsort(-b[i])[:5])) for i in range(len(a))]
    bad = [i for i in range(len(a)) if a[i].argmax() != b[i].argmax() or top[i] < 4 or kl[i] > kl_max]
    print(f"max KL {kl.max():.2e}, min top-5 overlap {min(top)}/5 over {len(a)} positions")
if len(bad):
    sys.exit(f"positions out of tolerance: {list(map(int, bad))}")
