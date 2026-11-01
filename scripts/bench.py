"""A/B speed comparison of two Krafla source trees on this machine.

  python3 scripts/bench.py --new DIR [--base DIR] [--reps N] [--threads LIST]
                           [--model PATH ...] [--isa LIST] [--out FILE.md]

Builds ml_bench (THREADS=pthread) for every ISA in --isa that this CPU runs,
in each tree, then runs the two builds alternately REPS times so that noise
from the machine falls on both: the kernel table of ml_bench without a model
and, for each --model, prompt prefill and decode of 128 tokens. Reports the
best run of each (and the new/base ratio) as Markdown.
"""
import argparse
import os
import platform
import re
import subprocess
import sys

X86 = ["avx512vnni", "avxvnni", "avx2", "sse41", "sse2"]
ARM = ["sve", "neon_i8mm", "neon_dotprod", "neon"]


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def build(tree, isa):
    r = sh(["make", "-s", f"ISA={isa}", "THREADS=pthread", f"build/{isa}-pt/ml_bench", f"build/{isa}-pt/ml_tok"],
           cwd=tree)
    return r.returncode == 0


def runs_here(tree, isa):
    """The ISA guard exits with 3 when the CPU lacks the build's ISA."""
    return sh([os.path.join(tree, "build", f"{isa}-pt", "ml_tok"), "--help"]).returncode != 3


def kernels(tree, isa):
    """ml_bench without a model -> {(op, shape): GOPS}."""
    out = sh([os.path.join(tree, "build", f"{isa}-pt", "ml_bench")]).stdout
    res = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 5 and ("matvec" in f[0] or "matmul" in f[0]) and f[4] != "-":
            res[(f[0], f[1])] = float(f[4])
    return res


def model_run(tree, isa, model, threads):
    out = sh([os.path.join(tree, "build", f"{isa}-pt", "ml_bench"), model, "-t", str(threads), "-n", "128"]).stdout
    m = re.search(r"prefill ([0-9.]+) tok/s, decode ([0-9.]+)", out)
    return (float(m[1]), float(m[2])) if m else None


def best(a, b):
    return {k: max(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b)}


def ratio(new, base):
    return f"{new / base:.2f}×" if base else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="source tree to measure")
    ap.add_argument("--base", help="source tree to compare against")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", default=f"1,{os.cpu_count() or 1}")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--isa", default=",".join(ARM if platform.machine() in ("aarch64", "arm64") else X86))
    ap.add_argument("--out")
    a = ap.parse_args()
    trees = {"new": os.path.abspath(a.new)}
    if a.base:
        trees["base"] = os.path.abspath(a.base)
    threads = sorted({int(t) for t in a.threads.split(",") if t})
    lines = [f"Machine: `{platform.machine()}`, {os.cpu_count()} CPUs; best of {a.reps} alternating runs.", ""]
    cpu = sh(["sh", "-c", "grep -m1 -E 'model name|^CPU part' /proc/cpuinfo"]).stdout.strip()
    if cpu:
        lines += [f"CPU: `{cpu}`", ""]
    for isa in a.isa.split(","):
        if not all(build(t, isa) for t in trees.values()):
            lines += [f"## {isa}", "", "build failed, skipped", ""]
            continue
        if not runs_here(trees["new"], isa):
            print(f"{isa}: this CPU lacks it, skipped", file=sys.stderr)
            continue
        print(f"{isa}: running", file=sys.stderr)
        k = {n: {} for n in trees}
        m = {n: {} for n in trees}
        for _ in range(a.reps):
            for n, t in trees.items():   # alternate the trees within every repetition
                k[n] = best(k[n], kernels(t, isa))
                for model in a.model:
                    for th in threads:
                        r = model_run(t, isa, model, th)
                        if r:
                            key = (os.path.basename(model), th)
                            old = m[n].get(key, (0.0, 0.0))
                            m[n][key] = (max(old[0], r[0]), max(old[1], r[1]))
        lines += [f"## {isa}", ""]
        if "base" in trees:
            lines += ["| Kernel | Shape | base GOPS | new GOPS | new/base |", "|---|---|---:|---:|---:|"]
            for op, shape in sorted(k["new"]):
                b, n = k["base"].get((op, shape), 0.0), k["new"][(op, shape)]
                lines.append(f"| {op} | {shape} | {b:.1f} | {n:.1f} | {ratio(n, b)} |")
        else:
            lines += ["| Kernel | Shape | GOPS |", "|---|---|---:|"]
            lines += [f"| {op} | {shape} | {v:.1f} |" for (op, shape), v in sorted(k["new"].items())]
        if a.model:
            lines.append("")
            if "base" in trees:
                lines += ["| Model | Threads | prefill base | prefill new | ratio | decode base | decode new | ratio |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for key in sorted(m["new"]):
                    (pb, db), (pn, dn) = m["base"].get(key, (0.0, 0.0)), m["new"][key]
                    lines.append(f"| {key[0]} | {key[1]} | {pb:.1f} | {pn:.1f} | {ratio(pn, pb)} | "
                                 f"{db:.1f} | {dn:.1f} | {ratio(dn, db)} |")
            else:
                lines += ["| Model | Threads | prefill tok/s | decode tok/s |", "|---|---:|---:|---:|"]
                lines += [f"| {key[0]} | {key[1]} | {p:.1f} | {d:.1f} |" for key, (p, d) in sorted(m["new"].items())]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
