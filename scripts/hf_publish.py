"""Converts a model of scripts/hf_models.json to Krafla files, measures them,
writes a model card and, with --upload, publishes them on the Hugging Face Hub
as <account>/<source name>-krafla. Run by .github/workflows/hf-models.yml.

    python3 scripts/hf_publish.py --list default          # ids of a selection, as JSON
    python3 scripts/hf_publish.py smollm2-135m-instruct --build build/avx2 \\
        --text wiki.txt --work /tmp/hf [--threads 4] [--upload]

Without --upload, the files and README.md stay in WORK/out for a look. Needs
numpy and huggingface_hub; HF_TOKEN (write access) for --upload and for gated
sources (Llama, Gemma), whose licenses the token's account must have accepted.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CATALOG = os.path.join(ROOT, "scripts", "hf_models.json")
REPO_URL = "https://github.com/JonPark0/krafla"
CHAT_PROMPT = "What is the capital of France? Answer in one sentence."
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"

NOTICES = {
    "llama": "Llama 3.2 is licensed under the Llama 3.2 Community License, "
             "Copyright © Meta Platforms, Inc. All Rights Reserved.\n",
    "gemma": "Gemma is provided under and subject to the Gemma Terms of Use found at "
             "ai.google.dev/gemma/terms\n",
}


def load_catalog():
    with open(CATALOG, encoding="utf-8") as f:
        return json.load(f)


def select(catalog, spec):
    """'default', 'all' or comma-separated ids -> catalog entries."""
    models = catalog["models"]
    if spec == "all":
        return models
    if spec == "default":
        return [m for m in models if m.get("default")]
    by_id = {m["id"]: m for m in models}
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        sys.exit(f"unknown model ids: {', '.join(unknown)} (see {os.path.relpath(CATALOG, ROOT)})")
    return [by_id[i] for i in ids]


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=True, text=True, **kw)
    except subprocess.CalledProcessError as e:
        sys.exit(f"{os.path.basename(cmd[1] if cmd[0] == sys.executable else cmd[0])} failed "
                 f"({e.returncode}):\n{(e.stderr or '')[-2000:]}")


def ml_version():
    with open(os.path.join(ROOT, "core", "ml.h"), encoding="utf-8") as f:
        return re.search(r'#define ML_VERSION "([^"]+)"', f.read()).group(1)


def git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def variants(model):
    """(file suffix, convert.py options, description) of each file."""
    out = [("Q4_0", ["--wtype", "q4"], "Q4_0, every matrix 4-bit")]
    if model.get("mixed"):
        out.append(("Q4_0-mixed", ["--wtype", "q4", "--q8", model["mixed"]],
                    f"Q4_0 with `{model['mixed']}` in Q8_0"))
    out.append(("Q8_0", ["--wtype", "q8"], "Q8_0, every matrix 8-bit"))
    return out


def measure(build, path, tok, text, threads, chat, prompt):
    ppl = run([sys.executable, os.path.join(ROOT, "tools", "perplexity.py"), os.path.join(build, "ml_run"), path,
               tok, text, "--ml-tok", os.path.join(build, "ml_tok"), "--ctx", "512", "--chunks", "4",
               "-t", str(threads)], capture_output=True).stdout
    bench = run([os.path.join(build, "ml_bench"), path, "-t", str(threads), "-n", "128"],
                capture_output=True).stdout
    m_ppl = re.search(r"perplexity ([0-9.]+) over (\d+) tokens", ppl)
    m_bench = re.search(r"prefill ([0-9.]+) tok/s, decode ([0-9.]+) tok/s", bench)
    if not m_ppl or not m_bench:
        sys.exit(f"could not parse measurements:\n{ppl}\n{bench}")
    reply = run([os.path.join(build, "ml_run"), path, "-z", tok, "--chat", chat, "-i", prompt,
                 "-n", "64", "-t", str(threads)], capture_output=True).stdout.strip()
    if not reply:
        sys.exit(f"{os.path.basename(path)}: empty chat reply")
    return {"perplexity": float(m_ppl.group(1)), "ppl_tokens": int(m_ppl.group(2)),
            "prefill": float(m_bench.group(1)), "decode": float(m_bench.group(2)), "reply": reply}


def model_card(model, repo_id, base, files, info):
    lic = model["license"]
    notice = model.get("notice")
    q4 = files[0]["name"]
    lines = [
        "---",
        f"license: {lic}",
        f"base_model: {model['source']}",
        "base_model_relation: quantized",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"- {t}" for t in ["krafla", "int4", "int8", "q4_0", "q8_0", "cpu", "sse2", "avx2", "avx512",
                              "arm-neon", "sve"]],
        "---",
        "",
        f"# {base} for Krafla",
        "",
    ]
    if notice == "llama":
        lines += ["**Built with Llama.**", ""]
    lines += [
        f"INT4 and INT8 conversions of [{model['source']}](https://huggingface.co/{model['source']}) for "
        f"[Krafla]({REPO_URL}), a dependency-free C11 runtime for CPU-only inference on x86-64 "
        "(SSE2 up to AVX-512 VNNI) and ARM (ARMv7 NEON up to i8mm and SVE).",
        "",
        "## Files",
        "",
        "| File | Weights | Size | Perplexity | Prefill | Decode | SHA-256 |",
        "|------|---------|-----:|-----------:|--------:|-------:|---------|",
    ]
    for f in files:
        m = f["measure"]
        lines.append(f"| `{f['name']}` | {f['desc']} | {f['size'] / 2**20:.0f} MiB | {m['perplexity']:.2f} | "
                     f"{m['prefill']:.0f} tok/s | {m['decode']:.1f} tok/s | `{f['sha256'][:16]}`… |")
    lines += [
        f"| `tokenizer.mlt` | tokenizer | {info['tok_size'] / 2**10:.0f} KiB | | | | `{info['tok_sha256'][:16]}`… |",
        "",
        f"Measured on {info['machine']} ({info['cpu']}, {info['threads']} threads) with Krafla's "
        f"`{info['isa']}` build. Perplexity is measured on the wikitext-2 test set, 4 chunks of 512 tokens "
        "(`tools/perplexity.py --ctx 512 --chunks 4`). Lower is better, and it is only comparable between "
        "files of the same model. Speed comes from `ml_bench -n 128` (128 prompt tokens, then 128 decoded "
        f"tokens). The files are converted with a {info['max_seq']}-position context. "
        "The full hashes are in `SHA256SUMS`.",
        "",
        "## Run",
        "",
        f"Download a [release]({REPO_URL}/releases) of Krafla (version {info['version']} or later; Linux "
        "x86-64, AArch64 and ARMv7, Windows x86-64) and two files from here:",
        "",
        "```sh",
        f"curl -LO https://huggingface.co/{repo_id}/resolve/main/{q4}",
        f"curl -LO https://huggingface.co/{repo_id}/resolve/main/tokenizer.mlt",
        f"./ml_run {q4} -z tokenizer.mlt --chat {model['chat']} -t 4",
        "```",
        "",
        "The release's `ml_run` runs the most capable build your CPU supports. A greedy one-turn reply from "
        f"`{q4}` to \"{CHAT_PROMPT + model.get('prompt_suffix', '')}\":",
        "",
        "```text",
        files[0]["measure"]["reply"],
        "```",
        "",
        "## How these files were made",
        "",
        f"With Krafla {info['version']} (commit [`{info['commit'][:7]}`]({REPO_URL}/commit/{info['commit']})) "
        f"on {info['date']}, from the safetensors of [{model['source']}](https://huggingface.co/{model['source']}):",
        "",
        "```sh",
        f"python3 tools/convert_tokenizer.py {base}/tokenizer.json tokenizer.mlt",
        *[f"python3 tools/convert.py hf {base} {f['name']} {' '.join(f['args'])} --max-seq {info['max_seq']}"
          for f in files],
        "```",
        "",
        "Q4_0 and Q8_0 keep one scale per block of 32 weights (f16 scales: 4.5 and 8.5 bits per weight). "
        f"See the [user guide]({REPO_URL}/blob/main/docs/user-guide.md#q8-q4-or-mixed) for the formats "
        f"and [the comparison with llama.cpp]({REPO_URL}/blob/main/docs/benchmarks.md).",
        "",
        "## License",
        "",
    ]
    if notice == "llama":
        lines.append("These files are derived from Llama 3.2 and are distributed under the "
                     "[Llama 3.2 Community License](LICENSE.txt) and its [Acceptable Use Policy](USE_POLICY.md). "
                     "See `NOTICE`.")
    elif notice == "gemma":
        lines.append("Gemma is provided under and subject to the Gemma Terms of Use found at "
                     "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms), including its use "
                     "restrictions. These files are modified (quantized) from the original. See `NOTICE`.")
    else:
        lines.append(f"The license of the original model, `{lic}`, applies"
                     + (" (copied in this repository)." if info["license_files"] else "."))
    lines.append("")
    return "\n".join(lines)


def publish(model, a):
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    base = model["source"].split("/")[1]
    src = os.path.join(a.work, "src", base)
    out = os.path.join(a.work, "out", base)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    snapshot_download(model["source"], local_dir=src, token=token,
                      allow_patterns=["*.json", "*.safetensors", "LICENSE*", "USE_POLICY*", "NOTICE*"])
    catalog = load_catalog()
    info = {"version": ml_version(), "commit": git_commit(), "cpu": cpu_name(), "threads": a.threads,
            "max_seq": catalog["max_seq"], "date": datetime.date.today().isoformat(),
            "machine": "a GitHub runner" if os.environ.get("GITHUB_ACTIONS") else "a local machine",
            "isa": os.path.basename(a.build).split("-")[0]}

    tok = os.path.join(out, "tokenizer.mlt")
    run([sys.executable, os.path.join(ROOT, "tools", "convert_tokenizer.py"), os.path.join(src, "tokenizer.json"), tok])
    info["tok_size"], info["tok_sha256"] = os.path.getsize(tok), sha256(tok)
    files = []
    for suffix, args, desc in variants(model):
        name = f"{base}-{suffix}.mlb"
        path = os.path.join(out, name)
        run([sys.executable, os.path.join(ROOT, "tools", "convert.py"), "hf", src, path, *args,
             "--max-seq", str(catalog["max_seq"])])
        f = {"name": name, "args": args, "desc": desc, "size": os.path.getsize(path), "sha256": sha256(path)}
        f["measure"] = measure(a.build, path, tok, a.text, a.threads, model["chat"],
                               CHAT_PROMPT + model.get("prompt_suffix", ""))
        print(f"{name}: {json.dumps(f['measure'])}", flush=True)
        files.append(f)

    info["license_files"] = []
    for name in sorted(os.listdir(src)):
        if re.match(r"(LICENSE|USE_POLICY|NOTICE)", name):
            shutil.copy(os.path.join(src, name), os.path.join(out, name))
            info["license_files"].append(name)
    if not info["license_files"] and model["license"] == "apache-2.0":
        # Apache 2.0 asks for a copy of the license with redistributed files
        import urllib.request
        with urllib.request.urlopen(APACHE_URL, timeout=60) as r, \
                open(os.path.join(out, "LICENSE"), "wb") as fh:
            fh.write(r.read())
        info["license_files"].append("LICENSE")
    if model.get("notice"):
        with open(os.path.join(out, "NOTICE"), "w", encoding="utf-8") as fh:
            fh.write(NOTICES[model["notice"]])
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        for f in files:
            fh.write(f"{f['sha256']}  {f['name']}\n")
        fh.write(f"{info['tok_sha256']}  tokenizer.mlt\n")

    shutil.rmtree(src, ignore_errors=True)   # disk space on the runner
    api = HfApi(token=token)
    owner = a.owner or (api.whoami()["name"] if token else "OWNER")
    repo_id = f"{owner}/{base}-krafla"
    card = model_card(model, repo_id, base, files, info)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    summary = {"id": model["id"], "repo": repo_id,
               "files": {f["name"]: {k: f["measure"][k] for k in ("perplexity", "prefill", "decode")} for f in files}}
    with open(os.path.join(a.work, f"{model['id']}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary), flush=True)

    if a.upload:
        if not token:
            sys.exit("--upload needs HF_TOKEN")
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=out, delete_patterns=["*.mlb"],
                          commit_message=f"Krafla {info['version']} ({info['commit'][:7]})")
        print(f"uploaded https://huggingface.co/{repo_id}", flush=True)
    else:
        print(f"not uploaded; files in {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="catalog id")
    ap.add_argument("--list", metavar="SPEC", help="print the ids of 'default', 'all' or a comma list as JSON")
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "avx2-pt"),
                    help="directory of ml_run, ml_tok and ml_bench (make print-build)")
    ap.add_argument("--text", help="evaluation text (wikitext-2 test set)")
    ap.add_argument("--work", default=os.path.join(ROOT, "build", "hf"), help="download and output directory")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--owner", help="Hugging Face account or organization (default: the token's)")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    catalog = load_catalog()
    if a.list:
        print(json.dumps([m["id"] for m in select(catalog, a.list)]))
        return
    if not a.model or not a.text:
        ap.error("give a model id and --text")
    a.build, a.text, a.work = (os.path.abspath(p) for p in (a.build, a.text, a.work))
    publish(select(catalog, a.model)[0], a)


if __name__ == "__main__":
    main()
