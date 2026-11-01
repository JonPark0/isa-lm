"""Tells IndexNow search engines (Bing, Yandex, Seznam, Naver, ...) which
pages of the docs site changed.

    python3 .github/scripts/indexnow.py docs/user-guide.md docs/faq.md
    python3 .github/scripts/indexnow.py --all --dry-run

The key is the name of docs/<32 hex digits>.txt, a file that holds the same
digits and is served next to the pages; it only proves that the submitter
controls the site, so it is public by design.
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SITE = "https://jonpark0.github.io/isa-lm/"
ENDPOINT = "https://api.indexnow.org/indexnow"


def find_key(docs):
    for path in glob.glob(os.path.join(docs, "*.txt")):
        name = os.path.basename(path)[:-4]
        if re.fullmatch(r"[0-9a-f]{32}", name):
            with open(path, encoding="ascii") as f:
                if f.read().strip() == name:
                    return name
    sys.exit("no IndexNow key file docs/<32 hex digits>.txt")


def excluded(docs):
    """Paths listed under exclude: in docs/_config.yml."""
    out, inside = set(), False
    with open(os.path.join(docs, "_config.yml"), encoding="utf-8") as f:
        for line in f:
            if re.match(r"^exclude:", line):
                inside = True
            elif inside and re.match(r"^\s+-\s+", line):
                out.add(line.split("-", 1)[1].strip())
            elif inside and line.strip() and not line.startswith(" "):
                inside = False
    return out


def page_url(rel):
    """docs/ path of a Markdown page -> its URL on the site, or None."""
    if not rel.endswith(".md") or any(p.startswith("_") for p in rel.split("/")):
        return None
    stem = rel[:-3]
    return SITE if stem == "index" else SITE + stem + ".html"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="changed files (repository paths)")
    ap.add_argument("--all", action="store_true", help="submit every page")
    ap.add_argument("--dry-run", action="store_true", help="print the request instead of sending it")
    a = ap.parse_args()
    docs = os.path.join(ROOT, "docs")
    skip = excluded(docs)
    if a.all:
        files = [os.path.relpath(p, docs).replace(os.sep, "/")
                 for p in glob.glob(os.path.join(docs, "**", "*.md"), recursive=True)]
    else:
        files = [f[len("docs/"):] for f in a.files if f.startswith("docs/")]
    urls = sorted({u for f in files if f not in skip and os.path.exists(os.path.join(docs, f))
                   for u in [page_url(f)] if u})
    if not urls:
        print("no pages to submit")
        return
    key = find_key(docs)
    body = {"host": "jonpark0.github.io", "key": key, "keyLocation": SITE + key + ".txt", "urlList": urls}
    if a.dry_run:
        print(json.dumps(body, indent=2))
        return
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=30) as r:   # 200 or 202; errors raise
        print(f"IndexNow {r.status}: {len(urls)} URLs")
        for u in urls:
            print(" ", u)


if __name__ == "__main__":
    main()
