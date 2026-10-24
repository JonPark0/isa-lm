"""Writes the GitHub wiki pages from README.md and docs/.

    git clone https://github.com/<owner>/<repo>.wiki.git wiki
    python3 tools/sync_wiki.py wiki --repo <owner>/<repo>
    cd wiki && git add -A && git commit -m "Sync from docs" && git push

docs/ stays the source of truth; links between docs become wiki page links and
links to other repository files become absolute GitHub URLs.
"""
import argparse
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

PAGES = {   # source file -> wiki page name
    "README.md": "Home",
    "docs/user-guide.md": "User-Guide",
    "docs/architecture.md": "Architecture",
    "docs/file-format.md": "File-Format",
    "docs/kernels.md": "Writing-Kernels",
    "docs/testing.md": "Testing",
    "docs/benchmarks.md": "Comparison-with-llama-cpp",
}

SIDEBAR = """**isa-lm**

- [[Home]]
- [[User Guide|User-Guide]]
- [[Architecture]]
- [[File Format|File-Format]]
- [[Writing Kernels|Writing-Kernels]]
- [[Testing]]
- [[Comparison with llama.cpp|Comparison-with-llama-cpp]]
"""

LINK = re.compile(r"\]\(([^)\s]+)\)")


def convert_links(text, src, repo, branch):
    base = os.path.dirname(src)

    def fix(m):
        target = m.group(1)
        if re.match(r"^[a-z]+://|^#|^mailto:", target):
            return m.group(0)
        path, _, anchor = target.partition("#")
        rel = os.path.normpath(os.path.join(base, path)).replace(os.sep, "/")
        if rel in PAGES:
            return f"]({PAGES[rel]}{'#' + anchor if anchor else ''})"
        kind = "tree" if os.path.isdir(os.path.join(ROOT, rel)) else "blob"
        return f"](https://github.com/{repo}/{kind}/{branch}/{rel}{'#' + anchor if anchor else ''})"

    return LINK.sub(fix, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wiki_dir")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()
    for src, page in PAGES.items():
        with open(os.path.join(ROOT, src), encoding="utf-8") as f:
            text = convert_links(f.read(), src, a.repo, a.branch)
        note = f"\n\n---\n_Generated from [`{src}`](https://github.com/{a.repo}/blob/{a.branch}/{src}); edit it there._\n"
        with open(os.path.join(a.wiki_dir, page + ".md"), "w", encoding="utf-8") as f:
            f.write(text.rstrip() + note)
    with open(os.path.join(a.wiki_dir, "_Sidebar.md"), "w", encoding="utf-8") as f:
        f.write(SIDEBAR)
    print(f"wrote {len(PAGES)} pages + sidebar to {a.wiki_dir}")


if __name__ == "__main__":
    main()
