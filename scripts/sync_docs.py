"""Writes the docs files that are derived from other files, or checks that
they are current (--check, run in CI):

  docs/assets/decode-ratio.svg   the decode ratios of docs/benchmarks.md's
                                 x86-64 and AArch64 tables, one thread
  docs/changelog.md              CHANGELOG.md, as a page of the docs site

    python3 scripts/sync_docs.py            # rewrite
    python3 scripts/sync_docs.py --check    # exit 1 if a file is stale
"""
import argparse
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BENCH = os.path.join(ROOT, "docs", "benchmarks.md")
CHART = os.path.join(ROOT, "docs", "assets", "decode-ratio.svg")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
CHANGELOG_PAGE = os.path.join(ROOT, "docs", "changelog.md")
CHANGELOG_FRONT = """---
title: "Changelog: what changed in each Krafla release"
description: "Release notes of Krafla: new models, kernels, formats, speed-ups and fixes in each version."
nav_order: 9
nav_title: "Changelog"
---
<!-- Generated from CHANGELOG.md by scripts/sync_docs.py; edit CHANGELOG.md instead. -->

"""


def bench_tables(path):
    """{section heading: [row dict]} for the tables with a 'Decode ratio' column."""
    tables, section, header = {}, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^#{2,3} (.+)$", line)
            if m:
                section, header = m.group(1).strip(), None
                continue
            if not line.startswith("|"):
                header = None
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = cells
            elif not set(line) <= set("|-: ") and "Decode ratio" in header:
                tables.setdefault(section, []).append(dict(zip(header, cells)))
    return tables


def chart_groups():
    tables = bench_tables(BENCH)
    groups = []
    for section, label in (("Speed", "x86-64 · Intel Xeon (Emerald Rapids), 4 vCPUs"),
                           ("AArch64", "AArch64 · GitHub Arm runner, 4 vCPUs")):
        rows = {}
        for r in tables.get(section, []):
            if r["Threads"] != "1":
                continue
            isa = r["ISA level"].replace(" (native)", "")
            rows.setdefault(isa, {})[r["Weights"]] = float(r["Decode ratio"].rstrip("×"))
        if not rows:
            sys.exit(f"no one-thread rows in the '{section}' table of {BENCH}")
        groups.append((label, list(rows.items())))
    return groups


def svg_chart(groups):
    W, left, right = 760, 170, 40
    row_h, bar_h, gap, group_h = 36, 12, 3, 30
    top = 86
    n_rows = sum(len(rows) for _, rows in groups)
    plot_h = n_rows * row_h + len(groups) * group_h
    H = top + plot_h + 44
    vmax = max(v for _, rows in groups for _, d in rows for v in d.values())
    xmax = max(4, int(vmax) + 1)
    sx = (W - left - right) / xmax
    x = lambda v: left + v * sx                                        # noqa: E731
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'role="img" aria-labelledby="dr-title dr-desc" font-family="system-ui, -apple-system, '
        'Segoe UI, Roboto, Helvetica, Arial, sans-serif">',
        '<title id="dr-title">Krafla decode speed relative to llama.cpp, one thread</title>',
        '<desc id="dr-desc">' + "; ".join(
            f"{isa}: Q4_0 {d.get('Q4_0', 0):.2f}x, Q8_0 {d.get('Q8_0', 0):.2f}x"
            for _, rows in groups for isa, d in rows) + "</desc>",
        "<style>",
        ".bg{fill:#fff;stroke:#e5e7eb}.t{fill:#1f2937}.m{fill:#6b7280}.g{stroke:#e5e7eb}"
        ".one{stroke:#6b7280;stroke-dasharray:4 3}.q4{fill:#2563eb}.q8{fill:#ea580c}.lb{fill:#fff}",
        "@media (prefers-color-scheme:dark){.bg{fill:#1f2023;stroke:#3a3b40}.t{fill:#e5e7eb}"
        ".m{fill:#9ca3af}.g{stroke:#374151}.one{stroke:#9ca3af}.q4{fill:#60a5fa}.q8{fill:#fb923c}"
        ".lb{fill:#1f2023}}",
        "</style>",
        f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="8"/>',
        '<text class="t" x="20" y="32" font-size="17" font-weight="600">Decode speed against llama.cpp, '
        'one thread</text>',
        '<text class="m" x="20" y="54" font-size="13">Qwen2.5 0.5B Instruct, same Q4_0 and Q8_0 formats; '
        'above 1× Krafla is faster</text>',
        f'<rect class="q4" x="{W - 200}" y="68" width="12" height="12" rx="2"/>'
        f'<text class="t" x="{W - 183}" y="79" font-size="13">Q4_0</text>',
        f'<rect class="q8" x="{W - 130}" y="68" width="12" height="12" rx="2"/>'
        f'<text class="t" x="{W - 113}" y="79" font-size="13">Q8_0</text>',
    ]
    y_end = top + plot_h
    for i in range(xmax + 1):
        out.append(f'<line class="g" x1="{x(i):.1f}" y1="{top}" x2="{x(i):.1f}" y2="{y_end}"/>')
        out.append(f'<text class="m" x="{x(i):.1f}" y="{y_end + 18}" font-size="12" '
                   f'text-anchor="middle">{i}×</text>')
    out.append(f'<line class="one" x1="{x(1):.1f}" y1="{top - 6}" x2="{x(1):.1f}" y2="{y_end}"/>')
    out.append(f'<text class="m" x="{x(1) + 4:.1f}" y="{y_end + 34}" font-size="12">llama.cpp = 1×</text>')
    y = top
    for label, rows in groups:
        # the group label sits on a strip of background over the grid lines
        out.append(f'<rect class="lb" x="12" y="{y + 6}" width="{W - 24}" height="{group_h - 10}"/>')
        out.append(f'<text class="m" x="20" y="{y + 20}" font-size="12" font-weight="600">{label}</text>')
        y += group_h
        for isa, d in rows:
            out.append(f'<text class="t" x="{left - 12}" y="{y + row_h / 2 + 4:.1f}" font-size="13" '
                       f'text-anchor="end">{isa}</text>')
            by = y + (row_h - 2 * bar_h - gap) / 2
            for cls, key in (("q4", "Q4_0"), ("q8", "Q8_0")):
                v = d.get(key)
                if v is not None:
                    out.append(f'<rect class="{cls}" x="{left}" y="{by:.1f}" width="{v * sx:.1f}" '
                               f'height="{bar_h}" rx="2"/>')
                    out.append(f'<text class="t" x="{x(v) + 5:.1f}" y="{by + bar_h - 2:.1f}" '
                               f'font-size="11">{v:.2f}×</text>')
                by += bar_h + gap
            y += row_h
    out.append("</svg>")
    return "\n".join(out) + "\n"


def changelog_page():
    with open(CHANGELOG, encoding="utf-8") as f:
        return CHANGELOG_FRONT + f.read()


def targets():
    return {CHART: svg_chart(chart_groups()), CHANGELOG_PAGE: changelog_page()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a derived file is not current")
    a = ap.parse_args()
    stale = []
    for path, text in targets().items():
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = None
        if current == text:
            continue
        rel = os.path.relpath(path, ROOT)
        if a.check:
            stale.append(rel)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote {rel}")
    if stale:
        sys.exit("stale: " + ", ".join(stale) + " (run python3 scripts/sync_docs.py)")


if __name__ == "__main__":
    main()
