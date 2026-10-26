#!/bin/sh
# Decides whether main carries a new release: prints "tag=vX.Y.Z" (from
# ML_VERSION in core/ml.h) and "release=true" when the remote has no such tag
# yet, "release=false" when it does. Fails when the version is new but
# CHANGELOG.md has no "## X.Y.Z" section, so a version bump without notes is
# caught in the pull request instead of after the merge.
# Usage: scripts/release_check.sh [REMOTE]   (default origin)
set -eu
ver=$(sed -n 's/^#define ML_VERSION "\(.*\)"/\1/p' core/ml.h)
[ -n "$ver" ] || { echo "release_check: no ML_VERSION in core/ml.h" >&2; exit 1; }
tag=v$ver
echo "tag=$tag"
if git ls-remote --exit-code --tags "${1:-origin}" "refs/tags/$tag" >/dev/null 2>&1; then
    echo "release=false"
    exit 0
fi
awk -v v="$ver" '/^## /{p = ($2 == v)} p' CHANGELOG.md | tail -n +2 | grep -q . ||
    { echo "release_check: ML_VERSION $ver has no tag and CHANGELOG.md has no section for it" >&2; exit 1; }
echo "release=true"
