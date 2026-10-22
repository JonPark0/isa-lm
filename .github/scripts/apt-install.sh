#!/usr/bin/env bash
# apt-install.sh [--configure-only] PKG...
# apt-get install for CI runners:
#  - picks the fastest Ubuntu mirror (GitHub's Azure mirror is sometimes ~200 kB/s),
#  - keeps downloaded .debs in $APT_CACHE_DIR (if set) so actions/cache can reuse them,
#  - network timeouts and retries, no man-db index rebuild, a timestamp per step.
# APT_ETC (default /etc/apt) and SUDO (default sudo) exist for testing.
set -euo pipefail
APT_ETC=${APT_ETC:-/etc/apt}
SUDO=${SUDO-sudo}
step() { echo "[$(date -u +%T)] $*"; }

configure_only=0
if [ "${1:-}" = "--configure-only" ]; then configure_only=1; shift; fi

step "configure apt"
if [ -z "${APT_ETC_TEST:-}" ]; then
    echo 'man-db man-db/auto-update boolean false' | $SUDO debconf-set-selections
    $SUDO rm -f /var/lib/man-db/auto-update
fi
$SUDO tee "$APT_ETC/apt.conf.d/99ci" >/dev/null <<'CONF'
Acquire::Retries "3";
Acquire::http::Timeout "20";
Acquire::https::Timeout "20";
Dpkg::Use-Pty "0";
CONF

arch=$(dpkg --print-architecture 2>/dev/null || echo amd64)
codename=$(. /etc/os-release 2>/dev/null && echo "${VERSION_CODENAME:-noble}")

# Bytes/s for a ~1-2 MB index file from mirror base URL $1; 0 unless it
# answered 200 (a proxy's error page must not count as a fast mirror).
mirror_speed() {
    local out
    out=$(curl -s -o /dev/null --connect-timeout 3 --max-time 6 -w '%{http_code} %{speed_download}' \
        "${1%/}/dists/$codename/main/binary-$arch/Packages.xz" 2>/dev/null) || true
    case $out in 200\ *) echo "${out#200 }" | cut -d. -f1 ;; *) echo 0 ;; esac
}

official_for() {   # the canonical mirror for an Ubuntu archive URL
    case $1 in *ubuntu-ports*) echo "http://ports.ubuntu.com/ubuntu-ports/" ;; *) echo "http://archive.ubuntu.com/ubuntu/" ;; esac
}

# Reorders mirror list file $1 (first field of each line = URI) fastest first,
# adding the official mirror as a candidate.
reorder_mirror_list() {
    local list=$1 uris best=0 u s
    uris=$( (awk '!/^#/ && NF {print $1}' "$list"; official_for "$(awk 'NF {print $1; exit}' "$list")") | awk '!seen[$0]++')
    : > /tmp/mirror-speeds.$$
    for u in $uris; do
        s=$(mirror_speed "$u"); s=${s:-0}
        step "mirror $u: $((s / 1024)) kB/s"
        echo "$s $u" >> /tmp/mirror-speeds.$$
    done
    sort -s -k1,1rn /tmp/mirror-speeds.$$ | awk '{print $2}' | $SUDO tee "$list" >/dev/null
    rm -f /tmp/mirror-speeds.$$
    step "mirror order in $list: $(tr '\n' ' ' < "$list")"
}

mapfile -t files < <(find "$APT_ETC" -maxdepth 2 \( -name sources.list -o -name '*.list' -o -name '*.sources' \) -type f)

# Plain Azure URIs in the sources: route them through a mirror list first.
n=0
for uri in $( (grep -hoE 'https?://azure\.(archive|ports)\.ubuntu\.com/[a-z-]+' "${files[@]}" 2>/dev/null || true) | sort -u); do
    n=$((n + 1))
    list="$APT_ETC/mirrors-$n.txt"
    printf '%s/\n' "$uri" | $SUDO tee "$list" >/dev/null
    $SUDO sed -i -E "s#${uri}/?([[:space:]]|\$)#mirror+file:${list}\\1#g" "${files[@]}"
done

# Every mirror list the sources use (GitHub runners ship /etc/apt/apt-mirrors.txt).
lists=$( (grep -hoE 'mirror\+file:[^[:space:]]+' "${files[@]}" 2>/dev/null || true) | sed 's/^mirror+file://' | sort -u)
if [ -z "$lists" ]; then
    step "no mirror list in sources; leaving them as they are"
else
    for list in $lists; do [ -f "$list" ] && reorder_mirror_list "$list"; done
fi
[ $configure_only -eq 0 ] || exit 0

cache_opt=()
if [ -n "${APT_CACHE_DIR:-}" ]; then
    mkdir -p "$APT_CACHE_DIR/partial"
    cache_opt=(-o "Dir::Cache::Archives=$APT_CACHE_DIR")
    step "package cache $APT_CACHE_DIR: $(find "$APT_CACHE_DIR" -maxdepth 1 -name '*.deb' | wc -l) .debs"
fi

step "apt-get update"
$SUDO apt-get update
step "apt-get install $*"
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${cache_opt[@]}" "$@"
if [ -n "${APT_CACHE_DIR:-}" ]; then   # let actions/cache (not root) read and save it
    $SUDO rm -f "$APT_CACHE_DIR/lock"
    $SUDO chown -R "$(id -u):$(id -g)" "$APT_CACHE_DIR"
fi
step "done"
