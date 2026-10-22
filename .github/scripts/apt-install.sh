#!/usr/bin/env bash
# apt-install.sh [--configure-only] PKG...
# apt-get install for CI runners: a fallback for Azure's Ubuntu mirror, network
# timeouts and retries, no man-db index rebuild, and timestamps for each step.
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

# GitHub runners point apt at azure.{archive,ports}.ubuntu.com. Route each such
# URI through a mirror+file list so apt moves on to the official mirror when the
# Azure one fails or stalls past the timeout.
mapfile -t files < <(find "$APT_ETC" -maxdepth 2 \( -name sources.list -o -name '*.list' -o -name '*.sources' \) -type f)
n=0
for uri in $(grep -hoE 'https?://azure\.(archive|ports)\.ubuntu\.com/[a-z-]+' "${files[@]}" 2>/dev/null | sort -u); do
    n=$((n + 1))
    list="$APT_ETC/mirrors-$n.txt"
    printf '%s/\n%s/\n' "$uri" "${uri/azure./}" | $SUDO tee "$list" >/dev/null
    $SUDO sed -i -E "s#${uri}/?([[:space:]]|\$)#mirror+file:${list}\\1#g" "${files[@]}"
    step "mirror fallback: $uri -> ${uri/azure./}"
done
[ $n -gt 0 ] || step "no Azure mirror in sources; leaving them as they are"
[ $configure_only -eq 0 ] || exit 0

step "apt-get update"
$SUDO apt-get update
step "apt-get install $*"
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
step "done"
