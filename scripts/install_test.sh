#!/bin/sh
# Run on Linux with systemctl installed; no running systemd or root required.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# Load only service configuration, redirect /etc into the fixture, and omit
# root-only ownership flags. Keep the migration and unit contents unchanged.
sed -n '/^configure_systemd() {$/,/^}$/p' "$SCRIPT_DIR/install.sh" |
    sed -e "s|/etc/systemd/system|$TEST_ROOT/etc/systemd/system|g" \
        -e 's/ -o0 -g0 / /g' > "$TEST_ROOT/configure.sh"
. "$TEST_ROOT/configure.sh"

SUDO=
OLLAMA_INSTALL_DIR="$TEST_ROOT/usr/local"
BINDIR=/usr/local/bin
IS_WSL2=false
UNIT_ETC="$TEST_ROOT/etc/systemd/system/ollama.service"
UNIT_VENDOR="$OLLAMA_INSTALL_DIR/lib/systemd/system/ollama.service"
mkdir -p "$(dirname "$UNIT_ETC")"

# Account management and daemon operations must not affect the host. Unit-file
# operations use real systemctl against the fixture, including mask detection.
id() { return 0; }
getent() { return 1; }
usermod() { :; }
status() { :; }
warning() { fail "$*"; }
systemctl() {
    case "$1" in
        is-system-running) echo running ;;
        daemon-reload) : ;;
        restart|try-restart|enable) fail "Unexpected service action: $*" ;;
        *) command systemctl --root="$TEST_ROOT" "$@" ;;
    esac
}

for scenario in empty-mask symlink-mask legacy-unit; do
    rm -f "$UNIT_ETC" "$UNIT_VENDOR"
    case "$scenario" in
        empty-mask) : > "$UNIT_ETC"; expected=masked ;;
        symlink-mask) ln -s /dev/null "$UNIT_ETC"; expected=masked ;;
        legacy-unit)
            cat > "$UNIT_ETC" <<'EOF'
[Service]
ExecStart=/bin/true
[Install]
WantedBy=default.target
EOF
            expected=disabled
            ;;
    esac
    before=$(systemctl is-enabled ollama 2>/dev/null || true)
    [ "$before" = "$expected" ] || fail "$scenario: invalid fixture ($before)"

    # Isolate the installer's EXIT trap. A disabled unit should use try-restart;
    # mask cases must never attempt to enable or restart the service.
    (
        configure_systemd
        if [ "$scenario" = legacy-unit ]; then trap - EXIT; fi
    )

    after=$(systemctl is-enabled ollama 2>/dev/null || true)
    [ "$after" = "$expected" ] || fail "$scenario: expected $expected, got $after"
    [ -s "$UNIT_VENDOR" ] || fail "$scenario: vendor unit missing"
    case "$scenario" in
        empty-mask)
            [ -f "$UNIT_ETC" ] && [ ! -s "$UNIT_ETC" ] && [ ! -L "$UNIT_ETC" ] ||
                fail 'Empty-file mask was not preserved'
            ;;
        symlink-mask)
            [ "$(readlink "$UNIT_ETC")" = /dev/null ] || fail 'Symlink mask was not preserved'
            ;;
        legacy-unit)
            [ ! -e "$UNIT_ETC" ] || fail 'Legacy unit was not migrated'
            ;;
    esac
    echo "PASS: $scenario"
done
