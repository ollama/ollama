#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WRAPPER="$ROOT/scripts/ollama.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

assert_eq() {
    local want=$1
    local got=$2
    local name=$3
    [[ "$got" == "$want" ]] || fail "$name: expected [$want], got [$got]"
}

cat > "$TMP/fake-ollama" <<'FAKE'
#!/usr/bin/env bash
if [[ "${1:-}" == list ]]; then
    printf '%s\n' "${OLLAMA_TEST_LIST:-NAME ID SIZE MODIFIED}"
    exit "${OLLAMA_TEST_LIST_EXIT:-0}"
fi
if [[ "${1:-}" == pull ]]; then
    printf '%s\n' "$2" >> "$OLLAMA_TEST_PULLS"
    [[ "$2" != "${OLLAMA_TEST_FAIL_PULL:-}" ]]
    exit $?
fi
printf '%s\n' "$@" > "$OLLAMA_TEST_ARGS"
printf '%s' "${OLLAMA_CONTEXT_LENGTH-unset}" > "$OLLAMA_TEST_CONTEXT"
exit "${OLLAMA_TEST_EXIT:-0}"
FAKE
cat > "$TMP/curl" <<'FAKE'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$OLLAMA_TEST_CURL_ARGS"
exit "${OLLAMA_TEST_CURL_EXIT:-0}"
FAKE
chmod +x "$TMP/fake-ollama" "$TMP/curl"

run_wrapper() {
    PATH="$TMP:$PATH" \
    OLLAMA_WRAPPER_BIN="$TMP/fake-ollama" \
    OLLAMA_TEST_ARGS="$TMP/args" \
    OLLAMA_TEST_CONTEXT="$TMP/context" \
    OLLAMA_TEST_CURL_ARGS="$TMP/curl-args" \
        "$@"
}

# Native commands must preserve argument boundaries and the native exit status.
set +e
run_wrapper "$WRAPPER" run 'model:tag' 'prompt with spaces' '!*' >/dev/null 2>&1
status=$?
set -e
assert_eq 0 "$status" 'native command exit status'
assert_eq $'run\nmodel:tag\nprompt with spaces\n!*' "$(<"$TMP/args")" 'native argument boundaries'

# stop MODEL remains native, unlike bare stop which manages the local server.
run_wrapper "$WRAPPER" stop 'model:tag' >/dev/null 2>&1 || fail 'stop MODEL should be passed through'
assert_eq $'stop\nmodel:tag' "$(<"$TMP/args")" 'stop MODEL routing'

# Help and local create commands must not require a running server.
: > "$TMP/curl-args"
run_wrapper "$WRAPPER" run --help >/dev/null 2>&1 || fail 'run --help should be forwarded without health check'
assert_eq $'run\n--help' "$(<"$TMP/args")" 'run help forwarding'
[[ ! -s "$TMP/curl-args" ]] || fail 'run --help must skip health check'
run_wrapper "$WRAPPER" create local-model >/dev/null 2>&1 || fail 'local create should be forwarded without health check'
assert_eq $'create\nlocal-model' "$(<"$TMP/args")" 'create forwarding'
[[ ! -s "$TMP/curl-args" ]] || fail 'local create must skip health check'

# Caller-provided configuration is not overwritten by wrapper defaults.
OLLAMA_CONTEXT_LENGTH=16384 run_wrapper "$WRAPPER" version >/dev/null 2>&1 || fail 'version should be passed through'
assert_eq 16384 "$(<"$TMP/context")" 'caller configuration precedence'

# An unset context length remains unset so Ollama can choose its automatic default.
env -u OLLAMA_CONTEXT_LENGTH \
    OLLAMA_WRAPPER_BIN="$TMP/fake-ollama" \
    OLLAMA_TEST_ARGS="$TMP/args" \
    OLLAMA_TEST_CONTEXT="$TMP/context" \
    "$WRAPPER" version >/dev/null 2>&1 || fail 'version should be passed through without context override'
assert_eq unset "$(<"$TMP/context")" 'automatic context length default'

# update all snapshots list output, continues after a pull failure, and reports failure.
export OLLAMA_TEST_PULLS="$TMP/pulls"
: > "$OLLAMA_TEST_PULLS"
export OLLAMA_TEST_FAIL_PULL='broken/model:latest'
export OLLAMA_TEST_LIST=$'NAME ID SIZE MODIFIED\nalpha/model:latest abc 1GB now\nbroken/model:latest def 2GB now\nomega/model:latest ghi 3GB now'
set +e
run_wrapper "$WRAPPER" update all >"$TMP/update.out" 2>&1
status=$?
set -e
assert_eq 1 "$status" 'update all aggregate failure'
assert_eq $'alpha/model:latest\nbroken/model:latest\nomega/model:latest' "$(<"$TMP/pulls")" 'update all continues after failure'

# An invalid model row must stop processing instead of silently skipping it.
: > "$TMP/pulls"
export OLLAMA_TEST_LIST=$'NAME ID SIZE MODIFIED\nalpha/model:latest abc 1GB now\n  abc 1GB now'
set +e
run_wrapper "$WRAPPER" update all >"$TMP/update-invalid.out" 2>&1
status=$?
set -e
assert_eq 1 "$status" 'malformed list row rejection'
assert_eq 'alpha/model:latest' "$(<"$TMP/pulls")" 'malformed row prevents partial continuation'

# Host normalization matches Ollama defaults, scheme, and wildcard handling.
: > "$TMP/curl-args"
OLLAMA_HOST=localhost run_wrapper "$WRAPPER" status >/dev/null 2>&1 || fail 'localhost status should be supported'
[[ "$(<"$TMP/curl-args")" == *'http://localhost:11434/api/version'* ]] || fail 'localhost default port'
: > "$TMP/curl-args"
OLLAMA_HOST=http://localhost run_wrapper "$WRAPPER" status >/dev/null 2>&1 || fail 'explicit HTTP status should be supported'
[[ "$(<"$TMP/curl-args")" == *'http://localhost:80/api/version'* ]] || fail 'explicit HTTP default port'
: > "$TMP/curl-args"
OLLAMA_HOST=https://example.com/api run_wrapper "$WRAPPER" status >/dev/null 2>&1 || fail 'path status should be supported'
[[ "$(<"$TMP/curl-args")" == *'https://example.com:443/api/api/version'* ]] || fail 'host path prefix'
: > "$TMP/curl-args"
OLLAMA_HOST=https://example.com run_wrapper "$WRAPPER" status >/dev/null 2>&1 || fail 'HTTPS status should be supported'
[[ "$(<"$TMP/curl-args")" == *'https://example.com:443/api/version'* ]] || fail 'HTTPS scheme and default port'
: > "$TMP/curl-args"
set +e
OLLAMA_HOST='[::]:11434' run_wrapper "$WRAPPER" status >"$TMP/ipv6.out" 2>&1
status=$?
set -e
assert_eq 0 "$status" 'IPv6 wildcard status exit'
[[ "$(<"$TMP/curl-args")" == *'http://[::1]:11434/api/version'* ]] || fail 'IPv6 wildcard probe address'

# A remote host must never be managed by local lifecycle commands.
set +e
OLLAMA_HOST=https://example.com:11434 run_wrapper "$WRAPPER" start >"$TMP/remote.out" 2>&1
status=$?
set -e
assert_eq 4 "$status" 'remote lifecycle rejection'
[[ "$(<"$TMP/remote.out")" == *'only supports local loopback'* ]] || fail 'remote lifecycle rejection message'
OLLAMA_HOST=https://example.com:11434 OLLAMA_TEST_CURL_EXIT=0 run_wrapper "$WRAPPER" status >"$TMP/remote-status.out" 2>&1 || fail 'remote status should be allowed'
[[ "$(<"$TMP/remote-status.out")" == *'not managed by this wrapper'* ]] || fail 'remote status output'

# A wrapper-managed daemon can be started, detected, and stopped by its exact PID.
cat > "$TMP/daemon" <<'DAEMON'
#!/usr/bin/env bash
if [[ "${1:-}" == serve ]]; then
    : > "$OLLAMA_TEST_RUNNING"
    trap 'rm -f "$OLLAMA_TEST_RUNNING"; exit 0' TERM INT
    while :; do sleep 0.1; done
fi
printf '%s\n' "$@" > "$OLLAMA_TEST_ARGS"
DAEMON
cat > "$TMP/curl" <<'FAKE'
#!/usr/bin/env bash
[[ -f "$OLLAMA_TEST_RUNNING" ]]
FAKE
cat > "$TMP/ps" <<'FAKE'
#!/usr/bin/env bash
if [[ "${1:-}" == -p && "${3:-}" == -o ]]; then
    case "${4:-}" in
        command=) printf '%s serve\\n' "$OLLAMA_TEST_DAEMON_PATH" ;;
        lstart=) printf '%s\\n' 'Mon Oct 13 10:00:00 2026' ;;
        *) exit 1 ;;
    esac
    exit 0
fi
exit 1
FAKE
chmod +x "$TMP/daemon" "$TMP/curl" "$TMP/ps"
export OLLAMA_TEST_RUNNING="$TMP/running"
export OLLAMA_TEST_DAEMON_PATH="$TMP/daemon"
export OLLAMA_WRAPPER_STATE_DIR="$TMP/state"
export OLLAMA_WRAPPER_BIN="$TMP/daemon"
export PATH="$TMP:$PATH"
if ! "$WRAPPER" start >"$TMP/start.out" 2>&1; then
    cat "$TMP/start.out" >&2
    fail 'managed start should succeed'
fi
[[ -s "$TMP/state/127.0.0.1_11434/server.pid" ]] || fail 'start should record the daemon PID'
"$WRAPPER" status >"$TMP/status.out" 2>&1 || fail 'managed status should succeed'
[[ "$(<"$TMP/status.out")" == *'managed by this wrapper'* ]] || fail 'managed status output'

if ! "$WRAPPER" stop >"$TMP/stop.out" 2>&1; then
    cat "$TMP/stop.out" >&2
    fail 'managed stop should succeed'
fi
[[ ! -e "$TMP/state/127.0.0.1_11434/server.pid" ]] || fail 'stop should remove PID state'

printf 'PASS: Unix wrapper routing and lifecycle\n'
