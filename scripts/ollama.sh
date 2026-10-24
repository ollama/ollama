#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
export OLLAMA_HOST

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && printf '%s/%s' "$PWD" "$(basename -- "${BASH_SOURCE[0]}")")

error() {
    printf 'ERROR: %s\n' "$*" >&2
}

resolve_binary() {
    local candidate=${OLLAMA_WRAPPER_BIN:-} resolved
    local -a candidates=()
    if [[ -n "$candidate" && "$candidate" != */* ]]; then
        candidate=$(command -v -- "$candidate" || true)
    fi
    if [[ -n "$candidate" ]]; then
        candidates+=("$candidate")
    else
        local old_ifs=$IFS path_entry
        IFS=:
        for path_entry in $PATH; do
            [[ -n "$path_entry" ]] || path_entry=.
            [[ -x "$path_entry/ollama" ]] || continue
            candidates+=("$path_entry/ollama")
        done
        IFS=$old_ifs
    fi
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate" ]] || continue
        resolved=$(cd -- "$(dirname -- "$candidate")" && printf '%s/%s' "$PWD" "$(basename -- "$candidate")")
        [[ "$resolved" == "$SCRIPT_PATH" ]] && continue
        printf '%s\n' "$resolved"
        return 0
    done
    error "Ollama binary not found or resolves only to this wrapper; set OLLAMA_WRAPPER_BIN."
    return 1
}

OLLAMA_BIN=$(resolve_binary) || exit 2

state_root=${OLLAMA_WRAPPER_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ollama-wrapper}

parse_host() {
    local scheme=http hostport host port path=
    if [[ "$OLLAMA_HOST" == *://* ]]; then
        scheme=${OLLAMA_HOST%%://*}
        hostport=${OLLAMA_HOST#*://}
        case "$scheme" in
            http) default_port=80 ;;
            https) default_port=443 ;;
        esac
    else
        hostport=$OLLAMA_HOST
        default_port=11434
    fi
    [[ "$scheme" == http || "$scheme" == https ]] || {
        error "Invalid OLLAMA_HOST scheme: $scheme"
        return 2
    }
    hostport=${hostport%%\#*}
    hostport=${hostport%%\?*}
    if [[ "$hostport" == */* ]]; then
        path=/${hostport#*/}
        hostport=${hostport%%/*}
    fi
    if [[ "$hostport" == *'@'* || -z "$hostport" ]]; then
        error "Invalid OLLAMA_HOST: $OLLAMA_HOST"
        return 2
    fi
    if [[ "$hostport" == \[*\] ]]; then
        host=${hostport%]*}]
        port=
    elif [[ "$hostport" == *:* ]]; then
        host=${hostport%:*}
        port=${hostport##*:}
    else
        host=$hostport
        port=
    fi
    if [[ -z "$port" ]]; then
        port=$default_port
    fi
    printf '%s\n' "$scheme|$host:$port|$path"
}

host_info=$(parse_host) || exit 2
scheme=${host_info%%|*}
rest=${host_info#*|}
host=${rest%%|*}
api_path=${rest#*|}
case "$host" in
    0.0.0.0:*) connect_host="127.0.0.1:${host#*:}" ;;
    '[::]'*) connect_host="[::1]:${host:5}" ;;
    *) connect_host=$host ;;
esac
case "$connect_host" in
    127.0.0.1:*|localhost:*|'[::1]':*) local_host=true ;;
    *) local_host=false ;;
esac
state_key=$(printf '%s' "$connect_host" | tr ':[]' '____')
state_dir=$state_root/$state_key
pid_file=$state_dir/server.pid
json_file=$state_dir/server.json
start_identity_file=$state_dir/server.start
lock_dir=$state_dir/lock

health_url() {
    printf '%s://%s%s/api/version' "$scheme" "$connect_host" "$api_path"
}

check_health() {
    command -v curl >/dev/null 2>&1 || {
        error "curl is required for server health checks."
        return 1
    }
    curl --silent --show-error --fail --max-time 0.8 "$(health_url)" >/dev/null 2>&1 || {
        error "Ollama server is not running."
        error "Run 'ollama start' or 'ollama serve'."
        return 1
    }
}

require_local() {
    if [[ "$local_host" != true ]]; then
        error "start, stop, and restart only supports local loopback hosts."
        return 4
    fi
}

lock_acquire() {
    mkdir -p "$state_dir" || return 1
    if ! mkdir "$lock_dir" 2>/dev/null; then
        error "Lifecycle operation is locked at $lock_dir."
        return 5
    fi
    printf '%s\n' "$$" > "$lock_dir/pid"
    trap 'rm -rf -- "$lock_dir"' EXIT
}

lock_release() {
    rm -rf -- "$lock_dir"
    trap - EXIT
}

read_pid() {
    [[ -r "$pid_file" ]] || return 1
    IFS= read -r server_pid < "$pid_file" || return 1
    [[ "$server_pid" =~ ^[0-9]+$ ]] || return 1
    [[ "$server_pid" -gt 1 ]]
}

pid_alive() {
    kill -0 "$1" 2>/dev/null
}

process_matches() {
    local pid=$1 command_line recorded_identity current_identity
    command_line=$(ps -p "$pid" -o command= 2>/dev/null) || return 1
    [[ "$command_line" == *"$OLLAMA_BIN"* && "$command_line" == *' serve'* ]] || return 1
    [[ -r "$start_identity_file" ]] || return 1
    IFS= read -r recorded_identity < "$start_identity_file" || return 1
    current_identity=$(ps -p "$pid" -o lstart= 2>/dev/null) || return 1
    [[ -n "$recorded_identity" && "$current_identity" == "$recorded_identity" ]]
}

managed_status() {
    if read_pid && pid_alive "$server_pid" && process_matches "$server_pid"; then
        printf 'Server is running, managed by this wrapper (PID %s).\n' "$server_pid"
        return 0
    fi
    if check_health; then
        printf '%s\n' 'Server is running, but it is not managed by this wrapper.'
        printf '%s\n' 'Wrapper configuration is not applied to this instance.'
        return 0
    fi
    printf '%s\n' 'Server is not running.'
}

start_server() {
    require_local || return $?
    lock_acquire || return $?
    if read_pid && pid_alive "$server_pid" && process_matches "$server_pid"; then
        printf 'Server is already running (PID %s).\n' "$server_pid"
        lock_release
        return 0
    fi
    if check_health; then
        error 'Server is already running, but is not managed by this wrapper.'
        lock_release
        return 4
    fi
    rm -f -- "$pid_file" "$json_file" "$start_identity_file"
    nohup "$OLLAMA_BIN" serve </dev/null >>"$state_dir/server.stdout.log" 2>>"$state_dir/server.stderr.log" &
    server_pid=$!
    printf '%s\n' "$server_pid" > "$pid_file.tmp"
    mv -f -- "$pid_file.tmp" "$pid_file"
    started_process=$(ps -p "$server_pid" -o lstart= 2>/dev/null) || started_process=''
    [[ -n "$started_process" ]] || {
        error 'Could not verify the started process identity.'
        kill -TERM "$server_pid" 2>/dev/null || true
        rm -f -- "$pid_file"
        lock_release
        return 4
    }
    printf '%s\n' "$started_process" > "$start_identity_file.tmp"
    mv -f -- "$start_identity_file.tmp" "$start_identity_file"
    started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    json_tmp=$json_file.tmp
    printf '{"pid":%s,"host":"%s","binary":"%s","started_at":"%s"}\n' \
        "$server_pid" "$connect_host" "$OLLAMA_BIN" "$started_at" > "$json_tmp"
    mv -f -- "$json_tmp" "$json_file"
    for ((attempt = 0; attempt < 50; attempt++)); do
        if check_health; then
            printf 'Server started (PID %s).\n' "$server_pid"
            lock_release
            return 0
        fi
        if ! pid_alive "$server_pid"; then
            error 'Ollama server exited before becoming healthy.'
            lock_release
            return 3
        fi
        sleep 0.2
    done
    error 'Ollama server did not become healthy within 10 seconds.'
    lock_release
    return 3
}

stop_server() {
    require_local || return $?
    lock_acquire || return $?
    if ! read_pid; then
        if check_health; then
            error 'Server is running, but it is not managed by this wrapper.'
            lock_release
            return 4
        fi
        printf '%s\n' 'Server is not running.'
        lock_release
        return 0
    fi
    if ! pid_alive "$server_pid"; then
        rm -f -- "$pid_file" "$json_file" "$start_identity_file"
        printf '%s\n' 'Server is not running.'
        lock_release
        return 0
    fi
    if ! process_matches "$server_pid"; then
        error "Cannot safely stop process $server_pid: process identity could not be verified."
        lock_release
        return 4
    fi
    started_identity=$(ps -p "$server_pid" -o lstart= 2>/dev/null) || started_identity=''
    [[ -n "$started_identity" ]] && [[ "$started_identity" == "$(<"$start_identity_file")" ]] || {
        error "Cannot safely stop process $server_pid: process identity changed."
        lock_release
        return 4
    }
    if ! kill -TERM "$server_pid"; then
        error "Cannot safely stop process $server_pid."
        lock_release
        return 4
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        rm -f -- "$pid_file" "$json_file" "$start_identity_file"
        printf '%s\n' 'Server stopped.'
        lock_release
        return 0
    fi
    if [[ "$(ps -p "$server_pid" -o lstart= 2>/dev/null)" != "$started_identity" ]]; then
        error "Cannot safely stop process $server_pid: process identity changed."
        lock_release
        return 4
    fi
    for ((attempt = 0; attempt < 50; attempt++)); do
        if ! pid_alive "$server_pid"; then
            rm -f -- "$pid_file" "$json_file" "$start_identity_file"
            if check_health; then
                error 'Server remains available after the managed process exited.'
                lock_release
                return 1
            fi
            printf '%s\n' 'Server stopped.'
            lock_release
            return 0
        fi
        sleep 0.2
    done
    error "Process $server_pid did not exit within 10 seconds; it was not force-killed."
    lock_release
    return 1
}

update_all() {
    check_health || return 3
    local listing line model failed=0 first=true
    listing=$("$OLLAMA_BIN" list) || {
        error 'Could not list local models.'
        return 1
    }
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
        if $first; then
            first=false
            continue
        fi
        model=${line%%[[:space:]]*}
        if [[ -z "$model" || "$model" == *[[:space:]]* || "$line" == "$model" ]]; then
            error "Cannot parse model row: $line"
            return 1
        fi
        if ! "$OLLAMA_BIN" pull "$model"; then
            error "Failed to update model: $model"
            failed=1
        fi
    done <<< "$listing"
    return "$failed"
}

case "${1:-}" in
    start)
        [[ $# -eq 1 ]] || { error 'Usage: ollama start'; exit 2; }
        start_server
        exit $?
        ;;
    stop)
        if (( $# == 1 )); then
            stop_server
            exit $?
        fi
        ;;
    restart)
        [[ $# -eq 1 ]] || { error 'Usage: ollama restart'; exit 2; }
        stop_server || exit $?
        start_server
        exit $?
        ;;
    status)
        [[ $# -eq 1 ]] || { error 'Usage: ollama status'; exit 2; }
        managed_status
        exit $?
        ;;
    update)
        if [[ "${2:-}" == all && $# -eq 2 ]]; then
            update_all
            exit $?
        fi
        ;;
esac

case "${1:-}" in
    run|pull|push|ps|list|show|rm)
        if [[ "${2:-}" != --help && "${2:-}" != -h ]]; then
            check_health || exit 3
        fi
        ;;
    stop)
        if (( $# > 1 )); then
            check_health || exit 3
        fi
        ;;
esac

exec "$OLLAMA_BIN" "$@"
