#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-9006}"
MENO_WEB_PORT="${MENO_WEB_PORT:-9012}"

PID_FILE="${PID_FILE:-$ROOT_DIR/var/meno-rag-api.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/meno-rag-api.log}"
API_BIN="${API_BIN:-$ROOT_DIR/.venv/bin/meno-rag-api}"
MIGRATE_BIN="${MIGRATE_BIN:-$ROOT_DIR/.venv/bin/meno-rag-migrate}"
RESET_BIN="${RESET_BIN:-$ROOT_DIR/.venv/bin/meno-rag-reset}"

# --fresh: wipe the application schema before bootstrap. Parsed once here so
# every subcommand can see it without per-command argparse plumbing.
FRESH=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=1 ;;
        *)       POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]:-start}"

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

# --fresh runs `meno-rag-reset --yes`, which DROPS every application table
# (conversations, messages, pipeline runs, sources, arena votes + leaderboard)
# plus alembic_version. It is permanent and cannot be undone. Because `start
# --fresh` passes --yes automatically, it bypasses the dry-run that the bare
# reset command has — so we gate it behind an explicit typed confirmation.
DESTRUCTIVE_TOKEN="REMOVE_ALL_DATABASES"

confirm_fresh() {
    cat >&2 <<EOF

############################################################################
# DANGER: --fresh will DROP ALL application data in the configured database
#   (DATABASE_URL=${DATABASE_URL:-<read from .env at runtime>})
# This permanently deletes every conversation, message, pipeline run, and the
# entire arena leaderboard. It CANNOT be undone.
############################################################################
EOF
    # Non-interactive opt-in for automation: set MENO_FRESH_CONFIRM to the token.
    if [[ -n "${MENO_FRESH_CONFIRM:-}" ]]; then
        if [[ "$MENO_FRESH_CONFIRM" == "$DESTRUCTIVE_TOKEN" ]]; then
            echo "MENO_FRESH_CONFIRM matches; proceeding with the destructive wipe." >&2
            return 0
        fi
        echo "MENO_FRESH_CONFIRM is set but does not equal '$DESTRUCTIVE_TOKEN'. Aborting; nothing was touched." >&2
        exit 3
    fi
    # No TTY and no env opt-in: refuse rather than silently wipe in a script/CI.
    if [[ ! -t 0 ]]; then
        echo "Refusing --fresh in a non-interactive shell. To proceed deliberately, run with" >&2
        echo "  MENO_FRESH_CONFIRM=$DESTRUCTIVE_TOKEN ./scripts/run_backend.sh start --fresh" >&2
        exit 3
    fi
    printf 'Type %s to confirm the irreversible wipe (anything else aborts): ' "$DESTRUCTIVE_TOKEN" >&2
    read -r reply || reply=""
    if [[ "$reply" != "$DESTRUCTIVE_TOKEN" ]]; then
        echo "Confirmation did not match. Aborting; no data was touched." >&2
        exit 3
    fi
    echo "Confirmed. Proceeding with the destructive wipe." >&2
}

ensure_venv() {
    # Self-heal: if the tooling is missing (fresh clone, after a pull that
    # adds new entry points, partial sync), run uv sync once. This keeps
    # `./scripts/run_backend.sh start` working as the single entry point.
    if [[ -x "$MIGRATE_BIN" && -x "$RESET_BIN" && -x "$API_BIN" ]]; then
        return 0
    fi
    echo "Backend tooling is missing in .venv; running uv sync --all-groups --frozen..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        return 1
    fi
    (cd "$ROOT_DIR" && uv sync --all-groups --frozen)
    if [[ ! -x "$MIGRATE_BIN" || ! -x "$RESET_BIN" || ! -x "$API_BIN" ]]; then
        echo "uv sync finished but one of the expected binaries is still missing:"
        [[ -x "$MIGRATE_BIN" ]] || echo "  - $MIGRATE_BIN"
        [[ -x "$RESET_BIN"   ]] || echo "  - $RESET_BIN"
        [[ -x "$API_BIN"     ]] || echo "  - $API_BIN"
        return 1
    fi
}

pid_from_file() {
    [[ -f "$PID_FILE" ]] || return 1
    cat "$PID_FILE"
}

pid_is_alive() {
    local pid="${1:-}"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

pid_command() {
    local pid="${1:-}"
    ps -p "$pid" -o command= 2>/dev/null || true
}

is_managed_process() {
    local pid="${1:-}"
    local command
    command="$(pid_command "$pid")"
    if [[ -z "$command" ]]; then
        return 0
    fi
    [[ "$command" == *"$API_BIN"* || "$command" == *"meno-rag-api"* ]]
}

is_running() {
    local pid
    pid="$(pid_from_file)" || return 1
    pid_is_alive "$pid" && is_managed_process "$pid"
}

start() {
    if is_running; then
        echo "Meno RAG API is already running with PID $(cat "$PID_FILE"); restarting it."
        stop
    fi

    if [[ -f "$PID_FILE" ]]; then
        local stale_pid
        stale_pid="$(cat "$PID_FILE")"
        if pid_is_alive "$stale_pid"; then
            echo "PID file points to a non-managed live process ($stale_pid); leaving it alone."
        fi
        rm -f "$PID_FILE"
    fi

    ensure_venv

    if [[ "$FRESH" -eq 1 ]]; then
        echo "--fresh: wiping the application schema before migrations..."
        (cd "$ROOT_DIR" && "$RESET_BIN" --yes)
    fi

    echo "Running database bootstrap + migrations..."
    # Wrap in if/else so set -e doesn't abort before we print the --fresh hint.
    if (cd "$ROOT_DIR" && "$MIGRATE_BIN"); then
        :
    else
        rc=$?
        if [[ "$rc" -eq 2 && "$FRESH" -eq 0 ]]; then
            printf '\nHint: to wipe the database and start clean, run:\n  ./scripts/run_backend.sh start --fresh\n'
        fi
        return "$rc"
    fi

    echo "Starting Meno RAG API in the background..."
    echo "Direct backend: http://${APP_HOST}:${APP_PORT}"
    echo "Meno-Web proxy endpoint: http://<meno-web-host>:${MENO_WEB_PORT}/v1"
    echo "Logs: $LOG_FILE"

    printf "\n--- %s starting Meno RAG API ---\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$LOG_FILE"

    APP_HOST="$APP_HOST" APP_PORT="$APP_PORT" PYTHONUNBUFFERED=1 \
        nohup "$API_BIN" >>"$LOG_FILE" 2>&1 < /dev/null &

    echo "$!" >"$PID_FILE"
    disown "$!" 2>/dev/null || true

    sleep 1
    if is_running; then
        echo "Started with PID $(cat "$PID_FILE")."
    else
        echo "Failed to start. Last log lines:"
        tail -n 40 "$LOG_FILE" || true
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if ! is_running; then
        echo "Meno RAG API is not running."
        if [[ -f "$PID_FILE" ]]; then
            local stale_pid
            stale_pid="$(cat "$PID_FILE")"
            if pid_is_alive "$stale_pid"; then
                echo "PID file points to a non-managed live process ($stale_pid); leaving it alone."
            fi
        fi
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "Stopping Meno RAG API with PID $pid..."
    kill "$pid"

    for _ in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "Stopped."
            return 0
        fi
        sleep 1
    done

    echo "Process did not stop gracefully; sending SIGKILL."
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
}

status() {
    if is_running; then
        echo "Meno RAG API is running with PID $(cat "$PID_FILE")."
        echo "Direct backend: http://${APP_HOST}:${APP_PORT}"
        echo "Meno-Web proxy endpoint: http://<meno-web-host>:${MENO_WEB_PORT}/v1"
        echo "Logs: $LOG_FILE"
    else
        echo "Meno RAG API is not running."
        if [[ -f "$PID_FILE" ]]; then
            echo "Removing stale PID file: $PID_FILE"
            rm -f "$PID_FILE"
        fi
    fi
}

logs() {
    touch "$LOG_FILE"
    tail -f "$LOG_FILE"
}

case "${1:-start}" in
    start)
        if [[ "$FRESH" -eq 1 ]]; then confirm_fresh; fi
        start
        ;;
    stop)
        stop
        ;;
    restart)
        if [[ "$FRESH" -eq 1 ]]; then confirm_fresh; fi
        stop
        start
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    foreground)
        ensure_venv
        APP_HOST="$APP_HOST" APP_PORT="$APP_PORT" PYTHONUNBUFFERED=1 "$API_BIN"
        ;;
    *)
        echo "Usage: $0 [--fresh] [start|stop|restart|status|logs|foreground]"
        echo ""
        echo "  --fresh     Drop ALL application tables before starting (permanent,"
        echo "              only valid with start/restart). Prompts for the typed"
        echo "              confirmation '$DESTRUCTIVE_TOKEN'; in a non-interactive"
        echo "              shell set MENO_FRESH_CONFIRM=$DESTRUCTIVE_TOKEN to proceed."
        exit 2
        ;;
esac
