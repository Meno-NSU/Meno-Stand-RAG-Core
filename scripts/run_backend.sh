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

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
    if is_running; then
        echo "Meno RAG API is already running with PID $(cat "$PID_FILE")."
        echo "Logs: $LOG_FILE"
        return 0
    fi

    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi

    echo "Starting Meno RAG API in the background..."
    echo "Direct backend: http://${APP_HOST}:${APP_PORT}"
    echo "Meno-Web proxy endpoint: http://<meno-web-host>:${MENO_WEB_PORT}/v1"
    echo "Logs: $LOG_FILE"

    if [[ ! -x "$API_BIN" ]]; then
        echo "Backend executable is missing: $API_BIN"
        echo "Run first: uv sync --all-groups --frozen"
        return 1
    fi

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
        rm -f "$PID_FILE"
        return 0
    fi

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
        start
        ;;
    stop)
        stop
        ;;
    restart)
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
        if [[ ! -x "$API_BIN" ]]; then
            echo "Backend executable is missing: $API_BIN"
            echo "Run first: uv sync --all-groups --frozen"
            exit 1
        fi
        APP_HOST="$APP_HOST" APP_PORT="$APP_PORT" PYTHONUNBUFFERED=1 "$API_BIN"
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status|logs|foreground]"
        exit 2
        ;;
esac
