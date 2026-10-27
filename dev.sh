#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -d frontend/node_modules ]; then
  echo "frontend/node_modules is missing; run: npm --prefix frontend install" >&2
  exit 1
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required (util-linux)" >&2
  exit 1
fi

if ! command -v fastapi >/dev/null 2>&1; then
  echo "fastapi is not on PATH; activate the virtualenv first" >&2
  exit 1
fi

npm --prefix frontend run build || true

setsid npm --prefix frontend run dev &
frontend_pid=$!

setsid fastapi dev &
backend_pid=$!

cleanup() {
  trap - INT TERM EXIT
  kill -TERM -"$frontend_pid" -"$backend_pid" 2>/dev/null || true
  trap 'trap - INT TERM EXIT; kill -KILL -"$frontend_pid" -"$backend_pid" 2>/dev/null || true; exit 1' INT TERM
  wait "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait "$backend_pid" || true
