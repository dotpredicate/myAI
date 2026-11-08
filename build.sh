#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -d frontend/node_modules ]; then
  echo "frontend/node_modules is missing; run: npm --prefix frontend install" >&2
  exit 1
fi

npm --prefix frontend run build
