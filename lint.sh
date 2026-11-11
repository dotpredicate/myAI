#!/bin/sh
set -eu
cd "$(dirname "$0")"

ruff check backend tests
mypy --check-untyped-defs backend tests
npm --prefix frontend run check
npm --prefix frontend run lint
