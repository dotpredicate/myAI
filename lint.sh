#!/bin/sh
set -eu
cd "$(dirname "$0")"

ruff check
mypy .
npm --prefix frontend run check
npm --prefix frontend run lint
