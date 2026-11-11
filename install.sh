#!/bin/sh
set -eu
cd "$(dirname "$0")"
ROOT=$(pwd -P)
TUI="$ROOT/frontend/tui/src/main.ts"
SERVER="$ROOT/myai-server"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: ./install.sh [--uninstall]

Installs the myAI terminal client as "myai" and the background server as
"myai-server".
EOF
    exit 0
    ;;
  --uninstall)
    rm -f "$BIN_DIR/myai" "$BIN_DIR/myai-server"
    echo "Removed $BIN_DIR/myai and $BIN_DIR/myai-server"
    exit 0
    ;;
esac

if [ ! -f "$TUI" ] || [ ! -f "$SERVER" ]; then
  echo "install.sh: expected $TUI and $SERVER" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "install.sh: node >= 22.18 is required" >&2
  exit 1
fi

node -e '
const [major, minor] = process.versions.node.split(".").map(Number);
if (major < 22 || (major === 22 && minor < 18)) {
  console.error(`install.sh: node >= 22.18 is required, found ${process.version}`);
  process.exit(1);
}
'

chmod +x "$TUI" "$SERVER"
mkdir -p "$BIN_DIR"
ln -sfn "$TUI" "$BIN_DIR/myai"
ln -sfn "$SERVER" "$BIN_DIR/myai-server"

echo "Installed $BIN_DIR/myai -> frontend/tui/src/main.ts"
echo "Installed $BIN_DIR/myai-server -> myai-server"
echo "Run: myai --help"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Note: $BIN_DIR is not on PATH; add it or call myai by full path." ;;
esac
