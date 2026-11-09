#!/bin/sh
set -eu
cd "$(dirname "$0")"
ROOT=$(pwd -P)
TARGET="$ROOT/frontend/tui/src/main.ts"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: ./install.sh [--uninstall]

Installs the myAI terminal client as the "myai" command by linking
frontend/tui/src/main.ts into ~/.local/bin (or $XDG_BIN_HOME).
EOF
    exit 0
    ;;
  --uninstall)
    rm -f "$BIN_DIR/myai"
    echo "Removed $BIN_DIR/myai"
    exit 0
    ;;
esac

if [ ! -f "$TARGET" ]; then
  echo "install.sh: $TARGET is missing" >&2
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

chmod +x "$TARGET"
mkdir -p "$BIN_DIR"
ln -sfn "$TARGET" "$BIN_DIR/myai"

echo "Installed $BIN_DIR/myai -> frontend/tui/src/main.ts"
echo "Run: myai --help"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Note: $BIN_DIR is not on PATH; add it or call myai by full path." ;;
esac
