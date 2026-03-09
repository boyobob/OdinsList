#!/usr/bin/env bash
set -euo pipefail

REPO="${ODINSLIST_REPO:-boyobob/OdinsList}"
INSTALL_BIN="$HOME/.local/bin"
INSTALL_DIR="$HOME/.odinslist"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux) OS_TAG="linux" ;;
  Darwin) OS_TAG="darwin" ;;
  *) echo "Error: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  x86_64) OS_ARCH="x64" ;;
  aarch64 | arm64) OS_ARCH="arm64" ;;
  *) echo "Error: Unsupported architecture: $ARCH"; exit 1 ;;
esac

PLATFORM="${OS_TAG}-${OS_ARCH}"

if [ -n "${1:-}" ]; then
  VERSION="$1"
else
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -1 | cut -d'"' -f4)"
fi

if [ -z "$VERSION" ]; then
  echo "Error: Could not determine latest version."
  exit 1
fi

TARBALL="odinslist-${PLATFORM}.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$TARBALL"

echo "Installing odinslist $VERSION ($PLATFORM)..."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "  Downloading $TARBALL..."
curl -fsSL "$URL" -o "$TMP/$TARBALL"

echo "  Extracting..."
tar -xzf "$TMP/$TARBALL" -C "$TMP"

if [ ! -f "$TMP/odinslist" ]; then
  echo "Error: Archive missing odinslist binary."
  exit 1
fi

if [ ! -d "$TMP/backend" ]; then
  echo "Error: Archive missing backend/ directory."
  exit 1
fi

mkdir -p "$INSTALL_BIN"
cp "$TMP/odinslist" "$INSTALL_BIN/odinslist"
chmod +x "$INSTALL_BIN/odinslist"

mkdir -p "$INSTALL_DIR"
rm -rf "$INSTALL_DIR/backend"
cp -R "$TMP/backend" "$INSTALL_DIR/backend"

echo "  Binary:  $INSTALL_BIN/odinslist"
echo "  Backend: $INSTALL_DIR/backend/"

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_BIN"; then
  echo ""
  echo "  WARNING: $INSTALL_BIN is not in your PATH."

  SHELL_NAME="$(basename "${SHELL:-}")"
  case "$SHELL_NAME" in
    bash) RC="$HOME/.bashrc" ;;
    zsh) RC="$HOME/.zshrc" ;;
    fish) RC="$HOME/.config/fish/config.fish" ;;
    *) RC="" ;;
  esac

  if [ -n "$RC" ]; then
    echo ""
    read -rp "  Add it to $RC? [Y/n] " answer
    answer="${answer:-Y}"
    if [[ "$answer" =~ ^[Yy] ]]; then
      if [ "$SHELL_NAME" = "fish" ]; then
        if ! grep -Fqx "fish_add_path $INSTALL_BIN" "$RC" 2>/dev/null; then
          echo "fish_add_path $INSTALL_BIN" >> "$RC"
        fi
      else
        if ! grep -Fqx 'export PATH="$HOME/.local/bin:$PATH"' "$RC" 2>/dev/null; then
          echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
        fi
      fi
      echo "  Added! Restart your shell or run: source $RC"
    else
      echo "  Add this to your shell config manually:"
      echo '    export PATH="$HOME/.local/bin:$PATH"'
    fi
  fi
fi

echo ""
echo "Success: odinslist $VERSION installed."
echo "Run 'odinslist' to get started."
