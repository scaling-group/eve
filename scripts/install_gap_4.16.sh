#!/usr/bin/env bash
set -euo pipefail

# Repo-local install:
#   - GAP tree: .agent-runtime/lib/gap-4.16.0
#   - launcher: .agent-runtime/bin/gap

gap_version="4.16.0"
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
runtime_dir="$repo_root/.agent-runtime"
install_dir="$runtime_dir/lib/gap-$gap_version"
launcher_dir="$runtime_dir/bin"
download_dir="${GAP416_DOWNLOAD_DIR:-$runtime_dir/downloads}"
archive_url="https://github.com/gap-system/gap/releases/download/v$gap_version/gap-$gap_version.tar.gz"
force=0

launcher="$launcher_dir/gap"

usage() {
  cat <<EOF
Usage:
  bash scripts/install_gap_4.16.sh [--download-dir DIR] [--force]

Installs GAP $gap_version to:
  $install_dir

Writes the gap launcher to:
  $launcher

EvE agents launched from this checkout receive this runtime bin directory on PATH:
  $launcher_dir

Downloads temporary files under:
  $download_dir

Manual shell use:
  $launcher
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --download-dir)
      [ "$#" -ge 2 ] || die "--download-dir needs a value"
      download_dir=$2
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "$download_dir" in
  /*) ;;
  *) die "download directory must be an absolute path: $download_dir" ;;
esac

case "$runtime_dir" in
  /*) ;;
  *) die "runtime directory must be an absolute path: $runtime_dir" ;;
esac

case "$install_dir" in
  /*) ;;
  *) die "install directory must be an absolute path: $install_dir" ;;
esac

case "$launcher_dir" in
  /*) ;;
  *) die "launcher directory must be an absolute path: $launcher_dir" ;;
esac

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) die "unsupported OS: $(uname -s). This script supports Linux and macOS." ;;
esac

for cmd in curl tar mktemp make cc sed; do
  need "$cmd"
done

if [ -e "$install_dir" ]; then
  [ "$force" = 1 ] || die "$install_dir already exists; rerun with --force"
  rm -rf "$install_dir"
fi

if [ -e "$launcher" ] || [ -L "$launcher" ]; then
  [ "$force" = 1 ] || die "$launcher already exists; rerun with --force"
  rm -f "$launcher"
fi

mkdir -p "$(dirname "$install_dir")" "$launcher_dir" "$download_dir"
tmp_dir=$(mktemp -d "$download_dir/gap-install.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

archive="$tmp_dir/gap-$gap_version.tar.gz"
extract_dir="$tmp_dir/extract"
mkdir -p "$extract_dir"

echo "Downloading $archive_url"
curl -fsSL "$archive_url" -o "$archive"

echo "Extracting GAP"
tar -xzf "$archive" -C "$extract_dir"
set -- "$extract_dir"/*
[ "$#" -eq 1 ] && [ -d "$1" ] || die "archive did not extract exactly one source directory"
src_dir=$1
mv "$src_dir" "$install_dir"

echo "Building GAP in $install_dir"
cd "$install_dir"
./configure
make

[ -x "$install_dir/gap" ] || die "build finished but $install_dir/gap is missing"

cat > "$launcher" <<EOF
#!/bin/sh
GAPROOT='$install_dir'
cd "\$GAPROOT" || exit 1
exec "\$GAPROOT/gap" -l "\$GAPROOT;" "\$@"
EOF
chmod 755 "$launcher"

echo "Running smoke check"
"$launcher" -q -c 'Print(Size(SymmetricGroup(3)),"\n"); QUIT;'

echo "GAP $gap_version installed."
echo "Run: $launcher"
echo "For manual command-name use: export PATH='$launcher_dir':\"\$PATH\""
