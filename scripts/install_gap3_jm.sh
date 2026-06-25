#!/usr/bin/env sh
#
# Install GAP3 from jmichel7/gap3-jm without compiling.
#
# This script supports Linux and macOS only. It downloads the upstream source
# tree and wires the platform's bundled 64-bit executable into the repo-local
# .agent-runtime/bin/gap3 launcher:
#   - Linux x86_64: bin/gap.linux
#   - macOS x86_64/arm64: bin/gap.mac
#
# Windows note:
#   This script intentionally does not run on Windows. On Windows, download or
#   clone https://github.com/jmichel7/gap3-jm, edit bin\gap.cmd so GAP_DIR
#   points to the gap3-jm directory and GAP_PRG points to gapcyg.exe, then put
#   gap.cmd, or a copy renamed gap3.cmd, in a directory on PATH. Keep
#   bin\cygwin1.dll beside bin\gapcyg.exe. If strict binary bitness matters for
#   a Windows deployment, audit the upstream gapcyg.exe first.
#
# Usage:
#   sh scripts/install_gap3_jm.sh [options]
#
# Options:
#   --download-dir DIR  directory for temporary downloads
#   --force             replace an existing install directory or launcher
#   -h, --help          show this help
#
# Environment overrides:
#   GAP3JM_DOWNLOAD_DIR, GAP3JM_ARCHIVE_URL, GAP3_MEM, GAP_MEM

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
runtime_dir="$repo_root/.agent-runtime"
repo_url="https://github.com/jmichel7/gap3-jm"
install_dir="$runtime_dir/lib/gap3-jm"
bin_dir="$runtime_dir/bin"
download_dir="${GAP3JM_DOWNLOAD_DIR:-$runtime_dir/downloads}"
archive_url="${GAP3JM_ARCHIVE_URL:-}"
force=0
tmp_dir=""

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  sh scripts/install_gap3_jm.sh [options]

Options:
  --download-dir DIR  directory for temporary downloads
  --force             replace an existing install directory or launcher
  -h, --help          show this help

Environment overrides:
  GAP3JM_DOWNLOAD_DIR, GAP3JM_ARCHIVE_URL, GAP3_MEM, GAP_MEM
EOF
}

cleanup() {
  if [ -n "$tmp_dir" ] && [ -d "$tmp_dir" ]; then
    rm -rf "$tmp_dir"
  fi
}
trap cleanup EXIT HUP INT TERM

strip_trailing_slashes() {
  path=$1
  while [ "$path" != "/" ] && [ "${path%/}" != "$path" ]; do
    path=${path%/}
  done
  printf '%s\n' "$path"
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

[ -n "$install_dir" ] || die "install directory is empty"
[ -n "$bin_dir" ] || die "bin directory is empty"
[ -n "$download_dir" ] || die "download directory is empty"

install_dir=$(strip_trailing_slashes "$install_dir")
bin_dir=$(strip_trailing_slashes "$bin_dir")
download_dir=$(strip_trailing_slashes "$download_dir")

case "$install_dir" in
  /*) ;;
  *) die "install directory must be absolute: $install_dir" ;;
esac

case "$bin_dir" in
  /*) ;;
  *) die "bin directory must be absolute: $bin_dir" ;;
esac

case "$download_dir" in
  /*) ;;
  *) die "download directory must be absolute: $download_dir" ;;
esac

case "$install_dir" in
  /)
    die "refusing unsafe install directory: $install_dir"
    ;;
esac

detect_gap_prg() {
  os_name=$(uname -s 2>/dev/null || printf unknown)
  machine=$(uname -m 2>/dev/null || printf unknown)

  case "$os_name" in
    Linux)
      case "$machine" in
        x86_64|amd64)
          printf '%s\n' "gap.linux"
          ;;
        *)
          die "gap3-jm ships a no-compile Linux binary for x86_64 only; found $machine"
          ;;
      esac
      ;;
    Darwin)
      case "$machine" in
        x86_64|arm64)
          printf '%s\n' "gap.mac"
          ;;
        *)
          die "gap3-jm ships a no-compile macOS binary for x86_64/arm64 only; found $machine"
          ;;
      esac
      ;;
    *)
      die "unsupported OS: $os_name. This installer supports Linux and macOS only."
      ;;
  esac
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

download() {
  url=$1
  dest=$2

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  else
    die "need curl or wget to download $repo_url"
  fi
}

single_quote() {
  printf "%s" "$1" | sed "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/"
}

write_launcher() {
  launcher_path=$1
  gap_prg=$2
  quoted_install_dir=$(single_quote "$install_dir")
  quoted_gap_prg=$(single_quote "$gap_prg")

  cat > "$launcher_path" <<EOF
#!/usr/bin/env sh
GAP_DIR=$quoted_install_dir
GAP_PRG=$quoted_gap_prg
GAP_MEM="\${GAP3_MEM:-\${GAP_MEM:-512m}}"
exec "\$GAP_DIR/bin/\$GAP_PRG" -m "\$GAP_MEM" -l "\$GAP_DIR/lib/" -h "\$GAP_DIR/doc/" "\$@"
EOF
  chmod 755 "$launcher_path"
}

validate_source_tree() {
  src_dir=$1
  gap_prg=$2

  [ -d "$src_dir/lib" ] || die "downloaded tree is missing lib/"
  [ -d "$src_dir/doc" ] || die "downloaded tree is missing doc/"
  [ -f "$src_dir/bin/$gap_prg" ] || die "downloaded tree is missing bin/$gap_prg"
}

find_extracted_dir() {
  parent=$1
  found=""

  for candidate in "$parent"/*; do
    [ -d "$candidate" ] || continue
    if [ -n "$found" ]; then
      die "archive extracted more than one top-level directory"
    fi
    found=$candidate
  done

  [ -n "$found" ] || die "archive did not extract a source directory"
  printf '%s\n' "$found"
}

gap_prg=$(detect_gap_prg)
launcher_path=$bin_dir/gap3

if [ -e "$install_dir" ] && [ "$force" -ne 1 ]; then
  die "$install_dir already exists; rerun with --force to replace it"
fi

if [ -e "$launcher_path" ] && [ "$force" -ne 1 ]; then
  die "$launcher_path already exists; rerun with --force to replace it"
fi

need_command uname
need_command tar
need_command mktemp
need_command sed
need_command dirname

if [ -z "$archive_url" ]; then
  archive_url="https://codeload.github.com/jmichel7/gap3-jm/tar.gz/master"
fi

mkdir -p "$download_dir"
tmp_dir=$(mktemp -d "$download_dir/gap3-install.XXXXXX")
archive_path=$tmp_dir/gap3-jm.tar.gz
extract_dir=$tmp_dir/extract
mkdir -p "$extract_dir"

log "Installing GAP3 from $repo_url (master)"
log "Using bundled executable: bin/$gap_prg"
log "Downloading $archive_url"
download "$archive_url" "$archive_path"

log "Extracting archive"
tar -xzf "$archive_path" -C "$extract_dir"
src_dir=$(find_extracted_dir "$extract_dir")
validate_source_tree "$src_dir" "$gap_prg"

mkdir -p "$(dirname "$install_dir")"
mkdir -p "$bin_dir"

if [ -e "$install_dir" ]; then
  log "Replacing $install_dir"
  rm -rf "$install_dir"
fi

mv "$src_dir" "$install_dir"
chmod 755 "$install_dir/bin/$gap_prg"

log "Writing launcher $launcher_path"
write_launcher "$launcher_path" "$gap_prg"

log "Running GAP3 smoke check"
if ! printf 'Print(Size(SymmetricGroup(3)),"\\n"); quit;\n' | "$launcher_path" -q; then
  die "GAP3 smoke check failed"
fi

log "GAP3 is installed."
log "Run it with: $launcher_path"
log "For manual command-name use: export PATH='$bin_dir':\"\$PATH\""
