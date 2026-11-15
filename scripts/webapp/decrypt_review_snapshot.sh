#!/usr/bin/env bash
# OPENSSL may point to Homebrew OpenSSL 3 when macOS's LibreSSL lacks PBKDF2.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 <artifact-dir> <private-key.pem> <out-dir>" >&2
  exit 2
fi
exec python3 "$(dirname -- "$0")/review_snapshot.py" decrypt "$@"
