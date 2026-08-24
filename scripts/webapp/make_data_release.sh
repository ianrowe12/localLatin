#!/usr/bin/env bash
#
# make_data_release.sh — package the gitignored webapp data payload as a
# versioned tarball for a GitHub Release.
#
# The deploy host (/homes/ipro222/localLatin) is reachable only through the
# self-hosted Actions runner, and deploy/deploy.sh only git-pulls. The
# prediction CSVs and IG attribution artifacts the webapp reads are gitignored,
# so they travel as a release asset instead: this script builds it here, and
# deploy/deploy.sh downloads + verifies + unpacks it when DATA_RELEASE_TAG is
# set.
#
# EVERY member of the tarball is under runs/active/. That is a hard structural
# guarantee, verified below and re-verified on the host before extraction, that
# unpacking can never touch the reviewer feedback database — the production
# config puts it at data/feedback.db, outside the payload prefix. The canon
# text corpora (data/canon_{un,}labelled/) are host-resident and are checked by
# export_webapp_data.sh rather than shipped, for the same reason.
#
# Usage:
#   bash scripts/webapp/make_data_release.sh [--tag TAG] [--repo-root DIR] [--out-dir DIR]
#
# Defaults: TAG=data-$(date +%Y%m%d), repo-root=$PWD, out-dir=$PWD/dist
#
# Then:
#   gh release create "$TAG" \
#       "$OUT/locallatin-data-$TAG.tar.gz" \
#       "$OUT/locallatin-data-$TAG.tar.gz.sha256" \
#       --notes "..." --title "$TAG"
set -euo pipefail

TAG="data-$(date +%Y%m%d)"
REPO_ROOT="$(pwd)"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)       TAG="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
VARIANTS=(raw abtt sif sif_abtt)

info()  { printf '\033[1;34m[data-release]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[data-release]\033[0m %s\n' "$*" >&2; }

# --- Collect the payload -----------------------------------------------------
MEMBERS=()
for variant in "${VARIANTS[@]}"; do
    MEMBERS+=("runs/active/resubmit/unlabelled/unlabelled_predictions_${variant}.csv")
done
MEMBERS+=("runs/active/ig_examples/phase12f_examples.csv")
MEMBERS+=("runs/active/ig_examples/artifacts")

for member in "${MEMBERS[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${member}" ]]; then
        error "Missing payload member: ${REPO_ROOT}/${member}"
        exit 1
    fi
    case "${member}" in
        runs/active/*) ;;
        *) error "Payload member outside runs/active/: ${member}"; exit 1 ;;
    esac
done

npz_count="$(find "${REPO_ROOT}/runs/active/ig_examples/artifacts" -name '*.npz' | wc -l | tr -d ' ')"
if [[ "${npz_count}" -eq 0 ]]; then
    error "No .npz attribution artifacts found — refusing to publish an empty payload"
    exit 1
fi
info "Attribution artifacts: ${npz_count} .npz files"

mkdir -p "${OUT_DIR}"
TARBALL="${OUT_DIR}/locallatin-${TAG}.tar.gz"
CHECKSUM="${TARBALL}.sha256"
MANIFEST="${OUT_DIR}/locallatin-${TAG}.manifest.txt"

# --- Build -------------------------------------------------------------------
# gzip -1: the .npz members are already deflate-compressed, so higher levels
# buy almost nothing and cost minutes on a ~350 MB payload.
info "Building ${TARBALL} ..."
tar -cf - -C "${REPO_ROOT}" "${MEMBERS[@]}" | gzip -1 > "${TARBALL}"

# --- Verify every member stays under runs/active/ ----------------------------
info "Verifying archive member paths..."
while IFS= read -r entry; do
    case "${entry}" in
        runs/active/*) ;;
        *)
            error "Archive contains a member outside runs/active/: ${entry}"
            rm -f "${TARBALL}"
            exit 1
            ;;
    esac
    case "${entry}" in
        *..*)
            error "Archive contains a path traversal member: ${entry}"
            rm -f "${TARBALL}"
            exit 1
            ;;
    esac
done < <(tar -tzf "${TARBALL}")

tar -tzf "${TARBALL}" > "${MANIFEST}"

( cd "${OUT_DIR}" && sha256sum "$(basename "${TARBALL}")" > "$(basename "${CHECKSUM}")" )

info "Done."
info "  tarball:  ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"
info "  checksum: ${CHECKSUM}"
info "  manifest: ${MANIFEST} ($(wc -l < "${MANIFEST}" | tr -d ' ') entries)"
info ""
info "Publish with:"
info "  gh release create ${TAG} '${TARBALL}' '${CHECKSUM}' --title ${TAG} --notes 'webapp data payload'"
info "Then set the repo variable so deploy/deploy.sh picks it up:"
info "  gh variable set DATA_RELEASE_TAG --body ${TAG}"
