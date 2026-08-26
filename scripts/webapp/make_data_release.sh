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
# SHARDING: a GitHub Release asset may not exceed 2 GB, and after the
# full-corpus attribution run (issue #84) the payload is several times that.
# --shard-bytes N splits it into `locallatin-<TAG>.partNN.tar.gz` parts, each
# below N, by greedily bin-packing the payload files (never splitting a file
# across parts, so every part is independently a valid tar). The parts together
# hold exactly the members a single-tarball build would, and
# `locallatin-<TAG>.parts.txt` lists them in order. Unsharded output is
# unchanged and is still what a small payload produces.
#
# Usage:
#   bash scripts/webapp/make_data_release.sh [--tag TAG] [--repo-root DIR] \
#        [--out-dir DIR] [--shard-bytes N]
#
# Defaults: TAG=data-$(date +%Y%m%d), repo-root=$PWD, out-dir=$PWD/dist,
#           shard-bytes=0 (one tarball)
#
# Then:
#   gh release create "$TAG" "$OUT"/locallatin-"$TAG"*.tar.gz \
#       "$OUT"/locallatin-"$TAG"*.sha256 "$OUT/locallatin-$TAG.parts.txt" \
#       --notes "..." --title "$TAG"
set -euo pipefail

TAG="data-$(date +%Y%m%d)"
REPO_ROOT="$(pwd)"
OUT_DIR=""
SHARD_BYTES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)          TAG="$2"; shift 2 ;;
        --repo-root)    REPO_ROOT="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --shard-bytes)  SHARD_BYTES="$2"; shift 2 ;;
        -h|--help)      sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! [[ "${SHARD_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "--shard-bytes must be a non-negative integer, got '${SHARD_BYTES}'" >&2
    exit 2
fi

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

# Query-query cosine matrices (issue #95), one ~10 MB .npz per model. Added by
# glob rather than by name: a deployment may serve a subset of the six models,
# and a hard-coded per-model list would fail the existence check below on any
# host that does. Absent entirely, reviewer directories simply cannot be
# scored, which is a degraded-but-working webapp -- so this is not fatal here.
shopt -s nullglob
QQ_MATRICES=("${REPO_ROOT}"/runs/active/resubmit/unlabelled/qq_sim_*.npz)
shopt -u nullglob
if [[ "${#QQ_MATRICES[@]}" -eq 0 ]]; then
    error "No qq_sim_*.npz found — reviewer-created directories will not be scorable."
    error "Build them with: python scripts/resubmit/build_qq_matrices.py"
else
    for matrix in "${QQ_MATRICES[@]}"; do
        MEMBERS+=("runs/active/resubmit/unlabelled/$(basename "${matrix}")")
    done
    info "Query-query matrices: ${#QQ_MATRICES[@]}"
fi

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
MANIFEST="${OUT_DIR}/locallatin-${TAG}.manifest.txt"
PARTS_LIST="${OUT_DIR}/locallatin-${TAG}.parts.txt"

# Verify one built archive: every member under runs/active/, no traversal.
# Called on each part, so a bad member never survives in a sharded build either.
verify_archive() {
    local archive="$1"
    while IFS= read -r entry; do
        case "${entry}" in
            runs/active/*) ;;
            *)
                error "Archive contains a member outside runs/active/: ${entry}"
                rm -f "${archive}"
                exit 1
                ;;
        esac
        case "${entry}" in
            *..*)
                error "Archive contains a path traversal member: ${entry}"
                rm -f "${archive}"
                exit 1
                ;;
        esac
    done < <(tar -tzf "${archive}")
}

# gzip -1: the .npz members are already deflate-compressed, so higher levels buy
# almost nothing and cost minutes on a multi-GB payload.
build_archive() {  # build_archive OUT_PATH FILE_LIST_PATH
    tar -cf - -C "${REPO_ROOT}" --files-from="$2" | gzip -1 > "$1"
}

# Expand the member list to concrete files, so the packer can bin-pack them and
# so both code paths pack exactly the same set.
FILE_LIST="$(mktemp)"
SIZED_LIST="$(mktemp)"
trap 'rm -f "${FILE_LIST}" "${SIZED_LIST}" "${FILE_LIST}".part.*' EXIT
# "<size>\t<path>", so the packer never has to stat 40k files one subprocess at
# a time.
( cd "${REPO_ROOT}" && find "${MEMBERS[@]}" -type f -printf '%s\t%p\n' ) \
    | LC_ALL=C sort -k2 > "${SIZED_LIST}"
cut -f2- < "${SIZED_LIST}" > "${FILE_LIST}"
file_count="$(wc -l < "${FILE_LIST}" | tr -d ' ')"
info "Payload files: ${file_count}"

PARTS=()
if [[ "${SHARD_BYTES}" -eq 0 ]]; then
    TARBALL="${OUT_DIR}/locallatin-${TAG}.tar.gz"
    info "Building ${TARBALL} ..."
    build_archive "${TARBALL}" "${FILE_LIST}"
    info "Verifying archive member paths..."
    verify_archive "${TARBALL}"
    PARTS=("${TARBALL}")
    tar -tzf "${TARBALL}" > "${MANIFEST}"
else
    # Greedy bin-pack on *uncompressed* size. gzip -1 over .npz members shrinks
    # them by only a few percent, so uncompressed size is a safe over-estimate
    # of the part size -- parts land under the limit, never over it.
    info "Sharding at ${SHARD_BYTES} bytes per part ..."
    LC_ALL=C awk -F'\t' -v limit="${SHARD_BYTES}" -v prefix="${FILE_LIST}.part." '
        {
            size = $1 + 0
            path = $2
            if (size > limit) {
                printf("single file %s (%d bytes) exceeds --shard-bytes %d\n", path, size, limit) > "/dev/stderr"
                exit 3
            }
            if (part == 0 || used + size > limit) { part++; used = 0 }
            print path >> (prefix sprintf("%02d", part))
            used += size
        }
    ' "${SIZED_LIST}"
    for part_list in "${FILE_LIST}".part.*; do
        [[ -e "${part_list}" ]] || { error "Sharding produced no parts"; exit 1; }
        suffix="${part_list##*.}"
        part="${OUT_DIR}/locallatin-${TAG}.part${suffix}.tar.gz"
        info "Building ${part} ($(wc -l < "${part_list}" | tr -d ' ') files) ..."
        build_archive "${part}" "${part_list}"
        verify_archive "${part}"
        PARTS+=("${part}")
    done
    : > "${MANIFEST}"
    for part in "${PARTS[@]}"; do
        tar -tzf "${part}" >> "${MANIFEST}"
    done
fi

# Every part gets its own sidecar, so a download can be verified part by part.
for part in "${PARTS[@]}"; do
    ( cd "${OUT_DIR}" && sha256sum "$(basename "${part}")" > "$(basename "${part}").sha256" )
done

# The parts list is deploy/deploy.sh's discovery mechanism: present means
# sharded, absent means the single-asset layout. So an unsharded build must NOT
# write one -- its published asset set stays exactly what it was before
# sharding existed, and the deploy path it exercises is the one every already
# published release uses.
if [[ "${#PARTS[@]}" -gt 1 ]] || [[ "${SHARD_BYTES}" -gt 0 ]]; then
    : > "${PARTS_LIST}"
    for part in "${PARTS[@]}"; do
        basename "${part}" >> "${PARTS_LIST}"
    done
else
    rm -f "${PARTS_LIST}"
fi

# The parts must reconstruct exactly the payload, no more and no less.
packed="$(LC_ALL=C sort -u < "${MANIFEST}" | grep -v '/$' | wc -l | tr -d ' ')"
if [[ "${packed}" -ne "${file_count}" ]]; then
    error "Packed ${packed} files but the payload has ${file_count}"
    exit 1
fi

info "Done."
for part in "${PARTS[@]}"; do
    info "  part:     ${part} ($(du -h "${part}" | cut -f1))  + .sha256"
done
if [[ -f "${PARTS_LIST}" ]]; then
    info "  parts:    ${PARTS_LIST} (${#PARTS[@]} archive(s)) — publish this too"
fi
info "  manifest: ${MANIFEST} ($(wc -l < "${MANIFEST}" | tr -d ' ') entries)"
info ""
info "Publish with:"
info "  gh release create ${TAG} ${OUT_DIR}/locallatin-${TAG}*.tar.gz ${OUT_DIR}/locallatin-${TAG}*.sha256 '${PARTS_LIST}' --title ${TAG} --notes 'webapp data payload'"
info "Then set the repo variable so deploy/deploy.sh picks it up:"
info "  gh variable set DATA_RELEASE_TAG --body ${TAG}"
